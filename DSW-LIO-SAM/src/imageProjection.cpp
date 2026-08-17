// 点云投影/预处理：接收点云→range image→提取语义标签

#include "utility.hpp"
#include "dsw_lio_sam/msg/cloud_info.hpp"
#include <condition_variable>

struct VelodynePointXYZIRT  // 点云类型定义  Velodyne格式
{
    PCL_ADD_POINT4D                  // x,y,z  三维坐标
    PCL_ADD_INTENSITY;               // intensity  反射强度
    uint16_t ring;                   // ring  激光线编号
    float time;                      // time  扫描时间
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW  // 
} EIGEN_ALIGN16;
POINT_CLOUD_REGISTER_POINT_STRUCT (VelodynePointXYZIRT,
    (float, x, x) (float, y, y) (float, z, z) (float, intensity, intensity)
    (uint16_t, ring, ring) (float, time, time)
)

struct OusterPointXYZIRT {
    PCL_ADD_POINT4D;
    float intensity;
    uint32_t t;
    uint16_t reflectivity;
    uint8_t ring;
    uint16_t noise;
    uint32_t range;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
} EIGEN_ALIGN16;
POINT_CLOUD_REGISTER_POINT_STRUCT(OusterPointXYZIRT,
    (float, x, x) (float, y, y) (float, z, z) (float, intensity, intensity)
    (uint32_t, t, t) (uint16_t, reflectivity, reflectivity)
    (uint8_t, ring, ring) (uint16_t, noise, noise) (uint32_t, range, range)
)

// Use the Velodyne point format as a common representation
using PointXYZIRT = VelodynePointXYZIRT;

const int queueLength = 2000;

class ImageProjection : public ParamServer  // ImageProjection类继承ParamServer
{
private:

    std::mutex imuLock;
    std::mutex odoLock;

    std::mutex semanticLock;

    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr subLaserCloud;
    rclcpp::CallbackGroup::SharedPtr callbackGroupLidar;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubLaserCloud;

    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubExtractedCloud;
    rclcpp::Publisher<dsw_lio_sam::msg::CloudInfo>::SharedPtr pubLaserCloudInfo;

    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr subImu;
    rclcpp::CallbackGroup::SharedPtr callbackGroupImu;
    std::deque<sensor_msgs::msg::Imu> imuQueue;

    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr subOdom;
    rclcpp::CallbackGroup::SharedPtr callbackGroupOdom;
    std::deque<nav_msgs::msg::Odometry> odomQueue;

    std::deque<sensor_msgs::msg::PointCloud2> cloudQueue;
    sensor_msgs::msg::PointCloud2 currentCloudMsg;

    std::vector<double> imuTime = std::vector<double>(queueLength);
    std::vector<double> imuRotX = std::vector<double>(queueLength);
    std::vector<double> imuRotY = std::vector<double>(queueLength);
    std::vector<double> imuRotZ = std::vector<double>(queueLength);

    int imuPointerCur;
    bool firstPointFlag;
    Eigen::Affine3f transStartInverse;

    pcl::PointCloud<PointXYZIRT>::Ptr laserCloudIn;
    pcl::PointCloud<OusterPointXYZIRT>::Ptr tmpOusterCloudIn;
    pcl::PointCloud<PointType>::Ptr   fullCloud;
    pcl::PointCloud<PointType>::Ptr   extractedCloud;

    int ringFlag = 0;
    int deskewFlag;
    cv::Mat rangeMat;

    bool odomDeskewFlag;
    float odomIncreX;
    float odomIncreY;
    float odomIncreZ;

    dsw_lio_sam::msg::CloudInfo cloudInfo;
    double timeScanCur;
    double timeScanEnd;
    std_msgs::msg::Header cloudHeader;

    vector<int> columnIdnCountVec;

    // ===== DSW-LIO-SAM: 语义相关部分 =====
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr subSemanticCloud;
    rclcpp::CallbackGroup::SharedPtr callbackGroupSemantic;
    std::deque<sensor_msgs::msg::PointCloud2> semanticCloudQueue;  // 语义缓存队列
    std::condition_variable semanticCv;
    pcl::PointCloud<PointTypeL>::Ptr semanticCloudIn;              // 带语义标签的点云  label来自Cylinder3D
    std::vector<uint32_t> pointSemanticLabels;                     // 当前帧每个点的语义标签
    std::vector<float> pointSemanticWeights;                       // 当前帧每个点的语义权重  核心创新！！！
    std::vector<uint32_t> fullCloudSemanticLabels;
    std::vector<float> fullCloudSemanticWeights;
    static constexpr int64_t semanticTimeToleranceNs = 1000000LL;  // 1 ms
    static constexpr int semanticWaitTimeoutMs = 5000;
    static constexpr size_t semanticQueueCapacity = kPointCloudQosDepth;
    static constexpr float semanticNearestSqDist = 0.05f * 0.05f;
    // ===== DSW-LIO-SAM 语义部分结束 =====

public:
    ImageProjection(const rclcpp::NodeOptions & options) :
            ParamServer("dsw_lio_sam_imageProjection", options), deskewFlag(0)
    {
        callbackGroupLidar = create_callback_group(
            rclcpp::CallbackGroupType::MutuallyExclusive);
        callbackGroupImu = create_callback_group(
            rclcpp::CallbackGroupType::MutuallyExclusive);
        callbackGroupOdom = create_callback_group(
            rclcpp::CallbackGroupType::MutuallyExclusive);

        auto lidarOpt = rclcpp::SubscriptionOptions();
        lidarOpt.callback_group = callbackGroupLidar;
        auto imuOpt = rclcpp::SubscriptionOptions();
        imuOpt.callback_group = callbackGroupImu;
        auto odomOpt = rclcpp::SubscriptionOptions();
        odomOpt.callback_group = callbackGroupOdom;

        subImu = create_subscription<sensor_msgs::msg::Imu>(
            imuTopic, qos_imu,
            std::bind(&ImageProjection::imuHandler, this, std::placeholders::_1),
            imuOpt);
        subOdom = create_subscription<nav_msgs::msg::Odometry>(
            odomTopic + "_incremental", qos_imu,
            std::bind(&ImageProjection::odometryHandler, this, std::placeholders::_1),
            odomOpt);
        subLaserCloud = create_subscription<sensor_msgs::msg::PointCloud2>(
            pointCloudTopic, qos_lidar,
            std::bind(&ImageProjection::cloudHandler, this, std::placeholders::_1),
            lidarOpt);


        pubExtractedCloud = create_publisher<sensor_msgs::msg::PointCloud2>(
            "dsw_lio_sam/deskew/cloud_deskewed", qos_point_cloud_pipeline);
        pubLaserCloudInfo = create_publisher<dsw_lio_sam::msg::CloudInfo>(
            "dsw_lio_sam/deskew/cloud_info", qos_point_cloud_pipeline);

        // ===== DSW-LIO-SAM: 初始化语义模块 =====
        if (semanticEnabled) {  // 构造函数：if (semanticEnabled)如果打开，创建订阅：subSemanticCloud，订阅：Cylinder3D输出topic
            callbackGroupSemantic = create_callback_group(
                rclcpp::CallbackGroupType::MutuallyExclusive);
            auto semanticOpt = rclcpp::SubscriptionOptions();
            semanticOpt.callback_group = callbackGroupSemantic;

            subSemanticCloud = create_subscription<sensor_msgs::msg::PointCloud2>(
                semanticCloudTopic, qos_point_cloud_pipeline,
                std::bind(&ImageProjection::semanticCloudHandler, this, std::placeholders::_1),
                semanticOpt);
        }
        // ===== DSW-LIO-SAM 语义初始化结束 =====

        allocateMemory();
        resetParameters();

        pcl::console::setVerbosityLevel(pcl::console::L_ERROR);
    }

    void allocateMemory()
    {
        laserCloudIn.reset(new pcl::PointCloud<PointXYZIRT>());
        tmpOusterCloudIn.reset(new pcl::PointCloud<OusterPointXYZIRT>());

        semanticCloudIn.reset(new pcl::PointCloud<PointTypeL>());  // DSW-LIO-SAM

        fullCloud.reset(new pcl::PointCloud<PointType>());
        extractedCloud.reset(new pcl::PointCloud<PointType>());

        fullCloud->points.resize(N_SCAN*Horizon_SCAN);

        cloudInfo.start_ring_index.assign(N_SCAN, 0);
        cloudInfo.end_ring_index.assign(N_SCAN, 0);

        cloudInfo.point_col_ind.assign(N_SCAN*Horizon_SCAN, 0);
        cloudInfo.point_range.assign(N_SCAN*Horizon_SCAN, 0);

        resetParameters();
    }

    void resetParameters()
    {
        laserCloudIn->clear();
        extractedCloud->clear();
        cloudInfo.point_semantic_labels.clear();
        cloudInfo.point_semantic_weights.clear();
        // reset range matrix for range image projection
        rangeMat = cv::Mat(N_SCAN, Horizon_SCAN, CV_32F, cv::Scalar::all(FLT_MAX));
        fullCloudSemanticLabels.assign(N_SCAN * Horizon_SCAN, static_cast<uint32_t>(SemanticLabel::UNKNOWN));
        fullCloudSemanticWeights.assign(N_SCAN * Horizon_SCAN, 1.0f);

        imuPointerCur = 0;
        firstPointFlag = true;
        odomDeskewFlag = false;

        for (int i = 0; i < queueLength; ++i)
        {
            imuTime[i] = 0;
            imuRotX[i] = 0;
            imuRotY[i] = 0;
            imuRotZ[i] = 0;
        }
        columnIdnCountVec.assign(N_SCAN, 0);
    }

    ~ImageProjection(){}

    void imuHandler(const sensor_msgs::msg::Imu::SharedPtr imuMsg)
    {
        sensor_msgs::msg::Imu thisImu = imuConverter(*imuMsg);

        std::lock_guard<std::mutex> lock1(imuLock);
        imuQueue.push_back(thisImu);

        // debug IMU data
        // cout << std::setprecision(6);
        // cout << "IMU acc: " << endl;
        // cout << "x: " << thisImu.linear_acceleration.x << 
        //       ", y: " << thisImu.linear_acceleration.y << 
        //       ", z: " << thisImu.linear_acceleration.z << endl;
        // cout << "IMU gyro: " << endl;
        // cout << "x: " << thisImu.angular_velocity.x << 
        //       ", y: " << thisImu.angular_velocity.y << 
        //       ", z: " << thisImu.angular_velocity.z << endl;
        // double imuRoll, imuPitch, imuYaw;
        // tf2::Quaternion orientation;
        // tf2::fromMsg(thisImu.orientation, orientation);
        // tf2::Matrix3x3(orientation).getRPY(imuRoll, imuPitch, imuYaw);
        // cout << "IMU roll pitch yaw: " << endl;
        // cout << "roll: " << imuRoll << ", pitch: " << imuPitch << ", yaw: " << imuYaw << endl << endl;
    }

    void odometryHandler(const nav_msgs::msg::Odometry::SharedPtr odometryMsg)
    {
        std::lock_guard<std::mutex> lock2(odoLock);
        odomQueue.push_back(*odometryMsg);
    }

    // ===== DSW-LIO-SAM: 语义点云回调 =====
    void semanticCloudHandler(const sensor_msgs::msg::PointCloud2::SharedPtr semanticCloudMsg)  // 语义输入入口：收到Cylinder3D语义点云，存入semanticCloudQueue
    {
        {
            std::lock_guard<std::mutex> lock(semanticLock);
            semanticCloudQueue.push_back(*semanticCloudMsg);
            while (semanticCloudQueue.size() > semanticQueueCapacity)
                semanticCloudQueue.pop_front();
        }
        semanticCv.notify_all();
    }

    bool cacheSemanticCloud()  // 最关键函数
    {
        if (!semanticEnabled) {
            int cloudSize = laserCloudIn->points.size();
            pointSemanticLabels.assign(cloudSize, static_cast<uint32_t>(SemanticLabel::UNKNOWN));  // 初始化
            pointSemanticWeights.assign(cloudSize, 1.0f);
            return true;
        }
        
        int cloudSize = laserCloudIn->points.size();
        pointSemanticLabels.assign(cloudSize, static_cast<uint32_t>(SemanticLabel::UNKNOWN));
        pointSemanticWeights.assign(cloudSize, 1.0f);
        
        sensor_msgs::msg::PointCloud2 semanticMsg;
        const int64_t targetStampNs = stampToNanoseconds(cloudHeader.stamp);
        {
            std::unique_lock<std::mutex> lock(semanticLock);

            auto try_pop_synced_semantic = [&]() -> bool {
                semanticCloudQueue.erase(
                    std::remove_if(
                        semanticCloudQueue.begin(), semanticCloudQueue.end(),
                        [&](const sensor_msgs::msg::PointCloud2& queued) {
                            return stampToNanoseconds(queued.header.stamp) <
                                   targetStampNs - semanticTimeToleranceNs;
                        }),
                    semanticCloudQueue.end());

                const auto bestIndex = findPointCloudByTimestamp(
                    semanticCloudQueue, targetStampNs, semanticTimeToleranceNs);
                if (!bestIndex)
                    return false;

                semanticMsg = semanticCloudQueue[*bestIndex];
                semanticCloudQueue.erase(
                    semanticCloudQueue.begin(),
                    semanticCloudQueue.begin() + static_cast<std::ptrdiff_t>(*bestIndex) + 1);
                return true;
            };

            bool hasSyncedSemantic = try_pop_synced_semantic();
            if (!hasSyncedSemantic)
            {
                const auto deadline = std::chrono::steady_clock::now() +
                    std::chrono::milliseconds(semanticWaitTimeoutMs);
                while (!hasSyncedSemantic && std::chrono::steady_clock::now() < deadline)
                {
                    semanticCv.wait_until(lock, deadline);
                    hasSyncedSemantic = try_pop_synced_semantic();
                }
            }

            if (!hasSyncedSemantic)
            {
                int64_t bestDiffNs = std::numeric_limits<int64_t>::max();
                for (const auto& queuedSemantic : semanticCloudQueue)
                {
                    const int64_t stampNs = stampToNanoseconds(queuedSemantic.header.stamp);
                    const int64_t diffNs = stampNs >= targetStampNs
                        ? stampNs - targetStampNs
                        : targetStampNs - stampNs;
                    bestDiffNs = std::min(bestDiffNs, diffNs);
                }

                if (semanticCloudQueue.empty())
                {
                    RCLCPP_WARN_THROTTLE(
                        get_logger(), *get_clock(), 2000,
                        "No semantic cloud within 1 ms for lidar stamp %lld; queue is empty after waiting %.1fs",
                        static_cast<long long>(targetStampNs), semanticWaitTimeoutMs / 1000.0);
                }
                else
                {
                    RCLCPP_WARN_THROTTLE(
                        get_logger(), *get_clock(), 2000,
                        "No semantic cloud within 1 ms for lidar stamp %lld after waiting %.1fs; best diff %.3fms",
                        static_cast<long long>(targetStampNs), semanticWaitTimeoutMs / 1000.0,
                        bestDiffNs / 1000000.0);
                }
                return !requireSemanticCloud;
            }
        }
        
        // 取时间最近的一帧语义点云
        pcl::fromROSMsg(semanticMsg, *semanticCloudIn);  // 时间同步
        
        // 点数相同：按索引匹配 - 只在点云完全对齐时有效  索引匹配
        // 点数不同：对每个原始点，在语义点云中找最近邻   KDTree最近邻搜索
        // int semanticSize = semanticCloudIn->points.size();
        // if (semanticSize == 0) {
        //     if (requireSemanticCloud) {
        //         RCLCPP_WARN_THROTTLE(
        //             get_logger(), *get_clock(), 2000,
        //             "Semantic cloud is required but the synced semantic cloud is empty.");
        //         return false;
        //     }
        //     return true;
        // }

        // if (semanticSize == cloudSize)  // 索引匹配  假设LiDAR[i]与Semantic[i]是一一对应的
        // {
        //     int matchedCount = 0;
        //     for (int i = 0; i < cloudSize; ++i) {
        //         const auto& lidarPoint = laserCloudIn->points[i];
        //         const auto& semanticPoint = semanticCloudIn->points[i];
        //         const float dx = lidarPoint.x - semanticPoint.x;
        //         const float dy = lidarPoint.y - semanticPoint.y;
        //         const float dz = lidarPoint.z - semanticPoint.z;
        //         const float sqDist = dx * dx + dy * dy + dz * dz;
        //         if (std::isfinite(sqDist) && sqDist <= semanticNearestSqDist) {  // semanticNearestSqDist = 0.05f * 0.05f; 即距离<5cm才认为索引匹配成功
        //             const uint32_t label = semanticPoint.label;
        //             pointSemanticLabels[i] = label;
        //             pointSemanticWeights[i] = computeSemanticWeight(
        //                 static_cast<SemanticLabel>(label),
        //                 semanticWeightAlpha);
        //             ++matchedCount;
        //         }
        //     }

        //     const float semanticCoverage =
        //         cloudSize > 0 ? static_cast<float>(matchedCount) / static_cast<float>(cloudSize) : 0.0f;
        //     if (semanticCoverage >= minimumSemanticCoverage) {
        //         return true;
        //     }

        //     RCLCPP_WARN_THROTTLE(
        //         get_logger(), *get_clock(), 2000,
        //         "Indexed semantic cloud failed geometry coverage %.3f < %.3f. Falling back to nearest-neighbor labels.",
        //         semanticCoverage, minimumSemanticCoverage);
        //     pointSemanticLabels.assign(cloudSize, static_cast<uint32_t>(SemanticLabel::UNKNOWN));
        //     pointSemanticWeights.assign(cloudSize, 1.0f);
        // }

        // if (semanticSize != cloudSize) {  // KDTree 最近邻搜索  情况1：点数不同
        //     RCLCPP_WARN_THROTTLE(
        //         get_logger(), *get_clock(), 2000,
        //         "Semantic cloud size mismatch: lidar=%d semantic=%d. Falling back to nearest-neighbor labels.",
        //         cloudSize, semanticSize);
        // }
        // // 建立过程
        // pcl::PointCloud<PointType>::Ptr semanticSearchCloud(new pcl::PointCloud<PointType>());  // 首先建立一个搜索点云
        // std::vector<int> semanticIndexMap;
        // semanticSearchCloud->reserve(semanticSize);
        // semanticIndexMap.reserve(semanticSize);
        // for (int i = 0; i < semanticSize; ++i) {
        //     const auto& semanticPoint = semanticCloudIn->points[i];
        //     if (!std::isfinite(semanticPoint.x) ||
        //         !std::isfinite(semanticPoint.y) ||
        //         !std::isfinite(semanticPoint.z))  // 然后把所有语义点加入
        //     {
        //         continue;
        //     }
        //     PointType searchPoint;
        //     searchPoint.x = semanticPoint.x;
        //     searchPoint.y = semanticPoint.y;
        //     searchPoint.z = semanticPoint.z;
        //     searchPoint.intensity = semanticPoint.intensity;
        //     semanticSearchCloud->push_back(searchPoint);
        //     semanticIndexMap.push_back(i);
        // }

        // if (semanticSearchCloud->empty()) {
        //     RCLCPP_WARN_THROTTLE(
        //         get_logger(), *get_clock(), 2000,
        //         "Semantic cloud has no finite points after filtering.");
        //     return !requireSemanticCloud;
        // }

        // pcl::KdTreeFLANN<PointType> semanticTree;
        // semanticTree.setInputCloud(semanticSearchCloud);  // 接着建立 KDTree
        // std::vector<int> pointSearchInd(1);
        // std::vector<float> pointSearchSqDis(1);
        // int matchedCount = 0;

        // // 查询过程
        // for (int i = 0; i < cloudSize; ++i) {
        //     PointType query;
        //     query.x = laserCloudIn->points[i].x;
        //     query.y = laserCloudIn->points[i].y;
        //     query.z = laserCloudIn->points[i].z;
        //     query.intensity = laserCloudIn->points[i].intensity;

        //     if (semanticTree.nearestKSearch(query, 1, pointSearchInd, pointSearchSqDis) > 0 &&  // K = 1表示寻找最近的一个语义点
        //         pointSearchSqDis[0] <= semanticNearestSqDist)
        //     {
        //         const int semanticIndex = semanticIndexMap[pointSearchInd[0]];
        //         const uint32_t label = semanticCloudIn->points[semanticIndex].label;
        //         pointSemanticLabels[i] = label;
        //         pointSemanticWeights[i] = computeSemanticWeight(
        //             static_cast<SemanticLabel>(label),
        //             semanticWeightAlpha);
        //         ++matchedCount;
        //     }
        // }

        // const float semanticCoverage =
        //     cloudSize > 0 ? static_cast<float>(matchedCount) / static_cast<float>(cloudSize) : 0.0f;
        // if (semanticCoverage < minimumSemanticCoverage) {  // KDTree 最近邻搜索 情况2：虽然点数相同，但是索引匹配成功率太低
        //     RCLCPP_WARN_THROTTLE(
        //         get_logger(), *get_clock(), 2000,
        //         "Semantic coverage %.3f is below required %.3f; dropping lidar frame.",
        //         semanticCoverage, minimumSemanticCoverage);
        //     return !requireSemanticCloud;
        // }
        // return true;
    
        // 直接全部使用KDTree最近邻搜索
        //============================
        // Build KDTree
        //============================
        const int semanticSize = static_cast<int>(semanticCloudIn->points.size());

        if (semanticSize != cloudSize)
        {
            RCLCPP_WARN_THROTTLE(
                get_logger(), *get_clock(), 2000,
                "Indexed semantic cloud size mismatch: lidar=%d semantic=%d",
                cloudSize, semanticSize);
            return !requireSemanticCloud;
        }

        if (semanticSize == 0)
        {
            RCLCPP_WARN_THROTTLE(
                get_logger(), *get_clock(), 2000,
                "Semantic cloud is empty.");

            return !requireSemanticCloud;
        }

        // 构建KDTree搜索点云
        int indexedMatchedCount = 0;
        for (int i = 0; i < cloudSize; ++i)
        {
            const auto& lidarPoint = laserCloudIn->points[i];
            const auto& semanticPoint = semanticCloudIn->points[i];
            const float dx = lidarPoint.x - semanticPoint.x;
            const float dy = lidarPoint.y - semanticPoint.y;
            const float dz = lidarPoint.z - semanticPoint.z;
            const float sqDist = dx * dx + dy * dy + dz * dz;
            if (!std::isfinite(sqDist) || sqDist > semanticNearestSqDist)
                continue;

            const uint32_t label = semanticPoint.label;
            pointSemanticLabels[i] = label;
            pointSemanticWeights[i] = computeSemanticWeight(
                static_cast<SemanticLabel>(label), semanticWeightAlpha);
            ++indexedMatchedCount;
        }

        const float indexedCoverage = cloudSize > 0
            ? static_cast<float>(indexedMatchedCount) / static_cast<float>(cloudSize)
            : 0.0f;
        RCLCPP_INFO_THROTTLE(
            get_logger(), *get_clock(), 3000,
            "Indexed semantic coverage: %.2f%% (%d/%d)",
            indexedCoverage * 100.0f, indexedMatchedCount, cloudSize);

        if (indexedCoverage < minimumSemanticCoverage)
        {
            RCLCPP_WARN_THROTTLE(
                get_logger(), *get_clock(), 2000,
                "Indexed semantic geometry coverage %.3f is below threshold %.3f",
                indexedCoverage, minimumSemanticCoverage);
            pointSemanticLabels.assign(
                cloudSize, static_cast<uint32_t>(SemanticLabel::UNKNOWN));
            pointSemanticWeights.assign(cloudSize, 1.0f);
            return !requireSemanticCloud;
        }

        return true;

#if 0
        pcl::PointCloud<PointType>::Ptr semanticSearchCloud(
            new pcl::PointCloud<PointType>());

        std::vector<int> semanticIndexMap;

        semanticSearchCloud->reserve(semanticSize);
        semanticIndexMap.reserve(semanticSize);

        for (int i = 0; i < semanticSize; ++i)
        {
            const auto& p = semanticCloudIn->points[i];

            if (!std::isfinite(p.x) ||
                !std::isfinite(p.y) ||
                !std::isfinite(p.z))
                continue;

            PointType pt;
            pt.x = p.x;
            pt.y = p.y;
            pt.z = p.z;
            pt.intensity = p.intensity;

            semanticSearchCloud->push_back(pt);
            semanticIndexMap.push_back(i);
        }

        if (semanticSearchCloud->empty())
        {
            RCLCPP_WARN_THROTTLE(
                get_logger(), *get_clock(), 2000,
                "No valid semantic points.");

            return !requireSemanticCloud;
        }

        //============================
        // KDTree
        //============================
        pcl::KdTreeFLANN<PointType> semanticTree;
        semanticTree.setInputCloud(semanticSearchCloud);

        std::vector<int> searchIdx(1);
        std::vector<float> searchDist(1);

        int matchedCount = 0;

        // 对每一个LiDAR点寻找最近语义点
        for (int i = 0; i < cloudSize; ++i)
        {
            PointType query;
            query.x = laserCloudIn->points[i].x;
            query.y = laserCloudIn->points[i].y;
            query.z = laserCloudIn->points[i].z;
            query.intensity = laserCloudIn->points[i].intensity;

            if (semanticTree.nearestKSearch(query, 1, searchIdx, searchDist) > 0)
            {
                if (searchDist[0] <= semanticNearestSqDist)
                {
                    int semanticIdx = semanticIndexMap[searchIdx[0]];

                    uint32_t label = semanticCloudIn->points[semanticIdx].label;

                    pointSemanticLabels[i] = label;

                    pointSemanticWeights[i] =
                        computeSemanticWeight(
                            static_cast<SemanticLabel>(label),
                            semanticWeightAlpha);

                    matchedCount++;
                }
            }
        }

        //============================
        // Coverage Check
        //============================
        float semanticCoverage =
            cloudSize > 0 ?
            static_cast<float>(matchedCount) /
            static_cast<float>(cloudSize)
            : 0.0f;

        RCLCPP_INFO_THROTTLE(
            get_logger(),
            *get_clock(),
            3000,
            "Semantic Coverage: %.2f%% (%d/%d)",
            semanticCoverage * 100.0f,
            matchedCount,
            cloudSize);

        if (semanticCoverage < minimumSemanticCoverage)
        {
            RCLCPP_WARN_THROTTLE(
                get_logger(),
                *get_clock(),
                2000,
                "Semantic coverage %.3f is below threshold %.3f",
                semanticCoverage,
                minimumSemanticCoverage);

            return !requireSemanticCloud;
        }

        return true;
#endif
    
    }

    void cloudHandler(const sensor_msgs::msg::PointCloud2::SharedPtr laserCloudMsg)
    {
        if (!cachePointCloud(laserCloudMsg))
            return;

        if (!deskewInfo())
            return;

        if (!cacheSemanticCloud())  // DSW-LIO-SAM: 缓存语义点云
            return;

        projectPointCloud();

        cloudExtraction();

        publishClouds();

        resetParameters();
    }

    bool cachePointCloud(const sensor_msgs::msg::PointCloud2::SharedPtr& laserCloudMsg)
    {
        // cache point cloud
        cloudQueue.push_back(*laserCloudMsg);
        if (cloudQueue.size() <= 2)
            return false;

        // convert cloud
        currentCloudMsg = std::move(cloudQueue.front());
        cloudQueue.pop_front();
        if (sensor == SensorType::VELODYNE || sensor == SensorType::LIVOX)
        {
            pcl::moveFromROSMsg(currentCloudMsg, *laserCloudIn);  
        }
        else if (sensor == SensorType::OUSTER)
        {
            // Convert to Velodyne format
            pcl::moveFromROSMsg(currentCloudMsg, *tmpOusterCloudIn);
            laserCloudIn->points.resize(tmpOusterCloudIn->size());
            laserCloudIn->is_dense = tmpOusterCloudIn->is_dense;
            for (size_t i = 0; i < tmpOusterCloudIn->size(); i++)
            {
                auto &src = tmpOusterCloudIn->points[i];
                auto &dst = laserCloudIn->points[i];
                dst.x = src.x;
                dst.y = src.y;
                dst.z = src.z;
                dst.intensity = src.intensity;
                dst.ring = src.ring;
                dst.time = src.t * 1e-9f;
            }
        }
        else
        {
            RCLCPP_ERROR_STREAM(get_logger(), "Unknown sensor type: " << int(sensor));
            rclcpp::shutdown();
        }

        // get timestamp
        cloudHeader = currentCloudMsg.header;
        timeScanCur = stamp2Sec(cloudHeader.stamp);
        timeScanEnd = timeScanCur + laserCloudIn->points.back().time;
    
        // remove Nan
        vector<int> indices;
        pcl::removeNaNFromPointCloud(*laserCloudIn, *laserCloudIn, indices);

        // check dense flag
        if (laserCloudIn->is_dense == false)
        {
            RCLCPP_ERROR(get_logger(), "Point cloud is not in dense format, please remove NaN points first!");
            rclcpp::shutdown();
        }

        // check ring channel
        // we will skip the ring check in case of velodyne - as we calculate the ring value downstream (line 572)
        if (ringFlag == 0)
        {
            ringFlag = -1;
            for (int i = 0; i < static_cast<int>(currentCloudMsg.fields.size()); ++i)
            {
                if (currentCloudMsg.fields[i].name == "ring")
                {
                    ringFlag = 1;
                    break;
                }
            }
            if (ringFlag == -1)
            {
                if (sensor == SensorType::VELODYNE) {
                    ringFlag = 2;
                } else {
                    RCLCPP_ERROR(get_logger(), "Point cloud ring channel not available, please configure your point cloud data!");
                    rclcpp::shutdown();
                }
            }
        }

        // check point time
        if (deskewFlag == 0)
        {
            deskewFlag = -1;
            for (auto &field : currentCloudMsg.fields)
            {
                if (field.name == "time" || field.name == "t")
                {
                    deskewFlag = 1;
                    break;
                }
            }
            if (deskewFlag == -1)
                RCLCPP_WARN(get_logger(), "Point cloud timestamp not available, deskew function disabled, system will drift significantly!");
        }

        return true;
    }

    bool deskewInfo()
    {
        std::lock_guard<std::mutex> lock1(imuLock);
        std::lock_guard<std::mutex> lock2(odoLock);

        // make sure IMU data available for the scan
        if (imuQueue.empty()) {
            RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
                "Waiting for IMU data: imuQueue is EMPTY");
            return false;
        }
        if (stamp2Sec(imuQueue.front().header.stamp) > timeScanCur) {
            RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
                "Waiting for IMU: earliest IMU (%.3f) is AFTER scan start (%.3f). Diff=%.3fs",
                stamp2Sec(imuQueue.front().header.stamp), timeScanCur,
                stamp2Sec(imuQueue.front().header.stamp) - timeScanCur);
            return false;
        }
        if (stamp2Sec(imuQueue.back().header.stamp) < timeScanEnd) {
            RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
                "Waiting for IMU: latest IMU (%.3f) is BEFORE scan end (%.3f). Diff=%.3fs",
                stamp2Sec(imuQueue.back().header.stamp), timeScanEnd,
                timeScanEnd - stamp2Sec(imuQueue.back().header.stamp));
            return false;
        }

        imuDeskewInfo();

        odomDeskewInfo();

        return true;
    }

    void imuDeskewInfo()
    {
        cloudInfo.imu_available = false;

        while (!imuQueue.empty())
        {
            if (stamp2Sec(imuQueue.front().header.stamp) < timeScanCur - 0.01)
                imuQueue.pop_front();
            else
                break;
        }

        if (imuQueue.empty())
            return;

        imuPointerCur = 0;

        for (int i = 0; i < static_cast<int>(imuQueue.size()); ++i)
        {
            sensor_msgs::msg::Imu thisImuMsg = imuQueue[i];
            double currentImuTime = stamp2Sec(thisImuMsg.header.stamp);

            // get roll, pitch, and yaw estimation for this scan
            if (currentImuTime <= timeScanCur)
                imuRPY2rosRPY(&thisImuMsg, &cloudInfo.imu_roll_init, &cloudInfo.imu_pitch_init, &cloudInfo.imu_yaw_init);
            if (currentImuTime > timeScanEnd + 0.01)
                break;

            if (imuPointerCur == 0){
                imuRotX[0] = 0;
                imuRotY[0] = 0;
                imuRotZ[0] = 0;
                imuTime[0] = currentImuTime;
                ++imuPointerCur;
                continue;
            }

            // get angular velocity
            double angular_x, angular_y, angular_z;
            imuAngular2rosAngular(&thisImuMsg, &angular_x, &angular_y, &angular_z);

            // integrate rotation
            double timeDiff = currentImuTime - imuTime[imuPointerCur-1];
            imuRotX[imuPointerCur] = imuRotX[imuPointerCur-1] + angular_x * timeDiff;
            imuRotY[imuPointerCur] = imuRotY[imuPointerCur-1] + angular_y * timeDiff;
            imuRotZ[imuPointerCur] = imuRotZ[imuPointerCur-1] + angular_z * timeDiff;
            imuTime[imuPointerCur] = currentImuTime;
            ++imuPointerCur;
        }

        --imuPointerCur;

        if (imuPointerCur <= 0)
        {
            RCLCPP_WARN(get_logger(), "imuDeskewInfo: imuPointerCur <= 0 after processing. cloudInfo.imu_available = false");
            return;
        }

        cloudInfo.imu_available = true;
    }

    void odomDeskewInfo()
    {
        cloudInfo.odom_available = false;

        while (!odomQueue.empty())
        {
            if (stamp2Sec(odomQueue.front().header.stamp) < timeScanCur - 0.01)
                odomQueue.pop_front();
            else
                break;
        }

        if (odomQueue.empty())
            return;

        if (stamp2Sec(odomQueue.front().header.stamp) > timeScanCur)
            return;

        // get start odometry at the beinning of the scan
        nav_msgs::msg::Odometry startOdomMsg;

        for (int i = 0; i < static_cast<int>(odomQueue.size()); ++i)
        {
            startOdomMsg = odomQueue[i];

            if (stamp2Sec(startOdomMsg.header.stamp) < timeScanCur)
                continue;
            else
                break;
        }

        tf2::Quaternion orientation;
        tf2::fromMsg(startOdomMsg.pose.pose.orientation, orientation);

        double roll, pitch, yaw;
        tf2::Matrix3x3(orientation).getRPY(roll, pitch, yaw);

        // Initial guess used in mapOptimization
        cloudInfo.initial_guess_x = startOdomMsg.pose.pose.position.x;
        cloudInfo.initial_guess_y = startOdomMsg.pose.pose.position.y;
        cloudInfo.initial_guess_z = startOdomMsg.pose.pose.position.z;
        cloudInfo.initial_guess_roll = roll;
        cloudInfo.initial_guess_pitch = pitch;
        cloudInfo.initial_guess_yaw = yaw;

        cloudInfo.odom_available = true;

        // get end odometry at the end of the scan
        odomDeskewFlag = false;

        if (stamp2Sec(odomQueue.back().header.stamp) < timeScanEnd)
            return;

        nav_msgs::msg::Odometry endOdomMsg;

        for (int i = 0; i < static_cast<int>(odomQueue.size()); ++i)
        {
            endOdomMsg = odomQueue[i];

            if (stamp2Sec(endOdomMsg.header.stamp) < timeScanEnd)
                continue;
            else
                break;
        }

        if (int(round(startOdomMsg.pose.covariance[0])) != int(round(endOdomMsg.pose.covariance[0])))
            return;

        Eigen::Affine3f transBegin = pcl::getTransformation(startOdomMsg.pose.pose.position.x, startOdomMsg.pose.pose.position.y, startOdomMsg.pose.pose.position.z, roll, pitch, yaw);

        tf2::fromMsg(endOdomMsg.pose.pose.orientation, orientation);
        tf2::Matrix3x3(orientation).getRPY(roll, pitch, yaw);
        Eigen::Affine3f transEnd = pcl::getTransformation(endOdomMsg.pose.pose.position.x, endOdomMsg.pose.pose.position.y, endOdomMsg.pose.pose.position.z, roll, pitch, yaw);

        Eigen::Affine3f transBt = transBegin.inverse() * transEnd;

        float rollIncre, pitchIncre, yawIncre;
        pcl::getTranslationAndEulerAngles(transBt, odomIncreX, odomIncreY, odomIncreZ, rollIncre, pitchIncre, yawIncre);

        odomDeskewFlag = true;
    }

    void findRotation(double pointTime, float *rotXCur, float *rotYCur, float *rotZCur)
    {
        *rotXCur = 0; *rotYCur = 0; *rotZCur = 0;

        int imuPointerFront = 0;
        while (imuPointerFront < imuPointerCur)
        {
            if (pointTime < imuTime[imuPointerFront])
                break;
            ++imuPointerFront;
        }

        if (pointTime > imuTime[imuPointerFront] || imuPointerFront == 0)
        {
            *rotXCur = imuRotX[imuPointerFront];
            *rotYCur = imuRotY[imuPointerFront];
            *rotZCur = imuRotZ[imuPointerFront];
        } else {
            int imuPointerBack = imuPointerFront - 1;
            double ratioFront = (pointTime - imuTime[imuPointerBack]) / (imuTime[imuPointerFront] - imuTime[imuPointerBack]);
            double ratioBack = (imuTime[imuPointerFront] - pointTime) / (imuTime[imuPointerFront] - imuTime[imuPointerBack]);
            *rotXCur = imuRotX[imuPointerFront] * ratioFront + imuRotX[imuPointerBack] * ratioBack;
            *rotYCur = imuRotY[imuPointerFront] * ratioFront + imuRotY[imuPointerBack] * ratioBack;
            *rotZCur = imuRotZ[imuPointerFront] * ratioFront + imuRotZ[imuPointerBack] * ratioBack;
        }
    }

    void findPosition(double relTime, float *posXCur, float *posYCur, float *posZCur)
    {
        *posXCur = 0; *posYCur = 0; *posZCur = 0;

        // If the sensor moves relatively slow, like walking speed, positional deskew seems to have little benefits. Thus code below is commented.

        // if (cloudInfo.odomAvailable == false || odomDeskewFlag == false)
        //     return;

        // float ratio = relTime / (timeScanEnd - timeScanCur);

        // *posXCur = ratio * odomIncreX;
        // *posYCur = ratio * odomIncreY;
        // *posZCur = ratio * odomIncreZ;
    }

    PointType deskewPoint(PointType *point, double relTime)
    {
        if (deskewFlag == -1 || cloudInfo.imu_available == false)
            return *point;

        double pointTime = timeScanCur + relTime;

        float rotXCur, rotYCur, rotZCur;
        findRotation(pointTime, &rotXCur, &rotYCur, &rotZCur);

        float posXCur, posYCur, posZCur;
        findPosition(relTime, &posXCur, &posYCur, &posZCur);

        if (firstPointFlag == true)
        {
            transStartInverse = (pcl::getTransformation(posXCur, posYCur, posZCur, rotXCur, rotYCur, rotZCur)).inverse();
            firstPointFlag = false;
        }

        // transform points to start
        Eigen::Affine3f transFinal = pcl::getTransformation(posXCur, posYCur, posZCur, rotXCur, rotYCur, rotZCur);
        Eigen::Affine3f transBt = transStartInverse * transFinal;

        PointType newPoint;
        newPoint.x = transBt(0,0) * point->x + transBt(0,1) * point->y + transBt(0,2) * point->z + transBt(0,3);
        newPoint.y = transBt(1,0) * point->x + transBt(1,1) * point->y + transBt(1,2) * point->z + transBt(1,3);
        newPoint.z = transBt(2,0) * point->x + transBt(2,1) * point->y + transBt(2,2) * point->z + transBt(2,3);
        newPoint.intensity = point->intensity;

        return newPoint;
    }

    void projectPointCloud()
    {
        int cloudSize = laserCloudIn->points.size();
        // range image projection
        for (int i = 0; i < cloudSize; ++i)
        {
            PointType thisPoint;
            thisPoint.x = laserCloudIn->points[i].x;
            thisPoint.y = laserCloudIn->points[i].y;
            thisPoint.z = laserCloudIn->points[i].z;
            thisPoint.intensity = laserCloudIn->points[i].intensity;

            float range = pointDistance(thisPoint);
            if (range < lidarMinRange || range > lidarMaxRange)
                continue;

            int rowIdn = laserCloudIn->points[i].ring;
            // if sensor is a velodyne (ringFlag = 2) calculate rowIdn based on number of scans
            if (ringFlag == 2) { 
                float verticalAngle =
                    atan2(thisPoint.z,
                        sqrt(thisPoint.x * thisPoint.x + thisPoint.y * thisPoint.y)) *
                    180 / M_PI;
                rowIdn = (verticalAngle + (N_SCAN - 1)) / 2.0;
            }

            if (rowIdn < 0 || rowIdn >= N_SCAN)
                continue;

            if (rowIdn % downsampleRate != 0)
                continue;

            int columnIdn = -1;
            if (sensor == SensorType::VELODYNE || sensor == SensorType::OUSTER)
            {
                float horizonAngle = atan2(thisPoint.x, thisPoint.y) * 180 / M_PI;
                static float ang_res_x = 360.0/float(Horizon_SCAN);
                columnIdn = -round((horizonAngle-90.0)/ang_res_x) + Horizon_SCAN/2;
                if (columnIdn >= Horizon_SCAN)
                    columnIdn -= Horizon_SCAN;
            }
            else if (sensor == SensorType::LIVOX)
            {
                columnIdn = columnIdnCountVec[rowIdn];
                columnIdnCountVec[rowIdn] += 1;
            }


            if (columnIdn < 0 || columnIdn >= Horizon_SCAN)
                continue;

            if (rangeMat.at<float>(rowIdn, columnIdn) != FLT_MAX)
                continue;

            thisPoint = deskewPoint(&thisPoint, laserCloudIn->points[i].time);

            rangeMat.at<float>(rowIdn, columnIdn) = range;

            int index = columnIdn + rowIdn * Horizon_SCAN;
            fullCloud->points[index] = thisPoint;

            fullCloudSemanticLabels[index] =
                pointSemanticLabels.size() > static_cast<size_t>(i) ? pointSemanticLabels[i] : static_cast<uint32_t>(SemanticLabel::UNKNOWN);
            fullCloudSemanticWeights[index] =
                pointSemanticWeights.size() > static_cast<size_t>(i) ? pointSemanticWeights[i] : 1.0f;

        }
    }

    void cloudExtraction()
    {
        int count = 0;
        // extract segmented cloud for lidar odometry
        for (int i = 0; i < N_SCAN; ++i)
        {
            cloudInfo.start_ring_index[i] = count - 1 + 5;
            for (int j = 0; j < Horizon_SCAN; ++j)
            {
                if (rangeMat.at<float>(i,j) != FLT_MAX)
                {
                    // mark the points' column index for marking occlusion later
                    cloudInfo.point_col_ind[count] = j;
                    // save range info
                    cloudInfo.point_range[count] = rangeMat.at<float>(i,j);

                    // DSW-LIO-SAM: 传递语义标签和权重
                    int index = j + i * Horizon_SCAN;

                    cloudInfo.point_semantic_labels.push_back(fullCloudSemanticLabels[index]);
                    cloudInfo.point_semantic_weights.push_back(fullCloudSemanticWeights[index]);

                    // save extracted cloud
                    extractedCloud->push_back(fullCloud->points[index]);
                    // size of extracted cloud
                    ++count;
                }
            }
            cloudInfo.end_ring_index[i] = count -1 - 5;
        }
    }
    
    void publishClouds()
    {
        cloudInfo.header = cloudHeader;
        publishCloud(pubExtractedCloud, extractedCloud, cloudHeader.stamp, lidarFrame);
        // Keep CloudInfo small. The deskewed cloud is published on its own topic and
        // consumed separately by featureExtraction.
        cloudInfo.cloud_deskewed = sensor_msgs::msg::PointCloud2();
        pubLaserCloudInfo->publish(cloudInfo);
    }
};

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);

    rclcpp::NodeOptions options;
    options.use_intra_process_comms(true);
    rclcpp::executors::MultiThreadedExecutor exec;

    auto IP = std::make_shared<ImageProjection>(options);
    exec.add_node(IP);

    RCLCPP_INFO(rclcpp::get_logger("rclcpp"), "\033[1;32m----> Image Projection Started.\033[0m");

    exec.spin();

    rclcpp::shutdown();
    return 0;
}
