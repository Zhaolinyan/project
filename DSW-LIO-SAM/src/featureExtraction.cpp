// 特征提取模块  核心作用：从去畸变后的激光点云中提取角点和面点，同时融合语义信息，利用语义过滤动态物体点。
#include "utility.hpp"
#include "dsw_lio_sam/msg/cloud_info.hpp"

struct smoothness_t{  // 保存：曲率值 + 点索引，用于后面排序
    float value;
    size_t ind;
};

struct by_value{ 
    bool operator()(smoothness_t const &left, smoothness_t const &right) { 
        return left.value < right.value;
    }
};

class FeatureExtraction : public ParamServer  // FeatureExtraction类继承ParamServer
{

public:

    rclcpp::Subscription<dsw_lio_sam::msg::CloudInfo>::SharedPtr subLaserCloudInfo;
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr subDeskewedCloud;

    rclcpp::Publisher<dsw_lio_sam::msg::CloudInfo>::SharedPtr pubLaserCloudInfo;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubCornerPoints;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubSurfacePoints;

    pcl::PointCloud<PointType>::Ptr extractedCloud;       // 接收端不改，因为fromROSMsg进来的还是XYZI
    pcl::PointCloud<PointTypeL>::Ptr cornerCloud;         // 改为带语义
    pcl::PointCloud<PointTypeL>::Ptr surfaceCloud;        // 改为带语义

    std::mutex inputQueueLock;
    std::deque<sensor_msgs::msg::PointCloud2> deskewedCloudQueue;
    std::deque<dsw_lio_sam::msg::CloudInfo> cloudInfoQueue;
    static constexpr int64_t inputTimeToleranceNs = 1000000LL;
    static constexpr size_t inputQueueCapacity = kPointCloudQosDepth;

    dsw_lio_sam::msg::CloudInfo cloudInfo;
    std_msgs::msg::Header cloudHeader;


    std::vector<smoothness_t> cloudSmoothness;
    std::vector<float> cloudCurvature;
    std::vector<int> cloudNeighborPicked;
    std::vector<int> cloudLabel;
    
    // ROS通信部分
    FeatureExtraction(const rclcpp::NodeOptions & options) :
        ParamServer("dsw_lio_sam_featureExtraction", options)
    {
        subLaserCloudInfo = create_subscription<dsw_lio_sam::msg::CloudInfo>(  // 输入1：subLaserCloudInfo
            "dsw_lio_sam/deskew/cloud_info", qos_point_cloud_pipeline,  // 订阅：/deskew/cloud_info
            std::bind(&FeatureExtraction::laserCloudInfoHandler, this, std::placeholders::_1));
        subDeskewedCloud = create_subscription<sensor_msgs::msg::PointCloud2>(  // 输入2：subDeskewedCloud
            "dsw_lio_sam/deskew/cloud_deskewed", qos_point_cloud_pipeline,  // 订阅：/deskew/cloud_deskewed
            std::bind(&FeatureExtraction::deskewedCloudHandler, this, std::placeholders::_1));

        pubLaserCloudInfo = create_publisher<dsw_lio_sam::msg::CloudInfo>(
            "dsw_lio_sam/feature/cloud_info", qos_point_cloud_pipeline);
        pubCornerPoints = create_publisher<sensor_msgs::msg::PointCloud2>(  // 输出角点：pubCornerPoints
            "dsw_lio_sam/feature/cloud_corner", 1);  // 发布：feature/cloud_corner
        pubSurfacePoints = create_publisher<sensor_msgs::msg::PointCloud2>(  // 输出面点：pubSurfacePoints
            "dsw_lio_sam/feature/cloud_surface", 1);  // 发布：feature/cloud_surface

        initializationValue();
    }

    // 点云类型  x、y、z、intensity、label
    void initializationValue()
    {
        cloudSmoothness.resize(N_SCAN*Horizon_SCAN);

        extractedCloud.reset(new pcl::PointCloud<PointType>());
        cornerCloud.reset(new pcl::PointCloud<PointTypeL>());  // 角点
        surfaceCloud.reset(new pcl::PointCloud<PointTypeL>());  // 面点

        cloudCurvature.assign(N_SCAN*Horizon_SCAN, 0.0f);
        cloudNeighborPicked.assign(N_SCAN*Horizon_SCAN, 0);
        cloudLabel.assign(N_SCAN*Horizon_SCAN, 0);
    }

    // 语义动态物体判断
    // DSW-LIO-SAM: 判断某个点是否属于动态物体（角点/面点提取共用，遵循 DRY）
    bool isDynamicLabel(int ind)
    {
        if (!semanticEnabled)
            return false;  // 如果关闭语义：所有点认为静态。保持：LIO-SAM原始效果。
        if (cloudInfo.point_semantic_labels.size() <= static_cast<unsigned>(ind))
            return false;
        const uint32_t label = cloudInfo.point_semantic_labels[ind];  // 获取标签。
        return label == static_cast<uint32_t>(SemanticLabel::CAR)   ||
               label == static_cast<uint32_t>(SemanticLabel::TRUCK) ||
               label == static_cast<uint32_t>(SemanticLabel::BUS)   ||
               label == static_cast<uint32_t>(SemanticLabel::PERSON)||
               label == static_cast<uint32_t>(SemanticLabel::BICYCLE)||
               label == static_cast<uint32_t>(SemanticLabel::MOTORCYCLE);  // 这些类别认为是动态 → 删除
    }

    void deskewedCloudHandler(const sensor_msgs::msg::PointCloud2::SharedPtr msgIn)  // 缓存去畸变点云
    {
        {
            std::lock_guard<std::mutex> lock(inputQueueLock);
            deskewedCloudQueue.push_back(*msgIn);
            while (deskewedCloudQueue.size() > inputQueueCapacity)
                deskewedCloudQueue.pop_front();
        }
        processQueuedFrames();
    }

    void laserCloudInfoHandler(const dsw_lio_sam::msg::CloudInfo::SharedPtr msgIn)
    {
        {
            std::lock_guard<std::mutex> lock(inputQueueLock);
            cloudInfoQueue.push_back(*msgIn);
            while (cloudInfoQueue.size() > inputQueueCapacity)
                cloudInfoQueue.pop_front();
        }
        processQueuedFrames();
    }

    void processQueuedFrames()
    {
        while (true) {
            sensor_msgs::msg::PointCloud2 deskewedMsg;
            dsw_lio_sam::msg::CloudInfo infoMsg;
            bool matched = false;

            {
                std::lock_guard<std::mutex> lock(inputQueueLock);
                while (!deskewedCloudQueue.empty() && !cloudInfoQueue.empty()) {
                    const int64_t cloudStampNs =
                        stampToNanoseconds(deskewedCloudQueue.front().header.stamp);
                    const int64_t infoStampNs =
                        stampToNanoseconds(cloudInfoQueue.front().header.stamp);
                    const int64_t diffNs = cloudStampNs >= infoStampNs
                        ? cloudStampNs - infoStampNs
                        : infoStampNs - cloudStampNs;

                    if (diffNs <= inputTimeToleranceNs) {
                        deskewedMsg = std::move(deskewedCloudQueue.front());
                        infoMsg = std::move(cloudInfoQueue.front());
                        deskewedCloudQueue.pop_front();
                        cloudInfoQueue.pop_front();
                        matched = true;
                        break;
                    }

                    if (cloudStampNs < infoStampNs) {
                        RCLCPP_WARN_THROTTLE(
                            get_logger(), *get_clock(), 2000,
                            "Dropping deskewed cloud without matching cloud_info: stamp=%lld",
                            static_cast<long long>(cloudStampNs));
                        deskewedCloudQueue.pop_front();
                    } else {
                        RCLCPP_WARN_THROTTLE(
                            get_logger(), *get_clock(), 2000,
                            "Dropping cloud_info without matching deskewed cloud: stamp=%lld",
                            static_cast<long long>(infoStampNs));
                        cloudInfoQueue.pop_front();
                    }
                }
            }

            if (!matched)
                return;

            cloudInfo = std::move(infoMsg);
            cloudHeader = cloudInfo.header;
            pcl::fromROSMsg(deskewedMsg, *extractedCloud);

            calculateSmoothness();
            markOccludedPoints();
            extractFeatures();
            publishFeatureCloud();
        }
    }

    uint32_t semanticLabelAt(size_t ind) const
    {
        if (ind < cloudInfo.point_semantic_labels.size())
            return cloudInfo.point_semantic_labels[ind];
        return static_cast<uint32_t>(SemanticLabel::UNKNOWN);
    }

    void calculateSmoothness()  // 计算点曲率
    {
        int cloudSize = extractedCloud->points.size();
        for (int i = 5; i < cloudSize - 5; i++)
        {
            float diffRange = cloudInfo.point_range[i-5] + cloudInfo.point_range[i-4]
                            + cloudInfo.point_range[i-3] + cloudInfo.point_range[i-2]
                            + cloudInfo.point_range[i-1] - cloudInfo.point_range[i] * 10
                            + cloudInfo.point_range[i+1] + cloudInfo.point_range[i+2]
                            + cloudInfo.point_range[i+3] + cloudInfo.point_range[i+4]
                            + cloudInfo.point_range[i+5];

            cloudCurvature[i] = diffRange*diffRange;//diffX * diffX + diffY * diffY + diffZ * diffZ;

            cloudNeighborPicked[i] = 0;
            cloudLabel[i] = 0;
            // cloudSmoothness for sorting
            cloudSmoothness[i].value = cloudCurvature[i];
            cloudSmoothness[i].ind = i;
        }
    }

    void markOccludedPoints()  // 遮挡点
    {
        int cloudSize = extractedCloud->points.size();
        // mark occluded points and parallel beam points
        for (int i = 5; i < cloudSize - 6; ++i)
        {
            // occluded points
            float depth1 = cloudInfo.point_range[i];
            float depth2 = cloudInfo.point_range[i+1];
            int columnDiff = std::abs(int(cloudInfo.point_col_ind[i+1] - cloudInfo.point_col_ind[i]));
            if (columnDiff < 10){
                // 10 pixel diff in range image
                if (depth1 - depth2 > 0.3){
                    cloudNeighborPicked[i - 5] = 1;
                    cloudNeighborPicked[i - 4] = 1;
                    cloudNeighborPicked[i - 3] = 1;
                    cloudNeighborPicked[i - 2] = 1;
                    cloudNeighborPicked[i - 1] = 1;
                    cloudNeighborPicked[i] = 1;
                }else if (depth2 - depth1 > 0.3){
                    cloudNeighborPicked[i + 1] = 1;
                    cloudNeighborPicked[i + 2] = 1;
                    cloudNeighborPicked[i + 3] = 1;
                    cloudNeighborPicked[i + 4] = 1;
                    cloudNeighborPicked[i + 5] = 1;
                    cloudNeighborPicked[i + 6] = 1;
                }
            }
            // parallel beam
            float diff1 = std::abs(float(cloudInfo.point_range[i-1] - cloudInfo.point_range[i]));
            float diff2 = std::abs(float(cloudInfo.point_range[i+1] - cloudInfo.point_range[i]));

            if (diff1 > 0.02 * cloudInfo.point_range[i] && diff2 > 0.02 * cloudInfo.point_range[i])
                cloudNeighborPicked[i] = 1;
        }
    }

    void extractFeatures()  // 核心
    {
        cornerCloud->clear();
        surfaceCloud->clear();  // 清空

        pcl::PointCloud<PointTypeL>::Ptr surfaceCloudScan(new pcl::PointCloud<PointTypeL>());
        pcl::PointCloud<PointTypeL>::Ptr surfaceCloudScanDS(new pcl::PointCloud<PointTypeL>());


        for (int i = 0; i < N_SCAN; i++)
        {
            surfaceCloudScan->clear();

            for (int j = 0; j < 6; j++)  // 每条scan分6段  目的：避免特征集中
            {

                int sp = (cloudInfo.start_ring_index[i] * (6 - j) + cloudInfo.end_ring_index[i] * j) / 6;
                int ep = (cloudInfo.start_ring_index[i] * (5 - j) + cloudInfo.end_ring_index[i] * (j + 1)) / 6 - 1;

                if (sp >= ep)
                    continue;

                std::sort(cloudSmoothness.begin()+sp, cloudSmoothness.begin()+ep, by_value());

                int largestPickedNum = 0;
                for (int k = ep; k >= sp; k--)  // 角点提取
                {
                    int ind = cloudSmoothness[k].ind;
                    // DSW-LIO-SAM: 跳过动态物体的角点
                    // bool isDynamic = isDynamicLabel(ind);
                    // if (cloudNeighborPicked[ind] == 0 && cloudCurvature[ind] > edgeThreshold && !isDynamic)  // 高曲率 非动态
                    
                    // 取消!isDynamic的硬删除
                    if (cloudNeighborPicked[ind] == 0 && cloudCurvature[ind] > edgeThreshold)
                    {
                        largestPickedNum++;
                        if (largestPickedNum <= 20){
                            cloudLabel[ind] = 1;
                            PointTypeL pt;
                            pt.x = extractedCloud->points[ind].x;
                            pt.y = extractedCloud->points[ind].y;
                            pt.z = extractedCloud->points[ind].z;
                            pt.intensity = extractedCloud->points[ind].intensity;
                            pt.label = semanticLabelAt(ind);
                            cornerCloud->push_back(pt);

                        } else {
                            break;
                        }

                        cloudNeighborPicked[ind] = 1;
                        for (int l = 1; l <= 5; l++)
                        {
                            int columnDiff = std::abs(int(cloudInfo.point_col_ind[ind + l] - cloudInfo.point_col_ind[ind + l - 1]));
                            if (columnDiff > 10)
                                break;
                            cloudNeighborPicked[ind + l] = 1;
                        }
                        for (int l = -1; l >= -5; l--)
                        {
                            int columnDiff = std::abs(int(cloudInfo.point_col_ind[ind + l] - cloudInfo.point_col_ind[ind + l + 1]));
                            if (columnDiff > 10)
                                break;
                            cloudNeighborPicked[ind + l] = 1;
                        }
                    }
                }

                for (int k = sp; k <= ep; k++)  // 面点提取
                {
                    int ind = cloudSmoothness[k].ind;
                    // DSW-LIO-SAM: 跳过动态物体的面点
                    // bool isDynamicSurf = isDynamicLabel(ind);
                    // if (cloudNeighborPicked[ind] == 0 && cloudCurvature[ind] < surfThreshold && !isDynamicSurf)

                    // 取消!isDynamic的硬删除
                    if (cloudNeighborPicked[ind] == 0 && cloudCurvature[ind] < surfThreshold)
                    {
                        cloudLabel[ind] = -1;
                        cloudNeighborPicked[ind] = 1;

                        for (int l = 1; l <= 5; l++) {
                            int columnDiff = std::abs(int(cloudInfo.point_col_ind[ind + l] - cloudInfo.point_col_ind[ind + l - 1]));
                            if (columnDiff > 10)
                                break;

                            cloudNeighborPicked[ind + l] = 1;
                        }
                        for (int l = -1; l >= -5; l--) {
                            int columnDiff = std::abs(int(cloudInfo.point_col_ind[ind + l] - cloudInfo.point_col_ind[ind + l + 1]));
                            if (columnDiff > 10)
                                break;

                            cloudNeighborPicked[ind + l] = 1;
                        }
                    }
                }

                for (int k = sp; k <= ep; k++)
                {
                    if (cloudLabel[k] <= 0){
                        PointTypeL pt;
                        pt.x = extractedCloud->points[k].x;
                        pt.y = extractedCloud->points[k].y;
                        pt.z = extractedCloud->points[k].z;
                        pt.intensity = extractedCloud->points[k].intensity;
                        pt.label = semanticLabelAt(k);
                        surfaceCloudScan->push_back(pt);
                    }
                }

            }

            surfaceCloudScanDS->clear();
            voxelDownsampleExperimentCloud(
                surfaceCloudScan,
                *surfaceCloudScanDS,
                odometrySurfLeafSize,
                originalLioSamMode);  // original 使用 PCL 质心，DSW 保留语义标签

            *surfaceCloud += *surfaceCloudScanDS;
        }
    }

    void freeCloudInfoMemory()
    {
        cloudInfo.start_ring_index.clear();
        cloudInfo.end_ring_index.clear();
        cloudInfo.point_col_ind.clear();
        cloudInfo.point_range.clear();

        cloudInfo.point_semantic_labels.clear();   // DSW-LIO-SAM
        cloudInfo.point_semantic_weights.clear();   // DSW-LIO-SAM

    }

    void publishFeatureCloud()  // 发布cornerCloud给MapOptimization
    {
        // free cloud info memory
        freeCloudInfoMemory();

        // save newly extracted features
        // ===== DSW-LIO-SAM: 手动转换并发布带语义的特征点云 =====
        sensor_msgs::msg::PointCloud2 cornerMsg, surfMsg;
        pcl::toROSMsg(*cornerCloud, cornerMsg);
        cornerMsg.header.stamp = cloudHeader.stamp;
        cornerMsg.header.frame_id = lidarFrame;
        // if (pubCornerPoints->get_subscription_count() != 0)
        //     pubCornerPoints->publish(cornerMsg);
        pubCornerPoints->publish(cornerMsg);
        cloudInfo.cloud_corner = cornerMsg;

        pcl::toROSMsg(*surfaceCloud, surfMsg);
        surfMsg.header.stamp = cloudHeader.stamp;
        surfMsg.header.frame_id = lidarFrame;
        // if (pubSurfacePoints->get_subscription_count() != 0)
        //     pubSurfacePoints->publish(surfMsg);
        pubSurfacePoints->publish(surfMsg);
        cloudInfo.cloud_surface = surfMsg;
        // ===== DSW-LIO-SAM 结束 =====

        // publish to mapOptimization
        pubLaserCloudInfo->publish(cloudInfo);
    }

};


int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    rclcpp::NodeOptions options;
    options.use_intra_process_comms(true);
    rclcpp::executors::SingleThreadedExecutor exec;

    auto FE = std::make_shared<FeatureExtraction>(options);

    exec.add_node(FE);

    RCLCPP_INFO(rclcpp::get_logger("rclcpp"), "\033[1;32m----> Feature Extraction Started.\033[0m");

    exec.spin();

    rclcpp::shutdown();
    return 0;
}
