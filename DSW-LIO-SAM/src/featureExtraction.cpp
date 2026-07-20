#include "utility.hpp"
#include "dsw_lio_sam/msg/cloud_info.hpp"

struct smoothness_t{ 
    float value;
    size_t ind;
};

struct by_value{ 
    bool operator()(smoothness_t const &left, smoothness_t const &right) { 
        return left.value < right.value;
    }
};

class FeatureExtraction : public ParamServer
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

    std::mutex deskewedCloudLock;
    std::deque<sensor_msgs::msg::PointCloud2> deskewedCloudQueue;

    dsw_lio_sam::msg::CloudInfo cloudInfo;
    std_msgs::msg::Header cloudHeader;


    std::vector<smoothness_t> cloudSmoothness;
    std::vector<float> cloudCurvature;
    std::vector<int> cloudNeighborPicked;
    std::vector<int> cloudLabel;

    FeatureExtraction(const rclcpp::NodeOptions & options) :
        ParamServer("dsw_lio_sam_featureExtraction", options)
    {
        subLaserCloudInfo = create_subscription<dsw_lio_sam::msg::CloudInfo>(
            "dsw_lio_sam/deskew/cloud_info", qos,
            std::bind(&FeatureExtraction::laserCloudInfoHandler, this, std::placeholders::_1));
        subDeskewedCloud = create_subscription<sensor_msgs::msg::PointCloud2>(
            "dsw_lio_sam/deskew/cloud_deskewed", rclcpp::QoS(5),
            std::bind(&FeatureExtraction::deskewedCloudHandler, this, std::placeholders::_1));

        pubLaserCloudInfo = create_publisher<dsw_lio_sam::msg::CloudInfo>(
            "dsw_lio_sam/feature/cloud_info", qos);
        pubCornerPoints = create_publisher<sensor_msgs::msg::PointCloud2>(
            "dsw_lio_sam/feature/cloud_corner", 1);
        pubSurfacePoints = create_publisher<sensor_msgs::msg::PointCloud2>(
            "dsw_lio_sam/feature/cloud_surface", 1);

        initializationValue();
    }

    void initializationValue()
    {
        cloudSmoothness.resize(N_SCAN*Horizon_SCAN);

        extractedCloud.reset(new pcl::PointCloud<PointType>());
        cornerCloud.reset(new pcl::PointCloud<PointTypeL>());
        surfaceCloud.reset(new pcl::PointCloud<PointTypeL>());

        cloudCurvature.assign(N_SCAN*Horizon_SCAN, 0.0f);
        cloudNeighborPicked.assign(N_SCAN*Horizon_SCAN, 0);
        cloudLabel.assign(N_SCAN*Horizon_SCAN, 0);
    }

    // DSW-LIO-SAM: 判断某个点是否属于动态物体（角点/面点提取共用，遵循 DRY）
    bool isDynamicLabel(int ind)
    {
        if (!semanticEnabled)
            return false;
        if (cloudInfo.point_semantic_labels.size() <= static_cast<unsigned>(ind))
            return false;
        const uint32_t label = cloudInfo.point_semantic_labels[ind];
        return label == static_cast<uint32_t>(SemanticLabel::CAR)   ||
               label == static_cast<uint32_t>(SemanticLabel::TRUCK) ||
               label == static_cast<uint32_t>(SemanticLabel::BUS)   ||
               label == static_cast<uint32_t>(SemanticLabel::PERSON)||
               label == static_cast<uint32_t>(SemanticLabel::BICYCLE)||
               label == static_cast<uint32_t>(SemanticLabel::MOTORCYCLE);
    }

    void deskewedCloudHandler(const sensor_msgs::msg::PointCloud2::SharedPtr msgIn)
    {
        std::lock_guard<std::mutex> lock(deskewedCloudLock);
        deskewedCloudQueue.push_back(*msgIn);
        while (deskewedCloudQueue.size() > 5)
            deskewedCloudQueue.pop_front();
    }

    bool cacheDeskewedCloud(const std_msgs::msg::Header& header)
    {
        std::lock_guard<std::mutex> lock(deskewedCloudLock);

        if (deskewedCloudQueue.empty())
        {
            RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
                "Waiting for deskewed point cloud on /dsw_lio_sam/deskew/cloud_deskewed");
            return false;
        }

        const double infoTime = stamp2Sec(header.stamp);
        while (!deskewedCloudQueue.empty() &&
               stamp2Sec(deskewedCloudQueue.front().header.stamp) < infoTime - 0.02)
        {
            deskewedCloudQueue.pop_front();
        }

        if (deskewedCloudQueue.empty())
            return false;

        const double cloudTime = stamp2Sec(deskewedCloudQueue.front().header.stamp);
        if (std::abs(cloudTime - infoTime) > 0.02)
        {
            RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
                "Deskewed cloud and cloud_info are not synchronized: cloud=%.6f info=%.6f",
                cloudTime, infoTime);
            return false;
        }

        pcl::fromROSMsg(deskewedCloudQueue.front(), *extractedCloud);
        deskewedCloudQueue.pop_front();
        return true;
    }

    uint32_t semanticLabelAt(size_t ind) const
    {
        if (ind < cloudInfo.point_semantic_labels.size())
            return cloudInfo.point_semantic_labels[ind];
        return static_cast<uint32_t>(SemanticLabel::UNKNOWN);
    }

    void laserCloudInfoHandler(const dsw_lio_sam::msg::CloudInfo::SharedPtr msgIn)
    {
        cloudInfo = *msgIn;
        cloudHeader = msgIn->header;
        if (!cacheDeskewedCloud(cloudHeader))
            return;

        calculateSmoothness();

        markOccludedPoints();

        extractFeatures();

        publishFeatureCloud();
    }


    void calculateSmoothness()
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

    void markOccludedPoints()
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

    void extractFeatures()
    {
        cornerCloud->clear();
        surfaceCloud->clear();

        pcl::PointCloud<PointTypeL>::Ptr surfaceCloudScan(new pcl::PointCloud<PointTypeL>());
        pcl::PointCloud<PointTypeL>::Ptr surfaceCloudScanDS(new pcl::PointCloud<PointTypeL>());


        for (int i = 0; i < N_SCAN; i++)
        {
            surfaceCloudScan->clear();

            for (int j = 0; j < 6; j++)
            {

                int sp = (cloudInfo.start_ring_index[i] * (6 - j) + cloudInfo.end_ring_index[i] * j) / 6;
                int ep = (cloudInfo.start_ring_index[i] * (5 - j) + cloudInfo.end_ring_index[i] * (j + 1)) / 6 - 1;

                if (sp >= ep)
                    continue;

                std::sort(cloudSmoothness.begin()+sp, cloudSmoothness.begin()+ep, by_value());

                int largestPickedNum = 0;
                for (int k = ep; k >= sp; k--)
                {
                    int ind = cloudSmoothness[k].ind;
                    // DSW-LIO-SAM: 跳过动态物体的角点
                    bool isDynamic = isDynamicLabel(ind);

                    if (cloudNeighborPicked[ind] == 0 && cloudCurvature[ind] > edgeThreshold && !isDynamic)

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

                for (int k = sp; k <= ep; k++)
                {
                    int ind = cloudSmoothness[k].ind;
                    // DSW-LIO-SAM: 跳过动态物体的面点
                    bool isDynamicSurf = isDynamicLabel(ind);

                    if (cloudNeighborPicked[ind] == 0 && cloudCurvature[ind] < surfThreshold && !isDynamicSurf)

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
            voxelDownsampleSemanticCloud(surfaceCloudScan, *surfaceCloudScanDS, odometrySurfLeafSize);

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

    void publishFeatureCloud()
    {
        // free cloud info memory
        freeCloudInfoMemory();

        // save newly extracted features
        // DSW-LIO-SAM: 手动转换并发布带语义的特征点云
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
