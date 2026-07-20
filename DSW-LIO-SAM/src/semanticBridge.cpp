#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "pcl_conversions/pcl_conversions.h"
#include "dsw_lio_sam/utility.hpp"

class SemanticBridge : public ParamServer {
public:
    SemanticBridge(const rclcpp::NodeOptions & options) : ParamServer("semantic_bridge", options) {
        // 话题名从 params.yaml 读取，不再硬编码
        sub = create_subscription<sensor_msgs::msg::PointCloud2>(
            pointCloudTopic, 10,
            std::bind(&SemanticBridge::callback, this, std::placeholders::_1));

        pub = create_publisher<sensor_msgs::msg::PointCloud2>(
            semanticCloudTopic, 10);

        RCLCPP_INFO(get_logger(), "语义桥接节点已启动: sub=%s pub=%s",
                    pointCloudTopic.c_str(), semanticCloudTopic.c_str());
    }

    void callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
        pcl::PointCloud<pcl::PointXYZI>::Ptr cloud(
            new pcl::PointCloud<pcl::PointXYZI>);
        pcl::fromROSMsg(*msg, *cloud);

        pcl::PointCloud<PointTypeL>::Ptr labeled(
            new pcl::PointCloud<PointTypeL>);
        labeled->resize(cloud->size());

        for (size_t i = 0; i < cloud->size(); ++i) {
            const auto& p = cloud->points[i];
            auto& q = labeled->points[i];
            q.x = p.x;
            q.y = p.y;
            q.z = p.z;
            q.intensity = p.intensity;

            SemanticLabel label = SemanticLabel::ROAD;

            if (p.z > pseudoCanopyMinZ)
                label = SemanticLabel::TREE;
            else if (p.z < pseudoGroundMaxZ)
                label = SemanticLabel::TERRAIN;
            else if (p.z > pseudoStructureMinZ || p.intensity > 0.8f)
                label = SemanticLabel::BUILDING;

            q.label = static_cast<uint32_t>(label);
        }

        sensor_msgs::msg::PointCloud2 out;
        pcl::toROSMsg(*labeled, out);
        out.header = msg->header;
        pub->publish(out);
    }

private:
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub;
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    rclcpp::NodeOptions options;
    options.use_intra_process_comms(true);
    rclcpp::spin(std::make_shared<SemanticBridge>(options));
    rclcpp::shutdown();
    return 0;
}
