// 头文件
#pragma once
#ifndef _UTILITY_LIDAR_ODOMETRY_H_
#define _UTILITY_LIDAR_ODOMETRY_H_

#include <iostream>
#include <rclcpp/rclcpp.hpp>

#include <std_msgs/msg/header.hpp>
#include <std_msgs/msg/string.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/msg/nav_sat_fix.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/path.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

#include <opencv2/opencv.hpp>

#include <pcl/kdtree/kdtree_flann.h>  // pcl include kdtree_flann throws error if PCL_NO_PRECOMPILE
                                      // is defined before
#define PCL_NO_PRECOMPILE
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/search/impl/search.hpp>
#include <pcl/range_image/range_image.h>
#include <pcl/common/common.h>
#include <pcl/common/transforms.h>
#include <pcl/registration/icp.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/crop_box.h>
#include <pcl_conversions/pcl_conversions.h>

#include <tf2/LinearMath/Quaternion.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_eigen/tf2_eigen.hpp>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

#include <vector>
#include <cmath>
#include <algorithm>
#include <queue>
#include <deque>
#include <iostream>
#include <fstream>
#include <ctime>
#include <cfloat>
#include <iterator>
#include <sstream>
#include <string>
#include <limits>
#include <iomanip>
#include <array>
#include <thread>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <map>
#include <unordered_map>

using namespace std;

// 语义类别表
// ===== DSW-LIO-SAM: 语义标签定义 =====
enum class SemanticLabel {
    UNKNOWN  = 0,   // 未知
    CAR      = 1,   // 汽车  （动态）
    TRUCK    = 2,   // 卡车  （动态）
    BUS      = 3,   // 公交车（动态）
    PERSON   = 4,   // 行人  （动态）
    BICYCLE  = 5,   // 自行车（动态）
    MOTORCYCLE = 6, // 摩托车（动态）
    BUILDING = 7,   // 建筑物（静态）
    ROAD     = 8,   // 路面  （静态）
    SIDEWALK = 9,   // 人行道（静态）
    TERRAIN  = 10,  // 地面  （静态）
    TREE     = 11,  // 树木  （半静态）
    POLE     = 12,  // 杆子  （静态）
    FENCE    = 13,  // 围栏  （静态）
    WALL     = 14   // 墙壁  （静态）
};

// 语义权重表
// 语义稳定性得分 S(c) ∈ [0, 1]
// 0 = 完全动态（不可靠），1 = 完全静态（最可靠）
inline float semanticStabilityScore(SemanticLabel label) {
    switch (label) {
        case SemanticLabel::CAR:        return 0.1f;         // 汽车  （动态）
        case SemanticLabel::TRUCK:      return 0.1f;         // 卡车  （动态）
        case SemanticLabel::BUS:        return 0.1f;         // 公交车（动态）
        case SemanticLabel::PERSON:     return 0.05f;        // 行人  （动态）
        case SemanticLabel::BICYCLE:    return 0.1f;         // 自行车（动态）
        case SemanticLabel::MOTORCYCLE: return 0.1f;         // 摩托车（动态）
        case SemanticLabel::BUILDING:   return 1.0f;         // 建筑物（静态）
        case SemanticLabel::ROAD:       return 0.95f;        // 路面  （静态）
        case SemanticLabel::SIDEWALK:   return 0.9f;         // 人行道（静态）
        case SemanticLabel::TERRAIN:    return 0.9f;         // 地面  （静态）
        case SemanticLabel::TREE:       return 0.6f;         // 树木  （半静态）
        case SemanticLabel::POLE:       return 0.95f;        // 杆子  （静态）
        case SemanticLabel::FENCE:      return 0.9f;         // 围栏  （静态）
        case SemanticLabel::WALL:       return 1.0f;         // 墙壁  （静态）
        default:                        return 0.5f;         // UNKNOWN
    }
}

// 语义权重函数
// 语义权重计算: w_i = 1 + α × (S(c_i) - 0.5)  稳定点权重大 动态点权重小
// α=0时，所有点权重=1，退化为原始LIO-SAM
inline float computeSemanticWeight(SemanticLabel label, float alpha = 0.0f) {
    float s = semanticStabilityScore(label);
    return std::max(0.05f, 1.0f + alpha * (s - 0.5f));
}

inline bool isPlausibleOptimizationStep(
    double translationMeters,
    double rotationRadians,
    double residualBefore,
    double residualAfter,
    double maxTranslationMeters,
    double maxRotationRadians)
{
    return std::isfinite(translationMeters) &&
           std::isfinite(rotationRadians) &&
           std::isfinite(residualBefore) &&
           std::isfinite(residualAfter) &&
           translationMeters <= maxTranslationMeters &&
           rotationRadians <= maxRotationRadians &&
           residualAfter <= residualBefore + 1e-9;
}

inline bool isPlausibleLidarCorrection(
    double deltaTimeSeconds,
    double translationMeters,
    double rotationRadians,
    double minimumTranslationLimitMeters = 5.0,
    double maximumSpeedMetersPerSecond = 20.0,
    double maximumRotationRadians = 45.0 * M_PI / 180.0)
{
    const double translationLimit = std::max(
        minimumTranslationLimitMeters,
        maximumSpeedMetersPerSecond * std::max(deltaTimeSeconds, 0.0));
    return deltaTimeSeconds > 0.0 &&
           std::isfinite(translationMeters) &&
           std::isfinite(rotationRadians) &&
           translationMeters <= translationLimit &&
           rotationRadians <= maximumRotationRadians;
}

inline SemanticLabel classifyPseudoSemanticPoint(
    float z,
    float normalizedIntensity,
    float groundMaxZ,
    float structureMinZ,
    float canopyMinZ)
{
    if (!std::isfinite(z)) {
        return SemanticLabel::UNKNOWN;
    }

    const float intensity = std::isfinite(normalizedIntensity)
        ? std::clamp(normalizedIntensity, 0.0f, 1.0f)
        : 0.0f;

    if (z < groundMaxZ) {
        return SemanticLabel::TERRAIN;
    }
    if (z > canopyMinZ) {
        return SemanticLabel::TREE;
    }
    if (z > structureMinZ || intensity > 0.8f) {
        return SemanticLabel::BUILDING;
    }
    return SemanticLabel::ROAD;
}

inline bool isDynamicSemanticLabel(SemanticLabel label) {  // 判断是不是动态物体
    return label == SemanticLabel::CAR ||
           label == SemanticLabel::TRUCK ||
           label == SemanticLabel::BUS ||
           label == SemanticLabel::PERSON ||
           label == SemanticLabel::BICYCLE ||
           label == SemanticLabel::MOTORCYCLE;
}
// ===== DSW-LIO-SAM 结束 =====


typedef pcl::PointXYZI PointType;

// ===== DSW-LIO-SAM: 带语义标签的点类型 =====
struct PointXYZIL {  // x、y、z、intensity、label  L-语义信息
    PCL_ADD_POINT4D
    float intensity;
    uint32_t label;  // SemanticLabel 枚举值
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
} EIGEN_ALIGN16;

POINT_CLOUD_REGISTER_POINT_STRUCT(PointXYZIL,
    (float, x, x) (float, y, y) (float, z, z)
    (float, intensity, intensity)
    (uint32_t, label, label)
)

typedef PointXYZIL PointTypeL;  // 带语义标签的点类型别名

struct SemanticVoxelKey {
    int x;
    int y;
    int z;

    bool operator==(const SemanticVoxelKey& other) const {
        return x == other.x && y == other.y && z == other.z;
    }

    bool operator<(const SemanticVoxelKey& other) const {
        return std::tie(x, y, z) < std::tie(other.x, other.y, other.z);
    }
};

struct SemanticVoxelKeyHash {
    std::size_t operator()(const SemanticVoxelKey& key) const {
        std::size_t h1 = std::hash<int>{}(key.x);
        std::size_t h2 = std::hash<int>{}(key.y);
        std::size_t h3 = std::hash<int>{}(key.z);
        return h1 ^ (h2 << 1U) ^ (h3 << 2U);
    }
};

inline void voxelDownsampleSemanticCloud(
    const pcl::PointCloud<PointTypeL>::ConstPtr& input,
    pcl::PointCloud<PointTypeL>& output,
    float leafSize)
{
    output.clear();
    if (!input || input->empty()) {
        return;
    }
    if (leafSize <= 0.0f) {
        output = *input;
        return;
    }

    struct VoxelAccumulator {
        double x = 0.0;
        double y = 0.0;
        double z = 0.0;
        double intensity = 0.0;
        std::size_t count = 0;
        std::array<std::size_t, 15> labelCounts{};
    };

    std::map<SemanticVoxelKey, VoxelAccumulator> voxels;
    for (const auto& point : input->points) {
        const SemanticVoxelKey key{
            static_cast<int>(std::floor(point.x / leafSize)),
            static_cast<int>(std::floor(point.y / leafSize)),
            static_cast<int>(std::floor(point.z / leafSize))};
        auto& voxel = voxels[key];
        voxel.x += point.x;
        voxel.y += point.y;
        voxel.z += point.z;
        voxel.intensity += point.intensity;
        ++voxel.count;
        const auto label = static_cast<std::size_t>(point.label);
        if (label < voxel.labelCounts.size()) {
            ++voxel.labelCounts[label];
        } else {
            ++voxel.labelCounts[static_cast<std::size_t>(SemanticLabel::UNKNOWN)];
        }
    }

    output.reserve(voxels.size());
    for (const auto& item : voxels) {
        const auto& voxel = item.second;
        PointTypeL point{};
        const double inverseCount = 1.0 / static_cast<double>(voxel.count);
        point.x = static_cast<float>(voxel.x * inverseCount);
        point.y = static_cast<float>(voxel.y * inverseCount);
        point.z = static_cast<float>(voxel.z * inverseCount);
        point.intensity = static_cast<float>(voxel.intensity * inverseCount);
        point.label = static_cast<uint32_t>(std::distance(
            voxel.labelCounts.begin(),
            std::max_element(voxel.labelCounts.begin(), voxel.labelCounts.end())));
        output.push_back(point);
    }
}

inline void voxelDownsampleOriginalLioSamCloud(
    const pcl::PointCloud<PointTypeL>::ConstPtr& input,
    pcl::PointCloud<PointTypeL>& output,
    float leafSize)
{
    output.clear();
    if (!input || input->empty()) {
        return;
    }
    if (leafSize <= 0.0f) {
        output = *input;
        for (auto& point : output.points) {
            point.label = static_cast<uint32_t>(SemanticLabel::UNKNOWN);
        }
        return;
    }

    pcl::PointCloud<PointType>::Ptr originalInput(new pcl::PointCloud<PointType>());
    originalInput->reserve(input->size());
    originalInput->header = input->header;
    originalInput->is_dense = input->is_dense;
    for (const auto& point : input->points) {
        PointType originalPoint;
        originalPoint.x = point.x;
        originalPoint.y = point.y;
        originalPoint.z = point.z;
        originalPoint.intensity = point.intensity;
        originalInput->push_back(originalPoint);
    }

    pcl::VoxelGrid<PointType> filter;
    filter.setLeafSize(leafSize, leafSize, leafSize);
    filter.setInputCloud(originalInput);
    pcl::PointCloud<PointType> filtered;
    filter.filter(filtered);

    output.reserve(filtered.size());
    output.header = filtered.header;
    output.is_dense = filtered.is_dense;
    for (const auto& point : filtered.points) {
        PointTypeL outputPoint;
        outputPoint.x = point.x;
        outputPoint.y = point.y;
        outputPoint.z = point.z;
        outputPoint.intensity = point.intensity;
        outputPoint.label = static_cast<uint32_t>(SemanticLabel::UNKNOWN);
        output.push_back(outputPoint);
    }
}

inline void voxelDownsampleExperimentCloud(
    const pcl::PointCloud<PointTypeL>::ConstPtr& input,
    pcl::PointCloud<PointTypeL>& output,
    float leafSize,
    bool originalLioSamMode)
{
    if (originalLioSamMode) {
        voxelDownsampleOriginalLioSamCloud(input, output, leafSize);
    } else {
        voxelDownsampleSemanticCloud(input, output, leafSize);
    }
}

inline float packDegeneracyIndicator(
    bool originalLioSamMode,
    bool isDegenerate,
    float degeneracyScore)
{
    return originalLioSamMode
        ? (isDegenerate ? 1.0f : 0.0f)
        : degeneracyScore;
}

inline bool isDegenerateFromIndicator(
    bool originalLioSamMode,
    float indicator,
    float threshold)
{
    return originalLioSamMode
        ? static_cast<int>(indicator) == 1
        : indicator > threshold;
}
// ===== DSW-LIO-SAM 结束 =====


enum class SensorType { VELODYNE, OUSTER, LIVOX };

class ParamServer : public rclcpp::Node
{
public:
    std::string robot_id;

    //Topics
    string pointCloudTopic;
    string imuTopic;
    string odomTopic;
    string gpsTopic;

    //Frames
    string lidarFrame;
    string baselinkFrame;
    string odometryFrame;
    string mapFrame;

    // GPS Settings
    bool useImuHeadingInitialization;
    bool useGpsElevation;
    float gpsCovThreshold;
    float poseCovThreshold;

    // Save pcd
    bool savePCD;
    string savePCDDirectory;

    // Lidar Sensor Configuration
    SensorType sensor = SensorType::OUSTER;
    int N_SCAN;
    int Horizon_SCAN;
    int downsampleRate;
    float lidarMinRange;
    float lidarMaxRange;

    // IMU
    float imuAccNoise;
    float imuGyrNoise;
    float imuAccBiasN;
    float imuGyrBiasN;
    float imuGravity;
    float imuRPYWeight;
    vector<double> extRotV;
    vector<double> extRPYV;
    vector<double> extTransV;
    Eigen::Matrix3d extRot;
    Eigen::Matrix3d extRPY;
    Eigen::Vector3d extTrans;
    Eigen::Quaterniond extQRPY;

    // LOAM
    float edgeThreshold;
    float surfThreshold;
    int edgeFeatureMinValidNum;
    int surfFeatureMinValidNum;

    // voxel filter paprams
    float odometrySurfLeafSize;
    float mappingCornerLeafSize;
    float mappingSurfLeafSize ;

    float z_tollerance;
    float rotation_tollerance;

    // CPU Params
    int numberOfCores;
    double mappingProcessInterval;

    // Surrounding map
    float surroundingkeyframeAddingDistThreshold;
    float surroundingkeyframeAddingAngleThreshold;
    float surroundingKeyframeDensity;
    float surroundingKeyframeSearchRadius;

    // Loop closure
    bool  loopClosureEnableFlag;
    float loopClosureFrequency;
    int   surroundingKeyframeSize;
    float historyKeyframeSearchRadius;
    float historyKeyframeSearchTimeDiff;
    int   historyKeyframeSearchNum;
    float historyKeyframeFitnessScore;

    // global map visualization radius
    float globalMapVisualizationSearchRadius;
    float globalMapVisualizationPoseDensity;
    float globalMapVisualizationLeafSize;
    // ===== DSW-LIO-SAM: 语义参数 =====
    bool originalLioSamMode;            // 是否恢复原始 LIO-SAM 算法路径
    bool semanticEnabled;              // 是否启用语义增强
    string semanticCloudTopic;         // 语义点云话题名
    float semanticWeightAlpha;         // 权重公式中的 α 参数
    float degeneracyThreshold;         // 退化检测阈值
    bool requireSemanticCloud;
    float minimumSemanticCoverage;
    float pseudoGroundMaxZ;
    float pseudoStructureMinZ;
    float pseudoCanopyMinZ;
    // ===== DSW-LIO-SAM 结束 =====


    ParamServer(std::string node_name, const rclcpp::NodeOptions & options) : Node(node_name, options)
    {
        declare_parameter("pointCloudTopic", "points");
        get_parameter("pointCloudTopic", pointCloudTopic);
        declare_parameter("imuTopic", "imu/data");
        get_parameter("imuTopic", imuTopic);
        declare_parameter("odomTopic", "dsw_lio_sam/odometry/imu");
        get_parameter("odomTopic", odomTopic);
        declare_parameter("gpsTopic", "dsw_lio_sam/odometry/gps");
        get_parameter("gpsTopic", gpsTopic);

        declare_parameter("lidarFrame", "laser_data_frame");
        get_parameter("lidarFrame", lidarFrame);
        declare_parameter("baselinkFrame", "base_link");
        get_parameter("baselinkFrame", baselinkFrame);
        declare_parameter("odometryFrame", "odom");
        get_parameter("odometryFrame", odometryFrame);
        declare_parameter("mapFrame", "map");
        get_parameter("mapFrame", mapFrame);

        declare_parameter("useImuHeadingInitialization", false);
        get_parameter("useImuHeadingInitialization", useImuHeadingInitialization);
        declare_parameter("useGpsElevation", false);
        get_parameter("useGpsElevation", useGpsElevation);
        declare_parameter("gpsCovThreshold", 2.0);
        get_parameter("gpsCovThreshold", gpsCovThreshold);
        declare_parameter("poseCovThreshold", 25.0);
        get_parameter("poseCovThreshold", poseCovThreshold);

        declare_parameter("savePCD", false);
        get_parameter("savePCD", savePCD);
        declare_parameter("savePCDDirectory", "/Downloads/LOAM/");
        get_parameter("savePCDDirectory", savePCDDirectory);

        std::string sensorStr;
        declare_parameter("sensor", "ouster");
        get_parameter("sensor", sensorStr);
        if (sensorStr == "velodyne")
        {
            sensor = SensorType::VELODYNE;
        }
        else if (sensorStr == "ouster")
        {
            sensor = SensorType::OUSTER;
        }
        else if (sensorStr == "livox")
        {
            sensor = SensorType::LIVOX;
        }
        else
        {
            RCLCPP_ERROR_STREAM(
                get_logger(),
                "Invalid sensor type (must be either 'velodyne' or 'ouster' or 'livox'): " << sensorStr);
            rclcpp::shutdown();
        }

        declare_parameter("N_SCAN", 64);
        get_parameter("N_SCAN", N_SCAN);
        declare_parameter("Horizon_SCAN", 512);
        get_parameter("Horizon_SCAN", Horizon_SCAN);
        declare_parameter("downsampleRate", 1);
        get_parameter("downsampleRate", downsampleRate);
        declare_parameter("lidarMinRange", 5.5);
        get_parameter("lidarMinRange", lidarMinRange);
        declare_parameter("lidarMaxRange", 1000.0);
        get_parameter("lidarMaxRange", lidarMaxRange);

        declare_parameter("imuAccNoise", 9e-4);
        get_parameter("imuAccNoise", imuAccNoise);
        declare_parameter("imuGyrNoise", 1.6e-4);
        get_parameter("imuGyrNoise", imuGyrNoise);
        declare_parameter("imuAccBiasN", 5e-4);
        get_parameter("imuAccBiasN", imuAccBiasN);
        declare_parameter("imuGyrBiasN", 7e-5);
        get_parameter("imuGyrBiasN", imuGyrBiasN);
        declare_parameter("imuGravity", 9.80511);
        get_parameter("imuGravity", imuGravity);
        declare_parameter("imuRPYWeight", 0.01);
        get_parameter("imuRPYWeight", imuRPYWeight);

        double ida[] = { 1.0,  0.0,  0.0,
                         0.0,  1.0,  0.0,
                         0.0,  0.0,  1.0};
        std::vector < double > id(ida, std::end(ida));
        declare_parameter("extrinsicRot", id);
        get_parameter("extrinsicRot", extRotV);
        declare_parameter("extrinsicRPY", id);
        get_parameter("extrinsicRPY", extRPYV);
        double zea[] = {0.0, 0.0, 0.0};
        std::vector < double > ze(zea, std::end(zea));
        declare_parameter("extrinsicTrans", ze);
        get_parameter("extrinsicTrans", extTransV);

        extRot = Eigen::Map<const Eigen::Matrix<double, -1, -1, Eigen::RowMajor>>(extRotV.data(), 3, 3);
        extRPY = Eigen::Map<const Eigen::Matrix<double, -1, -1, Eigen::RowMajor>>(extRPYV.data(), 3, 3);
        extTrans = Eigen::Map<const Eigen::Matrix<double, -1, -1, Eigen::RowMajor>>(extTransV.data(), 3, 1);
        extQRPY = Eigen::Quaterniond(extRPY);

        declare_parameter("edgeThreshold", 1.0);
        get_parameter("edgeThreshold", edgeThreshold);
        declare_parameter("surfThreshold", 0.1);
        get_parameter("surfThreshold", surfThreshold);
        declare_parameter("edgeFeatureMinValidNum", 10);
        get_parameter("edgeFeatureMinValidNum", edgeFeatureMinValidNum);
        declare_parameter("surfFeatureMinValidNum", 100);
        get_parameter("surfFeatureMinValidNum", surfFeatureMinValidNum);

        declare_parameter("odometrySurfLeafSize", 0.4);
        get_parameter("odometrySurfLeafSize", odometrySurfLeafSize);
        declare_parameter("mappingCornerLeafSize", 0.2);
        get_parameter("mappingCornerLeafSize", mappingCornerLeafSize);
        declare_parameter("mappingSurfLeafSize", 0.4);
        get_parameter("mappingSurfLeafSize", mappingSurfLeafSize);

        declare_parameter("z_tollerance", 1000.0);
        get_parameter("z_tollerance", z_tollerance);
        declare_parameter("rotation_tollerance", 1000.0);
        get_parameter("rotation_tollerance", rotation_tollerance);

        declare_parameter("numberOfCores", 4);
        get_parameter("numberOfCores", numberOfCores);
        declare_parameter("mappingProcessInterval", 0.15);
        get_parameter("mappingProcessInterval", mappingProcessInterval);

        declare_parameter("surroundingkeyframeAddingDistThreshold", 1.0);
        get_parameter("surroundingkeyframeAddingDistThreshold", surroundingkeyframeAddingDistThreshold);
        declare_parameter("surroundingkeyframeAddingAngleThreshold", 0.2);
        get_parameter("surroundingkeyframeAddingAngleThreshold", surroundingkeyframeAddingAngleThreshold);
        declare_parameter("surroundingKeyframeDensity", 2.0);
        get_parameter("surroundingKeyframeDensity", surroundingKeyframeDensity);
        declare_parameter("surroundingKeyframeSearchRadius", 50.0);
        get_parameter("surroundingKeyframeSearchRadius", surroundingKeyframeSearchRadius);

        declare_parameter("loopClosureEnableFlag", true);
        get_parameter("loopClosureEnableFlag", loopClosureEnableFlag);
        declare_parameter("loopClosureFrequency", 1.0);
        get_parameter("loopClosureFrequency", loopClosureFrequency);
        declare_parameter("surroundingKeyframeSize", 50);
        get_parameter("surroundingKeyframeSize", surroundingKeyframeSize);
        declare_parameter("historyKeyframeSearchRadius", 15.0);
        get_parameter("historyKeyframeSearchRadius", historyKeyframeSearchRadius);
        declare_parameter("historyKeyframeSearchTimeDiff", 30.0);
        get_parameter("historyKeyframeSearchTimeDiff", historyKeyframeSearchTimeDiff);
        declare_parameter("historyKeyframeSearchNum", 25);
        get_parameter("historyKeyframeSearchNum", historyKeyframeSearchNum);
        declare_parameter("historyKeyframeFitnessScore", 0.3);
        get_parameter("historyKeyframeFitnessScore", historyKeyframeFitnessScore);

        declare_parameter("globalMapVisualizationSearchRadius", 1000.0);
        get_parameter("globalMapVisualizationSearchRadius", globalMapVisualizationSearchRadius);
        declare_parameter("globalMapVisualizationPoseDensity", 10.0);
        get_parameter("globalMapVisualizationPoseDensity", globalMapVisualizationPoseDensity);
        declare_parameter("globalMapVisualizationLeafSize", 1.0);
        get_parameter("globalMapVisualizationLeafSize", globalMapVisualizationLeafSize);
        // ===== DSW-LIO-SAM: 读取实验与语义参数 =====
        declare_parameter("originalLioSamMode", false);
        get_parameter("originalLioSamMode", originalLioSamMode);
        declare_parameter("semanticEnabled", false);
        get_parameter("semanticEnabled", semanticEnabled);
        declare_parameter("semanticCloudTopic", "/semantic_cloud");
        get_parameter("semanticCloudTopic", semanticCloudTopic);
        declare_parameter("semanticWeightAlpha", 0.0);
        get_parameter("semanticWeightAlpha", semanticWeightAlpha);
        declare_parameter("degeneracyThreshold", 0.3);
        get_parameter("degeneracyThreshold", degeneracyThreshold);
        declare_parameter("requireSemanticCloud", false);
        get_parameter("requireSemanticCloud", requireSemanticCloud);
        declare_parameter("minimumSemanticCoverage", 0.0);
        get_parameter("minimumSemanticCoverage", minimumSemanticCoverage);
        declare_parameter("pseudoGroundMaxZ", -0.5);
        get_parameter("pseudoGroundMaxZ", pseudoGroundMaxZ);
        declare_parameter("pseudoStructureMinZ", 0.5);
        get_parameter("pseudoStructureMinZ", pseudoStructureMinZ);
        declare_parameter("pseudoCanopyMinZ", 2.0);
        get_parameter("pseudoCanopyMinZ", pseudoCanopyMinZ);
        if (originalLioSamMode &&
            (semanticEnabled || semanticWeightAlpha != 0.0f || requireSemanticCloud))
        {
            throw std::invalid_argument(
                "originalLioSamMode requires semanticEnabled=false, "
                "semanticWeightAlpha=0, and requireSemanticCloud=false");
        }
        // ===== DSW-LIO-SAM 结束 =====
    }

    sensor_msgs::msg::Imu imuConverter(const sensor_msgs::msg::Imu& imu_in)
    {
        sensor_msgs::msg::Imu imu_out = imu_in;
        // rotate acceleration
        Eigen::Vector3d acc(imu_in.linear_acceleration.x, imu_in.linear_acceleration.y, imu_in.linear_acceleration.z);
        acc = extRot * acc;
        imu_out.linear_acceleration.x = acc.x();
        imu_out.linear_acceleration.y = acc.y();
        imu_out.linear_acceleration.z = acc.z();
        // rotate gyroscope
        Eigen::Vector3d gyr(imu_in.angular_velocity.x, imu_in.angular_velocity.y, imu_in.angular_velocity.z);
        gyr = extRot * gyr;
        imu_out.angular_velocity.x = gyr.x();
        imu_out.angular_velocity.y = gyr.y();
        imu_out.angular_velocity.z = gyr.z();
        // rotate roll pitch yaw
        Eigen::Quaterniond q_from(imu_in.orientation.w, imu_in.orientation.x, imu_in.orientation.y, imu_in.orientation.z);
        Eigen::Quaterniond q_final = q_from * extQRPY;
        imu_out.orientation.x = q_final.x();
        imu_out.orientation.y = q_final.y();
        imu_out.orientation.z = q_final.z();
        imu_out.orientation.w = q_final.w();

        if (sqrt(q_final.x()*q_final.x() + q_final.y()*q_final.y() + q_final.z()*q_final.z() + q_final.w()*q_final.w()) < 0.1)
        {
            // RCLCPP_WARN(get_logger(), "IMU orientation not available (6-axis IMU), using identity quaternion.");  // 每条 IMU 都打印一次WARN
            RCLCPP_WARN_THROTTLE(
                get_logger(), *get_clock(), 5000,
                "IMU orientation not available (6-axis IMU), using identity quaternion.");
            imu_out.orientation.w = 1.0;  // 设为单位四元数，表示无旋转
            imu_out.orientation.x = 0.0;
            imu_out.orientation.y = 0.0;
            imu_out.orientation.z = 0.0;
        }


        return imu_out;
    }
};


sensor_msgs::msg::PointCloud2 publishCloud(rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr thisPub, pcl::PointCloud<PointType>::Ptr thisCloud, rclcpp::Time thisStamp, std::string thisFrame)
{
    sensor_msgs::msg::PointCloud2 tempCloud;
    pcl::toROSMsg(*thisCloud, tempCloud);
    tempCloud.header.stamp = thisStamp;
    tempCloud.header.frame_id = thisFrame;
    thisPub->publish(tempCloud);
    return tempCloud;
}

template<typename T>
double stamp2Sec(const T& stamp)
{
    return rclcpp::Time(stamp).seconds();
}

inline int64_t stampToNanoseconds(const builtin_interfaces::msg::Time& stamp)
{
    return static_cast<int64_t>(stamp.sec) * 1000000000LL +
           static_cast<int64_t>(stamp.nanosec);
}

inline std::optional<size_t> findPointCloudByTimestamp(
    const std::deque<sensor_msgs::msg::PointCloud2>& queue,
    int64_t targetStampNs,
    int64_t toleranceNs)
{
    std::optional<size_t> bestIndex;
    int64_t bestDiffNs = std::numeric_limits<int64_t>::max();

    for (size_t i = 0; i < queue.size(); ++i) {
        const int64_t stampNs = stampToNanoseconds(queue[i].header.stamp);
        const int64_t diffNs = stampNs >= targetStampNs
            ? stampNs - targetStampNs
            : targetStampNs - stampNs;
        if (diffNs <= toleranceNs && diffNs < bestDiffNs) {
            bestDiffNs = diffNs;
            bestIndex = i;
        }
    }

    return bestIndex;
}


template<typename T>
void imuAngular2rosAngular(sensor_msgs::msg::Imu *thisImuMsg, T *angular_x, T *angular_y, T *angular_z)
{
    *angular_x = thisImuMsg->angular_velocity.x;
    *angular_y = thisImuMsg->angular_velocity.y;
    *angular_z = thisImuMsg->angular_velocity.z;
}


template<typename T>
void imuAccel2rosAccel(sensor_msgs::msg::Imu *thisImuMsg, T *acc_x, T *acc_y, T *acc_z)
{
    *acc_x = thisImuMsg->linear_acceleration.x;
    *acc_y = thisImuMsg->linear_acceleration.y;
    *acc_z = thisImuMsg->linear_acceleration.z;
}


template<typename T>
void imuRPY2rosRPY(sensor_msgs::msg::Imu *thisImuMsg, T *rosRoll, T *rosPitch, T *rosYaw)
{
    double imuRoll, imuPitch, imuYaw;
    tf2::Quaternion orientation;
    tf2::fromMsg(thisImuMsg->orientation, orientation);
    tf2::Matrix3x3(orientation).getRPY(imuRoll, imuPitch, imuYaw);

    *rosRoll = imuRoll;
    *rosPitch = imuPitch;
    *rosYaw = imuYaw;
}


float pointDistance(PointType p)
{
    return sqrt(p.x*p.x + p.y*p.y + p.z*p.z);
}


float pointDistance(PointType p1, PointType p2)
{
    return sqrt((p1.x-p2.x)*(p1.x-p2.x) + (p1.y-p2.y)*(p1.y-p2.y) + (p1.z-p2.z)*(p1.z-p2.z));
}

rmw_qos_profile_t qos_profile{
    RMW_QOS_POLICY_HISTORY_KEEP_LAST,
    1,
    RMW_QOS_POLICY_RELIABILITY_RELIABLE,
    RMW_QOS_POLICY_DURABILITY_VOLATILE,
    RMW_QOS_DEADLINE_DEFAULT,
    RMW_QOS_LIFESPAN_DEFAULT,
    RMW_QOS_POLICY_LIVELINESS_SYSTEM_DEFAULT,
    RMW_QOS_LIVELINESS_LEASE_DURATION_DEFAULT,
    false
};

auto qos = rclcpp::QoS(
    rclcpp::QoSInitialization(
        qos_profile.history,
        qos_profile.depth
    ),
    qos_profile);

rmw_qos_profile_t qos_profile_imu{
    RMW_QOS_POLICY_HISTORY_KEEP_LAST,
    2000,
    RMW_QOS_POLICY_RELIABILITY_BEST_EFFORT,
    RMW_QOS_POLICY_DURABILITY_VOLATILE,
    RMW_QOS_DEADLINE_DEFAULT,
    RMW_QOS_LIFESPAN_DEFAULT,
    RMW_QOS_POLICY_LIVELINESS_SYSTEM_DEFAULT,
    RMW_QOS_LIVELINESS_LEASE_DURATION_DEFAULT,
    false
};

auto qos_imu = rclcpp::QoS(
    rclcpp::QoSInitialization(
        qos_profile_imu.history,
        qos_profile_imu.depth
    ),
    qos_profile_imu);

inline constexpr size_t kPointCloudQosDepth = 1024;

rmw_qos_profile_t qos_profile_lidar{
    RMW_QOS_POLICY_HISTORY_KEEP_LAST,
    kPointCloudQosDepth,
    RMW_QOS_POLICY_RELIABILITY_RELIABLE,
    RMW_QOS_POLICY_DURABILITY_VOLATILE,
    RMW_QOS_DEADLINE_DEFAULT,
    RMW_QOS_LIFESPAN_DEFAULT,
    RMW_QOS_POLICY_LIVELINESS_SYSTEM_DEFAULT,
    RMW_QOS_LIVELINESS_LEASE_DURATION_DEFAULT,
    false
};

auto qos_lidar = rclcpp::QoS(
    rclcpp::QoSInitialization(
        qos_profile_lidar.history,
        qos_profile_lidar.depth
    ),
    qos_profile_lidar);

rmw_qos_profile_t qos_profile_point_cloud_pipeline{
    RMW_QOS_POLICY_HISTORY_KEEP_LAST,
    kPointCloudQosDepth,
    RMW_QOS_POLICY_RELIABILITY_RELIABLE,
    RMW_QOS_POLICY_DURABILITY_VOLATILE,
    RMW_QOS_DEADLINE_DEFAULT,
    RMW_QOS_LIFESPAN_DEFAULT,
    RMW_QOS_POLICY_LIVELINESS_SYSTEM_DEFAULT,
    RMW_QOS_LIVELINESS_LEASE_DURATION_DEFAULT,
    false
};

auto qos_point_cloud_pipeline = rclcpp::QoS(
    rclcpp::QoSInitialization(
        qos_profile_point_cloud_pipeline.history,
        qos_profile_point_cloud_pipeline.depth
    ),
    qos_profile_point_cloud_pipeline);

#endif
