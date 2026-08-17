// DSW-LIO-SAM 语义评分单元测试
// 覆盖 semanticStabilityScore() 与 computeSemanticWeight() 的边界条件与映射正确性。
#include <gtest/gtest.h>
#include <cmath>
#include "utility.hpp"

// ---- semanticStabilityScore：语义稳定性得分 S(c) ∈ [0, 1] ----

TEST(SemanticStabilityScore, StaticLabelsAreHighlyStable)
{
    // 完全静态物体应接近 1.0
    EXPECT_FLOAT_EQ(semanticStabilityScore(SemanticLabel::BUILDING), 1.0f);
    EXPECT_FLOAT_EQ(semanticStabilityScore(SemanticLabel::WALL), 1.0f);
    EXPECT_GT(semanticStabilityScore(SemanticLabel::ROAD), 0.9f);
    EXPECT_GT(semanticStabilityScore(SemanticLabel::POLE), 0.9f);
}

TEST(SemanticStabilityScore, DynamicLabelsAreUnstable)
{
    // 动态物体应显著低于 0.5
    EXPECT_LT(semanticStabilityScore(SemanticLabel::CAR), 0.5f);
    EXPECT_LT(semanticStabilityScore(SemanticLabel::PERSON), 0.5f);
    EXPECT_LT(semanticStabilityScore(SemanticLabel::TRUCK), 0.5f);
    // 行人应为最不稳定的类别
    EXPECT_LE(semanticStabilityScore(SemanticLabel::PERSON),
              semanticStabilityScore(SemanticLabel::CAR));
}

TEST(SemanticStabilityScore, SemiStaticAndUnknown)
{
    // 半静态（树木）介于中间
    EXPECT_FLOAT_EQ(semanticStabilityScore(SemanticLabel::TREE), 0.6f);
    // 未知类别回退到中性 0.5
    EXPECT_FLOAT_EQ(semanticStabilityScore(SemanticLabel::UNKNOWN), 0.5f);
}

TEST(SemanticStabilityScore, AllScoresWithinUnitInterval)
{
    const SemanticLabel labels[] = {
        SemanticLabel::UNKNOWN, SemanticLabel::CAR, SemanticLabel::TRUCK,
        SemanticLabel::BUS, SemanticLabel::PERSON, SemanticLabel::BICYCLE,
        SemanticLabel::MOTORCYCLE, SemanticLabel::BUILDING, SemanticLabel::ROAD,
        SemanticLabel::SIDEWALK, SemanticLabel::TERRAIN, SemanticLabel::TREE,
        SemanticLabel::POLE, SemanticLabel::FENCE, SemanticLabel::WALL};
    for (auto l : labels) {
        float s = semanticStabilityScore(l);
        EXPECT_GE(s, 0.0f);
        EXPECT_LE(s, 1.0f);
    }
}

// ---- computeSemanticWeight：w = 1 + α × (S(c) - 0.5) ----

TEST(ComputeSemanticWeight, AlphaZeroDegradesToUnity)
{
    // α=0 时所有类别权重都应为 1（等同原始 LIO-SAM）
    EXPECT_FLOAT_EQ(computeSemanticWeight(SemanticLabel::BUILDING, 0.0f), 1.0f);
    EXPECT_FLOAT_EQ(computeSemanticWeight(SemanticLabel::PERSON, 0.0f), 1.0f);
    EXPECT_FLOAT_EQ(computeSemanticWeight(SemanticLabel::UNKNOWN, 0.0f), 1.0f);
}

TEST(ComputeSemanticWeight, StaticEnhancedDynamicSuppressed)
{
    const float alpha = 2.0f;
    // 静态类别权重 > 1（增强约束）
    EXPECT_GT(computeSemanticWeight(SemanticLabel::BUILDING, alpha), 1.0f);
    // 动态类别权重 < 1（削弱约束）
    EXPECT_LT(computeSemanticWeight(SemanticLabel::PERSON, alpha), 1.0f);
    // 未知类别（S=0.5）权重恒为 1
    EXPECT_FLOAT_EQ(computeSemanticWeight(SemanticLabel::UNKNOWN, alpha), 1.0f);
}

TEST(ComputeSemanticWeight, MatchesClosedFormFormula)
{
    const float alpha = 1.5f;
    const SemanticLabel l = SemanticLabel::TREE; // S=0.6
    float expected = 1.0f + alpha * (semanticStabilityScore(l) - 0.5f);
    EXPECT_FLOAT_EQ(computeSemanticWeight(l, alpha), expected);
}

TEST(ComputeSemanticWeight, RemainsPositiveForLargeAlpha)
{
    EXPECT_GT(computeSemanticWeight(SemanticLabel::PERSON, 10.0f), 0.0f);
}

TEST(SemanticLabels, DynamicClassificationIsCentralized)
{
    EXPECT_TRUE(isDynamicSemanticLabel(SemanticLabel::CAR));
    EXPECT_TRUE(isDynamicSemanticLabel(SemanticLabel::PERSON));
    EXPECT_FALSE(isDynamicSemanticLabel(SemanticLabel::BUILDING));
    EXPECT_FALSE(isDynamicSemanticLabel(SemanticLabel::UNKNOWN));
}

TEST(PseudoSemanticClassification, UsesNormalizedIntensity)
{
    EXPECT_EQ(
        classifyPseudoSemanticPoint(0.0f, 0.9f, -0.5f, 0.5f, 2.0f),
        SemanticLabel::BUILDING);
    EXPECT_EQ(
        classifyPseudoSemanticPoint(0.0f, 0.6f, -0.5f, 0.5f, 2.0f),
        SemanticLabel::ROAD);
}

TEST(PseudoSemanticClassification, HeightBandsAreReachable)
{
    EXPECT_EQ(
        classifyPseudoSemanticPoint(-1.0f, 0.0f, -0.5f, 0.5f, 2.0f),
        SemanticLabel::TERRAIN);
    EXPECT_EQ(
        classifyPseudoSemanticPoint(1.0f, 0.0f, -0.5f, 0.5f, 2.0f),
        SemanticLabel::BUILDING);
    EXPECT_EQ(
        classifyPseudoSemanticPoint(3.0f, 0.0f, -0.5f, 0.5f, 2.0f),
        SemanticLabel::TREE);
}

TEST(SemanticVoxelGrid, DoesNotInventAveragedLabelIds)
{
    auto input = std::make_shared<pcl::PointCloud<PointTypeL>>();
    PointTypeL first{};
    first.x = 0.01f;
    first.label = static_cast<uint32_t>(SemanticLabel::CAR);
    input->push_back(first);
    PointTypeL second{};
    second.x = 0.02f;
    second.label = static_cast<uint32_t>(SemanticLabel::BUILDING);
    input->push_back(second);

    pcl::PointCloud<PointTypeL> output;
    voxelDownsampleSemanticCloud(input, output, 1.0f);

    ASSERT_EQ(output.size(), 1U);
    const uint32_t label = output.front().label;
    EXPECT_TRUE(label == static_cast<uint32_t>(SemanticLabel::CAR) ||
                label == static_cast<uint32_t>(SemanticLabel::BUILDING));
}

TEST(SemanticVoxelGrid, UsesCentroidAndMajorityLabelDeterministically)
{
    auto input = std::make_shared<pcl::PointCloud<PointTypeL>>();
    for (int i = 0; i < 3; ++i) {
        PointTypeL point{};
        point.x = 0.1f + 0.1f * i;
        point.intensity = 1.0f + i;
        point.label = static_cast<uint32_t>(
            i == 0 ? SemanticLabel::CAR : SemanticLabel::BUILDING);
        input->push_back(point);
    }

    pcl::PointCloud<PointTypeL> firstOutput;
    pcl::PointCloud<PointTypeL> secondOutput;
    voxelDownsampleSemanticCloud(input, firstOutput, 1.0f);
    voxelDownsampleSemanticCloud(input, secondOutput, 1.0f);

    ASSERT_EQ(firstOutput.size(), 1U);
    ASSERT_EQ(secondOutput.size(), 1U);
    EXPECT_NEAR(firstOutput.front().x, 0.2f, 1e-6f);
    EXPECT_NEAR(firstOutput.front().intensity, 2.0f, 1e-6f);
    EXPECT_EQ(
        firstOutput.front().label,
        static_cast<uint32_t>(SemanticLabel::BUILDING));
    EXPECT_FLOAT_EQ(firstOutput.front().x, secondOutput.front().x);
    EXPECT_EQ(firstOutput.front().label, secondOutput.front().label);
}

TEST(OptimizationGuard, RejectsLargeOrResidualIncreasingStep)
{
    EXPECT_TRUE(isPlausibleOptimizationStep(
        0.2, 0.01, 10.0, 5.0, 2.0, 0.2));
    EXPECT_FALSE(isPlausibleOptimizationStep(
        2.1, 0.01, 10.0, 5.0, 2.0, 0.2));
    EXPECT_FALSE(isPlausibleOptimizationStep(
        0.2, 0.21, 10.0, 5.0, 2.0, 0.2));
    EXPECT_FALSE(isPlausibleOptimizationStep(
        0.2, 0.01, 10.0, 11.0, 2.0, 0.2));
}

TEST(LidarCorrectionGuard, RejectsTimeAndMotionDiscontinuities)
{
    EXPECT_TRUE(isPlausibleLidarCorrection(0.2, 1.0, 0.1));
    EXPECT_FALSE(isPlausibleLidarCorrection(0.0, 0.0, 0.0));
    EXPECT_FALSE(isPlausibleLidarCorrection(0.2, 6.0, 0.1));
    EXPECT_FALSE(isPlausibleLidarCorrection(0.2, 1.0, M_PI));
}

TEST(ExperimentVoxelGrid, OriginalModeUsesPclCentroid)
{
    auto input = std::make_shared<pcl::PointCloud<PointTypeL>>();
    PointTypeL first{};
    first.x = 0.1f;
    first.intensity = 1.0f;
    first.label = static_cast<uint32_t>(SemanticLabel::CAR);
    input->push_back(first);
    PointTypeL second{};
    second.x = 0.3f;
    second.intensity = 3.0f;
    second.label = static_cast<uint32_t>(SemanticLabel::BUILDING);
    input->push_back(second);

    pcl::PointCloud<PointTypeL> output;
    voxelDownsampleExperimentCloud(input, output, 1.0f, true);

    ASSERT_EQ(output.size(), 1U);
    EXPECT_NEAR(output.front().x, 0.2f, 1e-6f);
    EXPECT_NEAR(output.front().intensity, 2.0f, 1e-6f);
    EXPECT_EQ(output.front().label, static_cast<uint32_t>(SemanticLabel::UNKNOWN));
}

TEST(DegeneracyIndicator, OriginalModeUsesBinaryContract)
{
    EXPECT_FLOAT_EQ(packDegeneracyIndicator(true, false, 0.8f), 0.0f);
    EXPECT_FLOAT_EQ(packDegeneracyIndicator(true, true, 0.2f), 1.0f);
    EXPECT_TRUE(isDegenerateFromIndicator(true, 1.0f, 0.3f));
    EXPECT_FALSE(isDegenerateFromIndicator(true, 0.8f, 0.3f));
}

TEST(DegeneracyIndicator, DswModeUsesContinuousScore)
{
    EXPECT_FLOAT_EQ(packDegeneracyIndicator(false, true, 0.4f), 0.4f);
    EXPECT_TRUE(isDegenerateFromIndicator(false, 0.4f, 0.3f));
    EXPECT_FALSE(isDegenerateFromIndicator(false, 0.2f, 0.3f));
}

TEST(SemanticTimestampSync, RejectsAdjacentTenHertzFrame)
{
    std::deque<sensor_msgs::msg::PointCloud2> queue(1);
    queue.front().header.stamp.sec = 10;
    queue.front().header.stamp.nanosec = 100000000;

    const int64_t targetStampNs = 10LL * 1000000000LL;
    const auto match = findPointCloudByTimestamp(queue, targetStampNs, 1000000LL);

    EXPECT_FALSE(match.has_value());
}

TEST(SemanticTimestampSync, AcceptsOnlyFrameWithinOneMillisecond)
{
    std::deque<sensor_msgs::msg::PointCloud2> queue(2);
    queue[0].header.stamp.sec = 10;
    queue[0].header.stamp.nanosec = 100000000;
    queue[1].header.stamp.sec = 10;
    queue[1].header.stamp.nanosec = 500000;

    const int64_t targetStampNs = 10LL * 1000000000LL;
    const auto match = findPointCloudByTimestamp(queue, targetStampNs, 1000000LL);

    ASSERT_TRUE(match.has_value());
    EXPECT_EQ(*match, 1U);
}

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
