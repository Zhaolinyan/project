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

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
