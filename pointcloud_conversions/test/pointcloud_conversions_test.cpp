// SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// SPDX-License-Identifier: Apache-2.0

// Unit tests for point cloud layout metadata: fields, point_step, and row_step.
// allocate_point_cloud_msg allocates through the registered cuda_buffer backend,
// so these tests require a GPU-capable test environment. The conversion handles
// themselves are outside the scope of these tests.

#include <gtest/gtest.h>

#include <stdexcept>
#include <vector>

#include "pointcloud_conversions/pointcloud_conversions.hpp"

namespace pointcloud_conversions
{
namespace
{

TEST(PointcloudConversionsTest, AllocateXyzLayout)
{
  auto msg = allocate_point_cloud_msg(640, 480, /*use_color=*/false);
  EXPECT_EQ(msg->width, 640u);
  EXPECT_EQ(msg->height, 480u);
  EXPECT_FALSE(msg->is_bigendian);
  ASSERT_EQ(msg->fields.size(), 3u);
  EXPECT_EQ(msg->fields[0].name, "x");
  EXPECT_EQ(msg->fields[1].name, "y");
  EXPECT_EQ(msg->fields[2].name, "z");
  // 3 float32 => 12 bytes per point.
  EXPECT_EQ(msg->point_step, 12u);
  EXPECT_EQ(msg->row_step, 12u * 640u);
}

TEST(PointcloudConversionsTest, AllocateXyzRgbLayout)
{
  auto msg = allocate_point_cloud_msg(320, 240, /*use_color=*/true);
  ASSERT_EQ(msg->fields.size(), 4u);
  EXPECT_EQ(msg->fields[3].name, "rgb");
  // 4 float32 => 16 bytes per point.
  EXPECT_EQ(msg->point_step, 16u);
  EXPECT_EQ(msg->row_step, 16u * 320u);
}

TEST(PointcloudConversionsTest, AllocateCustomFields)
{
  namespace pf = sensor_msgs::msg;
  std::vector<PointFieldDef> fields{
    {"x", pf::PointField::FLOAT32, 1},
    {"y", pf::PointField::FLOAT32, 1},
    {"z", pf::PointField::FLOAT32, 1}};
  auto msg = allocate_point_cloud_msg(100, 1, fields);
  EXPECT_EQ(msg->fields.size(), 3u);
  EXPECT_EQ(msg->point_step, 12u);
  EXPECT_EQ(msg->row_step, 1200u);
}

TEST(PointcloudConversionsTest, EmptyFieldListThrows)
{
  const std::vector<PointFieldDef> fields;
  EXPECT_THROW(allocate_point_cloud_msg(10, 10, fields), std::invalid_argument);
}

}  // namespace
}  // namespace pointcloud_conversions

int main(int argc, char ** argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
