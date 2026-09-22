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

// Unit tests for the pure (non-GPU) logic of vpi_conversions: the
// vpi_format_from_encoding map. allocate_image_msg / from_input_image_msg /
// from_output_image_msg / to_image_msg touch device memory, VPI, and the
// registered cuda_buffer backend, so they belong to an on-target integration
// test rather than this host unit test.

#include <gtest/gtest.h>

#include <stdexcept>
#include <string>

#include "vpi_conversions/vpi_conversions.hpp"
#include "sensor_msgs/image_encodings.hpp"

namespace nvidia
{
namespace isaac_ros
{
namespace vpi_conversions
{
namespace
{

namespace enc = sensor_msgs::image_encodings;

TEST(VpiConversionsTest, FormatMono8)
{
  const VpiFormatInfo info = vpi_format_from_encoding(enc::MONO8);
  EXPECT_EQ(info.image_format, VPI_IMAGE_FORMAT_U8);
  ASSERT_EQ(info.pixel_types.size(), 1u);
  EXPECT_EQ(info.pixel_types[0], VPI_PIXEL_TYPE_U8);
  EXPECT_EQ(info.num_channels, 1);
  EXPECT_EQ(info.bytes_per_element, 1);
  EXPECT_FALSE(info.is_nv12);
}

TEST(VpiConversionsTest, FormatRgb8)
{
  const VpiFormatInfo info = vpi_format_from_encoding(enc::RGB8);
  EXPECT_EQ(info.image_format, VPI_IMAGE_FORMAT_RGB8);
  EXPECT_EQ(info.num_channels, 3);
  EXPECT_EQ(info.bytes_per_element, 1);
  EXPECT_FALSE(info.is_nv12);
}

TEST(VpiConversionsTest, FormatBgr8)
{
  const VpiFormatInfo info = vpi_format_from_encoding(enc::BGR8);
  EXPECT_EQ(info.image_format, VPI_IMAGE_FORMAT_BGR8);
  EXPECT_EQ(info.num_channels, 3);
  EXPECT_EQ(info.bytes_per_element, 1);
}

TEST(VpiConversionsTest, FormatRgba8)
{
  const VpiFormatInfo info = vpi_format_from_encoding(enc::RGBA8);
  EXPECT_EQ(info.image_format, VPI_IMAGE_FORMAT_RGBA8);
  EXPECT_EQ(info.num_channels, 4);
  EXPECT_EQ(info.bytes_per_element, 1);
}

TEST(VpiConversionsTest, FormatBgra8)
{
  const VpiFormatInfo info = vpi_format_from_encoding(enc::BGRA8);
  EXPECT_EQ(info.image_format, VPI_IMAGE_FORMAT_BGRA8);
  EXPECT_EQ(info.num_channels, 4);
  EXPECT_EQ(info.bytes_per_element, 1);
}

TEST(VpiConversionsTest, FormatMono16)
{
  const VpiFormatInfo info = vpi_format_from_encoding(enc::MONO16);
  EXPECT_EQ(info.image_format, VPI_IMAGE_FORMAT_U16);
  EXPECT_EQ(info.num_channels, 1);
  EXPECT_EQ(info.bytes_per_element, 2);
}

TEST(VpiConversionsTest, Format16UC1)
{
  const VpiFormatInfo info = vpi_format_from_encoding(enc::TYPE_16UC1);
  EXPECT_EQ(info.image_format, VPI_IMAGE_FORMAT_U16);
  EXPECT_EQ(info.num_channels, 1);
  EXPECT_EQ(info.bytes_per_element, 2);
}

TEST(VpiConversionsTest, Format32FC1)
{
  const VpiFormatInfo info = vpi_format_from_encoding(enc::TYPE_32FC1);
  EXPECT_EQ(info.image_format, VPI_IMAGE_FORMAT_F32);
  EXPECT_EQ(info.num_channels, 1);
  EXPECT_EQ(info.bytes_per_element, 4);
}

TEST(VpiConversionsTest, FormatNv12)
{
  const VpiFormatInfo info = vpi_format_from_encoding(kEncodingNV12);
  EXPECT_EQ(info.image_format, VPI_IMAGE_FORMAT_NV12);
  // NV12 is a two-plane format: Y (U8) then interleaved UV (2U8).
  ASSERT_EQ(info.pixel_types.size(), 2u);
  EXPECT_EQ(info.pixel_types[0], VPI_PIXEL_TYPE_U8);
  EXPECT_EQ(info.pixel_types[1], VPI_PIXEL_TYPE_2U8);
  EXPECT_EQ(info.num_channels, 1);
  EXPECT_EQ(info.bytes_per_element, 1);
  EXPECT_TRUE(info.is_nv12);
}

TEST(VpiConversionsTest, FormatUnsupportedThrows)
{
  EXPECT_THROW(vpi_format_from_encoding("not_an_encoding"), std::invalid_argument);
  EXPECT_THROW(vpi_format_from_encoding(""), std::invalid_argument);
  // NV21 is defined by ROS but intentionally unsupported here.
  EXPECT_THROW(vpi_format_from_encoding(enc::NV21), std::invalid_argument);
}

}  // namespace
}  // namespace vpi_conversions
}  // namespace isaac_ros
}  // namespace nvidia

int main(int argc, char ** argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
