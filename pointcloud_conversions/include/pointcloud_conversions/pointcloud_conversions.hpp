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

#ifndef POINTCLOUD_CONVERSIONS__POINTCLOUD_CONVERSIONS_HPP_
#define POINTCLOUD_CONVERSIONS__POINTCLOUD_CONVERSIONS_HPP_

// Header-only helpers for producing and consuming sensor_msgs/PointCloud2
// messages whose `data` field is a rosidl::Buffer<uint8_t> backed by CUDA
// device memory (via the registered cuda_buffer backend). This lets GPU
// producers (e.g. stereo point cloud reprojection) write directly into the
// published message's device buffer with no host round-trip, and lets GPU
// consumers read device pointers straight from a received message.

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "cuda_buffer/cuda_buffer_api.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "sensor_msgs/msg/point_field.hpp"

namespace pointcloud_conversions
{

// Describes one field of a point (name, datatype, element count).
struct PointFieldDef
{
  std::string name;
  uint8_t datatype;  // sensor_msgs::msg::PointField::FLOAT32, etc.
  uint32_t count{1};
};

namespace detail
{

inline uint32_t point_field_size(uint8_t datatype)
{
  using PointField = sensor_msgs::msg::PointField;
  switch (datatype) {
    case PointField::INT8:
    case PointField::UINT8:
      return 1;
    case PointField::INT16:
    case PointField::UINT16:
      return 2;
    case PointField::INT32:
    case PointField::UINT32:
    case PointField::FLOAT32:
      return 4;
    case PointField::FLOAT64:
      return 8;
    default:
      throw std::invalid_argument("pointcloud_conversions: unsupported PointField datatype");
  }
}

// Apply the field layout to `msg` (setting fields, point_step and row_step) and
// swap in a CUDA-backed data buffer sized for width * height points. The msg
// width/height must already be set.
inline void set_fields_and_allocate(
  sensor_msgs::msg::PointCloud2 & msg, const std::vector<PointFieldDef> & fields)
{
  if (fields.empty()) {
    throw std::invalid_argument("pointcloud_conversions: at least one field is required");
  }

  msg.fields.clear();
  msg.fields.reserve(fields.size());
  uint32_t offset = 0;
  for (const auto & field : fields) {
    auto & point_field = msg.fields.emplace_back();
    point_field.name = field.name;
    point_field.offset = offset;
    point_field.datatype = field.datatype;
    point_field.count = field.count;
    offset += field.count * point_field_size(field.datatype);
  }

  msg.point_step = offset;
  msg.row_step = msg.width * msg.point_step;
  const size_t bytes = static_cast<size_t>(msg.row_step) * static_cast<size_t>(msg.height);
  msg.data = cuda_buffer_backend::allocate_buffer(bytes);
}

}  // namespace detail

// Allocate a CUDA-backed PointCloud2 with the xyz (12 bytes/point) or xyz+rgb
// (16 bytes/point) layout used across the Isaac ROS stereo pipeline. Fills
// width/height/fields/point_step/row_step/data; the caller sets the header.
inline std::unique_ptr<sensor_msgs::msg::PointCloud2> allocate_point_cloud_msg(
  uint32_t width, uint32_t height, bool use_color)
{
  namespace pf = sensor_msgs::msg;
  auto msg = std::make_unique<sensor_msgs::msg::PointCloud2>();
  msg->width = width;
  msg->height = height;
  msg->is_bigendian = false;
  msg->is_dense = false;

  std::vector<PointFieldDef> fields{
    {"x", pf::PointField::FLOAT32, 1},
    {"y", pf::PointField::FLOAT32, 1},
    {"z", pf::PointField::FLOAT32, 1}};
  if (use_color) {
    fields.push_back({"rgb", pf::PointField::FLOAT32, 1});
  }
  detail::set_fields_and_allocate(*msg, fields);
  return msg;
}

// Allocate a CUDA-backed PointCloud2 with a caller-supplied point layout.
inline std::unique_ptr<sensor_msgs::msg::PointCloud2> allocate_point_cloud_msg(
  uint32_t width, uint32_t height, const std::vector<PointFieldDef> & fields)
{
  auto msg = std::make_unique<sensor_msgs::msg::PointCloud2>();
  msg->width = width;
  msg->height = height;
  msg->is_bigendian = false;
  msg->is_dense = false;
  detail::set_fields_and_allocate(*msg, fields);
  return msg;
}

// Write view of a PointCloud2 device buffer for CUDA kernel output. If the
// message is not already CUDA-backed, a device buffer is allocated and attached
// to the returned handle (substitute it back via get_promoted_buffer()).
inline cuda_buffer_backend::WriteHandle from_output_point_cloud_msg(
  sensor_msgs::msg::PointCloud2 & msg, cudaStream_t stream)
{
  return cuda_buffer_backend::from_output_buffer(msg.data, stream);
}

// Read view of a PointCloud2 device buffer for CUDA kernel input. If the
// message is CPU-backed, its contents are copied host-to-device first.
inline cuda_buffer_backend::ReadHandle from_input_point_cloud_msg(
  const sensor_msgs::msg::PointCloud2 & msg, cudaStream_t stream)
{
  return cuda_buffer_backend::from_input_buffer(msg.data, stream);
}

}  // namespace pointcloud_conversions

#endif  // POINTCLOUD_CONVERSIONS__POINTCLOUD_CONVERSIONS_HPP_
