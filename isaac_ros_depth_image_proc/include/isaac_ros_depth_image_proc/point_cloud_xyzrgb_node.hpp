// SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
// Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#pragma once

#include <chrono>
#include <string>
#include <utility>
#include <vector>

#include "message_filters/subscriber.hpp"
#include "message_filters/synchronizer.hpp"
#include "message_filters/sync_policies/exact_time.hpp"

#include "isaac_ros_common/cuda_stream.hpp"
#include "isaac_ros_common/qos.hpp"
#include "isaac_ros_depth_image_proc/depth_to_point_cloud_cuda.cu.hpp"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/camera_info.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"

namespace nvidia
{
namespace isaac_ros
{
namespace depth_image_proc
{

class PointCloudXyzrgbNode : public rclcpp::Node
{
public:
  explicit PointCloudXyzrgbNode(const rclcpp::NodeOptions & options);

  ~PointCloudXyzrgbNode();

  PointCloudXyzrgbNode(const PointCloudXyzrgbNode &) = delete;

  PointCloudXyzrgbNode & operator=(const PointCloudXyzrgbNode &) = delete;

private:
  void OnSynchronizedInputs(
    const sensor_msgs::msg::Image::ConstSharedPtr & depth_msg,
    const sensor_msgs::msg::Image::ConstSharedPtr & rgb_msg,
    const sensor_msgs::msg::CameraInfo::ConstSharedPtr & camera_info_msg);

  PointCloudProperties CreatePointCloudProperties(
    const sensor_msgs::msg::PointCloud2 & point_cloud_msg,
    const sensor_msgs::msg::CameraInfo::ConstSharedPtr & depth_info_msg,
    const int skip);

  DepthProperties CreateDepthProperties(
    const sensor_msgs::msg::CameraInfo::ConstSharedPtr & depth_info_msg);

  int skip_;
  uint16_t output_height_;
  uint16_t output_width_;
  int32_t input_qos_size_;
  int32_t output_qos_size_;

  message_filters::Subscriber<sensor_msgs::msg::Image> depth_sub_;
  message_filters::Subscriber<sensor_msgs::msg::Image> rgb_sub_;
  message_filters::Subscriber<sensor_msgs::msg::CameraInfo> camera_info_sub_;

  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr point_cloud_pub_;
  rclcpp::Publisher<sensor_msgs::msg::CameraInfo>::SharedPtr camera_info_pub_;

  using ExactSyncPolicy = message_filters::sync_policies::ExactTime<
    sensor_msgs::msg::Image,
    sensor_msgs::msg::Image,
    sensor_msgs::msg::CameraInfo
  >;
  message_filters::Synchronizer<ExactSyncPolicy> exact_sync_;

  ::nvidia::isaac_ros::common::CudaStreamPtr cuda_stream_;

  depth_image_proc::DepthToPointCloudCUDA cloud_compute_;
};

}  // namespace depth_image_proc
}  // namespace isaac_ros
}  // namespace nvidia
