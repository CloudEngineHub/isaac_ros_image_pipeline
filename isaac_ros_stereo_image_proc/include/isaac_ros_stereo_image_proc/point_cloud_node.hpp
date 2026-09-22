// SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
// Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "message_filters/subscriber.hpp"
#include "message_filters/synchronizer.hpp"
#include "message_filters/sync_policies/exact_time.hpp"

#include "isaac_ros_common/cuda_stream.hpp"
#include "isaac_ros_stereo_image_proc/point_cloud_cuda.cu.hpp"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/camera_info.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "stereo_msgs/msg/disparity_image.hpp"

namespace nvidia
{
namespace isaac_ros
{
namespace stereo_image_proc
{

class PointCloudNode : public rclcpp::Node
{
public:
  explicit PointCloudNode(const rclcpp::NodeOptions & options);

  ~PointCloudNode();

  PointCloudNode(const PointCloudNode &) = delete;
  PointCloudNode & operator=(const PointCloudNode &) = delete;

private:
  void PointCloudCallback(
    const sensor_msgs::msg::Image::ConstSharedPtr & left_image_msg,
    const stereo_msgs::msg::DisparityImage::ConstSharedPtr & disparity_msg,
    const sensor_msgs::msg::CameraInfo::ConstSharedPtr & left_camera_info_msg,
    const sensor_msgs::msg::CameraInfo::ConstSharedPtr & right_camera_info_msg);

  PointCloudProperties CreateCloudProperties(
    const sensor_msgs::msg::PointCloud2 & point_cloud_msg);
  DisparityProperties CreateDisparityProperties(
    const stereo_msgs::msg::DisparityImage::ConstSharedPtr & disparity_msg);
  CameraIntrinsics CreateCameraIntrinsics(
    const sensor_msgs::msg::CameraInfo::ConstSharedPtr & left_camera_info_msg,
    const sensor_msgs::msg::CameraInfo::ConstSharedPtr & right_camera_info_msg);
  RGBProperties CreateRGBProperties(
    const sensor_msgs::msg::Image::ConstSharedPtr & rgb_msg);
  bool SelectDisparityFormatAndCompute(
    float * point_cloud_output,
    const PointCloudProperties & cloud_properties,
    const stereo_msgs::msg::DisparityImage::ConstSharedPtr & disparity_msg,
    const DisparityProperties & disparity_properties,
    const sensor_msgs::msg::Image::ConstSharedPtr & rgb_msg,
    const RGBProperties & rgb_properties,
    const CameraIntrinsics & intrinsics);

  // Point cloud node parameters
  bool use_color_;
  float unit_scaling_;
  const int64_t input_queue_size_;
  const int64_t output_queue_size_;

  // Subscribers and publishers
  message_filters::Subscriber<sensor_msgs::msg::Image> left_image_sub_;
  message_filters::Subscriber<stereo_msgs::msg::DisparityImage> disparity_sub_;
  message_filters::Subscriber<sensor_msgs::msg::CameraInfo> left_camera_info_sub_;
  message_filters::Subscriber<sensor_msgs::msg::CameraInfo> right_camera_info_sub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr point_cloud_pub_;

  using ExactSyncPolicy = message_filters::sync_policies::ExactTime<
    sensor_msgs::msg::Image,
    stereo_msgs::msg::DisparityImage,
    sensor_msgs::msg::CameraInfo,
    sensor_msgs::msg::CameraInfo
  >;
  message_filters::Synchronizer<ExactSyncPolicy> exact_sync_;

  // Resources
  ::nvidia::isaac_ros::common::CudaStreamPtr cuda_stream_;
  PointCloudNodeCUDA cloud_compute_;
};

}  // namespace stereo_image_proc
}  // namespace isaac_ros
}  // namespace nvidia
