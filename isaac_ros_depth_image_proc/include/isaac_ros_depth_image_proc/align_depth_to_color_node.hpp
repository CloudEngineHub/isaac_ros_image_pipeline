// SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
// Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef ISAAC_ROS_DEPTH_IMAGE_PROC__ALIGN_DEPTH_TO_COLOR_NODE_HPP_
#define ISAAC_ROS_DEPTH_IMAGE_PROC__ALIGN_DEPTH_TO_COLOR_NODE_HPP_

#include <Eigen/Dense>
#include <memory>
#include <mutex>
#include <optional>

#include "message_filters/subscriber.hpp"
#include "message_filters/synchronizer.hpp"
#include "message_filters/sync_policies/exact_time.hpp"
#include "tf2_ros/buffer.h"
#include "tf2_ros/transform_listener.h"

#include "geometry_msgs/msg/transform_stamped.hpp"
#include "isaac_ros_common/cuda_stream.hpp"
#include "isaac_ros_common/qos.hpp"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/image_encodings.hpp"
#include "sensor_msgs/msg/camera_info.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "tf2_eigen/tf2_eigen.hpp"

namespace nvidia
{
namespace isaac_ros
{
namespace depth_image_proc
{

class AlignDepthToColorNode : public rclcpp::Node
{
public:
  explicit AlignDepthToColorNode(const rclcpp::NodeOptions & options);
  ~AlignDepthToColorNode();

private:
  const uint16_t input_qos_size_;
  const uint16_t output_qos_size_;
  bool use_cached_camera_info_;
  bool enable_performance_logging_;

  message_filters::Subscriber<sensor_msgs::msg::Image> depth_sub_;
  message_filters::Subscriber<sensor_msgs::msg::CameraInfo> depth_info_sub_;
  message_filters::Subscriber<sensor_msgs::msg::CameraInfo> color_info_sub_;

  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr aligned_depth_pub_;

  using ExactPolicy = message_filters::sync_policies::ExactTime<
    sensor_msgs::msg::Image, sensor_msgs::msg::CameraInfo, sensor_msgs::msg::CameraInfo>;
  std::unique_ptr<message_filters::Synchronizer<ExactPolicy>> exact_sync_;

  std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
  std::shared_ptr<tf2_ros::TransformListener> tf_listener_;

  std::optional<Eigen::Matrix4d> color_pose_depth_;

  mutable std::mutex camera_info_mutex_;

  std::optional<sensor_msgs::msg::CameraInfo> depth_camera_info_;
  std::optional<sensor_msgs::msg::CameraInfo> color_camera_info_;
  std::optional<sensor_msgs::msg::Image> depth_image_buffer_;

  void OnSynchronizedInputs(
    const sensor_msgs::msg::Image::ConstSharedPtr & depth_msg,
    const sensor_msgs::msg::CameraInfo::ConstSharedPtr & depth_info_msg,
    const sensor_msgs::msg::CameraInfo::ConstSharedPtr & color_info_msg);

  void DepthCallback(const sensor_msgs::msg::Image::ConstSharedPtr & msg);
  void DepthCameraInfoCallback(const sensor_msgs::msg::CameraInfo::ConstSharedPtr & msg);
  void ColorCameraInfoCallback(const sensor_msgs::msg::CameraInfo::ConstSharedPtr & msg);

  // Shared computation used by both synchronized and individual callbacks
  void ComputeAndPublishAlignedDepth(
    const sensor_msgs::msg::Image::ConstSharedPtr & msg,
    const sensor_msgs::msg::CameraInfo & depth_camera_info,
    const sensor_msgs::msg::CameraInfo & color_camera_info);

  ::nvidia::isaac_ros::common::CudaStreamPtr cuda_stream_;
};

}  // namespace depth_image_proc
}  // namespace isaac_ros
}  // namespace nvidia

#endif  // ISAAC_ROS_DEPTH_IMAGE_PROC__ALIGN_DEPTH_TO_COLOR_NODE_HPP_
