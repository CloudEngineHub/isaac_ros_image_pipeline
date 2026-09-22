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

#include "isaac_ros_common/cuda_stream.hpp"
#include "isaac_ros_common/qos.hpp"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "stereo_msgs/msg/disparity_image.hpp"

namespace nvidia
{
namespace isaac_ros
{
namespace stereo_image_proc
{

class DisparityToDepthNode : public rclcpp::Node
{
public:
  explicit DisparityToDepthNode(const rclcpp::NodeOptions & options);

  ~DisparityToDepthNode();

  DisparityToDepthNode(const DisparityToDepthNode &) = delete;
  DisparityToDepthNode & operator=(const DisparityToDepthNode &) = delete;

private:
  void DisparityToDepthCallback(
    const stereo_msgs::msg::DisparityImage::ConstSharedPtr & disparity_msg);

  // Parameters
  rclcpp::QoS input_qos_;
  rclcpp::QoS output_qos_;

  // Subscribers and publishers
  rclcpp::Subscription<stereo_msgs::msg::DisparityImage>::SharedPtr disparity_sub_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr depth_pub_;

  // Resources
  ::nvidia::isaac_ros::common::CudaStreamPtr cuda_stream_;
};

}  // namespace stereo_image_proc
}  // namespace isaac_ros
}  // namespace nvidia
