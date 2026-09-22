// SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
// Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef ISAAC_ROS_IMAGE_PROC__ALPHA_BLEND_NODE_HPP_
#define ISAAC_ROS_IMAGE_PROC__ALPHA_BLEND_NODE_HPP_

#include <memory>

#include <message_filters/subscriber.hpp>
#include <message_filters/synchronizer.hpp>
#include <message_filters/sync_policies/exact_time.hpp>

#include "isaac_ros_common/cuda_stream.hpp"
#include "isaac_ros_image_proc/alpha_blend.cu.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "rclcpp/rclcpp.hpp"

namespace nvidia
{
namespace isaac_ros
{
namespace image_proc
{

class AlphaBlendNode : public rclcpp::Node
{
public:
  explicit AlphaBlendNode(const rclcpp::NodeOptions & options);

  ~AlphaBlendNode();

private:
  // Callback function
  void InputCallback(
    const sensor_msgs::msg::Image::ConstSharedPtr & img_ptr,
    const sensor_msgs::msg::Image::ConstSharedPtr & mask_ptr);

  // Alpha blend node parameters
  double alpha_;
  int64_t input_queue_size_;
  int64_t output_queue_size_;

  // Subscribers and publishers
  message_filters::Subscriber<sensor_msgs::msg::Image> image_sub_;
  message_filters::Subscriber<sensor_msgs::msg::Image> mask_sub_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr image_pub_;

  // Exact message sync policy
  using ExactPolicy = ::message_filters::sync_policies::ExactTime<
    sensor_msgs::msg::Image, sensor_msgs::msg::Image>;
  message_filters::Synchronizer<ExactPolicy> sync_;

  // Resources
  ::nvidia::isaac_ros::common::CudaStreamPtr cuda_stream_;
};

}  // namespace image_proc
}  // namespace isaac_ros
}  // namespace nvidia

#endif  // ISAAC_ROS_IMAGE_PROC__ALPHA_BLEND_NODE_HPP_
