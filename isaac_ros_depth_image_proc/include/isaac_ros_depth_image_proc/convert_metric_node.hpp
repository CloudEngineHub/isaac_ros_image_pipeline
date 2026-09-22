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

#ifndef ISAAC_ROS_DEPTH_IMAGE_PROC__CONVERT_METRIC_NODE_HPP_
#define ISAAC_ROS_DEPTH_IMAGE_PROC__CONVERT_METRIC_NODE_HPP_

#include <memory>
#include <string>

#include "cvcuda/OpConvertTo.hpp"
#include "isaac_ros_common/cuda_stream.hpp"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"

namespace nvidia
{
namespace isaac_ros
{
namespace depth_image_proc
{

class ConvertMetricNode : public rclcpp::Node
{
public:
  explicit ConvertMetricNode(const rclcpp::NodeOptions options = rclcpp::NodeOptions());
  ~ConvertMetricNode();

private:
  void DepthCallback(const sensor_msgs::msg::Image::ConstSharedPtr msg);

  const uint16_t input_queue_size_;
  const uint16_t output_queue_size_;

  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr image_pub_;

  ::nvidia::isaac_ros::common::CudaStreamPtr cuda_stream_;

  cvcuda::ConvertTo convert_op_;
};

}  // namespace depth_image_proc
}  // namespace isaac_ros
}  // namespace nvidia

#endif  // ISAAC_ROS_DEPTH_IMAGE_PROC__CONVERT_METRIC_NODE_HPP_
