// SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
// Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef ISAAC_ROS_IMAGE_PROC__IMAGE_FLIP_NODE_HPP_
#define ISAAC_ROS_IMAGE_PROC__IMAGE_FLIP_NODE_HPP_

#include <string>

#include "cvcuda/OpFlip.hpp"
#include "rclcpp/rclcpp.hpp"
#include "isaac_ros_common/cuda_stream.hpp"
#include "cvcuda_conversions/cvcuda_conversions.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "nvcv/Tensor.hpp"

namespace nvidia
{
namespace isaac_ros
{
namespace image_proc
{

class ImageFlipNode : public rclcpp::Node
{
public:
  explicit ImageFlipNode(const rclcpp::NodeOptions & options);

  ~ImageFlipNode();

  ImageFlipNode(const ImageFlipNode &) = delete;

  ImageFlipNode & operator=(const ImageFlipNode &) = delete;

private:
  void imageSubCallback(const sensor_msgs::msg::Image::ConstSharedPtr msg);

  // Flip node parameters
  const std::string flip_mode_;
  const rclcpp::QoS input_qos_;
  const rclcpp::QoS output_qos_;

  // Flip node subscribers and publishers
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr image_pub_;

  // Resources
  ::nvidia::isaac_ros::common::CudaStreamPtr cuda_stream_;

  // Flip node CVCUDA operation
  cvcuda::Flip flip_op_;
};

}  // namespace image_proc
}  // namespace isaac_ros
}  // namespace nvidia

#endif  // ISAAC_ROS_IMAGE_PROC__IMAGE_FLIP_NODE_HPP_
