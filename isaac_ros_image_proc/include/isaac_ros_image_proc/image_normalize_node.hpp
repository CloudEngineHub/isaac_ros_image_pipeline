// SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
// Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef ISAAC_ROS_IMAGE_PROC__IMAGE_NORMALIZE_NODE_HPP_
#define ISAAC_ROS_IMAGE_PROC__IMAGE_NORMALIZE_NODE_HPP_

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "cvcuda/OpNormalize.hpp"
#include "cvcuda/OpConvertTo.hpp"
#include "isaac_ros_common/cuda_stream.hpp"
#include "isaac_ros_common/qos.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "nvcv/Tensor.hpp"
#include "rclcpp/rclcpp.hpp"

namespace nvidia
{
namespace isaac_ros
{
namespace image_proc
{

class ImageNormalizeNode : public rclcpp::Node
{
public:
  explicit ImageNormalizeNode(const rclcpp::NodeOptions & options);

  ~ImageNormalizeNode();

private:
  void imageSubCallback(const sensor_msgs::msg::Image::ConstSharedPtr msg);

  rclcpp::QoS input_qos_;
  rclcpp::QoS output_qos_;

  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr image_pub_;

  // Normalize node parameters
  const std::vector<double> mean_param_;
  const std::vector<double> stddev_param_;

  nvcv::Tensor mean_;
  nvcv::Tensor stddev_;
  nvcv::Tensor float_tensor_;
  uint32_t float_tensor_width_{0};
  uint32_t float_tensor_height_{0};
  std::string float_tensor_encoding_;

  // Resources
  ::nvidia::isaac_ros::common::CudaStreamPtr cuda_stream_;

  // CVCUDA operations
  cvcuda::Normalize norm_op_;
  cvcuda::ConvertTo convert_op_;
};

}  // namespace image_proc
}  // namespace isaac_ros
}  // namespace nvidia

#endif  // ISAAC_ROS_IMAGE_PROC__IMAGE_NORMALIZE_NODE_HPP_
