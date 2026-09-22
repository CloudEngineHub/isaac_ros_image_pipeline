// SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
// Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef ISAAC_ROS_IMAGE_PROC__IMAGE_FORMAT_CONVERTER_NODE_HPP_
#define ISAAC_ROS_IMAGE_PROC__IMAGE_FORMAT_CONVERTER_NODE_HPP_

#include <cstdint>
#include <memory>
#include <string>
#include <utility>

#include "cvcuda/OpAdvCvtColor.hpp"
#include "cvcuda/OpCvtColor.hpp"
#include "rclcpp/rclcpp.hpp"
#include "cvcuda_conversions/cvcuda_conversions.hpp"
#include "isaac_ros_common/cuda_stream.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "nvcv/ColorSpec.h"
#include "nvcv/Tensor.hpp"

namespace nvidia
{
namespace isaac_ros
{
namespace image_proc
{

class ImageFormatConverterNode : public rclcpp::Node
{
public:
  explicit ImageFormatConverterNode(const rclcpp::NodeOptions & options);

  ~ImageFormatConverterNode();

  ImageFormatConverterNode(const ImageFormatConverterNode &) = delete;

  ImageFormatConverterNode & operator=(const ImageFormatConverterNode &) = delete;

private:
  void imageSubCallback(const sensor_msgs::msg::Image::ConstSharedPtr msg);
  void convertMultiplanar(const sensor_msgs::msg::Image::ConstSharedPtr & msg);
  void convertMono8ToNV12(const sensor_msgs::msg::Image::ConstSharedPtr & msg);

  std::unique_ptr<sensor_msgs::msg::Image> allocateOutput(
    const sensor_msgs::msg::Image & msg);

  void publishOutput(
    std::unique_ptr<sensor_msgs::msg::Image> output_msg,
    const sensor_msgs::msg::Image & input_msg);

  // Parse a YUV color-spec string to a valid and corresponding NVCVColorSpec.
  static NVCVColorSpec ParseYuvColorSpec(const std::string & name);

  // Image format converter node parameters
  const std::string encoding_desired_;
  const int32_t image_width_;
  const int32_t image_height_;
  const NVCVColorSpec yuv_color_spec_;
  const rclcpp::QoS input_qos_;
  const rclcpp::QoS output_qos_;

  // Subscribers and publishers
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr image_pub_;

  // Resources
  ::nvidia::isaac_ros::common::CudaStreamPtr cuda_stream_;

  // CVCUDA operations
  cvcuda::CvtColor cvt_color_op_;
  // For handling the NV12 format.
  cvcuda::AdvCvtColor adv_cvt_color_op_;
};

}  // namespace image_proc
}  // namespace isaac_ros
}  // namespace nvidia

#endif  // ISAAC_ROS_IMAGE_PROC__IMAGE_FORMAT_CONVERTER_NODE_HPP_
