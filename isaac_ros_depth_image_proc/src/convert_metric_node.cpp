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

#include "isaac_ros_depth_image_proc/convert_metric_node.hpp"

#include <climits>

#include "cvcuda_conversions/cvcuda_conversions.hpp"
#include "isaac_ros_common/cuda_stream.hpp"
#include "sensor_msgs/image_encodings.hpp"

namespace nvidia
{
namespace isaac_ros
{
namespace depth_image_proc
{

namespace
{
constexpr float kMillimetresToMetres = 0.001f;
constexpr float kConvertOpBeta = 0.0f;
}  // namespace

ConvertMetricNode::ConvertMetricNode(const rclcpp::NodeOptions options)
: rclcpp::Node("convert_metric_node", options),
  input_queue_size_(declare_parameter<uint16_t>("input_queue_size", 10)),
  output_queue_size_(declare_parameter<uint16_t>("output_queue_size", 10))
{
  // Create CUDA stream
  cuda_stream_ = ::nvidia::isaac_ros::common::createCudaStream("ConvertMetricNode");

  const rclcpp::QoS input_qos = rclcpp::QoS(input_queue_size_).keep_last(input_queue_size_);
  const rclcpp::QoS output_qos = rclcpp::QoS(output_queue_size_).keep_last(output_queue_size_);

  rclcpp::SubscriptionOptions sub_options;
  sub_options.use_intra_process_comm = rclcpp::IntraProcessSetting::Enable;
  sub_options.acceptable_buffer_backends = "any";
  rclcpp::PublisherOptions pub_options;
  pub_options.use_intra_process_comm = rclcpp::IntraProcessSetting::Enable;

  image_sub_ = create_subscription<sensor_msgs::msg::Image>(
    "image_raw", input_qos,
    std::bind(&ConvertMetricNode::DepthCallback,
      this, std::placeholders::_1), sub_options);
  image_pub_ = create_publisher<sensor_msgs::msg::Image>(
    "image", output_qos, pub_options);
}

void ConvertMetricNode::DepthCallback(
  const sensor_msgs::msg::Image::ConstSharedPtr msg)
{
  if (msg->encoding != sensor_msgs::image_encodings::MONO16 &&
    msg->encoding != sensor_msgs::image_encodings::TYPE_16UC1)
  {
    RCLCPP_ERROR(
      get_logger(),
      "Input image format is not MONO16 or TYPE_16UC1 image."
      "This node only supports MONO16 or TYPE_16UC1 image."
      "The current image input is %s", msg->encoding.c_str());
    return;
  }

  auto output_msg = cvcuda_conversions::allocate_image_msg(
    msg->width, msg->height, sensor_msgs::image_encodings::TYPE_32FC1);
  output_msg->header = msg->header;

  {
    auto input_handle = cvcuda_conversions::from_input_image_msg(*msg, *cuda_stream_);
    auto output_handle = cvcuda_conversions::from_output_image_msg(*output_msg, *cuda_stream_);

    convert_op_(
      *cuda_stream_, input_handle, output_handle,
      kMillimetresToMetres, kConvertOpBeta);
  }

  image_pub_->publish(std::move(output_msg));
}

ConvertMetricNode::~ConvertMetricNode() {}

}  // namespace depth_image_proc
}  // namespace isaac_ros
}  // namespace nvidia

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(nvidia::isaac_ros::depth_image_proc::ConvertMetricNode)
