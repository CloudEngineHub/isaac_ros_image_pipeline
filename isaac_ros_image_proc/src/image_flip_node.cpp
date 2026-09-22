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

#include "isaac_ros_image_proc/image_flip_node.hpp"

#include <cuda_runtime.h>
#include <string>

#include "cvcuda/OpFlip.hpp"
#include "cuda_buffer/cuda_buffer_api.hpp"
#include "isaac_ros_common/cuda_stream.hpp"
#include "isaac_ros_common/qos.hpp"
#include "isaac_ros_cvcuda_utils/cvcuda_utilities.hpp"
#include "rclcpp/rclcpp.hpp"

namespace nvidia
{
namespace isaac_ros
{
namespace image_proc
{
ImageFlipNode::ImageFlipNode(const rclcpp::NodeOptions & options)
: rclcpp::Node("image_flip_node", options),
  flip_mode_(declare_parameter<std::string>("flip_mode", "BOTH")),
  input_qos_(::isaac_ros::common::AddQosParameter(*this, "DEFAULT", "input_qos", 10)),
  output_qos_(::isaac_ros::common::AddQosParameter(*this, "DEFAULT", "output_qos", 10))
{
  RCLCPP_DEBUG(get_logger(), "[ImageFlipNode] Constructor");

  cuda_stream_ = ::nvidia::isaac_ros::common::createCudaStream("ImageFlipNode");

  // Subscription options
  rclcpp::SubscriptionOptions sub_options;
  sub_options.use_intra_process_comm = rclcpp::IntraProcessSetting::Enable;
  // Accept GPU-backed image buffers; from_input_buffer promotes CPU buffers as needed.
  sub_options.acceptable_buffer_backends = "any";
  // Publisher options
  rclcpp::PublisherOptions pub_options;
  pub_options.use_intra_process_comm = rclcpp::IntraProcessSetting::Enable;

  // Create subscribers and publishers
  image_sub_ = create_subscription<sensor_msgs::msg::Image>(
    "image", input_qos_,
    std::bind(&ImageFlipNode::imageSubCallback, this, std::placeholders::_1), sub_options);
  image_pub_ = create_publisher<sensor_msgs::msg::Image>(
    "image_flipped", output_qos_, pub_options);
}

ImageFlipNode::~ImageFlipNode() {}

void ImageFlipNode::imageSubCallback(const sensor_msgs::msg::Image::ConstSharedPtr msg)
{
  auto output_msg = cvcuda_conversions::allocate_image_msg(msg->width, msg->height, msg->encoding);
  output_msg->header = msg->header;

  // Scope the tensors so the read/write CUDA events are recorded on the stream
  // before the output message is published.
  {
    auto input_tensor = cvcuda_conversions::from_input_image_msg(*msg, *cuda_stream_);
    auto output_tensor = cvcuda_conversions::from_output_image_msg(*output_msg, *cuda_stream_);

    int32_t flip_flag = cvcuda_utils::ToNVCVFlipMode(flip_mode_);
    flip_op_(*cuda_stream_, input_tensor, output_tensor, flip_flag);
  }

  image_pub_->publish(std::move(output_msg));
}

}  // namespace image_proc
}  // namespace isaac_ros
}  // namespace nvidia

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(nvidia::isaac_ros::image_proc::ImageFlipNode)
