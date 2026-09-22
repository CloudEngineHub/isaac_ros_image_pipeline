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

#include "isaac_ros_stereo_image_proc/disparity_to_depth_node.hpp"

#include <cmath>

#include "cuda_buffer/cuda_buffer_api.hpp"
#include "isaac_ros_common/cuda_stream.hpp"
#include "isaac_ros_common/qos.hpp"
#include "isaac_ros_stereo_image_proc/disparity_to_depth.cu.hpp"
#include "sensor_msgs/image_encodings.hpp"
#include "vpi_conversions/vpi_conversions.hpp"

namespace nvidia
{
namespace isaac_ros
{
namespace stereo_image_proc
{

DisparityToDepthNode::DisparityToDepthNode(const rclcpp::NodeOptions & options)
: rclcpp::Node("disparity_to_depth_node", options),
  input_qos_{::isaac_ros::common::AddQosParameter(*this, "DEFAULT", "input_qos")},
  output_qos_{::isaac_ros::common::AddQosParameter(*this, "DEFAULT", "output_qos")}
{
  RCLCPP_DEBUG(get_logger(), "[DisparityToDepthNode] Constructor");

  rclcpp::SubscriptionOptions sub_options;
  sub_options.use_intra_process_comm = rclcpp::IntraProcessSetting::Enable;
  sub_options.acceptable_buffer_backends = "any";
  disparity_sub_ = create_subscription<stereo_msgs::msg::DisparityImage>(
    "disparity", input_qos_, std::bind(&DisparityToDepthNode::DisparityToDepthCallback, this,
      std::placeholders::_1), sub_options);

  rclcpp::PublisherOptions pub_options;
  pub_options.use_intra_process_comm = rclcpp::IntraProcessSetting::Enable;
  depth_pub_ = create_publisher<sensor_msgs::msg::Image>(
    "depth", output_qos_, pub_options);

  cuda_stream_ = ::nvidia::isaac_ros::common::createCudaStream("DisparityToDepthNode");

  RCLCPP_DEBUG(get_logger(), "[DisparityToDepthNode] Setup complete");
}

DisparityToDepthNode::~DisparityToDepthNode() {}

void DisparityToDepthNode::DisparityToDepthCallback(
  const stereo_msgs::msg::DisparityImage::ConstSharedPtr & disparity_msg)
{
  RCLCPP_DEBUG(get_logger(), "[DisparityToDepthNode] DisparityToDepthCallback");
  const uint32_t width = disparity_msg->image.width;
  const uint32_t height = disparity_msg->image.height;
  const float baseline = std::abs(disparity_msg->t);
  const float focal_length = disparity_msg->f;

  auto depth_msg = vpi_conversions::allocate_image_msg(
    width, height, sensor_msgs::image_encodings::TYPE_32FC1);

  // Copy metadata from disparity message
  depth_msg->header = disparity_msg->header;

  {
    // Get read handle and device pointer for input disparity (32FC1)
    auto disparity_read_handle = cuda_buffer_backend::from_input_buffer(
      disparity_msg->image.data, *cuda_stream_);
    const float * disparity_ptr =
      reinterpret_cast<const float *>(disparity_read_handle.get_ptr());

    // Create output depth image (32FC1) and get write handle
    auto depth_write_handle = cuda_buffer_backend::from_output_buffer(
      depth_msg->data, *cuda_stream_);
    float * depth_ptr = reinterpret_cast<float *>(depth_write_handle.get_ptr());

    // Convert disparity to depth on GPU
    cudaError_t err = disparity_to_depth_cuda(
      disparity_ptr, depth_ptr, baseline, focal_length,
      static_cast<int>(height), static_cast<int>(width), *cuda_stream_);
    if (err != cudaSuccess) {
      RCLCPP_ERROR(get_logger(), "CUDA kernel launch failed: %s", cudaGetErrorString(err));
      return;
    }
  }

  depth_pub_->publish(std::move(depth_msg));
}

}  // namespace stereo_image_proc
}  // namespace isaac_ros
}  // namespace nvidia

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(nvidia::isaac_ros::stereo_image_proc::DisparityToDepthNode)
