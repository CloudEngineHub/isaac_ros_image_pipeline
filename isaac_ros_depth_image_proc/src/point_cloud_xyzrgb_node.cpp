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

#include "isaac_ros_depth_image_proc/point_cloud_xyzrgb_node.hpp"

#include <cstdio>
#include <limits>
#include <memory>
#include <string>
#include <utility>

#include "cuda_buffer/cuda_buffer_api.hpp"
#include "isaac_ros_common/qos.hpp"
#include "pointcloud_conversions/pointcloud_conversions.hpp"
#include "rclcpp/rclcpp.hpp"

namespace nvidia
{
namespace isaac_ros
{
namespace depth_image_proc
{

namespace PcConv = pointcloud_conversions;

PointCloudXyzrgbNode::PointCloudXyzrgbNode(const rclcpp::NodeOptions & options)
: rclcpp::Node("PointCloudXyzrgbNode", options),
  skip_(declare_parameter<int>("skip", 1)),
  output_height_(declare_parameter<uint16_t>("output_height", 1200)),
  output_width_(declare_parameter<uint16_t>("output_width", 1920)),
  input_qos_size_(declare_parameter<int32_t>("input_qos_size", 10)),
  output_qos_size_(declare_parameter<int32_t>("output_qos_size", 10)),
  depth_sub_{},
  rgb_sub_{},
  camera_info_sub_{},
  exact_sync_{ExactSyncPolicy(static_cast<int>(input_qos_size_)), depth_sub_,
    rgb_sub_, camera_info_sub_}
{
  RCLCPP_DEBUG(get_logger(), "[PointCloudXyzrgbNode] Constructor");

  cuda_stream_ = ::nvidia::isaac_ros::common::createCudaStream("PointCloudXyzrgbNode");

  if (skip_ < 1) {
    RCLCPP_ERROR(get_logger(), "skip must be strictly positive, %d was provided", skip_);
    throw std::invalid_argument("skip must be strictly positive");
  }

  const rclcpp::QoS input_qos = ::isaac_ros::common::AddQosParameter(
    *this, "DEFAULT", "input_qos").keep_last(input_qos_size_);
  const rclcpp::QoS output_qos = ::isaac_ros::common::AddQosParameter(
    *this, "DEFAULT", "output_qos").keep_last(output_qos_size_);

  rclcpp::SubscriptionOptions sub_options;
  sub_options.use_intra_process_comm = rclcpp::IntraProcessSetting::Enable;
  sub_options.acceptable_buffer_backends = "any";
  rclcpp::PublisherOptions pub_options;
  pub_options.use_intra_process_comm = rclcpp::IntraProcessSetting::Enable;

  exact_sync_.registerCallback(std::bind(&PointCloudXyzrgbNode::OnSynchronizedInputs, this,
    std::placeholders::_1, std::placeholders::_2, std::placeholders::_3));

  depth_sub_.subscribe(this, "depth_registered/image_rect", input_qos, sub_options);
  rgb_sub_.subscribe(this, "rgb/image_rect_color", input_qos, sub_options);
  camera_info_sub_.subscribe(this, "rgb/camera_info", input_qos, sub_options);

  point_cloud_pub_ = create_publisher<sensor_msgs::msg::PointCloud2>(
    "points", output_qos, pub_options);

  camera_info_pub_ = create_publisher<sensor_msgs::msg::CameraInfo>(
    "camera_info", output_qos, pub_options);

  RCLCPP_DEBUG(get_logger(), "[PointCloudXyzrgbNode] Setup complete");
}

PointCloudXyzrgbNode::~PointCloudXyzrgbNode() {}

PointCloudProperties PointCloudXyzrgbNode::CreatePointCloudProperties(
  const sensor_msgs::msg::PointCloud2 & point_cloud_msg,
  const sensor_msgs::msg::CameraInfo::ConstSharedPtr & depth_info_msg,
  const int skip)
{
  PointCloudProperties point_cloud_properties;
  const unsigned int floats_per_point =
    point_cloud_msg.point_step / static_cast<unsigned int>(sizeof(float));

  point_cloud_properties.n_points = depth_info_msg->height *
    depth_info_msg->width / skip;
  point_cloud_properties.point_step = floats_per_point;
  point_cloud_properties.x_offset = 0;
  point_cloud_properties.y_offset = 1;
  point_cloud_properties.z_offset = 2;
  point_cloud_properties.rgb_offset = 3;
  point_cloud_properties.bad_point = std::numeric_limits<float>::quiet_NaN();

  return point_cloud_properties;
}

DepthProperties PointCloudXyzrgbNode::CreateDepthProperties(
  const sensor_msgs::msg::CameraInfo::ConstSharedPtr & depth_info_msg)
{
  DepthProperties depth_properties;

  depth_properties.width = depth_info_msg->width;
  depth_properties.height = depth_info_msg->height;
  depth_properties.f_x = depth_info_msg->k[0];
  depth_properties.f_y = depth_info_msg->k[4];
  depth_properties.c_x = depth_info_msg->k[2];
  depth_properties.c_y = depth_info_msg->k[5];

  depth_properties.red_offset = 0;
  depth_properties.green_offset = 1;
  depth_properties.blue_offset = 2;

  return depth_properties;
}

void PointCloudXyzrgbNode::OnSynchronizedInputs(
  const sensor_msgs::msg::Image::ConstSharedPtr & depth_msg,
  const sensor_msgs::msg::Image::ConstSharedPtr & rgb_msg,
  const sensor_msgs::msg::CameraInfo::ConstSharedPtr & camera_info_msg)
{
  RCLCPP_DEBUG(get_logger(), "[PointCloudXyzrgbNode] OnSynchronizedInputs");

  const uint32_t n_points = camera_info_msg->height * camera_info_msg->width / skip_;
  auto point_cloud_msg = PcConv::allocate_point_cloud_msg(n_points, 1, /*use_color=*/true);
  point_cloud_msg->header = depth_msg->header;

  {
    auto depth_read_handle = cuda_buffer_backend::from_input_buffer(
      depth_msg->data, *cuda_stream_);
    const float * depth_ptr = reinterpret_cast<const float *>(depth_read_handle.get_ptr());

    auto rgb_read_handle = cuda_buffer_backend::from_input_buffer(
      rgb_msg->data, *cuda_stream_);
    const uint8_t * rgb_ptr = reinterpret_cast<const uint8_t *>(rgb_read_handle.get_ptr());

    auto point_cloud_write_handle = PcConv::from_output_point_cloud_msg(
      *point_cloud_msg, *cuda_stream_);
    float * point_cloud_ptr = reinterpret_cast<float *>(point_cloud_write_handle.get_ptr());

    PointCloudProperties point_cloud_properties = CreatePointCloudProperties(
      *point_cloud_msg, camera_info_msg, skip_);
    DepthProperties depth_properties = CreateDepthProperties(camera_info_msg);

    cloud_compute_.DepthToPointCloudCuda(
      depth_ptr,
      rgb_ptr,
      point_cloud_ptr,
      point_cloud_properties,
      depth_properties,
      true,
      skip_,
      *cuda_stream_);
  }

  point_cloud_pub_->publish(std::move(point_cloud_msg));
  camera_info_pub_->publish(*camera_info_msg);
}

}  // namespace depth_image_proc
}  // namespace isaac_ros
}  // namespace nvidia

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(nvidia::isaac_ros::depth_image_proc::PointCloudXyzrgbNode)
