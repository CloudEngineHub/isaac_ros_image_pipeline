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

#include "isaac_ros_stereo_image_proc/point_cloud_node.hpp"

#include <limits>
#include <memory>
#include <string>
#include <utility>

#include "cuda_buffer/cuda_buffer_api.hpp"
#include "isaac_ros_common/qos.hpp"
#include "pointcloud_conversions/pointcloud_conversions.hpp"
#include "sensor_msgs/image_encodings.hpp"

namespace nvidia
{
namespace isaac_ros
{
namespace stereo_image_proc
{

PointCloudNode::PointCloudNode(const rclcpp::NodeOptions & options)
: rclcpp::Node("point_cloud_node", options),
  use_color_(declare_parameter<bool>("use_color", false)),
  unit_scaling_(declare_parameter<float>("unit_scaling", 1.0)),
  input_queue_size_(declare_parameter<int64_t>("input_queue_size", 10)),
  output_queue_size_(declare_parameter<int64_t>("output_queue_size", 10)),
  left_image_sub_{},
  disparity_sub_{},
  left_camera_info_sub_{},
  right_camera_info_sub_{},
  exact_sync_{ExactSyncPolicy(static_cast<int>(input_queue_size_)), left_image_sub_,
    disparity_sub_, left_camera_info_sub_, right_camera_info_sub_}
{
  RCLCPP_DEBUG(get_logger(), "[PointCloudNode] Constructor");

  cuda_stream_ = ::nvidia::isaac_ros::common::createCudaStream("PointCloudNode");

  // Exact synchronization policy
  exact_sync_.registerCallback(std::bind(&PointCloudNode::PointCloudCallback, this,
    std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, std::placeholders::_4));

  const rclcpp::QoS input_qos = ::isaac_ros::common::AddQosParameter(
    *this, "DEFAULT", "input_qos").keep_last(input_queue_size_);
  const rclcpp::QoS output_qos = ::isaac_ros::common::AddQosParameter(
    *this, "DEFAULT", "output_qos").keep_last(output_queue_size_);

  // Subscribers (message_filters::Subscriber::subscribe takes rclcpp::QoS in ROS 2 Jazzy)
  rclcpp::SubscriptionOptions sub_options;
  sub_options.use_intra_process_comm = rclcpp::IntraProcessSetting::Enable;
  sub_options.acceptable_buffer_backends = "any";
  left_image_sub_.subscribe(this, "left/image_rect_color", input_qos, sub_options);
  disparity_sub_.subscribe(this, "disparity", input_qos, sub_options);
  left_camera_info_sub_.subscribe(this, "left/camera_info", input_qos, sub_options);
  right_camera_info_sub_.subscribe(this, "right/camera_info", input_qos, sub_options);

  // Publisher
  rclcpp::PublisherOptions pub_options;
  pub_options.use_intra_process_comm = rclcpp::IntraProcessSetting::Enable;
  point_cloud_pub_ = create_publisher<sensor_msgs::msg::PointCloud2>(
    "points2", output_qos, pub_options);

  cloud_compute_.SetUnitScaling(unit_scaling_);
  RCLCPP_DEBUG(get_logger(), "[PointCloudNode] Setup complete");
}

PointCloudNode::~PointCloudNode() {}

PointCloudProperties PointCloudNode::CreateCloudProperties(
  const sensor_msgs::msg::PointCloud2 & point_cloud_msg)
{
  PointCloudProperties cloud_properties;
  const int point_step = use_color_ ? 4 : 3;

  cloud_properties.point_row_step = point_step * point_cloud_msg.width;
  cloud_properties.point_step = point_step;
  cloud_properties.x_offset = 0;
  cloud_properties.y_offset = 1;
  cloud_properties.z_offset = 2;
  cloud_properties.is_bigendian = point_cloud_msg.is_bigendian;
  cloud_properties.bad_point = std::numeric_limits<float>::quiet_NaN();
  if (use_color_) {
    cloud_properties.rgb_offset = 3;
  }
  cloud_properties.buffer_size = point_cloud_msg.row_step * point_cloud_msg.height;

  return cloud_properties;
}

DisparityProperties PointCloudNode::CreateDisparityProperties(
  const stereo_msgs::msg::DisparityImage::ConstSharedPtr & disparity_msg)
{
  const unsigned int bytes_per_element = sensor_msgs::image_encodings::bitDepth(
    disparity_msg->image.encoding) / 8;

  DisparityProperties disparity_properties;
  const auto disp_step = disparity_msg->image.width * bytes_per_element;
  disparity_properties.row_step = disparity_msg->image.width;
  disparity_properties.height = disparity_msg->image.height;
  disparity_properties.width = disparity_msg->image.width;
  disparity_properties.buffer_size = disp_step * disparity_msg->image.height;

  return disparity_properties;
}

/*
 reprojection_matrix = [ FyTx  0     0   -FyCxTx     ]
                       [ 0     FxTx  0   -FxCyTx     ]
                       [ 0     0     0    FxFyTx     ]
                       [ 0     0     -Fy  Fy(Cx-Cx') ]
*/
CameraIntrinsics PointCloudNode::CreateCameraIntrinsics(
  const sensor_msgs::msg::CameraInfo::ConstSharedPtr & left_camera_info,
  const sensor_msgs::msg::CameraInfo::ConstSharedPtr & right_camera_info)
{
  CameraIntrinsics intrinsics;
  const double fx = left_camera_info->p[0];
  const double fy = left_camera_info->p[5];
  const double cx = left_camera_info->p[2];
  const double cx_prime = right_camera_info->p[2];
  const double cy = left_camera_info->p[6];
  const double tx = right_camera_info->p[0] == 0.0 ? right_camera_info->p[3] :
    right_camera_info->p[3] / right_camera_info->p[0];

  intrinsics.reprojection_matrix[0][0] = fy * tx;
  intrinsics.reprojection_matrix[0][3] = -fy * cx * tx;
  intrinsics.reprojection_matrix[1][1] = fx * tx;
  intrinsics.reprojection_matrix[1][3] = -fx * cy * tx;
  intrinsics.reprojection_matrix[2][3] = fx * fy * tx;
  intrinsics.reprojection_matrix[3][2] = -fy;
  // zero when disparities are pre-adjusted
  intrinsics.reprojection_matrix[3][3] = fy * (cx - cx_prime);

  return intrinsics;
}

RGBProperties PointCloudNode::CreateRGBProperties(
  const sensor_msgs::msg::Image::ConstSharedPtr & rgb_msg)
{
  RGBProperties rgb_properties;
  const auto rgb_step = rgb_msg->step;
  rgb_properties.row_step = rgb_step;
  rgb_properties.height = rgb_msg->height;
  rgb_properties.width = rgb_msg->width;
  rgb_properties.buffer_size = rgb_step * rgb_msg->height;

  // Only support 8 bit encodings (since point cloud can only support each color point with 8 bits)
  if (rgb_msg->encoding == "rgb8") {
    rgb_properties.red_offset = 0;
    rgb_properties.green_offset = 1;
    rgb_properties.blue_offset = 2;
    rgb_properties.color_step = 3;
  } else if (rgb_msg->encoding == "bgr8") {
    rgb_properties.blue_offset = 0;
    rgb_properties.green_offset = 1;
    rgb_properties.red_offset = 2;
    rgb_properties.color_step = 3;
  } else if (rgb_msg->encoding == "mono8") {
    rgb_properties.red_offset = 0;
    rgb_properties.green_offset = 0;
    rgb_properties.blue_offset = 0;
    rgb_properties.color_step = 1;
  } else {
    RCLCPP_ERROR(get_logger(), "[PointCloudNode] Unsupported encoding, Not publishing color");
    throw std::runtime_error("Unsupported encoding: " + rgb_msg->encoding);
  }
  return rgb_properties;
}

bool PointCloudNode::SelectDisparityFormatAndCompute(
  float * point_cloud_output,
  const PointCloudProperties & cloud_properties,
  const stereo_msgs::msg::DisparityImage::ConstSharedPtr & disparity_msg,
  const DisparityProperties & disparity_properties,
  const sensor_msgs::msg::Image::ConstSharedPtr & rgb_msg,
  const RGBProperties & rgb_properties,
  const CameraIntrinsics & intrinsics)
{
  // Select the appropriate disparity type so that the disparity image is interpreted correctly
  const auto & disparity_encoding = disparity_msg->image.encoding;

  if (disparity_encoding == sensor_msgs::image_encodings::TYPE_32FC1) {
    // Get read handles and device pointers for disparity and RGB inputs
    auto disparity_read_handle = cuda_buffer_backend::from_input_buffer(
      disparity_msg->image.data, *cuda_stream_);
    const float * disparity_buffer =
      reinterpret_cast<const float *>(disparity_read_handle.get_ptr());
    auto rgb_read_handle = cuda_buffer_backend::from_input_buffer(
      rgb_msg->data, *cuda_stream_);
    cloud_compute_.ComputePointCloudData<float>(
      point_cloud_output, cloud_properties, disparity_buffer, disparity_properties,
      rgb_read_handle.get_ptr(), rgb_properties, intrinsics, *cuda_stream_);
    CHECK_CUDA_ERROR(
      cudaStreamSynchronize(*cuda_stream_), "[PointCloudNode] cudaStreamSynchronize");
    return true;
  }

  RCLCPP_ERROR(get_logger(),
    "[PointCloudNode] Unsupported disparity encoding, Not computing point cloud");
  return false;
}

void PointCloudNode::PointCloudCallback(
  const sensor_msgs::msg::Image::ConstSharedPtr & left_image_msg,
  const stereo_msgs::msg::DisparityImage::ConstSharedPtr & disparity_msg,
  const sensor_msgs::msg::CameraInfo::ConstSharedPtr & left_camera_info_msg,
  const sensor_msgs::msg::CameraInfo::ConstSharedPtr & right_camera_info_msg)
{
  RCLCPP_DEBUG(get_logger(), "[PointCloudNode] PointCloudCallback");

  const uint32_t width = disparity_msg->image.width;
  const uint32_t height = disparity_msg->image.height;

  // Create output point cloud message
  auto point_cloud_msg =
    pointcloud_conversions::allocate_point_cloud_msg(width, height, use_color_);
  point_cloud_msg->header = left_image_msg->header;

  {
    // Get write handle for the output point cloud
    auto point_cloud_write_handle = pointcloud_conversions::from_output_point_cloud_msg(
      *point_cloud_msg, *cuda_stream_);
    float * point_cloud_ptr = reinterpret_cast<float *>(point_cloud_write_handle.get_ptr());

    // Format output point cloud message
    const auto cloud_properties = CreateCloudProperties(*point_cloud_msg);

    // Calculate camera intrinsics
    const auto intrinsics = CreateCameraIntrinsics(
      left_camera_info_msg, right_camera_info_msg);

    // Fulfill the disparity image properties
    const auto disparity_properties = CreateDisparityProperties(disparity_msg);

    // Fulfill the RGB image properties
    RGBProperties rgb_properties;
    if (use_color_) {
      rgb_properties = CreateRGBProperties(left_image_msg);
      cloud_compute_.SetUseColor(true);
    }

    // Compute the point cloud
    SelectDisparityFormatAndCompute(
      point_cloud_ptr, cloud_properties, disparity_msg, disparity_properties,
      left_image_msg, rgb_properties, intrinsics);
  }

  point_cloud_pub_->publish(std::move(point_cloud_msg));
}

}  // namespace stereo_image_proc
}  // namespace isaac_ros
}  // namespace nvidia

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(nvidia::isaac_ros::stereo_image_proc::PointCloudNode)
