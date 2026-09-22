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

#include "isaac_ros_image_proc/image_normalize_node.hpp"

#include <climits>

#include "cuda_buffer/cuda_buffer_api.hpp"
#include "isaac_ros_cvcuda_utils/cvcuda_utilities.hpp"
#include "isaac_ros_common/cuda_stream.hpp"
#include "cvcuda_conversions/cvcuda_conversions.hpp"
#include "nvcv/TensorDataAccess.hpp"
#include "sensor_msgs/image_encodings.hpp"

namespace nvidia
{
namespace isaac_ros
{
namespace image_proc
{
ImageNormalizeNode::ImageNormalizeNode(const rclcpp::NodeOptions & options)
: rclcpp::Node("image_normalize_node", options),
  input_qos_{::isaac_ros::common::AddQosParameter(*this, "DEFAULT", "input_qos")},
  output_qos_{::isaac_ros::common::AddQosParameter(*this, "DEFAULT", "output_qos")},
  mean_param_{declare_parameter<std::vector<double>>("mean", {0.5, 0.5, 0.5})},
  stddev_param_(declare_parameter<std::vector<double>>("stddev", {0.5, 0.5, 0.5}))
{
  cuda_stream_ = ::nvidia::isaac_ros::common::createCudaStream("ImageNormalizeNode");

  std::vector<float> mean_float(mean_param_.begin(), mean_param_.end());
  std::vector<float> stddev_float(stddev_param_.begin(), stddev_param_.end());
  nvcv::TensorShape::ShapeType shape{nvcv::TensorShape::ShapeType{1, 1, 1,
      static_cast<int64_t>(mean_param_.size())}};

  nvcv::TensorShape tensor_shape{shape, nvcv::TENSOR_NHWC};

  mean_ = nvcv::Tensor(tensor_shape, nvcv::TYPE_F32);
  stddev_ = nvcv::Tensor(tensor_shape, nvcv::TYPE_F32);

  auto mean_data = mean_.exportData<nvcv::TensorDataStridedCuda>();
  auto mean_access = nvcv::TensorDataAccessStridedImagePlanar::Create(*mean_data);
  auto stddev_data = stddev_.exportData<nvcv::TensorDataStridedCuda>();
  auto stddev_access = nvcv::TensorDataAccessStridedImagePlanar::Create(*stddev_data);

  rclcpp::SubscriptionOptions sub_options;
  sub_options.use_intra_process_comm = rclcpp::IntraProcessSetting::Enable;
  // Accept GPU-backed image buffers; from_input_buffer promotes CPU buffers as needed.
  sub_options.acceptable_buffer_backends = "any";
  rclcpp::PublisherOptions pub_options;
  pub_options.use_intra_process_comm = rclcpp::IntraProcessSetting::Enable;

  // Create subscribers and publishers
  image_sub_ = create_subscription<sensor_msgs::msg::Image>(
    "image", input_qos_,
    std::bind(&ImageNormalizeNode::imageSubCallback, this, std::placeholders::_1),
    sub_options);
  image_pub_ = create_publisher<sensor_msgs::msg::Image>(
    "normalized_image", output_qos_, pub_options);
  CHECK_CUDA_ERROR(
    cudaMemcpy2D(
      mean_access->sampleData(0), mean_access->rowStride(), mean_float.data(),
      mean_float.size() * sizeof(float), mean_float.size() * sizeof(float), 1,
      cudaMemcpyHostToDevice),
    "cudaMemcpy2D failed");
  CHECK_CUDA_ERROR(
    cudaMemcpy2D(
      stddev_access->sampleData(0), stddev_access->rowStride(), stddev_float.data(),
      stddev_float.size() * sizeof(float), stddev_float.size() * sizeof(float), 1,
      cudaMemcpyHostToDevice),
    "cudaMemcpy2D failed");
}

void ImageNormalizeNode::imageSubCallback(const sensor_msgs::msg::Image::ConstSharedPtr msg)
{
  const cvcuda_utils::NVCVImageFormat format = cvcuda_utils::ToNVCVFormat(msg->encoding);

  auto output_msg = cvcuda_conversions::allocate_image_msg(
    msg->width, msg->height, format.float_encoding);
  output_msg->header = msg->header;

  // Scope the tensors so the read/write CUDA events are recorded on the stream
  // before the output message is published.
  {
    auto input_tensor = cvcuda_conversions::from_input_image_msg(*msg, *cuda_stream_);
    auto output_tensor = cvcuda_conversions::from_output_image_msg(*output_msg, *cuda_stream_);
    if (float_tensor_width_ != msg->width || float_tensor_height_ != msg->height ||
      float_tensor_encoding_ != msg->encoding)
    {
      float_tensor_ = nvcv::Tensor(
        1, {static_cast<int32_t>(msg->width), static_cast<int32_t>(msg->height)},
        format.float_format);
      float_tensor_width_ = msg->width;
      float_tensor_height_ = msg->height;
      float_tensor_encoding_ = msg->encoding;
    }

    convert_op_(*cuda_stream_, input_tensor, float_tensor_, 1.0f, 0.0f);
    norm_op_(*cuda_stream_, float_tensor_, mean_, stddev_, output_tensor,
      1.0f, 0.0f, 0.0f, CVCUDA_NORMALIZE_SCALE_IS_STDDEV);
  }

  image_pub_->publish(std::move(output_msg));
}

ImageNormalizeNode::~ImageNormalizeNode() {}

}  // namespace image_proc
}  // namespace isaac_ros
}  // namespace nvidia

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(nvidia::isaac_ros::image_proc::ImageNormalizeNode)
