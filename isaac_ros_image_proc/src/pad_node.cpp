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

#include "isaac_ros_image_proc/pad_node.hpp"

#include <cuda_runtime.h>

#include "cuda_buffer/cuda_buffer_api.hpp"
#include "isaac_ros_common/cuda_stream.hpp"
#include "isaac_ros_common/qos.hpp"
#include "cvcuda_conversions/cvcuda_conversions.hpp"
#include "isaac_ros_cvcuda_utils/cvcuda_utilities.hpp"
#include "sensor_msgs/image_encodings.hpp"

namespace nvidia
{
namespace isaac_ros
{
namespace image_proc
{

namespace
{
const std::unordered_map<std::string, PaddingType> kStringToPaddingTypeMap({
          {"CENTER", PaddingType::kCenter},
          {"TOP_LEFT", PaddingType::kTopLeft},
          {"TOP_RIGHT", PaddingType::kTopRight},
          {"BOTTOM_LEFT", PaddingType::kBottomLeft},
          {"BOTTOM_RIGHT", PaddingType::kBottomRight}
        });

constexpr uint8_t kBitsInByte = 8;
constexpr uint8_t kBatchSize = 1;

uint32_t CalculateOffset(
  const uint16_t input_width,
  const uint16_t input_height,
  const uint16_t output_width,
  const uint16_t output_height,
  const PaddingType & padding_type,
  const std::vector<int64_t> & output_strides
)
{
  uint32_t offset = 0;
  switch (padding_type) {
    case PaddingType::kBottomRight: {
        offset = 0;
        return offset;
      }
    case PaddingType::kBottomLeft: {
        uint32_t start_x = output_width - input_width;
        offset = start_x * output_strides[2];
        return offset;
      }
    case PaddingType::kTopRight: {
        uint32_t start_y = output_height - input_height;
        offset = start_y * output_strides[1];
        return offset;
      }
    case PaddingType::kTopLeft: {
        uint32_t start_y = output_height - input_height;
        uint32_t start_x = output_width - input_width;
        offset = start_y * output_strides[1];
        offset += start_x * output_strides[2];
        return offset;
      }
    default: {
        throw std::invalid_argument("[PadNode] Unsupported Padding Type");
      }
  }
}

void CalculateStrides(
  const nvcv::ImageFormat & fmt,
  const uint16_t width,
  const uint16_t height,
  std::vector<int64_t> & strides
)
{
  nvcv::Tensor::Requirements tensor_reqs = nvcv::Tensor::CalcRequirements(
    kBatchSize, {width, height}, fmt);
  uint32_t input_image_channels = fmt.numChannels();
  uint32_t bytes_per_pixel = (
    nvcv::DataType{tensor_reqs.dtype}.bitsPerPixel() + kBitsInByte - 1) / kBitsInByte;
  strides.resize(4);
  strides[3] = bytes_per_pixel;
  strides[2] = input_image_channels * strides[3];
  strides[1] = width * strides[2];
  strides[0] = height * strides[1];
}

}  // namespace

PadNode::PadNode(const rclcpp::NodeOptions & options)
: rclcpp::Node("padding_node", options),
  input_qos_{::isaac_ros::common::AddQosParameter(*this, "DEFAULT", "input_qos")},
  output_qos_{::isaac_ros::common::AddQosParameter(*this, "DEFAULT", "output_qos")},
  output_image_width_(declare_parameter<uint16_t>("output_image_width", 1920)),
  output_image_height_(declare_parameter<uint16_t>("output_image_height", 1200)),
  padding_type_(declare_parameter<std::string>("padding_type", "CENTER")),
  border_type_(declare_parameter<std::string>("border_type", "CONSTANT")),
  border_pixel_color_value_(
    declare_parameter<std::vector<double>>("border_pixel_color_value", {0.0, 0.0, 0.0, 0.0}))
{
  auto img_padding_itr = kStringToPaddingTypeMap.find(padding_type_);
  if (img_padding_itr == std::end(kStringToPaddingTypeMap)) {
    RCLCPP_ERROR(get_logger(), "[PadNode] Unsupported padding type [%s]", padding_type_.c_str());
    throw std::invalid_argument("[PadNode] Unsupported padding type");
  }
  padding_type_val_ = img_padding_itr->second;
  border_type_val_ = cvcuda_utils::ToNVCVBorderType(border_type_);

  if (border_pixel_color_value_.size() != 4) {
    RCLCPP_ERROR(
      get_logger(),
      "[PadNode] Invalid length of border_pixel_channel_values. Needed 4, given %ld",
      border_pixel_color_value_.size());
    throw std::invalid_argument("[PadNode] Invalid length of border_pixel_channel_values");
  }
  for (const auto & val : border_pixel_color_value_) {
    border_values_float_.push_back(static_cast<float>(val));
  }
  cuda_stream_ = ::nvidia::isaac_ros::common::createCudaStream("PadNode");

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
    std::bind(&PadNode::imageSubCallback, this, std::placeholders::_1), sub_options);
  image_pub_ = create_publisher<sensor_msgs::msg::Image>("padded_image", output_qos_, pub_options);
}

PadNode::~PadNode() {}

void PadNode::imageSubCallback(const sensor_msgs::msg::Image::ConstSharedPtr msg)
{
  uint16_t input_width = msg->width;
  uint16_t input_height = msg->height;

  if ((input_width > output_image_width_) || (input_height > output_image_height_)) {
    RCLCPP_ERROR(
      get_logger(),
      "Input image dims > Output image dims."
    );
    throw std::runtime_error(
            "Error: Input image dims > Output image dims.");
  }
  auto input_format = cvcuda_utils::ToNVCVFormat(msg->encoding);
  int num_channels{sensor_msgs::image_encodings::numChannels(msg->encoding)};
  int bytes_per_channel = sensor_msgs::image_encodings::bitDepth(msg->encoding) / CHAR_BIT;
  RCLCPP_DEBUG(get_logger(),
    "[PadNode] Input width: %d, height: %d, num_channels: %d,"
    "bytes_per_channel: %d",
    msg->width, msg->height, num_channels, bytes_per_channel);

  auto output_msg = cvcuda_conversions::allocate_image_msg(
    output_image_width_, output_image_height_, msg->encoding);
  output_msg->header = msg->header;

  // Scope the tensors so the read/write CUDA events are recorded on the stream
  // before the output message is published.
  {
    auto input_tensor = cvcuda_conversions::from_input_image_msg(*msg, *cuda_stream_);
    auto output_tensor = cvcuda_conversions::from_output_image_msg(*output_msg, *cuda_stream_);

    void * input_data = cvcuda_conversions::tensor_base_ptr(input_tensor);
    uint8_t * output_data = cvcuda_conversions::tensor_base_ptr(output_tensor);

    if (padding_type_val_ == PaddingType::kCenter) {
      int top = (output_image_height_ - input_height) / 2;
      int left = (output_image_width_ - input_width) / 2;

      make_border_op_(
        *cuda_stream_, input_tensor,
        output_tensor, top, left, border_type_val_,
        {border_values_float_[0], border_values_float_[1],
          border_values_float_[2], border_values_float_[3]}
      );
    } else {
      // Initialize to 0 image.
      CHECK_CUDA_ERROR(
        cudaMemsetAsync(output_data, 0,
          output_image_height_ * output_image_width_ * num_channels * bytes_per_channel,
          *cuda_stream_),
        "cudaMemsetAsync failed");

      // Calculate offset for the specified type of padding
      std::vector<int64_t> input_strides;
      CalculateStrides(input_format.format, input_width, input_height, input_strides);
      std::vector<int64_t> output_strides;
      CalculateStrides(input_format.format, output_image_width_, output_image_height_,
        output_strides);

      uint32_t offset = CalculateOffset(
        input_width, input_height, output_image_width_, output_image_height_,
        padding_type_val_, output_strides
      );

      // Copy input image to to the corner.
      CHECK_CUDA_ERROR(
        cudaMemcpy2DAsync(
          reinterpret_cast<uint8_t *>(output_data + offset),
          output_strides[1],
          input_data,
          input_strides[1],
          input_strides[1],
          input_height,
          cudaMemcpyDefault,
          *cuda_stream_),
          "cudaMemcpy2DAsync failed");
    }
  }

  image_pub_->publish(std::move(output_msg));
}

}  // namespace image_proc
}  // namespace isaac_ros
}  // namespace nvidia

// Register as component
#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(nvidia::isaac_ros::image_proc::PadNode)
