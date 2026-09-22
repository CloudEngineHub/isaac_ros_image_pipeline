// SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef VPI_CONVERSIONS__VPI_CONVERSIONS_HPP_
#define VPI_CONVERSIONS__VPI_CONVERSIONS_HPP_

// Header-only converter between a ROS 2 sensor_msgs/Image (whose `data` field is
// a rosidl::Buffer<uint8_t>) and NVIDIA VPI's native VPIImage, riding on
// whichever rosidl::Buffer backend is registered at runtime (e.g. cuda_buffer).
//
// This mirrors the cvcuda_conversions design (allocate_* / from_input_* /
// from_output_* / to_*), but for VPI. Like nvcv::TensorWrapData, a wrapped
// VPIImage does NOT own its memory and VPI ops are asynchronous, so
// from_input/output return an ImageWrapper holder that keeps the buffer
// Read/WriteHandle alive (and owns the VPIImage) until the recorded completion
// event fires. The holder must outlive all VPI ops on the image and be destroyed
// before the message is published.
//
// It depends only on sensor_msgs, cuda_buffer, and VPI, so it is upstreamable
// next to cvcuda_conversions.

#include <cuda_runtime.h>

#include <vpi/CUDAInterop.h>
#include <vpi/Image.h>
#include <vpi/Status.h>
#include <vpi/Stream.h>
#include <vpi/Types.h>

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "cuda_buffer/cuda_buffer_api.hpp"
#include "sensor_msgs/image_encodings.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "stereo_msgs/msg/disparity_image.hpp"

namespace nvidia
{
namespace isaac_ros
{
namespace vpi_conversions
{

// ROS sensor_msgs::image_encodings defines NV21/NV24 but not NV12.
constexpr char kEncodingNV12[] = "nv12";

// VPI format plus the channel/element geometry needed to wrap a buffer.
struct VpiFormatInfo
{
  VPIImageFormat image_format;
  // Per-plane VPI pixel type: size 1 for packed formats, 2 for NV12 (Y + UV).
  std::vector<VPIPixelType> pixel_types;
  int num_channels;       // packed channels, used for row-stride math
  int bytes_per_element;  // bytes per channel element
  bool is_nv12;
};

// Derive the VPI image format from a ROS image encoding string.
inline VpiFormatInfo vpi_format_from_encoding(const std::string & encoding)
{
  namespace enc = sensor_msgs::image_encodings;
  static const std::unordered_map<std::string, VpiFormatInfo> kMap({
          {enc::MONO8, {VPI_IMAGE_FORMAT_U8, {VPI_PIXEL_TYPE_U8}, 1, 1, false}},
          {enc::RGB8, {VPI_IMAGE_FORMAT_RGB8, {VPI_PIXEL_TYPE_3U8}, 3, 1, false}},
          {enc::BGR8, {VPI_IMAGE_FORMAT_BGR8, {VPI_PIXEL_TYPE_3U8}, 3, 1, false}},
          {enc::RGBA8, {VPI_IMAGE_FORMAT_RGBA8, {VPI_PIXEL_TYPE_4U8}, 4, 1, false}},
          {enc::BGRA8, {VPI_IMAGE_FORMAT_BGRA8, {VPI_PIXEL_TYPE_4U8}, 4, 1, false}},
          {enc::MONO16, {VPI_IMAGE_FORMAT_U16, {VPI_PIXEL_TYPE_U16}, 1, 2, false}},
          {enc::TYPE_16UC1, {VPI_IMAGE_FORMAT_U16, {VPI_PIXEL_TYPE_U16}, 1, 2, false}},
          {enc::TYPE_32FC1, {VPI_IMAGE_FORMAT_F32, {VPI_PIXEL_TYPE_F32}, 1, 4, false}},
          {kEncodingNV12,
            {VPI_IMAGE_FORMAT_NV12, {VPI_PIXEL_TYPE_U8, VPI_PIXEL_TYPE_2U8}, 1, 1, true}},
        });
  auto it = kMap.find(encoding);
  if (it == kMap.end()) {
    throw std::invalid_argument("vpi_conversions: unsupported encoding: " + encoding);
  }
  return it->second;
}

namespace detail
{

inline void check_vpi(VPIStatus status, const char * what)
{
  if (status != VPI_SUCCESS) {
    char buffer[VPI_MAX_STATUS_MESSAGE_LENGTH];
    vpiGetLastStatusMessage(buffer, sizeof(buffer));
    throw std::runtime_error(
            std::string("vpi_conversions: ") + what + " failed: " +
            vpiStatusGetName(status) + ": " + buffer);
  }
}

// Build a CUDA-pitch-linear VPIImageData over a device pointer, deriving the
// plane layout from the message row stride. NV12 is laid out as a stacked
// buffer: a Y plane of `height` rows followed by an interleaved UV plane of
// `height / 2` rows, both at `msg.step` stride (matching allocate_image_msg).
//
// base_ptr is const because it may come from a read-only ReadHandle (see
// from_input_image_msg). VPIImagePlanePitchLinear::pBase is a non-const void*
// and VPI has no read-only image concept, so the const_cast that pBase requires
// is localized here, at the wrapping boundary -- the single place the hazard
// lives. A VPIImage wrapped over a ReadHandle's buffer must therefore be used
// only as a VPI input: submitting it as an op's output target would write into
// the read-only source message buffer with no compile-time error.
inline VPIImageData build_image_data(
  const sensor_msgs::msg::Image & msg, const uint8_t * base_ptr, const VpiFormatInfo & info)
{
  VPIImageData data{};
  data.bufferType = VPI_IMAGE_BUFFER_CUDA_PITCH_LINEAR;
  VPIImageBufferPitchLinear & pitch = data.buffer.pitch;
  pitch.format = info.image_format;

  uint8_t * base = const_cast<uint8_t *>(base_ptr);

  auto fill_plane = [](VPIImagePlanePitchLinear & plane, VPIPixelType pixel_type,
    int32_t width, int32_t height, int32_t pitch_bytes, int64_t offset, uint8_t * base) {
      plane.pixelType = pixel_type;
      plane.width = width;
      plane.height = height;
      plane.pitchBytes = pitch_bytes;
      plane.offsetBytes = offset;
      plane.pBase = base + offset;
    };

  if (info.is_nv12) {
    pitch.numPlanes = 2;
    fill_plane(
      pitch.planes[0], info.pixel_types[0],
      static_cast<int32_t>(msg.width), static_cast<int32_t>(msg.height),
      static_cast<int32_t>(msg.step), 0, base);
    fill_plane(
      pitch.planes[1], info.pixel_types[1],
      static_cast<int32_t>(msg.width / 2), static_cast<int32_t>(msg.height / 2),
      static_cast<int32_t>(msg.step),
      static_cast<int64_t>(msg.step) * static_cast<int64_t>(msg.height), base);
  } else {
    pitch.numPlanes = 1;
    fill_plane(
      pitch.planes[0], info.pixel_types[0],
      static_cast<int32_t>(msg.width), static_cast<int32_t>(msg.height),
      static_cast<int32_t>(msg.step), 0, base);
  }
  return data;
}

}  // namespace detail

// Owns the wrapped VPIImage and the buffer Read/WriteHandle for the lifetime of
// the wrapping image. Move-only: destroying the wrapper destroys the VPIImage
// and releases the buffer handle (recording its CUDA completion event).
template<typename HandleT>
class ImageWrapper
{
public:
  ImageWrapper() = default;

  ImageWrapper(VPIImage image, VPIImageData image_data, HandleT handle)
  : image_(image), image_data_(image_data), handle_(std::move(handle)) {}

  ImageWrapper(const ImageWrapper &) = delete;
  ImageWrapper & operator=(const ImageWrapper &) = delete;

  ImageWrapper(ImageWrapper && other) noexcept
  : image_(other.image_), image_data_(other.image_data_), handle_(std::move(other.handle_))
  {
    other.image_ = nullptr;
  }

  ImageWrapper & operator=(ImageWrapper && other) noexcept
  {
    if (this != &other) {
      reset();
      image_ = other.image_;
      image_data_ = other.image_data_;
      handle_ = std::move(other.handle_);
      other.image_ = nullptr;
    }
    return *this;
  }

  ~ImageWrapper() {reset();}

  VPIImage image() const {return image_;}

  VPIImageData & image_data() {return image_data_;}
  const VPIImageData & image_data() const {return image_data_;}

  // Re-point the wrapped VPIImage at a new device buffer (same layout/stride).
  // Used when reusing a VPIImage wrapper across frames with fresh input handles.
  bool update_data_pointer(uint8_t * data)
  {
    image_data_.buffer.pitch.planes[0].pBase = data;
    detail::check_vpi(vpiImageSetWrapper(image_, &image_data_), "vpiImageSetWrapper");
    return true;
  }

  // Base device pointer of the wrapped buffer (for raw cudaMemcpy paths).
  // Returns const uint8_t * for a ReadHandle and uint8_t * for a WriteHandle,
  // so the handle's const-ness is preserved at compile time.
  auto data_ptr() {return handle_.get_ptr();}

private:
  void reset()
  {
    if (image_ != nullptr) {
      vpiImageDestroy(image_);
      image_ = nullptr;
    }
  }

  VPIImage image_{nullptr};
  VPIImageData image_data_{};
  HandleT handle_;
};

// Move-only RAII wrapper turning an existing cudaStream_t into a VPIStream.
// The underlying CUDA stream is owned elsewhere; this only owns the VPI wrapper.
class VpiStream
{
public:
  explicit VpiStream(cudaStream_t stream, uint64_t backend_flags = VPI_BACKEND_CUDA)
  {
    detail::check_vpi(
      vpiStreamCreateWrapperCUDA(stream, backend_flags, &stream_),
      "vpiStreamCreateWrapperCUDA");
  }

  VpiStream(const VpiStream &) = delete;
  VpiStream & operator=(const VpiStream &) = delete;

  VpiStream(VpiStream && other) noexcept
  : stream_(other.stream_) {other.stream_ = nullptr;}

  VpiStream & operator=(VpiStream && other) noexcept
  {
    if (this != &other) {
      reset();
      stream_ = other.stream_;
      other.stream_ = nullptr;
    }
    return *this;
  }

  ~VpiStream() {reset();}

  VPIStream get() const {return stream_;}
  operator VPIStream() const {return stream_;}

private:
  void reset()
  {
    if (stream_ != nullptr) {
      vpiStreamDestroy(stream_);
      stream_ = nullptr;
    }
  }

  VPIStream stream_{nullptr};
};

// Allocate a CUDA-backed sensor_msgs/Image sized for (width, height, encoding).
// Fills width/height/encoding/step/data; the caller sets the header.
inline std::unique_ptr<sensor_msgs::msg::Image> allocate_image_msg(
  uint32_t width, uint32_t height, const std::string & encoding)
{
  const VpiFormatInfo info = vpi_format_from_encoding(encoding);
  const size_t step = static_cast<size_t>(width) * info.num_channels * info.bytes_per_element;
  const size_t rows = info.is_nv12 ? (static_cast<size_t>(height) * 3 / 2) : height;

  auto msg = std::make_unique<sensor_msgs::msg::Image>();
  msg->width = width;
  msg->height = height;
  msg->encoding = encoding;
  msg->is_bigendian = 0;
  msg->step = step;
  msg->data = cuda_buffer_backend::allocate_buffer(step * rows);
  return msg;
}

// Read view of msg.data as a wrapped VPIImage, format derived from msg.encoding.
inline ImageWrapper<cuda_buffer_backend::ReadHandle> from_input_image_msg(
  const sensor_msgs::msg::Image & msg, cudaStream_t stream,
  uint64_t backend_flags = VPI_BACKEND_CUDA)
{
  const VpiFormatInfo info = vpi_format_from_encoding(msg.encoding);
  auto handle = cuda_buffer_backend::from_input_buffer(msg.data, stream);
  VPIImageData data = detail::build_image_data(msg, handle.get_ptr(), info);
  VPIImage image{nullptr};
  detail::check_vpi(
    vpiImageCreateWrapper(&data, nullptr, backend_flags, &image), "vpiImageCreateWrapper");
  return ImageWrapper<cuda_buffer_backend::ReadHandle>(image, data, std::move(handle));
}

// Write view of msg.data as a wrapped VPIImage, format derived from msg.encoding.
inline ImageWrapper<cuda_buffer_backend::WriteHandle> from_output_image_msg(
  sensor_msgs::msg::Image & msg, cudaStream_t stream,
  uint64_t backend_flags = VPI_BACKEND_CUDA)
{
  const VpiFormatInfo info = vpi_format_from_encoding(msg.encoding);
  auto handle = cuda_buffer_backend::from_output_buffer(msg.data, stream);
  VPIImageData data = detail::build_image_data(msg, handle.get_ptr(), info);
  VPIImage image{nullptr};
  detail::check_vpi(
    vpiImageCreateWrapper(&data, nullptr, backend_flags, &image), "vpiImageCreateWrapper");
  return ImageWrapper<cuda_buffer_backend::WriteHandle>(image, data, std::move(handle));
}

// Copy a standalone VPIImage into a pre-allocated message's buffer. Unlike
// from_output_image_msg (which is zero-copy: the op writes directly into the
// message), this performs a device-to-device copy and is meant for producers
// that hold a VPIImage not already backed by the message's buffer. The message
// must already be sized for its encoding (e.g. via allocate_image_msg). The
// copy honors both the VPIImage's and the message's row strides.
inline void to_image_msg(
  sensor_msgs::msg::Image & msg, VPIImage image, cudaStream_t stream)
{
  const VpiFormatInfo info = vpi_format_from_encoding(msg.encoding);

  VPIImageData src{};
  detail::check_vpi(
    vpiImageLockData(image, VPI_LOCK_READ, VPI_IMAGE_BUFFER_CUDA_PITCH_LINEAR, &src),
    "vpiImageLockData");

  auto handle = cuda_buffer_backend::from_output_buffer(msg.data, stream);
  uint8_t * dst_base = handle.get_ptr();
  const VPIImageBufferPitchLinear & pitch = src.buffer.pitch;

  // Row byte width per plane: packed formats use the full channel width; NV12
  // uses `width` bytes for both the Y plane and the interleaved UV plane.
  const size_t packed_row_bytes = static_cast<size_t>(msg.width) *
    info.num_channels * info.bytes_per_element;

  size_t dst_offset = 0;
  cudaError_t err = cudaSuccess;
  for (int32_t p = 0; p < pitch.numPlanes && err == cudaSuccess; ++p) {
    const VPIImagePlanePitchLinear & plane = pitch.planes[p];
    const size_t row_bytes = info.is_nv12 ? static_cast<size_t>(msg.width) : packed_row_bytes;
    err = cudaMemcpy2DAsync(
      dst_base + dst_offset, msg.step,
      plane.pBase, plane.pitchBytes,
      row_bytes, static_cast<size_t>(plane.height),
      cudaMemcpyDeviceToDevice, stream);
    dst_offset += static_cast<size_t>(msg.step) * static_cast<size_t>(plane.height);
  }

  vpiImageUnlock(image);
  if (err != cudaSuccess) {
    throw std::runtime_error(
            std::string("vpi_conversions::to_image_msg: cudaMemcpy2DAsync failed: ") +
            cudaGetErrorString(err));
  }
}

// Allocate a CUDA-backed message for (width, height, encoding) and copy the
// VPIImage into it. Fills width/height/encoding/step/data; caller sets the
// header.
inline std::unique_ptr<sensor_msgs::msg::Image> to_image_msg(
  VPIImage image, const std::string & encoding,
  uint32_t width, uint32_t height, cudaStream_t stream)
{
  auto msg = allocate_image_msg(width, height, encoding);
  to_image_msg(*msg, image, stream);
  return msg;
}

// ---- stereo_msgs/DisparityImage helpers ----
//
// A DisparityImage carries a nested sensor_msgs/Image (single-plane 32FC1
// float disparities) plus stereo geometry (f, T, min/max disparity, ...). The
// buffer <-> VPIImage wrapping is identical to a 32FC1 image, so these helpers
// delegate to the sensor_msgs/Image functions operating on the nested `image`
// field. The stereo geometry fields are the caller's responsibility.

// Allocate a CUDA-backed DisparityImage sized for (width, height). The nested
// image is 32FC1; the caller sets the header and stereo geometry fields.
inline std::unique_ptr<stereo_msgs::msg::DisparityImage> allocate_disparity_image_msg(
  uint32_t width, uint32_t height)
{
  auto msg = std::make_unique<stereo_msgs::msg::DisparityImage>();
  auto image = allocate_image_msg(width, height, sensor_msgs::image_encodings::TYPE_32FC1);
  msg->image = std::move(*image);
  return msg;
}

// Allocate a CUDA-backed DisparityImage matching a stereo image pair. The
// output dimensions come from the left image and its header follows the right
// image, matching the disparity image's reference frame.
inline std::unique_ptr<stereo_msgs::msg::DisparityImage>
allocate_disparity_image_msg_from_input(
  const sensor_msgs::msg::Image::ConstSharedPtr & left_image_msg,
  const sensor_msgs::msg::Image::ConstSharedPtr & right_image_msg)
{
  auto msg = allocate_disparity_image_msg(left_image_msg->width, left_image_msg->height);
  msg->header = right_image_msg->header;
  return msg;
}

// Read view of a DisparityImage's nested image as a wrapped VPIImage.
inline ImageWrapper<cuda_buffer_backend::ReadHandle> from_input_disparity_image_msg(
  const stereo_msgs::msg::DisparityImage & msg, cudaStream_t stream,
  uint64_t backend_flags = VPI_BACKEND_CUDA)
{
  return from_input_image_msg(msg.image, stream, backend_flags);
}

// Write view of a DisparityImage's nested image as a wrapped VPIImage.
inline ImageWrapper<cuda_buffer_backend::WriteHandle> from_output_disparity_image_msg(
  stereo_msgs::msg::DisparityImage & msg, cudaStream_t stream,
  uint64_t backend_flags = VPI_BACKEND_CUDA)
{
  return from_output_image_msg(msg.image, stream, backend_flags);
}

// Copy a standalone VPIImage into a DisparityImage's pre-allocated nested image
// buffer (device-to-device). The nested image must already be sized (e.g. via
// allocate_disparity_image_msg).
inline void to_disparity_image_msg(
  stereo_msgs::msg::DisparityImage & msg, VPIImage image, cudaStream_t stream)
{
  to_image_msg(msg.image, image, stream);
}

// Allocate a CUDA-backed DisparityImage for (width, height) and copy the
// VPIImage into its nested 32FC1 image. The caller sets the header and stereo
// geometry fields.
inline std::unique_ptr<stereo_msgs::msg::DisparityImage> to_disparity_image_msg(
  VPIImage image, uint32_t width, uint32_t height, cudaStream_t stream)
{
  auto msg = allocate_disparity_image_msg(width, height);
  to_image_msg(msg->image, image, stream);
  return msg;
}

}  // namespace vpi_conversions
}  // namespace isaac_ros
}  // namespace nvidia

#endif  // VPI_CONVERSIONS__VPI_CONVERSIONS_HPP_
