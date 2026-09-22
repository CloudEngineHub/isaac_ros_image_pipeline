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

#ifndef CVCUDA_CONVERSIONS__CVCUDA_CONVERSIONS_HPP_
#define CVCUDA_CONVERSIONS__CVCUDA_CONVERSIONS_HPP_

// Header-only converter between a ROS 2 sensor_msgs/Image (whose `data` field is
// a rosidl::Buffer<uint8_t>) and CV-CUDA's native nvcv::Tensor, riding on
// whichever rosidl::Buffer backend is registered at runtime (e.g. cuda_buffer).
//
// This mirrors the ros2/rosidl_buffer_backends torch_conversions design
// (allocate_* / from_input_* / from_output_* / to_*), but for CV-CUDA. Just as
// torch_conversions hands the buffer Read/WriteHandle to the DLPack deleter,
// from_input/output hand it to the nvcv::TensorWrapData cleanup callback, so the
// returned nvcv::Tensor owns the handle. CV-CUDA ops are asynchronous, so the
// tensor must outlive every op on it and be destroyed before the message is
// published: dropping the last tensor reference releases the handle, which
// records the read/write completion event on the stream.
//
// It depends only on sensor_msgs, cuda_buffer, and CV-CUDA, so it is upstreamable
// next to torch_conversions.

#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "cuda_buffer/cuda_buffer_api.hpp"
#include "isaac_ros_tensor_msgs/msg/tensor_list.hpp"
#include "nvcv/Tensor.hpp"
#include "nvcv/TensorDataAccess.hpp"
#include "nvcv/TensorLayout.hpp"
#include "sensor_msgs/image_encodings.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "tensor_msgs/msg/experimental_tensor.hpp"

namespace cvcuda_conversions
{

using Tensor = tensor_msgs::msg::ExperimentalTensor;
using TensorList = isaac_ros_tensor_msgs::msg::TensorList;

// ROS sensor_msgs::image_encodings defines NV21/NV24 but not NV12.
constexpr char kEncodingNV12[] = "nv12";

// nvcv format plus the channel/element geometry needed to wrap a buffer.
struct ImageFormatInfo
{
  nvcv::ImageFormat format;
  int num_channels;
  int bytes_per_element;
  bool is_nv12;
};

// Derive the CV-CUDA image format from a ROS image encoding string.
inline ImageFormatInfo image_format_from_encoding(const std::string & encoding)
{
  namespace enc = sensor_msgs::image_encodings;
  static const std::unordered_map<std::string, ImageFormatInfo> kMap({
      {enc::MONO8, {nvcv::FMT_Y8, 1, 1, false}},
      {enc::RGB8, {nvcv::FMT_RGB8, 3, 1, false}},
      {enc::BGR8, {nvcv::FMT_BGR8, 3, 1, false}},
      {enc::RGBA8, {nvcv::FMT_RGBA8, 4, 1, false}},
      {enc::BGRA8, {nvcv::FMT_BGRA8, 4, 1, false}},
      {enc::MONO16, {nvcv::FMT_Y16, 1, 2, false}},
      {enc::TYPE_16UC1, {nvcv::FMT_Y16, 1, 2, false}},
      {enc::TYPE_32FC1, {nvcv::FMT_F32, 1, 4, false}},
      {enc::TYPE_32FC3, {nvcv::FMT_RGBf32, 3, 4, false}},
      {enc::TYPE_32FC4, {nvcv::FMT_RGBAf32, 4, 4, false}},
      {kEncodingNV12, {nvcv::FMT_Y8, 1, 1, true}},
    });
  auto it = kMap.find(encoding);
  if (it == kMap.end()) {
    throw std::invalid_argument("cvcuda_conversions: Unsupported encoding: " + encoding);
  }
  return it->second;
}

// Base device pointer of a wrapped tensor (for raw cudaMemcpy paths).
inline uint8_t * tensor_base_ptr(const nvcv::Tensor & tensor)
{
  auto data = tensor.exportData<nvcv::TensorDataStridedCuda>();
  return data ? reinterpret_cast<uint8_t *>(data->basePtr()) : nullptr;
}

namespace detail
{

// Cleanup callback target for nvcv::TensorWrapData: keeps the buffer
// Read/WriteHandle alive for as long as the wrapping tensor exists, the way
// torch_conversions' DLPack context does for at::Tensor.
template<typename HandleT>
struct HandleOwner
{
  void operator()(const nvcv::TensorData &) const {}

  HandleT handle;
};

// Wrap the handle's memory as an NHWC nvcv::Tensor using the message row
// stride. The tensor takes ownership of the handle.
template<typename HandleT>
nvcv::Tensor wrap_tensor(
  const sensor_msgs::msg::Image & msg, const ImageFormatInfo & info, HandleT handle)
{
  const int32_t tensor_height = info.is_nv12 ?
    static_cast<int32_t>(msg.height * 3 / 2) : static_cast<int32_t>(msg.height);

  nvcv::TensorDataStridedCuda::Buffer tensor_buffer;
  tensor_buffer.strides[3] = info.bytes_per_element;
  tensor_buffer.strides[2] = info.num_channels * tensor_buffer.strides[3];
  tensor_buffer.strides[1] = msg.step;
  tensor_buffer.strides[0] = tensor_height * tensor_buffer.strides[1];
  tensor_buffer.basePtr = const_cast<NVCVByte *>(
    reinterpret_cast<const NVCVByte *>(handle.get_ptr()));

  constexpr size_t kBatchSize{1};
  nvcv::Tensor::Requirements reqs{nvcv::Tensor::CalcRequirements(
      kBatchSize, {static_cast<int32_t>(msg.width), tensor_height}, info.format)};
  nvcv::TensorDataStridedCuda tensor_data{
    nvcv::TensorShape{reqs.shape, reqs.rank, reqs.layout},
    nvcv::DataType{reqs.dtype}, tensor_buffer};
  return nvcv::TensorWrapData(tensor_data, HandleOwner<HandleT>{std::move(handle)});
}

}  // namespace detail

// Allocate a CUDA-backed sensor_msgs/Image sized for (width, height, encoding).
// Fills width/height/encoding/step/data; the caller sets the header.
inline std::unique_ptr<sensor_msgs::msg::Image> allocate_image_msg(
  uint32_t width, uint32_t height, const std::string & encoding)
{
  const ImageFormatInfo info = image_format_from_encoding(encoding);
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

// Read view of msg.data as an nvcv::Tensor, format derived from msg.encoding.
// The tensor owns the buffer's ReadHandle: releasing the tensor records the
// read-completion event on `stream`.
inline nvcv::Tensor from_input_image_msg(
  const sensor_msgs::msg::Image & msg, cudaStream_t stream)
{
  const ImageFormatInfo info = image_format_from_encoding(msg.encoding);
  return detail::wrap_tensor(
    msg, info, cuda_buffer_backend::from_input_buffer(msg.data, stream));
}

// Write view of msg.data as an nvcv::Tensor, format derived from msg.encoding.
// The tensor owns the buffer's WriteHandle: releasing the tensor records the
// write-completion event on `stream`.
inline nvcv::Tensor from_output_image_msg(
  sensor_msgs::msg::Image & msg, cudaStream_t stream)
{
  const ImageFormatInfo info = image_format_from_encoding(msg.encoding);
  return detail::wrap_tensor(
    msg, info, cuda_buffer_backend::from_output_buffer(msg.data, stream));
}

// Copy a standalone nvcv::Tensor into a pre-allocated message's buffer. Unlike
// from_output_image_msg (which is zero-copy: the op writes directly into the
// message), this performs a device-to-device copy and is meant for producers
// that hold a tensor not already backed by the message's buffer. The message
// must already be sized for its encoding (e.g. via allocate_image_msg). The
// copy honors both the tensor's and the message's row strides.
inline void to_image_msg(
  sensor_msgs::msg::Image & msg, const nvcv::Tensor & tensor, cudaStream_t stream)
{
  const ImageFormatInfo info = image_format_from_encoding(msg.encoding);
  const size_t rows = info.is_nv12 ?
    (static_cast<size_t>(msg.height) * 3 / 2) : static_cast<size_t>(msg.height);
  const size_t row_bytes = static_cast<size_t>(msg.width) *
    info.num_channels * info.bytes_per_element;

  auto data = tensor.exportData<nvcv::TensorDataStridedCuda>();
  if (!data) {
    throw std::runtime_error(
            "cvcuda_conversions::to_image_msg: tensor does not hold strided CUDA data");
  }
  auto access = nvcv::TensorDataAccessStridedImagePlanar::Create(*data);
  if (!access) {
    throw std::runtime_error(
            "cvcuda_conversions::to_image_msg: could not access tensor data");
  }

  auto handle = cuda_buffer_backend::from_output_buffer(msg.data, stream);
  cudaError_t err = cudaMemcpy2DAsync(
    handle.get_ptr(), msg.step,
    access->sampleData(0), access->rowStride(),
    row_bytes, rows,
    cudaMemcpyDeviceToDevice, stream);
  if (err != cudaSuccess) {
    throw std::runtime_error(
            std::string("cvcuda_conversions::to_image_msg: cudaMemcpy2DAsync failed: ") +
            cudaGetErrorString(err));
  }
}

// Allocate a CUDA-backed message for (width, height, encoding) and copy the
// tensor into it. width/height are explicit because a ROS encoding and image
// dimensions are not uniquely recoverable from an nvcv::Tensor (e.g. NV12 uses
// a stacked Y8 layout). Fills width/height/encoding/step/data; caller sets the
// header.
inline std::unique_ptr<sensor_msgs::msg::Image> to_image_msg(
  const nvcv::Tensor & tensor, const std::string & encoding,
  uint32_t width, uint32_t height, cudaStream_t stream)
{
  auto msg = allocate_image_msg(width, height, encoding);
  to_image_msg(*msg, tensor, stream);
  return msg;
}

// ============================================================================
// tensor_msgs/ExperimentalTensor and isaac_ros_tensor_msgs/TensorList (DLPack-aligned)
// ============================================================================

// DLPack DLDataTypeCode values, the subset transportable to CV-CUDA. See
// https://dmlc.github.io/dlpack/latest/ and Tensor.msg.
enum class DLDataTypeCode : uint8_t
{
  kInt = 0,
  kUInt = 1,
  kFloat = 2,
  kBFloat = 4,
  kBool = 6,
};

// A DLPack DLDataType: {code, bits, lanes}. CV-CUDA has no vectorized element
// types, so dtype_lanes must be 1.
struct DLDataType
{
  uint8_t code;
  uint8_t bits;
  uint16_t lanes;
};

// Byte size of one element for a DLPack dtype: bits * lanes / 8.
inline int bytes_per_element(uint8_t dtype_bits, uint16_t dtype_lanes)
{
  if (dtype_bits == 0 || (dtype_bits % 8) != 0) {
    throw std::invalid_argument(
            "cvcuda_conversions: dtype_bits must be a positive multiple of 8, got " +
            std::to_string(dtype_bits));
  }
  return static_cast<int>(dtype_bits / 8) * static_cast<int>(dtype_lanes);
}

// Map a DLPack dtype {code, bits, lanes} to an nvcv::DataType.
inline nvcv::DataType to_nvcv_data_type(
  uint8_t dtype_code, uint8_t dtype_bits, uint16_t dtype_lanes)
{
  if (dtype_lanes != 1) {
    throw std::invalid_argument(
            "cvcuda_conversions: only scalar tensors (dtype_lanes == 1) are supported, got "
            "lanes=" + std::to_string(dtype_lanes));
  }
  switch (static_cast<DLDataTypeCode>(dtype_code)) {
    case DLDataTypeCode::kInt:
      switch (dtype_bits) {
        case 8: return nvcv::TYPE_S8;
        case 16: return nvcv::TYPE_S16;
        case 32: return nvcv::TYPE_S32;
        case 64: return nvcv::TYPE_S64;
      }
      break;
    case DLDataTypeCode::kUInt:
      switch (dtype_bits) {
        case 8: return nvcv::TYPE_U8;
        case 16: return nvcv::TYPE_U16;
        case 32: return nvcv::TYPE_U32;
        case 64: return nvcv::TYPE_U64;
      }
      break;
    case DLDataTypeCode::kFloat:
      switch (dtype_bits) {
        case 16: return nvcv::TYPE_F16;
        case 32: return nvcv::TYPE_F32;
        case 64: return nvcv::TYPE_F64;
      }
      break;
    case DLDataTypeCode::kBool:
      if (dtype_bits == 8) {return nvcv::TYPE_U8;}
      break;
    case DLDataTypeCode::kBFloat:
      break;
  }
  throw std::invalid_argument(
          "cvcuda_conversions: DLPack dtype {code=" + std::to_string(dtype_code) +
          ", bits=" + std::to_string(dtype_bits) +
          "} not representable as an nvcv::DataType");
}

// Map an nvcv::DataType back to a DLPack dtype {code, bits, lanes=1}.
inline DLDataType from_nvcv_data_type(nvcv::DataType dtype)
{
  constexpr auto kInt = static_cast<uint8_t>(DLDataTypeCode::kInt);
  constexpr auto kUInt = static_cast<uint8_t>(DLDataTypeCode::kUInt);
  constexpr auto kFloat = static_cast<uint8_t>(DLDataTypeCode::kFloat);
  constexpr auto kBool = static_cast<uint8_t>(DLDataTypeCode::kBool);
  switch (dtype) {
    case nvcv::TYPE_S8: return {kInt, 8, 1};
    case nvcv::TYPE_S16: return {kInt, 16, 1};
    case nvcv::TYPE_S32: return {kInt, 32, 1};
    case nvcv::TYPE_S64: return {kInt, 64, 1};
    case nvcv::TYPE_U8: return {kUInt, 8, 1};
    case nvcv::TYPE_U16: return {kUInt, 16, 1};
    case nvcv::TYPE_U32: return {kUInt, 32, 1};
    case nvcv::TYPE_U64: return {kUInt, 64, 1};
    case nvcv::TYPE_F16: return {kFloat, 16, 1};
    case nvcv::TYPE_F32: return {kFloat, 32, 1};
    case nvcv::TYPE_F64: return {kFloat, 64, 1};
    default:
      throw std::invalid_argument(
              "cvcuda_conversions: nvcv::DataType not representable as a DLPack dtype: " +
              std::to_string(static_cast<int>(dtype)));
  }
}

// Product of dimensions (element count) of a shape vector.
inline size_t num_elements(const std::vector<int64_t> & shape)
{
  size_t count = 1;
  for (const int64_t dim : shape) {
    if (dim < 0) {
      throw std::invalid_argument(
              "cvcuda_conversions: dynamic/negative dimension " + std::to_string(dim));
    }
    count *= static_cast<size_t>(dim);
  }
  return count;
}

// Look up a tensor in a TensorList by the parallel names[] entry.
inline const Tensor * find_tensor_by_name(const TensorList & tensor_list, const std::string & name)
{
  for (size_t i = 0; i < tensor_list.names.size(); ++i) {
    if (tensor_list.names[i] == name && i < tensor_list.tensors.size()) {
      return &tensor_list.tensors[i];
    }
  }
  return nullptr;
}

namespace tensor_detail
{

inline size_t rank_for_layout(nvcv::TensorLayout layout)
{
  if (layout == nvcv::TENSOR_NHWC || layout == nvcv::TENSOR_NCHW) {
    return 4;
  } else if (layout == nvcv::TENSOR_HWC || layout == nvcv::TENSOR_CHW) {
    return 3;
  }
  throw std::invalid_argument("cvcuda_conversions: unsupported nvcv::TensorLayout");
}

inline void compute_buffer_strides(
  const nvcv::TensorShape::ShapeType & shape,
  nvcv::TensorLayout layout,
  size_t bytes_per_element,
  nvcv::TensorDataStridedCuda::Buffer & buffer)
{
  if (layout == nvcv::TENSOR_HWC || layout == nvcv::TENSOR_CHW) {
    buffer.strides[2] = static_cast<int64_t>(bytes_per_element);
    buffer.strides[1] = shape[2] * buffer.strides[2];
    buffer.strides[0] = shape[1] * buffer.strides[1];
  } else if (layout == nvcv::TENSOR_NHWC || layout == nvcv::TENSOR_NCHW) {
    buffer.strides[3] = static_cast<int64_t>(bytes_per_element);
    buffer.strides[2] = shape[3] * buffer.strides[3];
    buffer.strides[1] = shape[2] * buffer.strides[2];
    buffer.strides[0] = shape[1] * buffer.strides[1];
  } else {
    throw std::invalid_argument("cvcuda_conversions: unsupported nvcv::TensorLayout");
  }
}

inline void set_buffer_strides_from_dlpack(
  const std::vector<int64_t> & element_strides,
  size_t bytes_per_element,
  nvcv::TensorLayout layout,
  nvcv::TensorDataStridedCuda::Buffer & buffer)
{
  const size_t rank = rank_for_layout(layout);
  if (element_strides.size() != rank) {
    throw std::invalid_argument(
            "cvcuda_conversions: stride rank does not match tensor layout rank");
  }
  for (size_t i = 0; i < rank; ++i) {
    buffer.strides[i] = element_strides[i] * static_cast<int64_t>(bytes_per_element);
  }
}

template<typename HandleT>
nvcv::Tensor wrap_tensor(
  const std::vector<int64_t> & shape_vec,
  const std::vector<int64_t> & element_strides,
  uint8_t dtype_bits,
  uint16_t dtype_lanes,
  uint64_t byte_offset,
  nvcv::DataType dtype,
  nvcv::TensorLayout layout,
  HandleT handle)
{
  const size_t rank = rank_for_layout(layout);
  if (shape_vec.size() != rank) {
    throw std::invalid_argument(
            "cvcuda_conversions: tensor rank " + std::to_string(shape_vec.size()) +
            " does not match layout rank " + std::to_string(rank));
  }

  nvcv::TensorShape::ShapeType shape(static_cast<int>(rank));
  for (size_t i = 0; i < rank; ++i) {
    shape[i] = shape_vec[i];
  }

  const size_t bpe = static_cast<size_t>(bytes_per_element(dtype_bits, dtype_lanes));
  nvcv::TensorDataStridedCuda::Buffer buffer{};
  buffer.basePtr = const_cast<NVCVByte *>(
    reinterpret_cast<const NVCVByte *>(handle.get_ptr() + byte_offset));

  if (element_strides.empty()) {
    compute_buffer_strides(shape, layout, bpe, buffer);
  } else {
    set_buffer_strides_from_dlpack(element_strides, bpe, layout, buffer);
  }

  nvcv::TensorShape tensor_shape{shape, layout};
  nvcv::TensorDataStridedCuda tensor_data{tensor_shape, dtype, buffer};
  return nvcv::TensorWrapData(
    tensor_data, detail::HandleOwner<HandleT>{std::move(handle)});
}

template<typename HandleT>
nvcv::Tensor wrap_tensor(
  const Tensor & tensor, nvcv::DataType dtype, nvcv::TensorLayout layout, HandleT handle)
{
  const std::vector<int64_t> shape_vec(tensor.shape.begin(), tensor.shape.end());
  const std::vector<int64_t> element_strides(tensor.strides.begin(), tensor.strides.end());
  return wrap_tensor(
    shape_vec, element_strides, tensor.dtype_bits, tensor.dtype_lanes, tensor.byte_offset,
    dtype, layout, std::move(handle));
}

// True when src strides match C-contiguous (row-major) layout for its shape.
inline bool is_c_contiguous(const nvcv::TensorDataStridedCuda & data)
{
  const int rank = data.rank();
  if (rank <= 0) {
    return true;
  }
  int64_t expected = data.dtype().strideBytes();
  for (int d = rank - 1; d >= 0; --d) {
    const int64_t extent = data.shape(d);
    if (extent < 0) {
      throw std::invalid_argument(
              "cvcuda_conversions: dynamic/negative dimension in source tensor");
    }
    if (extent > 1 && data.stride(d) != expected) {
      return false;
    }
    expected *= std::max<int64_t>(extent, 1);
  }
  return true;
}

// Lowest dimension index at which dims [index, rank) are C-contiguous.
inline int contiguous_suffix_start(const nvcv::TensorDataStridedCuda & data)
{
  const int rank = data.rank();
  int contiguous_from = rank;
  int64_t expected = data.dtype().strideBytes();
  for (int d = rank - 1; d >= 0; --d) {
    const int64_t extent = data.shape(d);
    if (extent > 1 && data.stride(d) != expected) {
      break;
    }
    contiguous_from = d;
    expected *= std::max<int64_t>(extent, 1);
  }
  return contiguous_from;
}

// Pack a (possibly pitch-padded) strided CUDA tensor into a contiguous destination.
inline void copy_strided_to_contiguous(
  void * dst, const nvcv::TensorDataStridedCuda & src, cudaStream_t stream)
{
  const int rank = src.rank();
  if (rank == 0) {
    return;
  }

  if (is_c_contiguous(src)) {
    size_t size_bytes = static_cast<size_t>(src.dtype().strideBytes());
    for (int d = 0; d < rank; ++d) {
      size_bytes *= static_cast<size_t>(src.shape(d));
    }
    if (size_bytes == 0) {
      return;
    }
    const cudaError_t err = cudaMemcpyAsync(
      dst, src.basePtr(), size_bytes, cudaMemcpyDeviceToDevice, stream);
    if (err != cudaSuccess) {
      throw std::runtime_error(
              std::string("cvcuda_conversions::to_tensor: cudaMemcpyAsync failed: ") +
              cudaGetErrorString(err));
    }
    return;
  }

  // dims [contiguous_from, rank) form a dense row; copy with cudaMemcpy2DAsync
  // across shape[contiguous_from - 1], iterating any outer dimensions.
  const int contiguous_from = contiguous_suffix_start(src);
  if (contiguous_from == 0 || contiguous_from >= rank) {
    throw std::runtime_error(
            "cvcuda_conversions::to_tensor: source tensor is not C-contiguous and "
            "cannot be packed with a 2-D strided copy (unsupported stride pattern)");
  }

  int64_t row_elems = 1;
  for (int d = contiguous_from; d < rank; ++d) {
    row_elems *= src.shape(d);
  }
  if (row_elems == 0) {
    return;
  }
  const size_t row_bytes =
    static_cast<size_t>(row_elems) * static_cast<size_t>(src.dtype().strideBytes());
  const int64_t rows = src.shape(contiguous_from - 1);
  if (rows == 0) {
    return;
  }
  const size_t src_pitch = static_cast<size_t>(src.stride(contiguous_from - 1));
  const size_t dst_pitch = row_bytes;
  if (src_pitch < row_bytes) {
    throw std::runtime_error(
            "cvcuda_conversions::to_tensor: source row stride is smaller than packed row size");
  }

  int64_t num_blocks = 1;
  for (int d = 0; d < contiguous_from - 1; ++d) {
    num_blocks *= src.shape(d);
  }
  if (num_blocks == 0) {
    return;
  }

  const size_t dst_block_bytes = static_cast<size_t>(rows) * row_bytes;
  uint8_t * dst_bytes = static_cast<uint8_t *>(dst);
  const uint8_t * src_base = reinterpret_cast<const uint8_t *>(src.basePtr());

  std::vector<int64_t> index(static_cast<size_t>(std::max(contiguous_from - 1, 0)), 0);
  for (int64_t block = 0; block < num_blocks; ++block) {
    size_t src_offset = 0;
    for (int d = 0; d < contiguous_from - 1; ++d) {
      src_offset += static_cast<size_t>(index[static_cast<size_t>(d)]) *
        static_cast<size_t>(src.stride(d));
    }
    const cudaError_t err = cudaMemcpy2DAsync(
      dst_bytes + static_cast<size_t>(block) * dst_block_bytes, dst_pitch,
      src_base + src_offset, src_pitch,
      row_bytes, static_cast<size_t>(rows),
      cudaMemcpyDeviceToDevice, stream);
    if (err != cudaSuccess) {
      throw std::runtime_error(
              std::string("cvcuda_conversions::to_tensor: cudaMemcpy2DAsync failed: ") +
              cudaGetErrorString(err));
    }

    for (int d = contiguous_from - 2; d >= 0; --d) {
      ++index[static_cast<size_t>(d)];
      if (index[static_cast<size_t>(d)] < src.shape(d)) {
        break;
      }
      index[static_cast<size_t>(d)] = 0;
    }
  }
}

}  // namespace tensor_detail

// Allocate a CUDA-backed Tensor sized for (shape, dtype). Fills dtype/shape/data
// with a contiguous row-major layout: strides is left empty (the DLPack convention
// for "contiguous, infer row-major from shape") and byte_offset is 0. Tensor has
// no name field; tensor names live in TensorList.names[] parallel to tensors[].
inline Tensor allocate_tensor(
  const std::vector<int64_t> & shape, uint8_t dtype_code, uint8_t dtype_bits,
  uint16_t dtype_lanes = 1)
{
  const int element_size = bytes_per_element(dtype_bits, dtype_lanes);

  Tensor tensor;
  tensor.dtype_code = dtype_code;
  tensor.dtype_bits = dtype_bits;
  tensor.dtype_lanes = dtype_lanes;
  tensor.shape.assign(shape.begin(), shape.end());
  tensor.byte_offset = 0;
  tensor.data = cuda_buffer_backend::allocate_buffer(
    num_elements(shape) * static_cast<size_t>(element_size));
  return tensor;
}

// Read view of tensor.data as an nvcv::Tensor; dtype derived from the message.
// The nvcv::Tensor owns the buffer's ReadHandle.
// `layout` must match the rank of `tensor.shape` (e.g. NCHW for 4D tensors).
inline nvcv::Tensor from_input_tensor(
  const Tensor & tensor, cudaStream_t stream, nvcv::TensorLayout layout = nvcv::TENSOR_NCHW)
{
  const nvcv::DataType dtype = to_nvcv_data_type(
    tensor.dtype_code, tensor.dtype_bits, tensor.dtype_lanes);
  return tensor_detail::wrap_tensor(
    tensor, dtype, layout, cuda_buffer_backend::from_input_buffer(tensor.data, stream));
}

// Like from_input_tensor, but interpret the buffer with an explicit shape (e.g. a
// reshape view). Element count must match `tensor.shape`.
inline nvcv::Tensor from_input_tensor(
  const Tensor & tensor, cudaStream_t stream, nvcv::TensorLayout layout,
  const std::vector<int64_t> & shape)
{
  if (num_elements(shape) !=
    num_elements(std::vector<int64_t>(tensor.shape.begin(), tensor.shape.end())))
  {
    throw std::invalid_argument(
            "cvcuda_conversions::from_input_tensor: shape override element count mismatch");
  }
  const nvcv::DataType dtype = to_nvcv_data_type(
    tensor.dtype_code, tensor.dtype_bits, tensor.dtype_lanes);
  // Contiguous strides for the view shape (ignore source strides that belong to
  // a different rank/layout).
  return tensor_detail::wrap_tensor(
    shape, {}, tensor.dtype_bits, tensor.dtype_lanes, tensor.byte_offset, dtype, layout,
    cuda_buffer_backend::from_input_buffer(tensor.data, stream));
}

// Write view of tensor.data as an nvcv::Tensor; dtype derived from the message.
// The nvcv::Tensor owns the buffer's WriteHandle.
inline nvcv::Tensor from_output_tensor(
  Tensor & tensor, cudaStream_t stream, nvcv::TensorLayout layout = nvcv::TENSOR_NCHW)
{
  const nvcv::DataType dtype = to_nvcv_data_type(
    tensor.dtype_code, tensor.dtype_bits, tensor.dtype_lanes);
  return tensor_detail::wrap_tensor(
    tensor, dtype, layout, cuda_buffer_backend::from_output_buffer(tensor.data, stream));
}

// Copy a standalone nvcv::Tensor into a pre-allocated Tensor message's buffer.
// The message must already be sized for its shape/dtype (e.g. via allocate_tensor).
inline void to_tensor(Tensor & tensor, const nvcv::Tensor & src, cudaStream_t stream)
{
  auto src_data = src.exportData<nvcv::TensorDataStridedCuda>();
  if (!src_data) {
    throw std::runtime_error(
            "cvcuda_conversions::to_tensor: source tensor does not hold strided CUDA data");
  }

  if (static_cast<size_t>(src_data->rank()) != tensor.shape.size()) {
    throw std::invalid_argument(
            "cvcuda_conversions::to_tensor: source rank " +
            std::to_string(src_data->rank()) + " does not match destination rank " +
            std::to_string(tensor.shape.size()));
  }
  for (size_t i = 0; i < tensor.shape.size(); ++i) {
    if (src_data->shape(static_cast<int>(i)) != tensor.shape[i]) {
      throw std::invalid_argument(
              "cvcuda_conversions::to_tensor: source shape does not match destination shape "
              "at dim " + std::to_string(i));
    }
  }

  const size_t element_bytes = static_cast<size_t>(
    bytes_per_element(tensor.dtype_bits, tensor.dtype_lanes));
  if (static_cast<size_t>(src_data->dtype().strideBytes()) != element_bytes) {
    throw std::invalid_argument(
            "cvcuda_conversions::to_tensor: source dtype size does not match destination");
  }

  auto handle = cuda_buffer_backend::from_output_buffer(tensor.data, stream);
  tensor_detail::copy_strided_to_contiguous(
    handle.get_ptr() + tensor.byte_offset, *src_data, stream);
}

// Allocate a CUDA-backed Tensor for (shape, dtype) and copy a device tensor into it.
inline Tensor to_tensor(
  const std::vector<int64_t> & shape, uint8_t dtype_code, uint8_t dtype_bits,
  const nvcv::Tensor & src, cudaStream_t stream, uint16_t dtype_lanes = 1)
{
  Tensor tensor = allocate_tensor(shape, dtype_code, dtype_bits, dtype_lanes);
  to_tensor(tensor, src, stream);
  return tensor;
}

}  // namespace cvcuda_conversions

#endif  // CVCUDA_CONVERSIONS__CVCUDA_CONVERSIONS_HPP_
