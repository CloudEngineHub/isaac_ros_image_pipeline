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

#include <gtest/gtest.h>

#include <cstdint>
#include <stdexcept>
#include <vector>

#include "cvcuda_conversions/cvcuda_conversions.hpp"

namespace
{

constexpr uint8_t kInt = static_cast<uint8_t>(cvcuda_conversions::DLDataTypeCode::kInt);
constexpr uint8_t kUInt = static_cast<uint8_t>(cvcuda_conversions::DLDataTypeCode::kUInt);
constexpr uint8_t kFloat = static_cast<uint8_t>(cvcuda_conversions::DLDataTypeCode::kFloat);
constexpr uint8_t kBool = static_cast<uint8_t>(cvcuda_conversions::DLDataTypeCode::kBool);

TEST(CVCudaConversionsTest, ToNVCVDataTypeSupported)
{
  EXPECT_EQ(cvcuda_conversions::to_nvcv_data_type(kInt, 8, 1), nvcv::TYPE_S8);
  EXPECT_EQ(cvcuda_conversions::to_nvcv_data_type(kInt, 32, 1), nvcv::TYPE_S32);
  EXPECT_EQ(cvcuda_conversions::to_nvcv_data_type(kUInt, 8, 1), nvcv::TYPE_U8);
  EXPECT_EQ(cvcuda_conversions::to_nvcv_data_type(kFloat, 16, 1), nvcv::TYPE_F16);
  EXPECT_EQ(cvcuda_conversions::to_nvcv_data_type(kFloat, 32, 1), nvcv::TYPE_F32);
  EXPECT_EQ(cvcuda_conversions::to_nvcv_data_type(kBool, 8, 1), nvcv::TYPE_U8);
}

TEST(CVCudaConversionsTest, FindTensorByNameUsesParallelNamesArray)
{
  cvcuda_conversions::TensorList list;
  list.names = {"input", "output"};
  cvcuda_conversions::Tensor input;
  cvcuda_conversions::Tensor output;
  input.shape = {1, 3, 224, 224};
  output.shape = {1, 1000};
  list.tensors = {input, output};

  const cvcuda_conversions::Tensor * found = cvcuda_conversions::find_tensor_by_name(list, "output");
  ASSERT_NE(found, nullptr);
  EXPECT_EQ(found->shape.size(), 2u);
  EXPECT_EQ(cvcuda_conversions::find_tensor_by_name(list, "missing"), nullptr);
}

TEST(CVCudaConversionsTest, AllocateTensorPopulatesDLPackFields)
{
  const std::vector<int64_t> shape{1, 3, 224, 224};
  cvcuda_conversions::Tensor tensor = cvcuda_conversions::allocate_tensor(shape, kFloat, 32, 1);

  EXPECT_EQ(tensor.dtype_code, kFloat);
  EXPECT_EQ(tensor.dtype_bits, 32);
  EXPECT_EQ(tensor.dtype_lanes, 1);
  EXPECT_EQ(tensor.shape, shape);
  EXPECT_TRUE(tensor.strides.empty());
  EXPECT_EQ(tensor.byte_offset, 0u);
}

}  // namespace

int main(int argc, char ** argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
