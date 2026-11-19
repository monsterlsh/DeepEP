/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See third-party/cann.LICENSE for the full text of the License.
 */
// Modified by SimpleBright_Man 2026

/*!
 * \file activation_common.h
 * \brief 门控激活公共数据结构、参数和常量。
 */

#ifndef MEGA_MOE_ARCH35_ACTIVATION_COMMON_H
#define MEGA_MOE_ARCH35_ACTIVATION_COMMON_H

#if defined(__DAV_C310__)
#if ASC_DEVKIT_MAJOR >= 9
#include "kernels/mega_moe_common/implementation_namespace.hpp"
#include "basic_api/kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "kernels/mega_moe_common/common/mega_moe_constants.hpp"

namespace DEEPEP_MEGAMOE_IMPL_NAMESPACE {
namespace Activation {

// 具体激活实现按 FP32 寄存器宽度划分主循环。
constexpr uint32_t VECTOR_LENGTH_FP32 = AscendC::VECTOR_REG_WIDTH / sizeof(float);

// 输入统一转换为 FP32，供各激活的寄存器计算流程使用。
constexpr AscendC::Reg::CastTrait CAST_INPUT_TO_FP32 = {AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::UNKNOWN,
                                                        AscendC::Reg::MaskMergeMode::ZEROING,
                                                        AscendC::RoundMode::UNKNOWN};

// 当前激活结果统一写入 BF16 UB；量化流程在激活之后独立执行。
constexpr AscendC::Reg::CastTrait CAST_FP32_TO_BF16 = {AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::NO_SAT,
                                                       AscendC::Reg::MaskMergeMode::ZEROING,
                                                       AscendC::RoundMode::CAST_RINT};

/*
 * 单个门控激活 tile 的完整执行上下文。
 *
 * Block Epilogue 在分派激活前直接计算地址、物理行距、循环次数和尾部 Mask，具体激活仅消费这些
 * 已物化的数据。此结构不保存 RegTensor、MaskReg、AddrReg 等 VecScope 内对象。
 */
template <typename InputType>
struct GatedActivationTileContext {
    __ubuf__ InputType* gate;
    __ubuf__ InputType* up;
    __ubuf__ bfloat16_t* output;
    __ubuf__ float* topkWeights;
    uint32_t topkWeightIndex;

    __ubuf__ InputType* gateTail;
    __ubuf__ InputType* upTail;
    __ubuf__ bfloat16_t* outputTail;
    __ubuf__ bfloat16_t* additionalPaddingOutput;

    uint32_t inputRowStrideElements;
    uint32_t outputRowStrideElements;
    uint16_t rowLoopCount;
    uint16_t fullVectorLoopCount;
    // 0/1 标志：为保持原 VecScope for-loop 形态，直接作为尾 Vector 计算循环上界。
    uint16_t needTailVectorCompute;
    // 0/1 标志：为保持原 VecScope for-loop 形态，直接作为额外补零写回循环上界。
    uint16_t needAdditionalPaddingStore;

    // UpdateMask 接收可写引用，具体激活需将以下计数复制到局部变量后再调用。
    uint32_t tailComputeMaskElementCount;
    uint32_t tailStoreMaskElementCount;
    uint32_t additionalPaddingStoreMaskElementCount;
};

// Abramowitz-Stegun 7.1.26 evaluates erf itself to |error| <= 1.5e-7.
// This is not the tanh GELU approximation; all arithmetic and route weighting stay FP32 until the BF16 cast.
#define MEGA_MOE_GELU_EVALUATE(outputReg, inputReg, weightReg, mask)                                               \
    do {                                                                                                          \
        AscendC::Reg::Muls(zReg, inputReg, inverseSqrtTwo, mask);                                                  \
        AscendC::Reg::Abs(absReg, zReg, mask);                                                                     \
        AscendC::Reg::Muls(tReg, absReg, erfP, mask);                                                              \
        AscendC::Reg::Adds(tReg, tReg, scalarOne, mask);                                                           \
        AscendC::Reg::Div(tReg, oneReg, tReg, mask);                                                               \
        AscendC::Reg::Mul(squareReg, absReg, absReg, mask);                                                        \
        AscendC::Reg::Muls(squareReg, squareReg, negScalarOne, mask);                                              \
        AscendC::Reg::Exp(expReg, squareReg, mask);                                                                \
        AscendC::Reg::Duplicate(polyReg, erfA5, mask);                                                             \
        AscendC::Reg::Mul(polyReg, polyReg, tReg, mask);                                                           \
        AscendC::Reg::Adds(polyReg, polyReg, erfA4, mask);                                                         \
        AscendC::Reg::Mul(polyReg, polyReg, tReg, mask);                                                           \
        AscendC::Reg::Adds(polyReg, polyReg, erfA3, mask);                                                         \
        AscendC::Reg::Mul(polyReg, polyReg, tReg, mask);                                                           \
        AscendC::Reg::Adds(polyReg, polyReg, erfA2, mask);                                                         \
        AscendC::Reg::Mul(polyReg, polyReg, tReg, mask);                                                           \
        AscendC::Reg::Adds(polyReg, polyReg, erfA1, mask);                                                         \
        AscendC::Reg::Mul(polyReg, polyReg, tReg, mask);                                                           \
        AscendC::Reg::Mul(polyReg, polyReg, expReg, mask);                                                         \
        AscendC::Reg::Muls(erfAbsReg, polyReg, negScalarOne, mask);                                                \
        AscendC::Reg::Adds(erfAbsReg, erfAbsReg, scalarOne, mask);                                                 \
        AscendC::Reg::Muls(negativeErfReg, erfAbsReg, negScalarOne, mask);                                         \
        AscendC::Reg::CompareScalar<float, AscendC::CMPMODE::GE>(positiveMask, zReg, 0.0F, mask);                  \
        AscendC::Reg::Select(erfReg, erfAbsReg, negativeErfReg, positiveMask);                                     \
        AscendC::Reg::Adds(erfReg, erfReg, scalarOne, mask);                                                       \
        AscendC::Reg::Mul(outputReg, inputReg, erfReg, mask);                                                      \
        AscendC::Reg::Muls(outputReg, outputReg, scalarHalf, mask);                                                \
        if constexpr (FuseTopkWeight) {                                                                            \
            AscendC::Reg::Mul(outputReg, outputReg, weightReg, mask);                                              \
        }                                                                                                          \
    } while (0)

template <typename InputType, bool FuseTopkWeight>
__aicore__ inline void RunGELU(const GatedActivationTileContext<InputType>& context)
{
    const float inverseSqrtTwo = 0.7071067811865475244F;
    const float erfP = 0.3275911F;
    const float erfA1 = 0.254829592F;
    const float erfA2 = -0.284496736F;
    const float erfA3 = 1.421413741F;
    const float erfA4 = -1.453152027F;
    const float erfA5 = 1.061405429F;
    const float scalarOne = 1.0F;
    const float negScalarOne = -1.0F;
    const float scalarHalf = 0.5F;
    bfloat16_t zeroValue = 0;
    const uint16_t rowLoopCount = context.rowLoopCount;
    const uint16_t fullVectorLoopCount = context.fullVectorLoopCount;
    const uint16_t needTailVectorCompute = context.needTailVectorCompute;
    const uint16_t needAdditionalPaddingStore = context.needAdditionalPaddingStore;
    uint32_t tailComputeMaskElementCount = context.tailComputeMaskElementCount;
    uint32_t tailStoreMaskElementCount = context.tailStoreMaskElementCount;
    uint32_t additionalPaddingStoreMaskElementCount = context.additionalPaddingStoreMaskElementCount;

    __VEC_SCOPE__
    {
        AscendC::Reg::RegTensor<InputType> inputStorageReg;
        AscendC::Reg::RegTensor<float> inputReg, zReg, absReg, tReg, squareReg, expReg, polyReg;
        AscendC::Reg::RegTensor<float> erfAbsReg, negativeErfReg, erfReg, outputReg, weightReg, oneReg;
        AscendC::Reg::RegTensor<bfloat16_t> outputBf16Reg, zeroReg;
        AscendC::Reg::MaskReg positiveMask;
        AscendC::Reg::MaskReg fullMask = AscendC::Reg::CreateMask<float, AscendC::Reg::MaskPattern::ALL>();
        AscendC::Reg::MaskReg tailComputeMask = AscendC::Reg::UpdateMask<float>(tailComputeMaskElementCount);
        AscendC::Reg::MaskReg tailStoreMask = AscendC::Reg::UpdateMask<float>(tailStoreMaskElementCount);
        AscendC::Reg::MaskReg additionalPaddingStoreMask =
            AscendC::Reg::UpdateMask<bfloat16_t>(additionalPaddingStoreMaskElementCount);
        AscendC::Reg::Duplicate(oneReg, scalarOne);

        for (uint16_t rowIndex = 0; rowIndex < rowLoopCount; ++rowIndex) {
            if constexpr (FuseTopkWeight) {
                AscendC::Reg::DataCopy<float, AscendC::Reg::LoadDist::DIST_BRC_B32>(
                    weightReg, context.topkWeights + rowIndex * INT32_PER_256B + context.topkWeightIndex);
            }
            for (uint16_t vectorIndex = 0; vectorIndex < fullVectorLoopCount; ++vectorIndex) {
                AscendC::Reg::AddrReg inputOffset = AscendC::Reg::CreateAddrReg<InputType>(
                    rowIndex, context.inputRowStrideElements, vectorIndex, VECTOR_LENGTH_FP32);
                AscendC::Reg::DataCopy<InputType, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(inputStorageReg, context.gate,
                                                                                           inputOffset);
                AscendC::Reg::Cast<float, InputType, CAST_INPUT_TO_FP32>(inputReg, inputStorageReg, fullMask);
                MEGA_MOE_GELU_EVALUATE(outputReg, inputReg, weightReg, fullMask);
                AscendC::Reg::Cast<bfloat16_t, float, CAST_FP32_TO_BF16>(outputBf16Reg, outputReg, fullMask);
                AscendC::Reg::AddrReg outputOffset = AscendC::Reg::CreateAddrReg<bfloat16_t>(
                    rowIndex, context.outputRowStrideElements, vectorIndex, VECTOR_LENGTH_FP32);
                AscendC::Reg::DataCopy<bfloat16_t, AscendC::Reg::StoreDist::DIST_PACK_B32>(context.output, outputBf16Reg,
                                                                                           outputOffset, fullMask);
            }
            AscendC::Reg::AddrReg tailInputOffset =
                AscendC::Reg::CreateAddrReg<InputType>(rowIndex, context.inputRowStrideElements);
            AscendC::Reg::AddrReg tailOutputOffset =
                AscendC::Reg::CreateAddrReg<bfloat16_t>(rowIndex, context.outputRowStrideElements);
            for (uint16_t tailIndex = 0; tailIndex < needTailVectorCompute; ++tailIndex) {
                AscendC::Reg::DataCopy<InputType, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(inputStorageReg,
                                                                                           context.gateTail,
                                                                                           tailInputOffset);
                AscendC::Reg::Cast<float, InputType, CAST_INPUT_TO_FP32>(inputReg, inputStorageReg, tailComputeMask);
                MEGA_MOE_GELU_EVALUATE(outputReg, inputReg, weightReg, tailComputeMask);
                AscendC::Reg::Cast<bfloat16_t, float, CAST_FP32_TO_BF16>(outputBf16Reg, outputReg, tailComputeMask);
                AscendC::Reg::DataCopy<bfloat16_t, AscendC::Reg::StoreDist::DIST_PACK_B32>(
                    context.outputTail, outputBf16Reg, tailOutputOffset, tailStoreMask);
            }
            for (uint16_t paddingIndex = 0; paddingIndex < needAdditionalPaddingStore; ++paddingIndex) {
                AscendC::Reg::Duplicate(zeroReg, zeroValue);
                AscendC::Reg::DataCopy<bfloat16_t>(context.additionalPaddingOutput, zeroReg, tailOutputOffset,
                                                   additionalPaddingStoreMask);
            }
        }
    }
}

#undef MEGA_MOE_GELU_EVALUATE

}  // namespace Activation
}  // namespace DEEPEP_MEGAMOE_IMPL_NAMESPACE

#endif  // defined(__DAV_C310__)
#endif  // MEGA_MOE_ARCH35_ACTIVATION_COMMON_H
