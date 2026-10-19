// Copyright (c) 2026, Lu Lu
// Modified by SimpleBright_Man 2026
#pragma once

#include "mx_quant_common.hpp"
#include "../../common/fp4_expanded_policy.hpp"

#if defined(__DAV_C310__)
namespace MegaMoeImpl::MxQuant {

// A5 MX MMAD has no mixed FP4 x FP8 operand pair. Quantize to the E2M1
// lattice and write its exact E4M3 representation. Both GEMMs therefore use
// MXFP4 activation precision with FP8 physical storage/communication/MMAD.
// No additional scale adjustment or E4M3 rounding is performed.
__aicore__ inline void QuantizeMxFp4ExpandedData(__ubuf__ bfloat16_t* input, __ubuf__ uint16_t* reciprocalScale,
                                                 __ubuf__ int8_t* output, uint32_t dataCount)
{
    const uint16_t loops =
        static_cast<uint16_t>((dataCount + DATA_ELEMENT_COUNT_PER_LOOP - 1) / DATA_ELEMENT_COUNT_PER_LOOP);
    __ubuf__ uint16_t* output16 = reinterpret_cast<__ubuf__ uint16_t*>(output);
    __VEC_SCOPE__
    {
        AscendC::Reg::RegTensor<bfloat16_t> input0, input1;
        AscendC::Reg::RegTensor<uint16_t> reciprocal, magnitudeMask, signMask;
        AscendC::Reg::RegTensor<uint16_t> magnitude, sign, candidate, code0, code1;
        AscendC::Reg::MaskReg selected;
        auto full = AscendC::Reg::CreateMask<uint16_t, AscendC::Reg::MaskPattern::ALL>();
        AscendC::Reg::Duplicate(magnitudeMask, uint16_t{0x7fff});
        AscendC::Reg::Duplicate(signMask, uint16_t{0x8000});

#define MEGAMOE_FP4_ROUND_STEP(code, comparison, threshold, encoded)                                              \
    AscendC::Reg::CompareScalar<uint16_t, AscendC::CMPMODE::comparison>(selected, magnitude, uint16_t{threshold}, \
                                                                        full);                                    \
    AscendC::Reg::Duplicate(candidate, uint16_t{encoded});                                                        \
    AscendC::Reg::Select(code, candidate, code, selected)

#define MEGAMOE_FP4_ENCODE(code, normalized)                                                           \
    AscendC::Reg::And(magnitude, (AscendC::Reg::RegTensor<uint16_t>&)normalized, magnitudeMask, full); \
    AscendC::Reg::And(sign, (AscendC::Reg::RegTensor<uint16_t>&)normalized, signMask, full);           \
    AscendC::Reg::ShiftRights(sign, sign, int16_t{8}, full);                                           \
    AscendC::Reg::Duplicate(code, uint16_t{0});                                                        \
    MEGAMOE_FP4_ROUND_STEP(code, GT, Fp4Expanded::kQuarter, 0x30);                                     \
    MEGAMOE_FP4_ROUND_STEP(code, GE, Fp4Expanded::kThreeQuarters, 0x38);                               \
    MEGAMOE_FP4_ROUND_STEP(code, GT, Fp4Expanded::kOneAndQuarter, 0x3c);                               \
    MEGAMOE_FP4_ROUND_STEP(code, GE, Fp4Expanded::kOneAndThreeQuarters, 0x40);                         \
    MEGAMOE_FP4_ROUND_STEP(code, GT, Fp4Expanded::kTwoAndHalf, 0x44);                                  \
    MEGAMOE_FP4_ROUND_STEP(code, GE, Fp4Expanded::kThreeAndHalf, 0x48);                                \
    MEGAMOE_FP4_ROUND_STEP(code, GT, Fp4Expanded::kFive, 0x4c);                                        \
    MEGAMOE_FP4_ROUND_STEP(code, GT, 0x7f80, 0x7f);                                                    \
    AscendC::Reg::Or(code, code, sign, full)

        for (uint16_t i = 0; i < loops; ++i) {
            // Each register holds alternating elements; each scale is repeated
            // 16 times for its half of the original 32-element MX group.
            AscendC::Reg::DataCopy<bfloat16_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE,
                                   AscendC::Reg::LoadDist::DIST_DINTLV_B16>(input0, input1, input,
                                                                            DATA_ELEMENT_COUNT_PER_LOOP);
            AscendC::Reg::DataCopy<uint16_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE,
                                   AscendC::Reg::LoadDist::DIST_E2B_B16>(reciprocal, reciprocalScale,
                                                                         SCALE_ELEMENT_COUNT_PER_DATA_LOOP);
            AscendC::Reg::Mul(input0, input0, (AscendC::Reg::RegTensor<bfloat16_t>&)reciprocal, full);
            AscendC::Reg::Mul(input1, input1, (AscendC::Reg::RegTensor<bfloat16_t>&)reciprocal, full);
            MEGAMOE_FP4_ENCODE(code0, input0);
            MEGAMOE_FP4_ENCODE(code1, input1);
            AscendC::Reg::Interleave(code0, code1, code0, code1);
            auto first = AscendC::Reg::UpdateMask<uint16_t>(dataCount);
            auto second = AscendC::Reg::UpdateMask<uint16_t>(dataCount);
            AscendC::Reg::DataCopy<uint16_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE,
                                   AscendC::Reg::StoreDist::DIST_PACK_B16>(output16, code0, 64, first);
            AscendC::Reg::DataCopy<uint16_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE,
                                   AscendC::Reg::StoreDist::DIST_PACK_B16>(output16, code1, 64, second);
        }
#undef MEGAMOE_FP4_ENCODE
#undef MEGAMOE_FP4_ROUND_STEP
    }
}

}  // namespace MegaMoeImpl::MxQuant
#endif
