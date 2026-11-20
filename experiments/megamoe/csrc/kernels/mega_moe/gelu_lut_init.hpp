#pragma once

#include <cstdint>
#include "basic_api/kernel_basic_intf.h"
#if __has_include("version/asc_devkit_version.h")
#include "version/asc_devkit_version.h"
#endif
#if defined(__NPU_DEVICE__)
#include "kernels/mega_moe_common/epilogue/activation/activation_common.hpp"

namespace MegaMoeGeluSetup {
using namespace AscendC;
constexpr uint32_t kChunk = 1024;
__simd_vf__ inline void FactorBatch(__ubuf__ float* dst, uint32_t begin)
{
    namespace R = AscendC::Reg;
    auto mask = R::CreateMask<float, R::MaskPattern::ALL>();
    R::RegTensor<int32_t> unpacked;
    R::RegTensor<float> x, factor;
    for (uint16_t i = 0; i < kChunk; i += MegaMoeImpl::Activation::VECTOR_LENGTH_FP32) {
        // DIST_UNPACK_B16 yields zero-extended raw bits in the low 16 bits of each lane.
        R::Arange(unpacked, static_cast<int32_t>(begin + i));
        auto& raw = reinterpret_cast<R::RegTensor<bfloat16_t>&>(unpacked);
        R::Cast<float, bfloat16_t, MegaMoeImpl::Activation::CAST_INPUT_TO_FP32>(x, raw, mask);
        MegaMoeImpl::Activation::ComputeGeluFactor(factor, x, mask);
        R::StoreAlign(dst + i, factor, mask);
    }
}
}  // namespace MegaMoeGeluSetup
#endif

__global__ __vector__ void megamoe_gelu_factor_kernel(__gm__ uint8_t* output)
{
#if defined(__NPU_DEVICE__)
    using namespace AscendC;
    using MegaMoeGeluSetup::kChunk;
    InitSocState();
    LocalTensor<float> local(TPosition::VECCALC, 0U, kChunk);
    GlobalTensor<float> gm;
    gm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(output));
    for (uint32_t begin = GetBlockIdx() * kChunk; begin < MegaMoeImpl::MTE_GELU_LUT_KEY_COUNT;
         begin += GetBlockNum() * kChunk) {
        asc_vf_call<MegaMoeGeluSetup::FactorBatch>(reinterpret_cast<__ubuf__ float*>(local.GetPhyAddr()), begin);
        SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
        DataCopy(gm[begin], local, kChunk);
        PipeBarrier<PIPE_ALL>();
    }
#endif
}

extern "C" __attribute__((visibility("default"))) void megamoe_init_gelu_factors(void* stream, uint8_t* output,
                                                                                uint32_t blocks)
{
    megamoe_gelu_factor_kernel<<<blocks, 0, stream>>>(output);
}
