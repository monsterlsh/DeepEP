// Copyright (c) 2026, Lu Lu
// Modified by SimpleBright_Man 2026

// The hierarchical device stages and tiling are based on revision 0c68d94.
#define DEEPEP_MEGAMOE_HIERARCHICAL_IMPL 1

#if __has_include("version/asc_devkit_version.h")
#include "version/asc_devkit_version.h"
#endif
#include "basic_api/kernel_basic_intf.h"
#include "kernel_launch.hpp"
#include "mega_moe_tiling.hpp"
#include "common/mega_moe_constants.hpp"
#if defined(__NPU_DEVICE__)
#include "mega_moe_wave_a8w4.hpp"
#endif

using namespace AscendC;

template <uint8_t GroupedMode, bool EarlyWeights, bool FuseGateUp = false>
__schedmode__(1) __global__
    __mix__(1, 2) void mega_moe_hierarchical_kernel(
        GM_ADDR context, GM_ADDR x, GM_ADDR x_scales, GM_ADDR topk_ids, GM_ADDR topk_weights, GM_ADDR weight1,
        GM_ADDR weight2, GM_ADDR weight_scales1, GM_ADDR weight_scales2, GM_ADDR shared_weight1, GM_ADDR shared_weight2,
        GM_ADDR shared_weight_scales1, GM_ADDR shared_weight_scales2, GM_ADDR y, GM_ADDR expert_token_nums,
        GM_ADDR workspace, GM_ADDR timer, HierarchicalMegaMoeTilingData tiling, uint64_t ffts_addr)
{
#if defined(__NPU_DEVICE__)
    InitSocState();
    using HierarchicalMegaMoeImpl::MegaMoeActMode;
    const auto activation = static_cast<MegaMoeActMode>(tiling.actMode);
    if (tiling.groupedMatmulMode != GroupedMode || !HierarchicalMegaMoeImpl::UseHierarchicalRoce(&tiling) ||
        (activation != MegaMoeActMode::SWIGLU && activation != MegaMoeActMode::GELU) ||
        tiling.bs == 0 ||
        tiling.bs > HierarchicalMegaMoeImpl::HIERARCHICAL_MAX_TOKENS) {
        trap();
    }
    util_set_ffts_config(ffts_addr);
    constexpr bool nativeFp4 = GroupedMode == HierarchicalMegaMoeImpl::GROUPED_MATMUL_MODE_A4W4_NZ;
    constexpr int32_t quantMode = nativeFp4 ? HierarchicalMegaMoeImpl::E2M1_QUANT : 4;  // MXFP8 E4M3
    using Weight = std::conditional_t<GroupedMode == HierarchicalMegaMoeImpl::GROUPED_MATMUL_MODE_GENERAL,
                                      fp8_e4m3fn_t, fp4x2_e2m1_t>;
    HierarchicalMegaMoeTilingData invocation = tiling;
    HierarchicalMegaMoeImpl::MegaMoeA8W4Wave<bfloat16_t, bfloat16_t, float, Weight, quantMode,
                                            HierarchicalMegaMoeImpl::COMBINE_NO_QUANT, EarlyWeights, FuseGateUp>
        op;
    op.Init(context, x, topk_ids, topk_weights, weight1, weight2, nullptr, weight_scales1, weight_scales2, x_scales,
            shared_weight1, shared_weight2, shared_weight_scales1, shared_weight_scales2, y, expert_token_nums,
            workspace, &invocation, nullptr, timer);
    op.Process();
#endif
}

namespace {

struct KernelArguments {
    uint8_t *context, *x, *x_scales, *topk_ids, *topk_weights, *weight1, *weight2, *weight_scales1, *weight_scales2;
    uint8_t *shared_weight1, *shared_weight2, *shared_weight_scales1, *shared_weight_scales2;
    uint8_t *y, *expert_token_nums, *workspace, *timer;
};

template <uint8_t GroupedMode, bool EarlyWeights, bool FuseGateUp = false>
void Launch(uint32_t blocks, uint32_t ub_bytes, void* stream, const KernelArguments& args,
            const HierarchicalMegaMoeTilingData& tiling, uint64_t ffts_addr)
{
    mega_moe_hierarchical_kernel<GroupedMode, EarlyWeights, FuseGateUp><<<blocks, ub_bytes, stream>>>(
        args.context, args.x, args.x_scales, args.topk_ids, args.topk_weights, args.weight1, args.weight2,
        args.weight_scales1, args.weight_scales2, args.shared_weight1, args.shared_weight2, args.shared_weight_scales1,
        args.shared_weight_scales2, args.y, args.expert_token_nums, args.workspace, args.timer, tiling, ffts_addr);
}

template <uint8_t GroupedMode, bool FuseGateUp = false>
void SelectWeighting(uint32_t blocks, uint32_t ub_bytes, void* stream, const KernelArguments& args,
                     const HierarchicalMegaMoeTilingData& tiling, uint64_t ffts_addr)
{
    if (tiling.topkWeightsPrefetch) {
        Launch<GroupedMode, true, FuseGateUp>(blocks, ub_bytes, stream, args, tiling, ffts_addr);
    } else {
        Launch<GroupedMode, false, FuseGateUp>(blocks, ub_bytes, stream, args, tiling, ffts_addr);
    }
}

}  // namespace

extern "C" __attribute__((visibility("default"))) void mega_moe_hierarchical_kernel_do(
    uint32_t block_dim, uint32_t ub_bytes, void* stream, bool is_a8w4, uint8_t* context, uint8_t* x, uint8_t* x_scales,
    uint8_t* topk_ids, uint8_t* topk_weights, uint8_t* weight1, uint8_t* weight2, uint8_t* weight_scales1,
    uint8_t* weight_scales2, uint8_t* shared_weight1, uint8_t* shared_weight2, uint8_t* shared_weight_scales1,
    uint8_t* shared_weight_scales2, uint8_t* y, uint8_t* expert_token_nums, uint8_t* workspace, uint8_t* timer,
    const uint8_t* tiling_host, uint64_t ffts_addr)
{
    static_assert(sizeof(HierarchicalMegaMoeTilingData) == 368, "Hierarchical MegaMoE tiling ABI changed");
    const auto& tiling = *reinterpret_cast<const HierarchicalMegaMoeTilingData*>(tiling_host);
    using namespace HierarchicalMegaMoeImpl;
    const KernelArguments args{context, x, x_scales, topk_ids, topk_weights, weight1, weight2, weight_scales1,
                               weight_scales2, shared_weight1, shared_weight2, shared_weight_scales1,
                               shared_weight_scales2, y, expert_token_nums, workspace, timer};
    if (is_a8w4) {
        if (tiling.actMode != static_cast<uint8_t>(MegaMoeActMode::GELU)) {
            SelectWeighting<GROUPED_MATMUL_MODE_A8W4, true>(block_dim, ub_bytes, stream, args, tiling, ffts_addr);
        } else {
            SelectWeighting<GROUPED_MATMUL_MODE_A8W4>(block_dim, ub_bytes, stream, args, tiling, ffts_addr);
        }
    } else if (tiling.groupedMatmulMode == GROUPED_MATMUL_MODE_A4W4_NZ) {
        SelectWeighting<GROUPED_MATMUL_MODE_A4W4_NZ>(block_dim, ub_bytes, stream, args, tiling, ffts_addr);
    } else {
        SelectWeighting<GROUPED_MATMUL_MODE_GENERAL>(block_dim, ub_bytes, stream, args, tiling, ffts_addr);
    }
}
