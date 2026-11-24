// Copyright (c) 2026, Lu Lu
// Modified by SimpleBright_Man 2026

#include "forward.hpp"
#include "kernel_launch.hpp"
#include "urma_workspace.hpp"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <vector>
#include "kernels/mega_moe_urma/common/mega_moe_compute_contract.hpp"
#include "kernels/mega_moe_urma/common/mega_moe_route_plan.hpp"

#include <acl/acl.h>
#include "shmem.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/core/npu/NPUCachingAllocator.h"
#include "tiling/platform/platform_ascendc.h"

#include "check.hpp"
#include "gelu_lut_setup.hpp"
#include "kernels/mega_moe_common/common/ascend_timer_v2.hpp"
#include "kernels/mega_moe_common/common/mega_moe_constants.hpp"
#include "kernels/mega_moe_common/common/mega_moe_workspace.hpp"
#include "kernels/mega_moe_common/mc2_common/mc2_moe_context.hpp"
#include "kernels/mega_moe_hierarchical/common/ascend_timer_v2.hpp"

namespace deepep::megamoe {

PlatformInfo GetPlatformInfo()
{
    auto* platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    TORCH_CHECK(platform != nullptr, "failed to get AscendC platform information");
    uint64_t ub = 0;
    platform->GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub);
    const uint32_t aic = platform->GetCoreNumAic();
    const uint32_t aiv = platform->GetCoreNumAiv();
    TORCH_CHECK(aic > 0 && aiv > 0 && ub > 0, "invalid A5 platform resources");
    return {aic, aiv, static_cast<uint32_t>(ub), platform->CalcTschBlockDim(aiv, aic, aiv),
            platform->GetLibApiWorkSpaceSize()};
}

void CheckTensor(const at::Tensor& t, int64_t dims, const char* name)
{
    TORCH_CHECK(t.defined() && t.device().type() == c10::DeviceType::PrivateUse1, name, " must be an NPU tensor");
    TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
    TORCH_CHECK(t.dim() == dims, name, " must be ", dims, "D");
}

namespace {

template <typename T>
T CeilDiv(T value, T divisor)
{
    return (value + divisor - 1) / divisor;
}

template <typename T>
T AlignUp(T value, T alignment)
{
    return CeilDiv(value, alignment) * alignment;
}

uint32_t DispatchCopyBytes(const MegaMoeTilingData& t)
{
    return static_cast<uint32_t>(MegaMoeImpl::CalcQuantTokenScaleBytes(
        t.h, t.topK, t.topkWeightsPrefetch != 0));
}

MegaMoeDispatchBufferConfig DispatchConfig(const MegaMoeTilingData& t, uint32_t ub)
{
    MegaMoeDispatchBufferConfig c{};
    // Compact-route receive stores at most one local-expert index per source
    // token.  The old mask route used numMaxTokens * topK here.
    const uint64_t total = static_cast<uint64_t>(t.numMaxTokensPerRank);
    const uint64_t aligned = AlignUp(total, uint64_t{256});
    c.copyBufferBytes = DispatchCopyBytes(t);
    const uint32_t fixed =
        AlignUp(t.epWorldSize * t.moeExpertPerRank * 4U, 32U) + AlignUp(t.moeExpertPerRank * 4U, 32U);
    c.routeItemsPerBatch = static_cast<int32_t>(std::min<uint64_t>(aligned, BASE_RECV_ROUTE_ITEMS_PER_BATCH));
    c.routeBatchCount = static_cast<int32_t>(CeilDiv(total, static_cast<uint64_t>(c.routeItemsPerBatch)));
    const uint32_t route = static_cast<uint32_t>(c.routeItemsPerBatch);
    const uint32_t without_slots = fixed + route * sizeof(int32_t);
    const uint32_t budget = ub > without_slots ? ub - without_slots : 0U;
    c.bufferCount = std::clamp(static_cast<int32_t>(budget / (c.copyBufferBytes + 32U)), MIN_DISPATCH_BUFFER_COUNT,
                               MAX_DISPATCH_BUFFER_COUNT);
    if (static_cast<uint64_t>(c.routeItemsPerBatch) < total) {
        const uint32_t fixed_with_slots = fixed + static_cast<uint32_t>(c.bufferCount) * (c.copyBufferBytes + 32U);
        const uint32_t route_budget = ub > fixed_with_slots ? ub - fixed_with_slots : 0U;
        uint32_t expanded = route_budget / sizeof(int32_t);
        expanded = expanded / 256U * 256U;
        expanded = static_cast<uint32_t>(std::min<uint64_t>(expanded, aligned));
        if (expanded > static_cast<uint32_t>(c.routeItemsPerBatch)) {
            c.routeItemsPerBatch = static_cast<int32_t>(expanded);
            c.routeBatchCount = static_cast<int32_t>(CeilDiv(total, static_cast<uint64_t>(expanded)));
        }
    }
    return c;
}

MegaMoeSendMaskBufferConfig SendMaskConfig(const MegaMoeTilingData& t, uint32_t fixed, uint32_t owned_experts,
                                           uint32_t ub)
{
    MegaMoeSendMaskBufferConfig c{};
    const uint64_t total = static_cast<uint64_t>(t.numMaxTokensPerRank) * t.topK;
    const uint64_t aligned = AlignUp(total, uint64_t{256});
    c.routeItemsPerBatch = static_cast<int32_t>(std::min<uint64_t>(aligned, BASE_SEND_ROUTE_ITEMS_PER_BATCH));
    c.routeBatchCount = static_cast<int32_t>(CeilDiv(total, static_cast<uint64_t>(c.routeItemsPerBatch)));
    auto route_slot_bytes = [&](uint32_t route_items) {
        // A batch can begin in the middle of a token's top-k segment.  Reserve
        // one matched index per covered token, including both boundary tokens.
        const uint64_t matched = CeilDiv<uint64_t>(static_cast<uint64_t>(route_items) + t.topK - 1U, t.topK);
        const uint64_t valid_index_bytes = AlignUp<uint64_t>(matched * sizeof(int32_t), 32U);
        return static_cast<uint32_t>(route_items / 8U + valid_index_bytes);
    };
    c.bufferBytes = route_slot_bytes(static_cast<uint32_t>(c.routeItemsPerBatch));
    const uint32_t route_bytes = static_cast<uint32_t>(c.routeItemsPerBatch) * 4U;
    const uint32_t without_masks = fixed + 2U * route_bytes;
    const uint32_t budget = ub > without_masks ? ub - without_masks : 0U;
    c.bufferCount = std::min(static_cast<int32_t>(budget / c.bufferBytes), MAX_SEND_MASK_BUFFER_COUNT);
    const uint64_t pushes = static_cast<uint64_t>(c.routeBatchCount) * owned_experts;
    if (pushes > 0 && static_cast<uint64_t>(c.bufferCount) > pushes) {
        c.bufferCount = static_cast<int32_t>(pushes);
    }
    c.bufferCount = std::max(c.bufferCount, MIN_SEND_MASK_BUFFER_COUNT);
    if (static_cast<uint64_t>(c.routeItemsPerBatch) < total) {
        const uint64_t fixed_with_padding =
            static_cast<uint64_t>(fixed) + static_cast<uint64_t>(c.bufferCount) * (32U + 2U * sizeof(int32_t));
        const uint64_t route_budget = ub > fixed_with_padding ? ub - fixed_with_padding : 0U;
        const uint64_t ring_bits =
            static_cast<uint64_t>(c.bufferCount) +
            CeilDiv<uint64_t>(static_cast<uint64_t>(c.bufferCount) * sizeof(int32_t) * 8U, t.topK);
        uint64_t expanded = route_budget * 8U / (2U * sizeof(int32_t) * 8U + ring_bits);
        expanded = expanded / 256U * 256U;
        expanded = std::min<uint64_t>(expanded, aligned);
        if (expanded > static_cast<uint64_t>(c.routeItemsPerBatch)) {
            c.routeItemsPerBatch = static_cast<int32_t>(expanded);
            c.routeBatchCount = static_cast<int32_t>(CeilDiv(total, static_cast<uint64_t>(expanded)));
            c.bufferBytes = route_slot_bytes(static_cast<uint32_t>(expanded));
        }
    }
    return c;
}

uint64_t HostFlagElements(const MegaMoeTilingData& t)
{
    const uint64_t waves = CeilDiv<uint64_t>(t.maxOutputSize, 256U);
    const uint64_t wave_slots = waves * MegaMoeImpl::INT_CACHELINE;
    uint64_t count = static_cast<uint64_t>(t.moeExpertPerRank) *
                     (wave_slots + wave_slots + static_cast<uint64_t>(MegaMoeImpl::INT_CACHELINE) * t.aicNum);
    const bool is_w4 = t.groupedMatmulMode == MegaMoeImpl::GROUPED_MATMUL_MODE_A8W4;
    if (is_w4 || (t.topoType == MegaMoeImpl::TOPO_TYPE_MTE && t.combineQuantMode == MegaMoeImpl::COMBINE_NO_QUANT)) {
        // Pairwise GMM2/combine publishes one sequence per AIC.
        count += static_cast<uint64_t>(t.aicNum) * MegaMoeImpl::INT_CACHELINE;
    }
    if (t.topoType == MegaMoeImpl::TOPO_TYPE_MTE && (t.combineQuantMode != MegaMoeImpl::COMBINE_NO_QUANT ||
                                                     t.groupedMatmulMode == MegaMoeImpl::GROUPED_MATMUL_MODE_A8W4)) {
        // Match WorkspaceInfo: the atomic A8W4 path owns one counter per M-group,
        // while quantized Combine retains one ready slot per AIC.
        const uint64_t ready_slots_per_expert = t.combineQuantMode == MegaMoeImpl::COMBINE_NO_QUANT
                                                    ? wave_slots
                                                    : static_cast<uint64_t>(t.aicNum) * MegaMoeImpl::INT_CACHELINE;
        count += static_cast<uint64_t>(t.moeExpertPerRank) * ready_slots_per_expert;
    }
    if (t.sharedExpertNum > 0U) {
        count += CeilDiv<uint64_t>(t.bs, 256U) * t.sharedExpertNum * MegaMoeImpl::INT_CACHELINE;
        if (t.topoType == MegaMoeImpl::TOPO_TYPE_MTE && t.groupedMatmulMode == MegaMoeImpl::GROUPED_MATMUL_MODE_A8W4) {
            count += 4ULL * t.aicNum * MegaMoeImpl::INT_CACHELINE;
        }
    }
    return count;
}

void SetSendMaskConfigs(MegaMoeTilingData& t, uint32_t ub)
{
    const uint32_t reset_per_core =
        static_cast<uint32_t>(CeilDiv(HostFlagElements(t), static_cast<uint64_t>(t.blockAivNum)));
    const uint32_t reset_count = std::min(reset_per_core, static_cast<uint32_t>(DISPATCH_RESET_BATCH));
    const uint32_t reset_bytes = AlignUp(reset_count, uint32_t{MegaMoeImpl::INT32_PER_256B}) * 4U;
    const uint32_t quant_out = DispatchCopyBytes(t);
    const uint32_t quant_in = AlignUp(t.h, 128U) * 2U;
    const auto sharedPolicy = MegaMoeImpl::SelectSharedExpertPolicy(
        t.groupedMatmulMode == MegaMoeImpl::GROUPED_MATMUL_MODE_A8W4 &&
            t.actMode != static_cast<uint8_t>(MegaMoeImpl::MegaMoeActMode::GELU),
        t.sharedExpertNum, t.bs);
    const uint32_t route_cores = sharedPolicy.routeAiv1Only ? t.aicNum : t.blockAivNum;
    const uint32_t max_experts = MegaMoeImpl::CalcSendMaskExpertCapacity(&t, route_cores);
    const uint32_t accumulator = AlignUp(max_experts * 4U, 32U);
    const uint32_t fixed = reset_bytes + 2048U + 2U * quant_out + 2U * quant_in + accumulator;
    const uint32_t experts = t.epWorldSize * t.moeExpertPerRank;
    const uint32_t base = experts / route_cores;
    t.sendMaskCoreCountWithExtraExpert = experts % route_cores;
    const bool rail = MegaMoeImpl::UseDecodeRailRouting(&t);
    t.sendMaskConfigForCoreWithExtraExpert = SendMaskConfig(t, fixed, rail ? max_experts : base + 1U, ub);
    t.sendMaskConfigForCoreWithoutExtraExpert = SendMaskConfig(t, fixed, rail ? max_experts : base, ub);
}

MegaMoeUnpermuteBufferConfig UnpermuteConfig(const MegaMoeTilingData& t, uint32_t core_tokens, uint32_t ub)
{
    MegaMoeUnpermuteBufferConfig c{};
    if (core_tokens == 0) return c;
    const uint32_t bf16_bytes = AlignUp(t.h * 2U, 32U);
    const uint32_t fp32_bytes = AlignUp(t.h * 4U, 32U);
    const uint32_t slot_bytes = bf16_bytes + fp32_bytes;
    c.bf16SlotElementCount = bf16_bytes / 2U;
    c.fp32SlotElementCount = fp32_bytes / 4U;
    const uint32_t base_tokens = UNPERMUTE_WEIGHT_ITEMS_PER_BATCH / t.topK;
    c.tokensPerBatch = static_cast<int32_t>(std::min(base_tokens, core_tokens));
    auto set_weight_bytes = [&]() {
        c.topKWeightsBufferBytes = AlignUp(static_cast<uint32_t>(c.tokensPerBatch) * t.topK * 4U, 32U);
        c.topKWeightsConversionBufferBytes = 0U;
    };
    set_weight_bytes();
    const uint32_t before_inputs = c.topKWeightsBufferBytes + slot_bytes;
    const uint32_t input_budget = ub > before_inputs ? ub - before_inputs : 0U;
    c.inputBufferCount = std::clamp(static_cast<int32_t>(input_budget / slot_bytes), MIN_UNPERMUTE_INPUT_BUFFER_COUNT,
                                    MAX_UNPERMUTE_INPUT_BUFFER_COUNT);
    c.inputBufferCount = std::min(c.inputBufferCount, c.tokensPerBatch * static_cast<int32_t>(t.topK));
    c.inputBufferCount = std::max(c.inputBufferCount, MIN_UNPERMUTE_INPUT_BUFFER_COUNT);
    if (base_tokens < core_tokens) {
        const uint32_t fixed = (static_cast<uint32_t>(c.inputBufferCount) + 1U) * slot_bytes;
        const uint32_t weight_budget = ub > fixed + UNPERMUTE_WEIGHT_ALIGNMENT_RESERVE_BYTES
                                           ? ub - fixed - UNPERMUTE_WEIGHT_ALIGNMENT_RESERVE_BYTES
                                           : 0U;
        const uint32_t expanded = std::min(weight_budget / (t.topK * 4U), core_tokens);
        if (expanded > static_cast<uint32_t>(c.tokensPerBatch)) {
            c.tokensPerBatch = static_cast<int32_t>(expanded);
            set_weight_bytes();
        }
    }
    return c;
}

void SetUnpermuteConfigs(MegaMoeTilingData& t, uint32_t ub)
{
    const uint32_t full = CeilDiv(t.bs, t.blockAivNum);
    const uint32_t active = CeilDiv(t.bs, full);
    const uint32_t tail = t.bs - (active - 1U) * full;
    const bool tail_full = tail == full;
    t.unpermuteFullTokenChunkCoreCount = tail_full ? active : active - 1U;
    t.unpermuteConfigForFullTokenChunk = UnpermuteConfig(t, full, ub);
    t.unpermuteConfigForTailTokenChunk = tail_full ? MegaMoeUnpermuteBufferConfig{} : UnpermuteConfig(t, tail, ub);
}

MegaMoeTilingData MakeTiling(int64_t bs, int64_t h, int64_t hidden_dim, int64_t topk, int64_t local_experts,
                             int64_t world, int64_t max_tokens, int64_t max_recv, bool is_a8w4,
                             int64_t shared_experts, const PlatformInfo& p, int64_t topk_weights_type,
                             const std::string& activation)
{
    MegaMoeTilingData t{};
    t.moeExpertPerRank = static_cast<uint32_t>(local_experts);
    t.bs = static_cast<uint32_t>(bs);
    t.h = static_cast<uint32_t>(h);
    t.hiddenDim = static_cast<uint32_t>(hidden_dim);
    t.epWorldSize = static_cast<uint32_t>(world);
    t.blockNumPerEP = std::max(1U, p.aic / t.epWorldSize);
    t.numMaxTokensPerRank = static_cast<uint32_t>(max_tokens == 0 ? bs : max_tokens);
    t.maxOutputSize = static_cast<uint32_t>(
        max_recv == 0 ? t.numMaxTokensPerRank * t.epWorldSize * std::min<uint32_t>(topk, local_experts) : max_recv);
    t.topK = static_cast<uint32_t>(topk);
    t.aicNum = p.aic;
    t.blockAivNum = p.aiv;
    t.combineQuantMode = MegaMoeImpl::COMBINE_NO_QUANT;
    t.clampLimit = std::numeric_limits<float>::max();
    t.groupedMatmulMode = is_a8w4 ? MegaMoeImpl::GROUPED_MATMUL_MODE_A8W4 : MegaMoeImpl::GROUPED_MATMUL_MODE_GENERAL;
    t.topoType = MegaMoeImpl::TOPO_TYPE_MTE;
    t.sharedExpertNum = static_cast<uint32_t>(shared_experts);
    t.combineSyncSlotCountPerExpert = 0;
    t.topkWeightsPrefetch = topk_weights_type;
    // Prefetch switches activation readiness to per-expert tile flags. Each
    // token selects an expert at most once; reserve all EP source tokens and
    // the full GMM1 N span (also covers fused small-BS scheduling).
    t.maxTilesPerExpert = topk_weights_type
        ? CeilDiv<uint32_t>(t.numMaxTokensPerRank * t.epWorldSize, 256U) * CeilDiv<uint32_t>(t.hiddenDim, 256U)
        : 0U;
    t.actMode = static_cast<uint8_t>(activation == "gelu" ? MegaMoeImpl::MegaMoeActMode::GELU
                                                           : MegaMoeImpl::MegaMoeActMode::SWIGLU);
    t.actSubMode = static_cast<uint8_t>(MegaMoeImpl::MegaMoeActSubMode::DEFAULT);
    t.activationAlpha = 1.0F;
    t.activationBeta = 1.0F;
    // 按本次每卡 BS 选择，不用预分配容量；小 batch 缩短 wave，较早释放 GMM2。
    if (is_a8w4 && activation != "gelu" && t.bs <= MegaMoeImpl::W4A8_SMALL_BATCH_MAX_TOKENS) {
        const uint64_t gmm1_tasks_per_m_group =
            CeilDiv<uint64_t>(t.hiddenDim / MegaMoeImpl::ACTIVATION_N_HALF, MegaMoeImpl::GMM1_FUSED_TILE_N);
        t.mGroupsPerWave =
            static_cast<uint32_t>(CeilDiv<uint64_t>(static_cast<uint64_t>(p.aic) * 2U, gmm1_tasks_per_m_group));
    } else {
        // 大 batch 和 A8W8 的预算包括 GMM2 的最低任务量约束。
        const uint64_t gmm1_tiles = CeilDiv<uint64_t>(t.hiddenDim, 256U);
        const uint64_t gmm2_tiles = CeilDiv<uint64_t>(t.h, 256U);
        t.mGroupsPerWave = static_cast<uint32_t>(std::max(
            CeilDiv<uint64_t>(static_cast<uint64_t>(p.aic) * 4U, gmm1_tiles), CeilDiv<uint64_t>(p.aic, gmm2_tiles)));
    }
    t.isPerExpertWeightTensor = false;
    t.dispatchBufferConfig = DispatchConfig(t, p.ub);
    SetSendMaskConfigs(t, p.ub);
    SetUnpermuteConfigs(t, p.ub);
    return t;
}

uint64_t Address(const torch::Tensor& tensor)
{
    return reinterpret_cast<uint64_t>(tensor.data_ptr());
}

std::vector<uint32_t> BuildGeluLookupTable(const torch::Tensor& example, uint32_t blocks)
{
    // Calibrate once per plan with the device's BF16 cast and CANN Erf, including nonfinite inputs.
    constexpr int64_t keys = MegaMoeImpl::MTE_GELU_LUT_KEY_COUNT;
    auto scratch = torch::empty({keys}, example.options().dtype(torch::kInt32));
    const auto stream = c10_npu::getCurrentNPUStream();
    megamoe_init_gelu_factors(stream.stream(), static_cast<uint8_t*>(scratch.data_ptr()), blocks);
    ASCEND_DEEPEP_CHECK_ACL(aclrtSynchronizeStream(stream.stream()));
    std::vector<uint32_t> factors(keys);
    ASCEND_DEEPEP_CHECK_ACL(aclrtMemcpy(factors.data(), factors.size() * sizeof(uint32_t), scratch.data_ptr(),
                                        scratch.nbytes(), ACL_MEMCPY_DEVICE_TO_HOST));
    return PackGeluLookupTable(factors);
}

int64_t MteWorkspaceAllocationBytes(const MegaMoeTilingData& tiling, const PlatformInfo& platform)
{
    const MegaMoeImpl::WorkspaceInfo layout(nullptr, &tiling);
    const int64_t overhead = static_cast<int64_t>(platform.system_workspace) + kReservedWorkspaceBytes;
    TORCH_CHECK(layout.workspaceSize >= 0 && layout.workspaceSize <= std::numeric_limits<int64_t>::max() - overhead,
                "MegaMoE workspace size overflow");
    return layout.workspaceSize + overhead;
}

constexpr int64_t kTraceRowElements = MegaMoeUrma::kTimerCores * MegaMoeUrma::kTraceRecordElements;

int64_t ValidateTimer(const torch::Tensor& timer, bool urma)
{
#ifndef DEEPEP_MEGAMOE_TIMER
    TORCH_CHECK(false, "Rebuild with DEEPEP_MEGAMOE_TIMER=ON before passing timer");
#endif
    CheckTensor(timer, 1, "timer");
    TORCH_CHECK(timer.scalar_type() == torch::kInt64, "timer must be int64");
    if (!urma) {
        TORCH_CHECK(timer.numel() == AscendTimer::TOTAL_BUFFER_SIZE, "Invalid MTE timer allocation");
        return 0;
    }
    const int64_t trace_elements = timer.numel() - int64_t{MegaMoeUrma::kTimerElements};
    TORCH_CHECK(Address(timer) % 128U == 0 && trace_elements % kTraceRowElements == 0 &&
                    trace_elements >= 2 * kTraceRowElements &&
                    trace_elements <= (int64_t{MegaMoeUrma::kTraceMaxCapacity} + 1) * kTraceRowElements,
                "Invalid URMA fine trace allocation");
    return trace_elements;
}

}  // namespace

struct UrmaWorkspace {
    MegaMoeUrma::Arguments args_{};
    std::vector<torch::Tensor> buffers_;

    UrmaWorkspace(torch::Tensor example, int64_t world, int64_t intermediate, int64_t topk, int64_t cv_rows,
                  int64_t max_tokens, int64_t rank, uint64_t ffts, uint64_t symmetric_address, int64_t shared_experts,
                  int64_t shared_intermediate, int64_t num_experts, const PlatformInfo& platform, int64_t topk_weights_type,
                  const std::string& activation, bool windowed, int64_t timeoutSeconds, bool fp8Weights)
    {
        const auto layout = MakeUrmaSymmetricLayout(world, max_tokens, topk, example.size(1), cv_rows, topk_weights_type, windowed);
        args_.config = layout.config;
        args_.config.timeoutCycles = uint64_t(timeoutSeconds) * 1000000000ULL;
        args_.config.windowedTransport = windowed ? 1U : 0U;
        auto& config = args_.config;
        config.fp8Weights = fp8Weights;
        config.activationMode = activation == "gelu"
                                    ? static_cast<uint32_t>(DEEPEP_MEGAMOE_IMPL_NAMESPACE::MegaMoeActMode::GELU)
                                    : static_cast<uint32_t>(DEEPEP_MEGAMOE_IMPL_NAMESPACE::MegaMoeActMode::SWIGLU);
        config.bs = example.size(0);
        config.intermediate = intermediate;
        config.localExperts = num_experts / world;
        config.rank = rank;
        config.gmm1AheadMGroups = 12U;
        args_.sharedExperts = shared_experts;
        args_.sharedIntermediate = shared_experts ? shared_intermediate : 0;
        args_.symmetric = symmetric_address;
        args_.ffts = ffts;
        const auto options = example.options().dtype(torch::kUInt8);
        BindUrmaScratch(args_, layout, max_tokens, platform.system_workspace, [&](uint64_t bytes) {
            buffers_.push_back(torch::empty({static_cast<int64_t>(bytes)}, options));
            return Address(buffers_.back());
        });
        if (args_.geluLookup && activation == "gelu") {
            const auto factors = BuildGeluLookupTable(example, platform.aic);
            if (!factors.empty() && factors.size() <= MegaMoeUrma::WindowGeluUb::tableWords) {
                const auto bytes = factors.size() * sizeof(uint32_t);
                ASCEND_DEEPEP_CHECK_ACL(aclrtMemcpy(reinterpret_cast<void*>(args_.geluLookup),
                    MegaMoeUrma::WindowGeluUb::tableBytes, factors.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE));
                args_.geluLookupWords = static_cast<uint32_t>(factors.size());
            }
        }
    }

    template <class Visitor>
    void for_each_tensor(Visitor&& visit) const
    {
        for (const auto& buffer : buffers_) visit(buffer);
    }
};

Plan::Plan(torch::Tensor example, int64_t intermediate, int64_t topk, int64_t experts, int64_t world,
           int64_t max_tokens, bool is_a8w4, int64_t shared_experts, int64_t ranks_per_node, uint64_t ffts,
           uint64_t symmetric_address, int64_t symmetric_bytes, const std::string& backend, int64_t topk_weights_type,
           const std::string& activation, const std::string& mma_type, bool windowed,
           int64_t timeout_seconds)
    : backend_(backend == "mte" ? Backend::Mte : Backend::Urma),
      topk_weights_type_(topk_weights_type == -1 ? static_cast<int64_t>(activation == "gelu") : topk_weights_type),
      ffts_(ffts),
      example_(example),
      intermediate_(intermediate),
      topk_(topk),
      experts_(experts),
      world_(world),
      max_tokens_(max_tokens),
      shared_experts_(shared_experts),
      is_a8w4_(is_a8w4),
      windowed_(windowed),
      timeout_seconds_(timeout_seconds),
      activation_(activation),
      symmetric_address_(symmetric_address),
      symmetric_bytes_(symmetric_bytes)
{
    TORCH_CHECK(topk_weights_type_ == 0 || topk_weights_type_ == 1, "topk_weights_type must be 0 or 1");
    TORCH_CHECK(activation_ == "swiglu" || activation_ == "gelu", "activation must be 'swiglu' or 'gelu'");
    const std::string resolved_mma_type = mma_type.empty() ? (is_a8w4 ? "fp8xfp4" : "fp8xfp8") : mma_type;
    TORCH_CHECK(resolved_mma_type == "fp8xfp4" || resolved_mma_type == "fp8xfp8",
                "mma_type must be fp8xfp4 or fp8xfp8");
    TORCH_CHECK(is_a8w4_ == (resolved_mma_type == "fp8xfp4"),
                "is_a8w4 and mma_type describe different matrix modes");
    TORCH_CHECK(backend == "mte" || backend == "urma", "MegaMoE backend must be mte or urma");
    CheckTensor(example, 2, "plan input");
    TORCH_CHECK(example.scalar_type() == torch::kBFloat16 || example.scalar_type() == c10::ScalarType::Float8_e4m3fn,
                "Plan input must be BF16 or FP8 E4M3");
    TORCH_CHECK(example.size(0) > 0 && example.size(0) <= max_tokens, "BS outside allocated capacity");
    TORCH_CHECK(ranks_per_node == 0 || (ranks_per_node > 0 && world % ranks_per_node == 0),
                "ranks_per_node must divide EP");
    const bool urma = backend_ == Backend::Urma;
    TORCH_CHECK(!windowed_ ||
                    (urma && shared_experts == 0 &&
                     (activation_ == "gelu" ||
                      (activation_ == "swiglu" && topk_weights_type_ == 0 && (world == 16 || world == 32)))),
                "Windowed MegaMoE requires URMA S0 with GELU or EP16/EP32 SwiGLU with Finalize weighting");
    TORCH_CHECK(timeout_seconds_ > 0 && timeout_seconds_ <= 3600, "Plan timeout_seconds must be in [1,3600]");
    TORCH_CHECK(!windowed_ || (world == 32 ? ranks_per_node == 16 : (ranks_per_node == 0 || ranks_per_node == world)),
                "Windowed MegaMoE requires one host for EP8/16 or two contiguous 16-rank hosts for EP32");
    if (urma) {
        TORCH_CHECK(is_a8w4_ || (windowed_ && (world == 16 || world == 32)),
                    "URMA routed FP8xFP8 requires windowed EP16/EP32 without shared experts");
        TORCH_CHECK(intermediate >= 256 && intermediate <= 16384 && intermediate % 256 == 0,
                    "URMA intermediate width must be divisible by 256, in [256,16384]");
        TORCH_CHECK(shared_experts == 0 || shared_experts == 1, "URMA supports S0/S1");
    } else {
        TORCH_CHECK(intermediate >= 256 && intermediate <= 4096 && intermediate % 128 == 0,
                    "Plan intermediate_hidden must be divisible by 128, in [256,4096]");
        TORCH_CHECK(shared_experts >= 0 && shared_experts <= 8 && (is_a8w4_ || shared_experts == 0),
                    "MTE shared experts require A8W4, with count in [0,8]");
    }
    const int64_t input_hidden = example.size(1);
    const auto required = buffer_size(world, experts, max_tokens, topk, input_hidden, backend, topk_weights_type_,
                                      resolved_mma_type, windowed_);
    TORCH_CHECK(symmetric_address != 0 && symmetric_bytes >= required, "MegaMoE symmetric buffer needs ", required,
                " bytes, got ", symmetric_bytes);
    platform_ = GetPlatformInfo();
    if (urma) {
        TORCH_CHECK(platform_.aic >= MegaMoeUrma::kAic && platform_.aiv >= 2U * MegaMoeUrma::kAic &&
                        platform_.ub >= MegaMoeUrma::kRequiredUbBytes,
                    "URMA requires 28 AIC/56 AIV and 248 KiB usable UB");
        auto* device_platform = platform_ascendc::PlatformAscendCManager::GetInstance();
        platform_.aic = MegaMoeUrma::kAic;
        platform_.aiv = 2U * MegaMoeUrma::kAic;
        platform_.block_dim = device_platform->CalcTschBlockDim(platform_.aiv, platform_.aic, platform_.aiv);
        TORCH_CHECK(aclshmem_n_pes() == world, "MegaMoE plan does not match its SHMEM runtime");
    }

    // Both backends resolve peer mappings once during plan preparation.
    // Only MTE uploads this context; URMA passes the symmetric base in its ABI.
    Mc2Aclnn::Mc2MoeContext context{};
    const int rank = aclshmem_my_pe();
    TORCH_CHECK(rank >= 0 && rank < world && world <= Mc2Aclnn::HCCL_MAX_RANK_SIZE, "Invalid SHMEM rank/world: ", rank,
                "/", world);
    context.epRankId = static_cast<uint32_t>(rank);
    context.rankSizePerServer = static_cast<uint32_t>(world);
    for (int64_t peer = 0; peer < world; ++peer) {
        if (urma && world == 32 && rank / 16 != peer / 16) continue;  // RDMA peers have no MTE mapping.
        void* address = aclshmem_ptr(reinterpret_cast<void*>(symmetric_address), static_cast<int32_t>(peer));
        TORCH_CHECK(address != nullptr, "rank=", rank, " prepare: cannot resolve peer=", peer);
        context.epHcclBuffer_[peer] = reinterpret_cast<uint64_t>(address);
    }
    if (!urma) {
        tiling_ = MakeTiling(example.size(0), input_hidden, intermediate * (activation_ == "gelu" ? 1 : 2), topk,
                             experts, world, max_tokens, 0, is_a8w4_, shared_experts, platform_,
                             topk_weights_type_, activation_);
        tiling_.dispatchRanksPerNode = static_cast<uint32_t>(ranks_per_node);
        static_assert(sizeof(MegaMoeTilingData) == 224, "MegaMoE tiling ABI changed");
        std::vector<uint32_t> calibratedTable;
        if (MegaMoeImpl::MteGeluLutFits(tiling_, platform_.ub)) {
            calibratedTable = BuildGeluLookupTable(example, platform_.aic);
            tiling_.geluLutWords = static_cast<uint16_t>(calibratedTable.size());
        }
        const int64_t bytes = MteWorkspaceAllocationBytes(tiling_, platform_);
        const size_t lutBytes = tiling_.geluLutWords * sizeof(uint32_t);
        workspace_ = torch::empty({bytes}, example.options().dtype(torch::kUInt8));
        if (lutBytes) {
            ASCEND_DEEPEP_CHECK_ACL(aclrtMemcpy(workspace_.data_ptr(),
                                                lutBytes, calibratedTable.data(), lutBytes, ACL_MEMCPY_HOST_TO_DEVICE));
        }
        context_ = torch::empty({static_cast<int64_t>(sizeof(context))}, workspace_.options());
        ASCEND_DEEPEP_CHECK_ACL(
            aclrtMemcpy(context_.data_ptr(), sizeof(context), &context, sizeof(context), ACL_MEMCPY_HOST_TO_DEVICE));
    } else if (windowed_) {
        // S0 fixes all workspace dimensions at prepare, before any peer enters the wave.
        urma_ = std::make_unique<UrmaWorkspace>(example_, world_, intermediate_, topk_, UrmaCvRows(world_, max_tokens_),
                                               max_tokens_, rank, ffts_, symmetric_address_, shared_experts_, 0,
                                               world_ * experts_, platform_, topk_weights_type_, activation_, windowed_,
                                               timeout_seconds_, !is_a8w4_);
    }
    // Other URMA plans still need the shared tensor dimensions from forward.
}

Plan::~Plan() = default;

int64_t Plan::workspace_bytes() const
{
    if (backend_ == Backend::Mte) return workspace_.defined() ? workspace_.nbytes() : 0;
    int64_t bytes = 0;
    if (urma_)
        urma_->for_each_tensor([&](const torch::Tensor& t) {
            if (t.defined()) bytes += t.nbytes();
        });
    return bytes;
}

std::tuple<torch::Tensor, torch::Tensor> Plan::forward(
    torch::Tensor x, c10::optional<torch::Tensor> x_scales, torch::Tensor topk_ids, torch::Tensor topk_weights,
    torch::Tensor weight1, torch::Tensor weight2, torch::Tensor weight_scales1, torch::Tensor weight_scales2,
    c10::optional<torch::Tensor> timer, c10::optional<torch::Tensor> shared_weight1,
    c10::optional<torch::Tensor> shared_weight2, c10::optional<torch::Tensor> shared_weight_scales1,
    c10::optional<torch::Tensor> shared_weight_scales2, c10::optional<torch::Tensor> out, int64_t generation, float activation_clamp)
{
    const bool urma = backend_ == Backend::Urma;
    CheckTensor(x, 2, "x");
    CheckTensor(topk_ids, 2, "topk_ids");
    CheckTensor(topk_weights, 2, "topk_weights");
    CheckTensor(weight1, 3, "weight1");
    CheckTensor(weight2, 3, "weight2");
    CheckTensor(weight_scales1, 4, "weight_scales1");
    CheckTensor(weight_scales2, 4, "weight_scales2");
    const bool prequantized = x_scales.has_value();
    const int64_t bs = x.size(0);
    const int64_t h = x.size(1);
    const int64_t hidden = intermediate_ * (activation_ == "gelu" ? 1 : 2);
    const bool valid_bs = windowed_ ? (bs > 0 && bs <= max_tokens_) : bs == example_.size(0);
    TORCH_CHECK(x.device() == example_.device() && valid_bs && h == example_.size(1),
                "Input shape/device differs from prepared plan");
    TORCH_CHECK(topk_ids.scalar_type() == torch::kInt32 && topk_ids.sizes() == torch::IntArrayRef({bs, topk_}),
                "topk_ids must be int32 [BS,TopK]");
    TORCH_CHECK(topk_weights.scalar_type() == torch::kFloat32 && topk_weights.sizes() == topk_ids.sizes(),
                "topk_weights must be FP32 [BS,TopK]");
    TORCH_CHECK(activation_clamp >= 0.0F, "activation_clamp must be nonnegative and not NaN");
    TORCH_CHECK(activation_ != "gelu" || activation_clamp == std::numeric_limits<float>::max(),
                "activation_clamp is only defined for SwiGLU");
    tiling_.clampLimit = std::min(activation_clamp, std::numeric_limits<float>::max());
    TORCH_CHECK(!urma || prequantized, "URMA requires prequantized FP8 input with E8M0 scales");
    const auto input_type = prequantized ? c10::ScalarType::Float8_e4m3fn : torch::kBFloat16;
    TORCH_CHECK(x.scalar_type() == input_type,
                "x must be BF16 or FP8 E4M3 with E8M0 scales");
    if (prequantized) {
        if (urma && x_scales->dim() == 3) {
            CheckTensor(*x_scales, 3, "x_scales");
            TORCH_CHECK(x_scales->sizes() == torch::IntArrayRef({bs, h / 64, 2}), "Invalid URMA input scale shape");
            x_scales = x_scales->reshape({bs, h / 32});
        }
        CheckTensor(*x_scales, 2, "x_scales");
        TORCH_CHECK(x_scales->scalar_type() == c10::ScalarType::Float8_e8m0fnu &&
                        x_scales->sizes() == torch::IntArrayRef({bs, CeilDiv(h, int64_t{32})}),
                    "x_scales must be E8M0 [BS,H/32]");
    }
    const auto weight_type = is_a8w4_ ? torch::kUInt8 : c10::ScalarType::Float8_e4m3fn;
    TORCH_CHECK(weight1.scalar_type() == weight_type && weight2.scalar_type() == weight_type,
                "Weights differ from prepared matrix mode");
    TORCH_CHECK(weight1.sizes() == torch::IntArrayRef({experts_, hidden, is_a8w4_ ? h / 2 : h}) &&
                    weight2.sizes() == torch::IntArrayRef({experts_, h, is_a8w4_ ? intermediate_ / 2 : intermediate_}),
                "Routed weight shape differs from plan");
    TORCH_CHECK(weight_scales1.scalar_type() == c10::ScalarType::Float8_e8m0fnu &&
                    weight_scales2.scalar_type() == c10::ScalarType::Float8_e8m0fnu &&
                    weight_scales1.sizes() == torch::IntArrayRef({experts_, hidden, CeilDiv(h, int64_t{64}), 2}) &&
                    weight_scales2.sizes() == torch::IntArrayRef({experts_, h, CeilDiv(intermediate_, int64_t{64}), 2}),
                "Invalid routed weight scales");
    for (const auto* tensor : {&topk_ids, &topk_weights, &weight1, &weight2, &weight_scales1, &weight_scales2})
        TORCH_CHECK(tensor->device() == x.device(), "MegaMoE tensors must be on the same NPU");
    for (const auto* tensor :
         {&x_scales, &timer, &out, &shared_weight1, &shared_weight2, &shared_weight_scales1, &shared_weight_scales2})
        TORCH_CHECK(!tensor->has_value() || tensor->value().device() == x.device(),
                    "Optional MegaMoE tensors must be on the same NPU");

    const bool has_shared = shared_weight1.has_value();
    TORCH_CHECK(shared_weight2.has_value() == has_shared && shared_weight_scales1.has_value() == has_shared &&
                    shared_weight_scales2.has_value() == has_shared && has_shared == (shared_experts_ > 0),
                "All shared weight/scale tensors must match the configured shared expert count");
    int64_t shared_i = 0;
    if (has_shared) {
        CheckTensor(*shared_weight1, 3, "shared_weight1");
        CheckTensor(*shared_weight2, 3, "shared_weight2");
        CheckTensor(*shared_weight_scales1, 4, "shared_weight_scales1");
        CheckTensor(*shared_weight_scales2, 4, "shared_weight_scales2");
        TORCH_CHECK(activation_ == "gelu" || shared_weight1->size(1) % 2 == 0,
                    "Shared SwiGLU GMM1 must contain gate/up pairs");
        shared_i = shared_weight1->size(1) / (activation_ == "gelu" ? 1 : 2);
        TORCH_CHECK(urma ? (shared_i >= 256 && shared_i <= 16384 && shared_i % 256 == 0) : shared_i == intermediate_,
                    "Shared intermediate is unsupported for the selected backend");
        const auto shared_type = urma ? c10::ScalarType::Float8_e4m3fn : torch::kUInt8;
        TORCH_CHECK(
            is_a8w4_ && shared_weight1->scalar_type() == shared_type &&
                shared_weight2->scalar_type() == shared_type &&
                shared_weight1->sizes() ==
                    torch::IntArrayRef({shared_experts_, shared_i * (activation_ == "gelu" ? 1 : 2), urma ? h : h / 2}) &&
                shared_weight2->sizes() == torch::IntArrayRef({shared_experts_, h, urma ? shared_i : shared_i / 2}),
            "Shared weights have invalid dtype or shape for the selected backend");
        TORCH_CHECK(shared_weight_scales1->scalar_type() == c10::ScalarType::Float8_e8m0fnu &&
                        shared_weight_scales2->scalar_type() == c10::ScalarType::Float8_e8m0fnu &&
                        shared_weight_scales1->sizes() ==
                            torch::IntArrayRef({shared_experts_, shared_i * (activation_ == "gelu" ? 1 : 2),
                                                CeilDiv(h, int64_t{64}), 2}) &&
                        shared_weight_scales2->sizes() ==
                            torch::IntArrayRef({shared_experts_, h, CeilDiv(shared_i, int64_t{64}), 2}),
                    "Invalid shared weight scales");
    }
    uint64_t trace_capacity = 0;
    int64_t trace_elements = 0;
    if (timer) {
        TORCH_CHECK(is_a8w4_ || (urma && windowed_),
                    "MegaMoE timer requires routed A8W4 or windowed URMA A8W8");
        trace_elements = ValidateTimer(*timer, urma);
        if (urma) trace_capacity = trace_elements / kTraceRowElements - 1;
    }
    const int64_t output_sizes[] = {bs, h};
    const torch::IntArrayRef output_shape(output_sizes);
    torch::Tensor y = out ? *out : torch::empty(output_shape, x.options().dtype(torch::kBFloat16));
    CheckTensor(y, 2, "out");
    TORCH_CHECK(y.scalar_type() == torch::kBFloat16 && y.sizes() == output_shape, "out must be BF16 [BS,H]");
    torch::Tensor expert_tokens = torch::empty({experts_}, x.options().dtype(torch::kInt32));
    const auto stream = c10_npu::getCurrentNPUStream();
    // Prepare backend state before assembling either device argument block.
    if (urma) {
        TORCH_CHECK(generation > last_generation_ && generation < (1LL << 31),
                    "URMA generation must be positive and increase across the Buffer lifetime");
        if (!urma_ || shared_intermediate_ != shared_i) {
            if (urma_) ASCEND_DEEPEP_CHECK_ACL(aclrtSynchronizeStream(stream.stream()));
            urma_.reset();
            urma_ = std::make_unique<UrmaWorkspace>(example_, world_, intermediate_, topk_, UrmaCvRows(world_, bs),
                                                    max_tokens_, aclshmem_my_pe(), ffts_, symmetric_address_,
                                                    shared_experts_, shared_i, world_ * experts_, platform_, topk_weights_type_,
                                                    activation_, windowed_, timeout_seconds_, !is_a8w4_);
            shared_intermediate_ = shared_i;
        }
        urma_->args_.config.bs = bs;
        // Invocation flags are reset by AIV workers before the first route join.
        if (timer) {
            ASCEND_DEEPEP_CHECK_ACL(
                aclrtMemsetAsync(timer->data_ptr(), timer->nbytes(), 0, timer->nbytes(), stream.stream()));
        }
    }
    // Common dispatch stage; only the device argument representation differs.
    auto ptr = [](const torch::Tensor& t) { return static_cast<uint8_t*>(t.data_ptr()); };
    auto opt = [&](const c10::optional<torch::Tensor>& t) { return t ? ptr(*t) : nullptr; };
    MteLaunchArgs mte{};
    MegaMoeUrma::Arguments* urma_args = nullptr;
    if (urma) {
        auto& args = urma_->args_;
        args.activationClamp = tiling_.clampLimit;
        args.config.epoch = generation;
        args.x = Address(x);
        args.xScale = Address(*x_scales);
        args.topkIds = Address(topk_ids);
        args.topkIdBytes = topk_ids.element_size();
        args.weights = Address(topk_weights);
        args.weight1 = Address(weight1);
        args.weight2 = Address(weight2);
        args.weightScale1 = Address(weight_scales1);
        args.weightScale2 = Address(weight_scales2);
        args.output = Address(y);
        args.expertCountsOutput = Address(expert_tokens);
        // Aggregate counters occupy the trailer of the same public timer buffer.
        args.timer = timer ? Address(*timer) + trace_elements * sizeof(int64_t) : 0;
        args.trace = timer ? Address(*timer) : 0;
        args.traceCapacity = trace_capacity;
        args.coarseTrace = 0;
        if (has_shared) {
            args.sharedWeight1 = Address(*shared_weight1);
            args.sharedWeight2 = Address(*shared_weight2);
            args.sharedScale1 = Address(*shared_weight_scales1);
            args.sharedScale2 = Address(*shared_weight_scales2);
        }
        urma_args = &args;
    } else {
        mte = {is_a8w4_,
               ptr(context_),
               ptr(x),
               opt(x_scales),
               ptr(topk_ids),
               ptr(topk_weights),
               ptr(weight1),
               ptr(weight2),
               ptr(weight_scales1),
               ptr(weight_scales2),
               opt(shared_weight1),
               opt(shared_weight2),
               opt(shared_weight_scales1),
               opt(shared_weight_scales2),
               ptr(y),
               ptr(expert_tokens),
               ptr(workspace_),
               opt(timer),
               reinterpret_cast<const uint8_t*>(&tiling_),
               ffts_};
    }
    launch_kernel(backend_, platform_.block_dim, platform_.ub, stream.stream(), mte, urma_args);
    if (urma) last_generation_ = generation;
    // Shared lifetime handling for both backends. Input scopes and output ABI do
    // not depend on which private device parameter block was dispatched.
    auto record = [&](const torch::Tensor& t) {
        if (t.defined()) c10_npu::NPUCachingAllocator::recordStream(t.storage().data_ptr(), stream);
    };
    for (const auto& t :
         {x, topk_ids, topk_weights, weight1, weight2, weight_scales1, weight_scales2, y, expert_tokens})
        record(t);
    for (const auto* t :
         {&x_scales, &timer, &shared_weight1, &shared_weight2, &shared_weight_scales1, &shared_weight_scales2})
        if (t->has_value()) record(t->value());
    if (urma)
        urma_->for_each_tensor(record);
    else {
        record(workspace_);
        record(context_);
    }
    return {y, expert_tokens};
}

torch::Tensor allocate_timer(torch::Tensor example, const std::string& backend)
{
#ifndef DEEPEP_MEGAMOE_TIMER
    TORCH_CHECK(false, "Rebuild with DEEPEP_MEGAMOE_TIMER=ON before allocating timer");
#endif
    CheckTensor(example, 2, "timer example");
    TORCH_CHECK(backend == "mte" || backend == "urma" || backend == "mte_roce",
                "MegaMoE backend must be mte, urma or mte_roce");
    int64_t elements = AscendTimer::TOTAL_BUFFER_SIZE;
    if (backend == "mte_roce") {
        elements = HierarchicalAscendTimer::TOTAL_BUFFER_SIZE;
    } else if (backend == "urma") {
        elements = int64_t{MegaMoeUrma::kTimerCores} *
                       (MegaMoeUrma::kTraceHeaderElements +
                        int64_t{MegaMoeUrma::kTraceMaxCapacity} * MegaMoeUrma::kTraceRecordElements) +
                   MegaMoeUrma::kTimerElements;
    }
    return torch::zeros({elements}, example.options().dtype(torch::kInt64));
}

void validate_timer(const torch::Tensor& timer, const std::string& backend)
{
    if (backend == "mte_roce") {
#ifndef DEEPEP_MEGAMOE_TIMER
        TORCH_CHECK(false, "Rebuild with DEEPEP_MEGAMOE_TIMER=ON before passing timer");
#endif
        CheckTensor(timer, 1, "timer");
        TORCH_CHECK(timer.scalar_type() == torch::kInt64 &&
                        timer.numel() == HierarchicalAscendTimer::TOTAL_BUFFER_SIZE,
                    "Invalid MTE+RoCE timer allocation");
        return;
    }
    TORCH_CHECK(backend == "mte" || backend == "urma", "MegaMoE backend must be mte, urma or mte_roce");
    ValidateTimer(timer, backend == "urma");
}

std::tuple<int64_t, int64_t> urma_window_bytes(int64_t world, int64_t experts, int64_t tokens, int64_t topk,
                                               int64_t hidden, int64_t intermediate, const std::string& mma_type)
{
    TORCH_CHECK(mma_type == "fp8xfp4" || (mma_type == "fp8xfp8" && (world == 16 || world == 32)),
                "URMA window supports routed FP8xFP4 or EP16/EP32 FP8xFP8");
    const auto layout = MakeUrmaSymmetricLayout(world, tokens, topk, hidden, UrmaCvRows(world, tokens), 1, true);
    TORCH_CHECK(experts >= 1 && experts <= 128 && topk <= world * experts, "Invalid URMA local expert count");
    TORCH_CHECK(intermediate >= 256 && intermediate <= 16384 && intermediate % 256 == 0,
                "Invalid URMA intermediate width");
    MegaMoeUrma::Arguments args{};
    args.config = layout.config;
    args.config.localExperts = experts;
    args.config.windowedTransport = 1U;
    args.config.intermediate = intermediate;
    args.config.fp8Weights = mma_type == "fp8xfp8";
    uint64_t scratch = 0;
    BindUrmaScratch(args, layout, tokens, GetPlatformInfo().system_workspace, [&](uint64_t bytes) {
        scratch += bytes;
        return uint64_t{0};
    });
    return {layout.config.symmetricBytes, scratch};
}

int64_t mte_plan_bytes(int64_t world, int64_t experts, int64_t tokens, int64_t capacity, int64_t topk,
                       int64_t hidden, int64_t intermediate, const std::string& activation, const std::string& mma_type,
                       int64_t shared_experts, int64_t topk_weights_type)
{
    TORCH_CHECK(tokens > 0 && tokens <= capacity, "MTE plan token count outside capacity");
    TORCH_CHECK(intermediate >= 256 && intermediate <= 4096 && intermediate % 128 == 0,
                "Plan intermediate_hidden must be divisible by 128, in [256,4096]");
    TORCH_CHECK(activation == "gelu" || activation == "swiglu", "activation must be 'swiglu' or 'gelu'");
    const bool gelu = activation == "gelu";
    const bool w4 = mma_type == "fp8xfp4";
    const int64_t weight_mode = topk_weights_type == -1 ? static_cast<int64_t>(gelu) : topk_weights_type;
    TORCH_CHECK(weight_mode == 0 || weight_mode == 1, "topk_weights_type must be 0 or 1");
    TORCH_CHECK(shared_experts >= 0 && shared_experts <= 8 && (w4 || shared_experts == 0),
                "Plan shared experts require W4A8, with count in [0,8]");
    // Validate routed dimensions and MMA mode through the same entry point as Plan.
    buffer_size(world, experts, capacity, topk, hidden, "mte", weight_mode, mma_type, false);
    const auto platform = GetPlatformInfo();
    auto tiling = MakeTiling(tokens, hidden, intermediate * (gelu ? 1 : 2), topk, experts, world, capacity,
                             0, w4, shared_experts, platform, weight_mode, activation);
    if (gelu && w4 && shared_experts == 0) {
        // A different tail can pass the UB guard even when the current shape misses it.
        // Budget the largest calibrated table for any tail at this capacity.
        tiling.geluLutWords = static_cast<uint16_t>(MegaMoeImpl::MTE_GELU_LUT_WORD_CAPACITY);
    }
    return MteWorkspaceAllocationBytes(tiling, platform) + sizeof(Mc2Aclnn::Mc2MoeContext);
}

int64_t buffer_size(int64_t world, int64_t experts, int64_t tokens, int64_t topk, int64_t hidden,
                    const std::string& backend, int64_t topk_weights_type, const std::string& mma_type, bool windowed)
{
    TORCH_CHECK(topk_weights_type == 0 || topk_weights_type == 1, "topk_weights_type must be 0 or 1");
    TORCH_CHECK(backend == "mte" || backend == "urma", "MegaMoE backend must be mte or urma");
    TORCH_CHECK(!windowed || backend == "urma", "Windowed native MegaMoE plans require URMA");
    const std::string resolved_mma_type = mma_type.empty() ? "fp8xfp8" : mma_type;
    TORCH_CHECK(resolved_mma_type == "fp8xfp4" || resolved_mma_type == "fp8xfp8",
                "mma_type must be fp8xfp4 or fp8xfp8");
    if (backend == "urma") {
        TORCH_CHECK(resolved_mma_type == "fp8xfp4" || (mma_type == "fp8xfp8" && windowed && (world == 16 || world == 32)),
                    "URMA requires routed FP8xFP4 or windowed EP16/EP32 FP8xFP8");
        const auto layout = MakeUrmaSymmetricLayout(world, tokens, topk, hidden, 128, topk_weights_type, windowed);
        TORCH_CHECK(experts >= 1 && experts <= 128 && topk <= world * experts, "Invalid URMA expert count");
        return layout.config.symmetricBytes;
    }
    TORCH_CHECK(world >= 2 && world <= 128 && experts >= 1 && experts <= 128 && tokens >= 1 && tokens <= 65536 &&
                    topk >= 1 && topk <= 32 && topk <= world * experts && hidden >= 1024 && hidden <= 8192 &&
                    hidden % 64 == 0,
                "Invalid buffer shape");
    // The buffer outlives BS/weight-mode changes. Reserve the decode rail/count
    // layout as well; prefill continues to bind its original device offsets.
    const MegaMoeImpl::PeermemSizeParams params{tokens,
                                                topk,
                                                hidden,
                                                experts,
                                                world,
                                                2,
                                                topk_weights_type == 1,
                                                false,
                                                MegaMoeImpl::TOPO_TYPE_MTE,
                                                true};
    return AlignUp<int64_t>(MegaMoeImpl::EXCEPTION_DUMP_REGION_SIZE + MegaMoeImpl::CalcPeermemLeastSize(params),
                            2LL << 20);
}

}  // namespace deepep::megamoe
