// Copyright (c) 2026, Lu Lu
#pragma once

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

#include "host_platform.hpp"
#include "roce_arena.hpp"

namespace deepep::megamoe::roce {

using ::deepep::megamoe::PlatformInfo;

inline void CheckTilingCondition(bool valid, const char* message)
{
    if (!valid) throw std::invalid_argument(message);
}

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

inline uint8_t MmaMode(const std::string& mma_type)
{
    using namespace HierarchicalMegaMoeImpl;
    if (mma_type == "fp8xfp4") return GROUPED_MATMUL_MODE_A8W4;
    if (mma_type == "fp8xfp8") return GROUPED_MATMUL_MODE_GENERAL;
    if (mma_type == "fp4xfp4") return GROUPED_MATMUL_MODE_A4W4_NZ;
    throw std::invalid_argument("MTE+RoCE mma_type must be fp8xfp4, fp8xfp8, or fp4xfp4");
}

inline bool IsGelu(const HierarchicalMegaMoeTilingData& t)
{
    return t.actMode == static_cast<uint8_t>(HierarchicalMegaMoeImpl::MegaMoeActMode::GELU);
}

inline uint32_t IntermediateWidth(const HierarchicalMegaMoeTilingData& t)
{
    return t.hiddenDim / HierarchicalMegaMoeImpl::ActivationFactor(t.actMode);
}

inline uint32_t QuantizedRecordBytes(const HierarchicalMegaMoeTilingData& t)
{
    using namespace HierarchicalMegaMoeImpl;
    return static_cast<uint32_t>(CalcQuantTokenScaleBytes(
        t.h, DispatchElementsPerByte(t.groupedMatmulMode), t.topK, t.topkWeightsPrefetch != 0));
}

inline uint32_t DispatchCopyBytes(const HierarchicalMegaMoeTilingData& t)
{
    using namespace HierarchicalMegaMoeImpl;
    return HierarchicalDispatchPacketCount(&t) * DISPATCH_PACKET_DATA;
}

inline HierarchicalMegaMoeDispatchBufferConfig DispatchConfig(const HierarchicalMegaMoeTilingData& t, uint32_t ub)
{
    HierarchicalMegaMoeDispatchBufferConfig c{};
    // Compact-route receive stores at most one local-expert index per source
    // token.  The old mask route used numMaxTokens * topK here.
    const uint64_t total = static_cast<uint64_t>(t.numMaxTokensPerRank);
    const uint64_t aligned = AlignUp(total, uint64_t{256});
    c.copyBufferBytes = DispatchCopyBytes(t);
    CheckTilingCondition(!HierarchicalMegaMoeImpl::UseSharedRouteTable(&t) ||
                    c.copyBufferBytes >= HierarchicalMegaMoeImpl::CalcRailTopkIdsSlotDataBytes(t.bs, t.topK) + 256,
                "Insufficient dispatch UB for the per-source route histogram");
    const uint32_t fixed =
        AlignUp(t.epWorldSize * t.moeExpertPerRank * 4U, 32U) + AlignUp(t.moeExpertPerRank * 4U, 32U);
    c.routeItemsPerBatch = static_cast<int32_t>(std::min<uint64_t>(aligned, BASE_RECV_ROUTE_ITEMS_PER_BATCH));
    c.routeBatchCount = static_cast<int32_t>(CeilDiv(total, static_cast<uint64_t>(c.routeItemsPerBatch)));
    const uint32_t route = static_cast<uint32_t>(c.routeItemsPerBatch);
    const uint32_t without_slots = fixed + route * sizeof(int32_t);
    const uint32_t budget = ub > without_slots ? ub - without_slots : 0U;
    const uint32_t buffer_bytes = c.copyBufferBytes + HierarchicalMegaMoeImpl::INT32_PER_256B * sizeof(int32_t);
    CheckTilingCondition(budget >= MIN_DISPATCH_BUFFER_COUNT * buffer_bytes, "Insufficient UB for Dispatch copy buffers");
    c.bufferCount = std::min(static_cast<int32_t>(budget / buffer_bytes), MAX_DISPATCH_BUFFER_COUNT);
    if (static_cast<uint64_t>(c.routeItemsPerBatch) < total) {
        const uint32_t fixed_with_slots = fixed + static_cast<uint32_t>(c.bufferCount) * buffer_bytes;
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

inline HierarchicalMegaMoeSendMaskBufferConfig SendMaskConfig(const HierarchicalMegaMoeTilingData& t, uint32_t fixed, uint32_t owned_experts,
                                           uint32_t ub)
{
    HierarchicalMegaMoeSendMaskBufferConfig c{};
    const uint64_t total = static_cast<uint64_t>(t.numMaxTokensPerRank) * t.topK;
    const bool rail = HierarchicalMegaMoeImpl::UseDecodeRailRouting(&t);
    const uint64_t storage =
        rail ? CeilDiv<uint64_t>(total, HierarchicalMegaMoeImpl::RAIL_PACKET_DATA_WORDS) * HierarchicalMegaMoeImpl::RAIL_PACKET_WORDS : total;
    const uint64_t aligned = AlignUp(storage, uint64_t{256});
    const auto payload_capacity = [rail](uint64_t slots) {
        return rail ? slots / HierarchicalMegaMoeImpl::RAIL_PACKET_WORDS * HierarchicalMegaMoeImpl::RAIL_PACKET_DATA_WORDS : slots;
    };
    c.routeItemsPerBatch = static_cast<int32_t>(std::min<uint64_t>(aligned, BASE_SEND_ROUTE_ITEMS_PER_BATCH));
    c.routeBatchCount = static_cast<int32_t>(CeilDiv(total, payload_capacity(c.routeItemsPerBatch)));
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
    if (payload_capacity(c.routeItemsPerBatch) < total) {
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
            c.routeBatchCount = static_cast<int32_t>(CeilDiv(total, payload_capacity(expanded)));
            c.bufferBytes = route_slot_bytes(static_cast<uint32_t>(expanded));
        }
    }
    if (HierarchicalMegaMoeImpl::UseRouteAllGather(&t)) {
        const uint32_t words =
            CeilDiv(t.bs * t.topK, HierarchicalMegaMoeImpl::RAIL_PACKET_DATA_WORDS) * HierarchicalMegaMoeImpl::RAIL_PACKET_WORDS;
        c.routeItemsPerBatch = static_cast<int32_t>(AlignUp(words, 256U));
        c.routeBatchCount = 1;
        c.bufferCount = MIN_SEND_MASK_BUFFER_COUNT;
        const uint32_t indices = AlignUp(t.bs, 32U) * t.moeExpertPerRank * sizeof(int32_t);
        c.bufferBytes = std::max(route_slot_bytes(c.routeItemsPerBatch),
                                 AlignUp(CeilDiv(indices, static_cast<uint32_t>(c.bufferCount)), 32U));
    }
    CheckTilingCondition(fixed + 2ULL * c.routeItemsPerBatch * sizeof(int32_t) +
                    static_cast<uint64_t>(c.bufferCount) * c.bufferBytes <= ub,
                "Insufficient UB for route packing and relay");
    return c;
}

inline void SetSendMaskConfigs(HierarchicalMegaMoeTilingData& t, uint32_t ub)
{
    using namespace HierarchicalMegaMoeImpl;
    const WorkspaceInfo workspace(nullptr, &t);
    const uint64_t flag_elements = std::max(workspace.flagResetElementCount, workspace.gmm1TileStatusElementCount);
    const uint32_t reset_per_core =
        static_cast<uint32_t>(CeilDiv(flag_elements, static_cast<uint64_t>(t.blockAivNum)));
    const uint32_t reset_count = std::min(reset_per_core, static_cast<uint32_t>(DISPATCH_RESET_BATCH));
    const uint32_t reset_bytes = AlignUp(reset_count, uint32_t{HierarchicalMegaMoeImpl::INT32_PER_256B}) * 4U;
    const uint32_t quant_out = QuantizedRecordBytes(t);
    const uint32_t quant_in = AlignUp(t.h, 128U) * 2U;
    const auto sharedPolicy = HierarchicalMegaMoeImpl::SelectSharedExpertPolicy(
        t.groupedMatmulMode == HierarchicalMegaMoeImpl::GROUPED_MATMUL_MODE_A8W4, t.sharedExpertNum, t.bs);
    const uint32_t route_cores = sharedPolicy.routeAiv1Only ? t.aicNum : t.blockAivNum;
    const uint32_t max_experts = HierarchicalMegaMoeImpl::CalcSendMaskExpertCapacity(&t, route_cores);
    const uint32_t accumulator = AlignUp(max_experts * 4U, 32U);
    const uint32_t fixed = reset_bytes + 2048U + 2U * quant_out + 2U * quant_in + accumulator;
    const uint32_t experts = t.epWorldSize * t.moeExpertPerRank;
    const uint32_t base = experts / route_cores;
    t.sendMaskCoreCountWithExtraExpert = experts % route_cores;
    const bool rail = HierarchicalMegaMoeImpl::UseDecodeRailRouting(&t);
    t.sendMaskConfigForCoreWithExtraExpert = SendMaskConfig(t, fixed, rail ? max_experts : base + 1U, ub);
    t.sendMaskConfigForCoreWithoutExtraExpert = SendMaskConfig(t, fixed, rail ? max_experts : base, ub);
}

inline HierarchicalMegaMoeUnpermuteBufferConfig UnpermuteConfig(const HierarchicalMegaMoeTilingData& t, uint32_t core_tokens, uint32_t ub)
{
    HierarchicalMegaMoeUnpermuteBufferConfig c{};
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
    CheckTilingCondition(input_budget >= MIN_UNPERMUTE_INPUT_BUFFER_COUNT * slot_bytes,
                "Insufficient UB for Unpermute reduction buffers");
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

inline void SetUnpermuteConfigs(HierarchicalMegaMoeTilingData& t, uint32_t ub)
{
    const uint32_t full = CeilDiv(t.bs, HierarchicalMegaMoeImpl::UnpermuteJobCount(&t));
    const uint32_t active = CeilDiv(t.bs, full);
    const uint32_t tail = t.bs - (active - 1U) * full;
    const bool tail_full = tail == full;
    t.unpermuteFullTokenChunkCoreCount = tail_full ? active : active - 1U;
    t.unpermuteConfigForFullTokenChunk = UnpermuteConfig(t, full, ub);
    t.unpermuteConfigForTailTokenChunk = tail_full ? HierarchicalMegaMoeUnpermuteBufferConfig{} : UnpermuteConfig(t, tail, ub);
}

inline void CheckArenaGeometry(int64_t world, int64_t experts, int64_t capacity, int64_t topk, int64_t hidden,
                               int64_t topk_weights_type)
{
    using namespace HierarchicalMegaMoeImpl;
    CheckTilingCondition(
        world == HIERARCHICAL_WORLD_SIZE && experts >= 1 && experts <= 128 &&
            capacity >= 1 && capacity <= HIERARCHICAL_MAX_TOKENS && topk >= 1 && topk <= 32 &&
            topk <= world * experts && hidden >= 1024 && hidden <= 8192 && hidden % 64 == 0,
        "MTE+RoCE requires EP32, 1 to 128 experts per rank, K in [1,32], H in [1024,8192] divisible by 64, "
        "and capacity in [1,256]");
    CheckTilingCondition(topk_weights_type == 0 || topk_weights_type == 1,
                         "MTE+RoCE topk_weights_type must be 0 (late) or 1 (early)");
}

inline bool FusesGateUp(const HierarchicalMegaMoeTilingData& t)
{
    return !IsGelu(t) && t.groupedMatmulMode == HierarchicalMegaMoeImpl::GROUPED_MATMUL_MODE_A8W4;
}

inline uint32_t Gmm1TilesPerMGroup(const HierarchicalMegaMoeTilingData& t)
{
    using namespace HierarchicalMegaMoeImpl;
    return CeilDiv(IntermediateWidth(t), FusesGateUp(t) ? GMM1_FUSED_TILE_N : L1_TILE_N);
}

inline void SetWaveSize(HierarchicalMegaMoeTilingData& t)
{
    using namespace HierarchicalMegaMoeImpl;
    const auto first_tiles = Gmm1TilesPerMGroup(t);
    if (FusesGateUp(t)) {
        t.mGroupsPerWave = CeilDiv(t.aicNum * (t.sharedFp8 ? 3U : 2U), first_tiles);
    } else {
        const auto second_tiles = CeilDiv(t.h, L1_TILE_N);
        t.mGroupsPerWave = std::max(CeilDiv(t.aicNum * 4U, first_tiles), CeilDiv(t.aicNum, second_tiles));
    }
}

inline HierarchicalMegaMoeTilingData MakeTiling(int64_t bs, int64_t h, int64_t intermediate, int64_t topk,
                                              int64_t local_experts, int64_t world, int64_t capacity,
                                              int64_t shared_experts, const PlatformInfo& p,
                                              const std::string& activation, uint8_t mma_mode, int64_t topk_weights_type)
{
    using namespace HierarchicalMegaMoeImpl;
    CheckArenaGeometry(world, local_experts, capacity, topk, h, topk_weights_type);
    CheckTilingCondition(bs > 0 && bs <= capacity, "MTE+RoCE Plan BS must be in [1, capacity]");
    CheckTilingCondition(intermediate >= 256 && intermediate <= 4096 && intermediate % 128 == 0,
                         "MTE+RoCE intermediate_hidden must be divisible by 128 and in [256,4096]");
    CheckTilingCondition(activation == "gelu" || activation == "swiglu", "MTE+RoCE activation must be gelu or swiglu");
    CheckTilingCondition(shared_experts >= 0 && shared_experts <= 8, "MTE+RoCE shared expert count must be in [0,8]");
    CheckTilingCondition(shared_experts == 0 || (mma_mode == GROUPED_MATMUL_MODE_A8W4 && activation == "swiglu"),
                         "MTE+RoCE shared experts require SwiGLU W4A8");

    HierarchicalMegaMoeTilingData t{};
    t.moeExpertPerRank = static_cast<uint32_t>(local_experts);
    t.bs = static_cast<uint32_t>(bs);
    t.h = static_cast<uint32_t>(h);
    t.hiddenDim = static_cast<uint32_t>(activation == "gelu" ? intermediate : 2 * intermediate);
    t.epWorldSize = static_cast<uint32_t>(world);
    t.blockNumPerEP = std::max(1U, p.aic / t.epWorldSize);
    t.numMaxTokensPerRank = static_cast<uint32_t>(capacity);
    t.topK = static_cast<uint32_t>(topk);
    t.maxOutputSize = t.numMaxTokensPerRank * t.epWorldSize * std::min(t.topK, t.moeExpertPerRank);
    t.aicNum = p.aic;
    t.blockAivNum = p.aiv;
    t.combineQuantMode = COMBINE_NO_QUANT;
    t.clampLimit = std::numeric_limits<float>::max();
    t.groupedMatmulMode = mma_mode;
    t.dispatchRanksPerNode = HIERARCHICAL_RANKS_PER_HOST;
    t.topoType = TOPO_TYPE_MTE_ROCE;
    CheckTilingCondition(p.aic >= HIERARCHICAL_DISPATCH_WORKERS && p.aiv == 2U * p.aic,
                         "MTE+RoCE requires at least 14 AICs and two AIVs per AIC");
    CheckTilingCondition(p.ub > HIERARCHICAL_ROCE_UB_BYTES, "Not enough UB for MegaMoE RoCE scratch");
    const uint32_t stage_ub = p.ub - HIERARCHICAL_ROCE_UB_BYTES;
    CheckTilingCondition(stage_ub >= CalcGatewayCombineReduceUbBytes(&t),
                         "Not enough UB for gateway readiness and FP32 reduction");
    t.roceScratchUbOffset = stage_ub;
    t.sharedExpertNum = static_cast<uint32_t>(shared_experts);
    t.combineSyncSlotCountPerExpert = 0;
    t.topkWeightsPrefetch = static_cast<int32_t>(topk_weights_type);
    t.actMode = static_cast<uint8_t>(activation == "gelu" ? MegaMoeActMode::GELU : MegaMoeActMode::SWIGLU);
    // Unique top-k IDs bound one expert by capacity times EP, even if all tokens select it.
    t.maxTilesPerExpert = topk_weights_type == 1
                             ? CeilDiv(t.numMaxTokensPerRank * t.epWorldSize, L1_TILE_M_256) * Gmm1TilesPerMGroup(t)
                             : 0U;
    t.actSubMode = static_cast<uint8_t>(MegaMoeActSubMode::DEFAULT);
    t.activationAlpha = 1.0F;
    t.activationBeta = 1.0F;
    SetWaveSize(t);
    t.isPerExpertWeightTensor = false;
    t.dispatchBufferConfig = DispatchConfig(t, stage_ub);
    SetSendMaskConfigs(t, stage_ub);
    SetUnpermuteConfigs(t, stage_ub);
    return t;
}

inline int64_t WorkspaceBytes(const HierarchicalMegaMoeTilingData& t, const PlatformInfo& platform)
{
    const HierarchicalMegaMoeImpl::WorkspaceInfo info(nullptr, &t);
    return info.workspaceSize + static_cast<int64_t>(platform.system_workspace) + kReservedWorkspaceBytes;
}

}  // namespace deepep::megamoe::roce
