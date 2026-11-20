// Copyright (c) 2026, Lu Lu
// Modified by SimpleBrightMan 2026

#pragma once

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>

#include "kernels/mega_moe_urma/common/mega_moe_route_plan.hpp"

namespace deepep::megamoe {

// Host-only. Both the preflight estimate and the actual allocations use these layouts.
// Input limits are checked before any dimension product. Even the full-batch legacy
// shape has at most 2^21 receiver rows and H,I <= 2^14, so products fit uint64_t.
inline uint64_t UrmaAlign(uint64_t bytes, uint64_t alignment = 512)
{
    if (bytes > std::numeric_limits<uint64_t>::max() - (alignment - 1))
        throw std::overflow_error("URMA allocation alignment overflow");
    return (bytes + alignment - 1) / alignment * alignment;
}

struct UrmaSymmetricLayout {
    MegaMoeUrma::Config config{};
    uint64_t expandedX = 0, expandedScale = 0, gmm2Output = 0;
};

inline uint32_t UrmaCvRows(int64_t world, int64_t bs)
{
    return world == 16 && bs <= 2560 ? 128U : 256U;
}

inline UrmaSymmetricLayout MakeUrmaSymmetricLayout(int64_t world, int64_t tokens, int64_t topk, int64_t hidden,
                                                   uint32_t cvRows, int64_t weightMode, bool windowed)
{
    using namespace MegaMoeUrma;
    if (hidden < 256 || hidden > 8192 || hidden % 256 || (cvRows != 128 && cvRows != 256) ||
        (weightMode != 0 && weightMode != 1))
        throw std::invalid_argument("Invalid URMA hidden dimension, CV height or route weight mode");
    if (windowed) {
        if ((world != 8 && world != 16 && world != 32) || tokens < 1 || tokens > kWindowTokens || topk < 1 ||
            topk > kWindowMaxTopk || weightMode != 1)
            throw std::invalid_argument("Windowed URMA requires EP8/16/32, W1..256, K1..32 and early weights");
    } else if ((world != 4 && world != 8 && world != 16) || tokens < 1 || tokens > kMaxTokensPerRank || topk < 1 ||
               topk > 8) {
        throw std::invalid_argument("Full-batch URMA requires EP4/8/16, B1..16384 and K1..8");
    }
    UrmaSymmetricLayout layout{};
    auto& c = layout.config;
    c.world = world;
    c.hidden = hidden;
    c.topk = topk;
    c.cvRows = cvRows;
    c.topkWeightsType = weightMode;
    c.dispatchScaleStride = weightMode ? UrmaAlign(hidden / 32, 32) + 32 : hidden / 32;
    c.sourceRowCapacity = uint64_t(tokens) * topk;
    c.maxPeerChunks = (c.sourceRowCapacity + cvRows - 1) / cvRows;
    const uint64_t received = world * c.sourceRowCapacity;
    // Reserve for both CV heights so tails and legacy shape changes share all peer offsets.
    const uint64_t flags = world * ((c.sourceRowCapacity + 127) / 128) * kFlagBytes;
    uint64_t bytes = 0;
    auto region = [&](uint64_t size) {
        const uint64_t start = bytes;
        bytes += UrmaAlign(size);
        return start;
    };
    c.receiveXOffset = region(received * hidden);
    c.receiveScaleOffset = region(received * c.dispatchScaleStride);
    c.returnYOffset = region(c.sourceRowCapacity * hidden * 2);
    c.dispatchReadyOffset = region(flags);
    c.combineReadyOffset = region(flags);
    c.barrierOffset = region(4ULL * world * kFlagBytes);
    c.routeInboxOffset = region(world * kRouteMailboxBytes);
    c.routeEntryOffset = region(world * kFlagBytes);
    c.routeReadyOffset = region(world * kFlagBytes);
    if (world == 32) {
        // Inter-host RDMA reads its local source through the registered symmetric MR.
        c.routeOutboxOffset = region(world * kRouteMailboxBytes);
        c.epochSourceOffset = region(kFlagBytes);
        layout.expandedX = region(c.sourceRowCapacity * hidden);
        layout.expandedScale = region(c.sourceRowCapacity * c.dispatchScaleStride);
        layout.gmm2Output = region(received * hidden * 2);
    }
    c.symmetricBytes = UrmaAlign(bytes, 2ULL << 20);
    return layout;
}

inline bool ReserveUrmaGeluLookup(const MegaMoeUrma::Arguments& a)
{
    const auto& c = a.config;
    // The public preflight has no activation argument. Reserve the same local
    // region for this geometry in the estimate and the Plan. Only GELU fills it.
    // H4096 fixes the scale-gather bound in WindowGeluUb. Other shapes keep
    // their original cooperative schedule until evaluated on the target topology.
    return c.windowedTransport && !a.sharedExperts && c.topkWeightsType == 1U &&
           c.hidden == 4096U && c.intermediate == 1536U && (c.topk == 25U || c.topk == 32U) &&
           c.world * c.localExperts == 800U;
}

// Allocate receives each actual local buffer's byte count and returns its device
// address. A counting visitor can instead return zero without creating any tensors.
template <typename Allocate>
inline void BindUrmaScratch(MegaMoeUrma::Arguments& a, const UrmaSymmetricLayout& layout, uint32_t maxTokens,
                            uint64_t systemWorkspace, Allocate allocate)
{
    using namespace MegaMoeUrma;
    const auto& c = a.config;
    const uint64_t sent = c.sourceRowCapacity, received = sent * c.world;
    const uint64_t groups = RouteGroupCapacity(received, c.localExperts);
    const uint64_t chunks = uint64_t(c.world) * c.maxPeerChunks;
    const uint64_t experts = uint64_t(c.localExperts) * c.world;
    // Include the minimum tensor allocation and the device allocation granularity.
    auto region = [&](uint64_t bytes) { return allocate(UrmaAlign(std::max<uint64_t>(bytes, 1))); };
    if (a.sharedExperts) {
        const uint64_t si = a.sharedIntermediate, sharedGroups = (maxTokens + 255U) / 256U;
        const auto gelu = static_cast<uint32_t>(DEEPEP_MEGAMOE_IMPL_NAMESPACE::MegaMoeActMode::GELU);
        a.sharedGmm1 = region(maxTokens * si * (c.activationMode == gelu ? 2 : 4));
        a.sharedActivation = region(maxTokens * si);
        a.sharedActivationScale = region(maxTokens * (si / 32));
        a.sharedOutput = region(uint64_t(maxTokens) * c.hidden * 2);
        a.sharedRoutedOutput = region(uint64_t(maxTokens) * c.hidden * 4);
        a.sharedGmm1Done = region(sharedGroups * (si / 256) * kFlagBytes);
        a.sharedActivationDone = region(sharedGroups * kFlagBytes);
    }
    if (c.world == 32) {
        a.expandedX = a.symmetric + layout.expandedX;
        a.expandedScale = a.symmetric + layout.expandedScale;
        a.gmm2Output = a.symmetric + layout.gmm2Output;
    } else {
        a.expandedX = region(sent * c.hidden);
        a.expandedScale = region(sent * c.dispatchScaleStride);
        a.gmm2Output = region(received * c.hidden * 2);
    }
    a.gatheredInputScale = region(received * (c.hidden / 32));
    a.gatheredRouteWeights = region(c.topkWeightsType ? received * 32U : 0U);
    a.activation = region(received * c.intermediate);
    a.activationScale = region(received * (c.intermediate / 32));
    a.computeBatches = region(groups * sizeof(ComputeBatch));
    a.computeTasks = region(2 * groups * sizeof(ComputeBatchTask));
    a.packReady = region(chunks * kFlagBytes);
    a.inputScaleDone = region(groups * kFlagBytes);
    a.activationDone = region(groups * kFlagBytes);
    a.gmm2Done = region(groups * kFlagBytes);
    a.reduceIndex = region(sent * sizeof(int64_t));
    a.expertRoutes = region(experts * sent * sizeof(int32_t));
    a.routeCounts = region(experts * sizeof(int64_t));
    a.routeOffsets = region((experts + 1) * sizeof(int64_t));
    a.routeHistogram = region(experts * sizeof(uint32_t));
    a.routeInfo = region(sizeof(DeviceRouteInfo));
    a.routeConfig = region(sizeof(Config));
    a.localFlags = region((a.sharedExperts ? kSharedLocalFlagCount : 16U) * kFlagBytes);
    a.gmmIssued = region(kAic * kFlagBytes);
    a.systemWorkspace = region(systemWorkspace + (2ULL << 20));
    if (ReserveUrmaGeluLookup(a)) a.geluLookup = region(WindowGeluUb::tableBytes);
    // Preserve valid backing addresses for reserved fields in the kernel argument ABI.
    a.gmm1Output = region(0);
    a.fragments = region(0);
    a.segments = region(0);
    a.chunkGroupIndices = region(0);
    a.sendChunks = region(0);
    a.recvChunks = region(0);
    a.packRounds = region(0);
    a.gmm1Done = region(0);
    a.order = region(0);
    a.reduceDependencies = region(0);
    a.gmm2Contiguous = region(0);
    a.peerReorderRanges = region(0);
    a.peerReorderSpans = region(0);
}

}  // namespace deepep::megamoe
