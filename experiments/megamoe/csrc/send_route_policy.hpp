// Copyright (c) 2026, Lu Lu
// Modified by SimpleBright_Man 2026

#pragma once

#include <cstdint>

namespace deepep::megamoe {

// Compact routing keeps both a match bitset and one int32 index for every
// covered token. A batch may start partway through a token's top-k segment.
inline constexpr uint64_t SendRouteSlotBytes(uint32_t route_items, uint32_t topk)
{
    const uint64_t covered_tokens = (static_cast<uint64_t>(route_items) + 2ULL * topk - 2U) / topk;
    const uint64_t index_bytes = (covered_tokens * sizeof(int32_t) + 31U) / 32U * 32U;
    return route_items / 8U + index_bytes;
}

inline constexpr uint64_t SendRouteUbBytes(uint32_t route_items, uint32_t topk, uint32_t fixed_bytes,
                                           uint32_t buffer_count)
{
    return static_cast<uint64_t>(fixed_bytes) + 2ULL * route_items * sizeof(int32_t) +
           static_cast<uint64_t>(buffer_count) * SendRouteSlotBytes(route_items, topk);
}

// Find the largest 256-item batch that fits the actual UB budget. In particular,
// the 12288-item baseline is not a safe minimum after compact indices were added
// to every ring slot: topk=1 can require a smaller batch even with two slots.
// Zero means that the caller must reject the configuration before launching.
inline constexpr uint32_t FitSendRouteBatch(uint32_t max_items, uint32_t topk, uint32_t fixed_bytes,
                                            uint32_t buffer_count, uint32_t ub_bytes)
{
    if (topk == 0U || buffer_count == 0U) return 0U;
    uint32_t low = 0U;
    uint32_t high = max_items / 256U;
    while (low < high) {
        const uint32_t middle = low + (high - low + 1U) / 2U;
        if (SendRouteUbBytes(middle * 256U, topk, fixed_bytes, buffer_count) <= ub_bytes) {
            low = middle;
        } else {
            high = middle - 1U;
        }
    }
    return low * 256U;
}

}  // namespace deepep::megamoe
