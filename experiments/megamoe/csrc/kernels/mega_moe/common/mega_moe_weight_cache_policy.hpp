// Copyright (c) 2026, Lu Lu
// Modified by SimpleBright_Man 2026
#pragma once
#include <cstdint>

#if defined(__NPU_DEVICE__)
#define MEGAMOE_WEIGHT_CACHE_INLINE __aicore__ inline
#else
#define MEGAMOE_WEIGHT_CACHE_INLINE inline
#endif

namespace MegaMoeImpl {
// ScaleBDN stores [N, ceil(K/64), 2], with K-scale pairs contiguous.
// Each N row must start on a cache line before fragmented K-window reads bypass L2.
// N256 tile offsets preserve this alignment, including a final partial N tile.
MEGAMOE_WEIGHT_CACHE_INLINE constexpr bool CanBypassScaleBDNL2(bool coversWholeExpert, uint32_t rows, uint32_t tileRows,
                                                               uint32_t scaleK, uint32_t scaleElementBytes)
{
    constexpr uint64_t cacheLineBytes = 128U;
    const uint64_t rowStrideBytes = static_cast<uint64_t>(scaleK) * scaleElementBytes;
    return coversWholeExpert && rows <= tileRows && rowStrideBytes != 0U && rowStrideBytes % cacheLineBytes == 0U;
}
}  // namespace MegaMoeImpl
#undef MEGAMOE_WEIGHT_CACHE_INLINE
