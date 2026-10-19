// Copyright (c) 2026, Lu Lu
// Modified by SimpleBright_Man 2026
#pragma once
#include <cstdint>

#if defined(__NPU_DEVICE__)
#define MEGAMOE_A8W8_POLICY_INLINE __aicore__ inline
#else
#define MEGAMOE_A8W8_POLICY_INLINE inline
#endif

namespace MegaMoeImpl {
// Preserve the M256 dispatch/scheduling group, but stage planar gate/up in
// M128 pieces. Two M256 N256 BF16 inputs would exceed Ascend950's 248 KiB UB.
constexpr uint32_t A8W8_GMM1_UB_TILE_M = 128U;
constexpr uint32_t A8W8_GMM1_UB_TILE_N = 256U;
constexpr uint32_t A8W8_GMM1_UB_INPUT_BYTES = 2U * A8W8_GMM1_UB_TILE_M * A8W8_GMM1_UB_TILE_N * sizeof(uint16_t);
static_assert(A8W8_GMM1_UB_INPUT_BYTES <= 248U * 1024U, "A8W8 gate/up inputs exceed available UB");

MEGAMOE_A8W8_POLICY_INLINE constexpr uint32_t Gmm1UbSubtileRows(uint32_t logicalRows, uint32_t rowOffset,
                                                                uint32_t ubTileRows)
{
    if (rowOffset >= logicalRows) {
        return 0U;
    }
    const uint32_t remaining = logicalRows - rowOffset;
    return remaining < ubTileRows ? remaining : ubTileRows;
}

MEGAMOE_A8W8_POLICY_INLINE constexpr bool IsLastGmm1UbSubtile(uint32_t logicalRows, uint32_t rowOffset,
                                                              uint32_t subtileRows)
{
    return subtileRows != 0U && rowOffset < logicalRows && subtileRows == logicalRows - rowOffset;
}
}  // namespace MegaMoeImpl
#undef MEGAMOE_A8W8_POLICY_INLINE
