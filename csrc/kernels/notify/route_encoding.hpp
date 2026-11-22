// Copyright (c) 2026, Lu Lu

#pragma once

#include <cstdint>
#include "../ht_limits.h"

namespace ascend_deepep {

// Keep the rank and the low four bits of each slot at their existing wire
// offsets. Bits 17 and 18 carry the two extra slot bits for TopK32.
inline constexpr uint32_t kNotifyWireRankMask = 0x1ffU;
inline constexpr uint32_t kNotifyWireSlotMask = 0xfU;
inline constexpr uint32_t kNotifyWirePrimaryShift = 9U;
inline constexpr uint32_t kNotifyWireSecondaryShift = 13U;
inline constexpr uint32_t kNotifyWirePrimaryHigh = 17U;
inline constexpr uint32_t kNotifyWireSecondaryHigh = 18U;
static_assert(kHtMaxTopk <= 32);

#if defined(__CCE_AICORE__) || defined(__CCE_KT_TEST__)
__aicore__
#endif
inline constexpr uint32_t PackNotifyRoute(uint32_t rank, uint32_t primary, uint32_t secondary)
{
    return rank | ((primary & kNotifyWireSlotMask) << kNotifyWirePrimaryShift) |
           ((secondary & kNotifyWireSlotMask) << kNotifyWireSecondaryShift) |
           ((primary >> 4U) << kNotifyWirePrimaryHigh) | ((secondary >> 4U) << kNotifyWireSecondaryHigh);
}

#if defined(__CCE_AICORE__) || defined(__CCE_KT_TEST__)
__aicore__
#endif
inline constexpr uint32_t NotifyRoutePrimarySlot(uint32_t row)
{
    return ((row >> kNotifyWirePrimaryShift) & kNotifyWireSlotMask) | (((row >> kNotifyWirePrimaryHigh) & 1U) << 4U);
}

#if defined(__CCE_AICORE__) || defined(__CCE_KT_TEST__)
__aicore__
#endif
inline constexpr uint32_t NotifyRouteSecondarySlot(uint32_t row)
{
    return ((row >> kNotifyWireSecondaryShift) & kNotifyWireSlotMask) |
           (((row >> kNotifyWireSecondaryHigh) & 1U) << 4U);
}

}  // namespace ascend_deepep
