// Modified by SimpleBright_Man 2026
#pragma once

#include "kernels/mega_moe_common/implementation_namespace.hpp"
#include <cstdint>

#if defined(__NPU_DEVICE__)
#define MEGA_MOE_WAVE_WINDOW_INLINE __aicore__ inline
#else
#define MEGA_MOE_WAVE_WINDOW_INLINE inline
#endif

namespace DEEPEP_MEGAMOE_IMPL_NAMESPACE {
struct WaveRowMapping {
    uint64_t logicalStart = 0U;
    uint64_t storageStart = 0U;
};

MEGA_MOE_WAVE_WINDOW_INLINE constexpr uint32_t WaveWindowSlotCount(bool enabled, bool isA8W4)
{
    return enabled ? (isA8W4 ? 3U : 2U) : 1U;
}

MEGA_MOE_WAVE_WINDOW_INLINE constexpr uint64_t WaveWindowRowCapacity(uint64_t maxRows, uint32_t mGroupsPerWave,
                                                                     uint32_t windowSlots)
{
    if (mGroupsPerWave == 0U || windowSlots <= 1U) {
        return maxRows;
    }
    const uint64_t capacity = static_cast<uint64_t>(mGroupsPerWave) * 256U * windowSlots;
    return maxRows < capacity ? maxRows : capacity;
}

MEGA_MOE_WAVE_WINDOW_INLINE constexpr WaveRowMapping MakeWaveRowMapping(uint64_t maxRows, uint64_t storageRows,
                                                                        uint32_t mGroupsPerWave, uint32_t windowSlots,
                                                                        uint64_t waveId, uint64_t logicalStart)
{
    if (storageRows >= maxRows || mGroupsPerWave == 0U || windowSlots <= 1U) {
        return {};
    }
    return {logicalStart, (waveId % windowSlots) * static_cast<uint64_t>(mGroupsPerWave) * 256U};
}

MEGA_MOE_WAVE_WINDOW_INLINE constexpr uint64_t MapWaveRow(const WaveRowMapping& mapping, uint64_t logicalRow)
{
    return mapping.storageStart + logicalRow - mapping.logicalStart;
}

MEGA_MOE_WAVE_WINDOW_INLINE constexpr bool CanScheduleReadyCombine(uint64_t dispatchedRows, uint64_t combineEndRow,
                                                                     uint64_t requiredLeadRows, bool dispatchAfterCombine)
{
    return !dispatchAfterCombine && dispatchedRows >= combineEndRow && dispatchedRows - combineEndRow >= requiredLeadRows;
}
}  // namespace DEEPEP_MEGAMOE_IMPL_NAMESPACE

#undef MEGA_MOE_WAVE_WINDOW_INLINE
