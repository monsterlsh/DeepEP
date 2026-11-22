// Exercise the device scheduler with delayed or immediately ready GMM2 slices.
#include <cassert>
#include <cstdint>
#include <initializer_list>
#include "kernels/mega_moe_common/common/mega_moe_wave_window.hpp"

#define __aicore__
using HierarchicalMegaMoeImpl::CanScheduleReadyCombine;
constexpr unsigned GMM1_TILE_M = 256, GROUPED_MATMUL_MODE_A8W4 = 1;
enum class MegaMoeActMode : uint8_t { GELU = 4 };
struct Tiling { unsigned groupedMatmulMode = GROUPED_MATMUL_MODE_A8W4, actMode = 4; };
struct Params { const Tiling* tilingData; bool inputScaleGmAddr = true; };
struct ExpertTokenPosition { unsigned expertIdx = 0, globalTokenIndex = 0; };
struct AivTaskCursor {
    struct { ExpertTokenPosition end; } slice;
    unsigned nextTile = 0, tileCount = 1;
};
struct SharedActivationCursor { bool done = false; };
struct Timer {
    bool Enabled() { return false; }
    void ClearSlice() {}
    void SetWave(int) {}
};
struct HierarchicalAscendTimer { static constexpr int NO_WAVE = -1; };
namespace AscendC { inline int64_t GetSystemCycle() { static int64_t tick = 0; return ++tick; } }
constexpr int64_t GM_FLAG_POLL_BACKOFF_CYCLES = 1;
bool UseEarlyClosure(const Tiling*) { return true; }
bool UseHierarchicalRoce(const Tiling*) { return true; }
bool UseAiv0CombineOwners(const Tiling*) { return true; }
bool IsHierarchicalCombineOwner(const Tiling*, unsigned) { return false; }
int HierarchicalCombineLane(const Tiling*, unsigned) { return 0; }
int HierarchicalRoceScratch(const Params&) { return 0; }
template <class... Args> bool TryProgressHierarchicalReducedCombine(Args&&...) { return false; }

struct Scheduler {
    Tiling tiling;
    Params params_{&tiling};
    struct { unsigned moeExpertPerRank = 25; } commonConfig_;
    struct Job { unsigned jobIndex = 0, totalJobs = 1; } aivJob_;
    struct { Job blockJob; } gmmExecutionConfig_;
    struct { unsigned blockIdx = 0; } countWorkspace_;
    struct { bool inFlight = false; void Drain(Timer&) { inFlight = false; } } combinePipeline_;
    Timer timer_;
    unsigned mGroupsPerWave_ = 19, sharedExpertNum_ = 0, hierarchicalCombine_ = 0;
    bool TopkWeightsPrefetch = true, delayed = false, closure = false, combinedSinceDispatch = false;
    unsigned dispatched = 0, combined = 0, firstCombineAt = 0;
    void DispatchNextExpertSlice(ExpertTokenPosition& pos, uint32_t& groups, uint64_t) {
        assert(!combinePipeline_.inFlight && !closure);
        ++groups;
        pos.expertIdx = ++dispatched;
        pos.globalTokenIndex += 256;
        combinedSinceDispatch = false;
    }
    void PublishCoreClosure(const Params&, unsigned, bool) {
        assert(!closure && dispatched == commonConfig_.moeExpertPerRank);
        closure = true;
    }
    bool NextAivSlice(AivTaskCursor& cur, bool activation) {
        if (activation || combined == commonConfig_.moeExpertPerRank) return false;
        cur.slice.end.globalTokenIndex += 256;
        return true;
    }
    bool TryRunReadyCombine(const AivTaskCursor& cur) {
        if (delayed && !closure) return false;
        assert(cur.slice.end.globalTokenIndex == (combined + 1) * 256);
        if (!closure) assert(!combinedSinceDispatch);
        if (combined++ == 0) firstCombineAt = dispatched;
        combinePipeline_.inFlight = combinedSinceDispatch = true;
        return true;
    }
    bool IsActivationReady(AivTaskCursor&, int32_t) { return false; }
    void RunReadyActivation(AivTaskCursor&, int32_t&) { assert(false); }
    bool TryRunSharedExpertActivation(SharedActivationCursor&, void*) { return false; }
    void Run(int32_t& readySequence);
};

/* DEVICE SCHEDULER */

int main()
{
    for (bool early : {true, false}) for (bool delayed : {true, false}) {
        Scheduler run;
        run.TopkWeightsPrefetch = early;
        run.delayed = delayed;
        int32_t readySequence = 0;
        run.Run(readySequence);
        assert(run.firstCombineAt == (early && !delayed ? 20u : 25u));
        assert(run.dispatched == 25 && run.combined == 25 && run.closure && !run.combinePipeline_.inFlight);
    }
}
