// Replay the production wave handoff with host-side Dispatch copies.
#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <vector>

#define __aicore__
#include "kernels/mega_moe_common/common/mega_moe_peer_rows.hpp"
#include "kernels/mega_moe_common/common/mega_moe_wave_window.hpp"

using namespace MegaMoeImpl;

constexpr uint32_t kWorld = 32, kTopK = 25, kLocalExperts = 12;
constexpr uint32_t GMM1_TILE_M = 256, W4A8_SMALL_BATCH_MAX_TOKENS = 256, COMBINE_NO_QUANT = 0;
struct ExpertTokenPosition { uint32_t globalTokenIndex = 0, expertIdx = 0; };
struct ExpertTokenRange { ExpertTokenPosition begin, end; };
struct BlockJobContext { uint32_t jobIndex = 0, totalJobs = 1; };
struct Common { uint32_t tokenNum, rank, moeExpertPerRank; };
struct SplitExpert { uint32_t firstWaveRows, totalRows; };
struct Params { struct { SplitExpert expertRevTokenNumsPtr; } workspaceInfo; };
struct CountTable {
    std::array<uint32_t, kWorld> bySource{};
    uint32_t Get(uint32_t peer) const { return bySource[peer]; }
};
struct RouteCopies {
    CountTable counts;
    std::array<std::vector<uint32_t>, kWorld> visits;
    std::vector<bool> peerBalanced;
};

// Model each copy as a visit to one route in the original source queue. The
// production peer ordering and prefix helpers determine each copied interval.
template <typename, typename, uint32_t, bool, bool PeerBalanced = false>
void DispatchTokenRange(int, const Common& common, const BlockJobContext&, int, const Params&, int, RouteCopies& copies,
                        const ExpertTokenRange& range, WaveRowMapping)
{
    copies.peerBalanced.push_back(PeerBalanced);
    const uint32_t begin = range.begin.globalTokenIndex;
    const uint32_t end = range.end.globalTokenIndex;
    const auto& counts = copies.counts;
    const auto limits = GetPeerRowLimits(counts, kWorld);
    auto beginCut = GetPeerRowCut(counts, kWorld, limits, begin);
    auto endCut = GetPeerRowCut(counts, kWorld, limits, end);
    uint32_t sourcePrefix = 0;
    for (uint32_t ordinal = 0; ordinal < kWorld; ++ordinal) {
        const auto peer = PeerBalanced ? GetDispatchPeer(ordinal, common.rank, kWorld, 8) : ordinal;
        const auto count = counts.Get(peer);
        const auto sliceStart = std::clamp(begin, sourcePrefix, sourcePrefix + count) - sourcePrefix;
        const auto sliceEnd = std::clamp(end, sourcePrefix, sourcePrefix + count) - sourcePrefix;
        const auto first = PeerBalanced ? TakePeerRowPrefix(count, beginCut) : sliceStart;
        const auto last = PeerBalanced ? TakePeerRowPrefix(count, endCut) : sliceEnd;
        for (uint32_t row = first; row < last; ++row) ++copies.visits[peer][row];
        sourcePrefix += count;
    }
}

template <uint32_t>
ExpertTokenRange PlanNextExpertTokenRangeInWave(SplitExpert split, int, uint32_t, uint32_t, uint32_t& groups,
                                               const ExpertTokenPosition& begin)
{
    ++groups;
    // Other experts add no information to the coverage check. Replay the
    // partial expert at the exact K37 startup boundary and then its remainder.
    return {begin, begin.globalTokenIndex == 0 ? ExpertTokenPosition{split.firstWaveRows, 0}
                                              : ExpertTokenPosition{split.totalRows, 1}};
}

void ReloadDispatchCumsumRange(const Common&, RouteCopies&, uint32_t, uint32_t) {}
struct AscendTimer { static constexpr int AIV_DISPATCH = 0; };
struct Probe {
    using ActivationType = uint8_t;
    using QuantScaleOutType = uint8_t;
    static constexpr bool TopkWeightsPrefetch = false;
    static constexpr uint32_t CombineQuantMode = COMBINE_NO_QUANT;
    Common commonConfig_;
    Params params_;
    int tokenDispatchConfig_ = 0, syncWorkspaceLayout_ = 0, g_winRankAddr_ = 0, countWorkspace_ = 0;
    uint32_t mGroupsPerWave_ = 1;
    struct { BlockJobContext blockJob; } gmmExecutionConfig_;
    struct { void SetWave(int64_t) {} void StartSpan(int) {} void EndSpan(int) {} } timer_;
    RouteCopies tokenDispatchScratch_;

    WaveRowMapping WaveStorage(uint64_t, uint64_t) const { return {}; }
    bool IsPositionWithinWave(const ExpertTokenPosition& position, uint32_t groups) const {
        return position.expertIdx < commonConfig_.moeExpertPerRank && groups < mGroupsPerWave_;
    }
    void DispatchWaveRange(const ExpertTokenRange& range, const BlockJobContext& dispatchJob, WaveRowMapping mapping)
    /* SELECTOR */;
    void DispatchNextExpertSlice(ExpertTokenPosition& position, uint32_t& waveMGroupCount,
                                 uint64_t waveId, uint64_t waveLogicalStart)
    /* NEXT SLICE */

    void Run() {
        ExpertTokenPosition dispatchPosition{};
        /* FIRST WAVE */
        uint32_t nextGroups = 0;
        DispatchNextExpertSlice(dispatchPosition, nextGroups, 1, dispatchPosition.globalTokenIndex);
        assert(dispatchPosition.expertIdx == 1);
    }
};

int main(int argc, char** argv)
{
    assert(argc == 2);
    const auto window = static_cast<uint32_t>(std::strtoul(argv[1], nullptr, 10));
    const uint32_t logicalBs = window == 2048 ? 4096 : window;
    const uint32_t splitExpert = window == 2048 ? 0 : 3;
    const uint32_t splitRows = window == 2048 ? 2816 : 512;
    for (uint32_t offset = 0; offset < logicalBs; offset += window) {
        std::array<CountTable, kWorld> byOwner{};
        for (uint32_t source = 0; source < kWorld; ++source) {
            const auto start = (source * logicalBs + offset) * kTopK;
            for (uint32_t route = start; route < start + window * kTopK; ++route) {
                // Match benchmark.inputs() balanced routes for EP32 and E384.
                if (route / kWorld % kLocalExperts == splitExpert) ++byOwner[route % kWorld].bySource[source];
            }
        }
        for (uint32_t owner = 0; owner < kWorld; ++owner) {
            Probe probe{};
            probe.commonConfig_ = {window, owner, 1};
            probe.tokenDispatchScratch_.counts = byOwner[owner];
            uint32_t total = 0;
            for (uint32_t source = 0; source < kWorld; ++source) {
                const auto rows = byOwner[owner].Get(source);
                probe.tokenDispatchScratch_.visits[source].resize(rows);
                total += rows;
            }
            assert(splitRows < total);
            probe.params_.workspaceInfo.expertRevTokenNumsPtr = {splitRows, total};
            probe.Run();
            for (const auto& source : probe.tokenDispatchScratch_.visits) {
                assert(std::all_of(source.begin(), source.end(), [](uint32_t visits) { return visits == 1; }));
            }
            const bool prefill = window > 256;
            assert((probe.tokenDispatchScratch_.peerBalanced == std::vector<bool>{prefill, prefill}));
        }
    }
}
