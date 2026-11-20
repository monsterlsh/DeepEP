#include "kernels/mega_moe_urma/common/mega_moe_route_plan.hpp"
#include <array>
#include <cassert>
#include <cstddef>
#include <vector>
#define __aicore__
#define __gm__

namespace MegaMoeUrma {
enum class Phase { Dispatch, Combine, Activation };
using CoarsePhase = Phase;
struct Timer {
    void Begin(Phase) {}
    void End(Phase, uint64_t, bool) {}
    void CoarseBegin(Phase) {}
    void CoarseEnd(Phase) {}
    uint64_t TraceBegin()
    {
        return 0;
    }
    void TraceEnd(TraceEvent, uint64_t, TraceMeta) {}
    void MarkSendComplete(Phase) {}
};
struct DispatchReceiveObserver {
    void Init(Timer&) {}
    void Arm(const Arguments&, uint32_t, uint32_t) {}
    void Poll() {}
};
struct SharedIncoming {
    const Arguments& a;
    uint32_t owner;
    bool peer = false, self = false;
    DispatchReceiveObserver dispatch;
    void Poll() {}
};
enum class TPosition { VECCALC };
struct CooperativeUb {
    static constexpr uint32_t submitted = kAiv1PersistentUbBase, selfCounts = submitted + kMaxPeerChunks * 4U;
};
alignas(64) uint32_t ub[kRequiredUbBytes / 4U];
template <class T>
struct LocalTensor {
    uint32_t offset, count;
    LocalTensor(TPosition, uint32_t bytes, uint32_t n) : offset(bytes / 4U), count(n)
    {
        assert(bytes + n * sizeof(T) <= sizeof(ub));
    }
    void SetValue(uint32_t i, T value)
    {
        assert(i < count);
        ub[offset + i] = value;
    }
    T GetValue(uint32_t i) const
    {
        assert(i < count);
        return ub[offset + i];
    }
};
template <class T>
T ReadGmByPassDCache(T* v)
{
    return *v;
}
template <class T>
void WriteGmByPassDCache(T* dst, T value)
{
    *dst = value;
}
struct PackBatch {
    PackBatch(const Arguments&, Timer&) {}
    void FlushReady() {}
    void Close() {}
};
DeviceRouteInfo info;
std::array<uint32_t, kMaxWorld> sends{}, dispatchFlags{}, combineFlags{}, finishes{}, quiets{};
uint32_t selfReports = 0;
int32_t ready = 128;
constexpr uint32_t kMaxPutBytes = 1U << 28U, kReorderCvSubmitted = UINT32_MAX, kCombinePutRows = 128;
uint32_t PeerForOwner(uint32_t rank, uint32_t owner)
{
    return owner + (owner >= rank);
}
uint64_t PeerPhaseSlot(const Arguments&, uint32_t phase, uint32_t rank)
{
    return (phase * kMaxWorld + rank) * kFlagBytes;
}
const DeviceRouteInfo* RouteInfo(const Arguments&)
{
    return &info;
}
uint32_t RuntimeSendCvCount(const Arguments&, uint32_t p)
{
    return sends[p];
}
Chunk RuntimeSendChunk(const Arguments& a, uint32_t p, uint32_t cv)
{
    return {p, cv, uint64_t(cv) * a.config.cvRows, uint64_t(cv) * a.config.cvRows, a.config.cvRows};
}
int32_t* LocalCompletionFlag(const Arguments&, uint32_t)
{
    return &ready;
}
int32_t* PackReadyFlag(const Arguments&, uint32_t, uint32_t)
{
    return &ready;
}
uint32_t DispatchBatchLimit(uint32_t, uint32_t, bool)
{
    return 1U;
}
bool AdjacentChunks(const Chunk&, uint32_t, uint32_t, const Chunk&)
{
    return false;
}
TraceMeta CooperativeChunkMeta(const Chunk&, uint32_t, uint32_t)
{
    return {};
}
Chunk RuntimeReceiveChunkByIndex(const Arguments& a, uint32_t i)
{
    const uint32_t p = PlanPeerForCv(i, a.config, info, false), cv = i - info.recvCvBase[p];
    return RuntimeSendChunk(a, p, cv);
}
uint32_t PairedReorderedPrefix(const Arguments&, const Chunk& c)
{
    return c.rows;
}
void Put(const Arguments&, uint64_t, uint8_t*, uint32_t, uint32_t) {}
void MarkReorderComplete(const Arguments&)
{
    ++selfReports;
}
void PublishOrdered(const Arguments& a, uint64_t offset, uint32_t p, uint64_t epoch)
{
    assert(epoch == a.config.epoch);
    if (offset == PeerPhaseSlot(a, 1, a.config.rank))
        ++finishes[p];
    else if (offset >= a.config.combineReadyOffset)
        ++combineFlags[p];
    else
        ++dispatchFlags[p];
}
void Quiet(const Arguments&, uint32_t p)
{
    ++quiets[p];
}
void FinishCombinePeer(const Arguments&, uint32_t, bool&, bool&)
{
    assert(quiets[15] && quiets[31]);
}
struct CooperativeNoWait {};
struct UbRoutedActivation {
    void Init(const Arguments&) {}
};
struct DirectReorderWork {
    Arguments& a;
    PackBatch& pack;
    uint32_t core;
    UbRoutedActivation& activation;
};
struct RoutedCompletionConsumer {
    RoutedCompletionConsumer(const Arguments&, uint32_t) {}
    void Init() {}
    bool Done()
    {
        return true;
    }
};
struct SharedActivationState {
    SharedActivationState(const Arguments&, uint32_t, Timer&) {}
    bool Done()
    {
        return true;
    }
    bool TryOne(CooperativeNoWait&)
    {
        return false;
    }
};
constexpr uint32_t kPackStartupRounds = 2U;
void PackOneRound(const Arguments&, uint32_t, uint32_t, uint32_t, PackBatch&) {}
void MarkPackWorkerComplete(const Arguments&) {}
bool RouteSummaryReady(const Arguments&)
{
    return true;
}
void AcquireRouteSummary(const Arguments&) {}
Fragment LoadFragment(const Arguments&, uint32_t)
{
    return {};
}
bool AreGroupInputsReady(const Arguments&, const Fragment&)
{
    return true;
}
void GatherInputScale(const Arguments&, const Fragment&, uint32_t, Timer&) {}
uint32_t ConsumeIssuedReorders(RoutedCompletionConsumer&, DirectReorderWork&, Timer&)
{
    return 0;
}
namespace AscendC {
uint64_t GetSystemCycle()
{
    static uint64_t tick = 0;
    return ++tick;
}
}  // namespace AscendC
using AscendC::GetSystemCycle;
struct Timeout {
    uint32_t peer, phase;
};
void aclshmemi_kernel_abort(const char*, uint32_t rank, uint32_t peer, unsigned long long epoch, uint32_t phase)
{
    assert(rank == 0U && epoch == 17U);
    throw Timeout{peer, phase};
}
#include "ep32_peer_scheduler.inc"

void Run(bool empty)
{
    Arguments a{};
    a.config.world = 32U;
    a.config.rank = 0U;
    a.config.topk = 32U;
    a.config.bs = 256U;
    a.config.hidden = 4096U;
    a.config.localExperts = 25U;
    a.config.epoch = 17U;
    a.config.windowedTransport = 1U;
    a.config.cvRows = 128U;
    a.config.maxPeerChunks = kWindowPeerChunks;
    a.config.dispatchReadyOffset = 2 * kMaxWorld * kFlagBytes;
    a.config.combineReadyOffset = a.config.dispatchReadyOffset + kMaxWorld * kWindowPeerChunks * kFlagBytes;
    std::vector<uint64_t> memory((a.config.combineReadyOffset + kWindowPeerChunks * kFlagBytes) / 8U, 0U);
    a.symmetric = reinterpret_cast<uint64_t>(memory.data());
    sends = {};
    dispatchFlags = {};
    combineFlags = {};
    finishes = {};
    quiets = {};
    selfReports = 0;
    info = {};
    if (!empty) {
        sends[15] = 20U;
        sends[31] = 44U;
    }
    // The two remote source ranges and the self source may each have 64 receive CVs.
    for (uint32_t p = 0; p < kMaxWorld; ++p) {
        const uint32_t cvs = !empty && (p == 0 || p == 15 || p == 31) ? kWindowPeerChunks : 0U;
        info.recvCvBase[p + 1] = info.recvCvBase[p] + cvs;
        info.incoming[p][a.config.localExperts] = cvs * a.config.cvRows;
    }
    Timer timer;
    PackBatch pack(a, timer);
    SharedIncoming left{a, 14U}, right{a, 30U};
    CooperativePeer first(a, PeerOwnerForSlot(a.config.world, 14, 0), 0, timer, left.dispatch);
    CooperativePeer second(a, PeerOwnerForSlot(a.config.world, 14, 1), 1, timer, right.dispatch);
    memory[PeerPhaseSlot(a, 0U, 31U) / 8U] = a.config.epoch;
    memory[PeerPhaseSlot(a, 0U, 15U) / 8U] = a.config.epoch - 1U;
    for (uint32_t step = 0; step < 1024U && !(first.Done() && second.Done()); ++step) {
        // Let peer 31 advance independently before peer 15 starts. For
        // nonempty streams their private submitted-CV counters overlap in time.
        if (step == 65U) memory[PeerPhaseSlot(a, 0U, 15U) / 8U] = a.config.epoch;
        first.TryDispatch(pack, true);
        second.TryDispatch(pack, true);
        first.TryCombine(true);
        second.TryCombine(true);
        if (step == 64U) {
            assert(!first.Done() && !dispatchFlags[15] && !finishes[15]);
            assert(empty ? second.Done() : (combineFlags[31] > 0U && !second.Done()));
        }
    }
    assert(first.Done() && second.Done());
    first.DrainOutgoing(left);
    second.DrainOutgoing(right);
    first.WaitIncoming(left);
    second.WaitIncoming(right);
    for (uint32_t p : {15U, 31U}) {
        assert(dispatchFlags[p] == sends[p]);
        assert(combineFlags[p] == (empty ? 0U : kWindowPeerChunks));
        assert(finishes[p] == 1U && quiets[p] == 1U);
    }
    assert(selfReports == 1U);
    for (uint32_t cv = 0; cv < kWindowPeerChunks; ++cv) {
        const auto epoch = memory[(a.config.combineReadyOffset + cv * kFlagBytes) / 8U];
        assert(epoch == (!empty && cv % 16U == 14U ? a.config.epoch : 0U));
    }
}
// Run the production owner loop with a permanently stale PeerStart on an empty
// edge. The other peer must publish its finished marker before the timeout.
void CheckTimeout(uint32_t stalled)
{
    Arguments a{};
    a.config.world = 32U;
    a.config.rank = 0U;
    a.config.epoch = 17U;
    a.config.timeoutCycles = 256U;
    a.config.windowedTransport = 1U;
    alignas(128) uint64_t memory[2U * kMaxWorld * kFlagBytes / sizeof(uint64_t)]{};
    a.symmetric = reinterpret_cast<uint64_t>(memory);
    sends = {};
    finishes = {};
    info = {};
    const uint32_t live = stalled == 15U ? 31U : 15U;
    memory[PeerPhaseSlot(a, 0U, live) / 8U] = a.config.epoch;
    memory[PeerPhaseSlot(a, 0U, stalled) / 8U] = a.config.epoch - 1U;
    Timer timer;
    SharedIncoming incoming{a, 14U};
    try {
        RunCooperativeAiv1(a, 14U, timer, incoming);
    } catch (const Timeout& t) {
        assert(t.peer == stalled && t.phase == kWaitPeerProgress);
        assert(finishes[live] == 1U && finishes[stalled] == 0U);
        return;
    }
    assert(false);
}
}  // namespace MegaMoeUrma
int main()
{
    MegaMoeUrma::Run(false);
    MegaMoeUrma::Run(true);
    MegaMoeUrma::CheckTimeout(15U);
    MegaMoeUrma::CheckTimeout(31U);
}
