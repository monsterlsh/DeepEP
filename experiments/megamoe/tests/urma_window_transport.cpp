// Host-only SHMEM doubles. Each PE has its own address space and each RC QP
// retains source pointers until a successful completion poll.
#include <array>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

#define __aicore__
#define __ubuf__
#define __gm__
constexpr int PIPE_S = 0;
constexpr int ACLSHMEMX_UDMA_ODR_RO = 0, ACLSHMEMX_UDMA_OP_CONFIG_SO = 1;
struct aclshmemx_udma_op_config_t { unsigned a, b, c, d; };

using Heap = std::array<uint8_t, 4096>;
alignas(128) Heap heaps[32];
alignas(128) uint8_t ub[256U * 1024U];
uint32_t localRank, cqError;
uint64_t clockTicks;
struct Transfer { uint32_t offset; uint8_t* source; uint32_t bytes; };
struct aclshmemi_rdma_sq_ctx { uint64_t head_addr; };
aclshmemi_rdma_sq_ctx sq[64];
uint32_t heads[64];
std::vector<Transfer> pending[64];
enum class aclshmemi_rdma_backend_t { HNS_1825 };
constexpr auto ACLSHMEMI_K_RDMA_BACKEND = aclshmemi_rdma_backend_t::HNS_1825;
struct DeviceInfo { unsigned qp_num; uint64_t sq_ptr; } info{2, reinterpret_cast<uint64_t>(sq)};
auto* aclshmemi_qp_info_fetch() { return &info; }
uint32_t aclshmemi_hns_1825_read_u32_gm(uint64_t addr) { return *reinterpret_cast<uint32_t*>(addr); }
template<class... Args> void aclshmemi_kernel_abort(const char* message, Args...) { throw std::runtime_error(message); }

namespace AscendC {
enum class TPosition { VECCALC };
enum class HardEvent { S_MTE3, MTE3_S };
using TEventID = unsigned;
template<HardEvent> void SetFlag(unsigned) {}
template<HardEvent> void WaitFlag(unsigned) {}
template<class T> struct LocalTensor {
    T* ptr;
    LocalTensor(TPosition, unsigned offset, unsigned elements) : ptr(reinterpret_cast<T*>(ub + offset))
    { assert(offset + elements * sizeof(T) <= sizeof(ub)); }
    T* GetPhyAddr() { return ptr; }
    void SetValue(unsigned i, T v) { ptr[i] = v; }
};
template<class T> struct GlobalTensor { T* ptr; void SetGlobalBuffer(T* p) { ptr = p; } };
template<class T> void DataCopy(GlobalTensor<T> dst, LocalTensor<T> src, uint32_t n)
{ std::memcpy(dst.ptr, src.ptr, n * sizeof(T)); }
template<class T> T ReadGmByPassDCache(T* p) { return *p; }
void AtomicAdd(int32_t* p, int32_t delta) { *p += delta; }
uint64_t GetSystemCycle() { return clockTicks++; }
}

uint32_t offsetOf(const uint8_t* dst, uint32_t size) {
    auto offset = reinterpret_cast<uintptr_t>(dst) - reinterpret_cast<uintptr_t>(heaps[localRank].data());
    assert(offset <= heaps[localRank].size() && size <= heaps[localRank].size() - offset);
    return uint32_t(offset);
}
void* aclshmem_ptr(uint8_t* dst, unsigned peer) {
    assert(peer / 16U == localRank / 16U);
    return heaps[peer].data() + offsetOf(dst, 1U);
}
void aclshmemx_mte_put_nbi(uint8_t* dst, uint8_t* src, uint8_t*, unsigned, unsigned bytes, unsigned peer, unsigned)
{ std::memcpy(aclshmem_ptr(dst, peer), src, bytes); }
template<class, int, const aclshmemx_udma_op_config_t&> void aclshmemx_udma_qp_put_nbi(
    uint8_t*, uint8_t*, void*, unsigned, unsigned, unsigned, unsigned) { assert(false); }
template<class, int> void aclshmemx_udma_atomic_add(uint64_t*, uint64_t, unsigned) { assert(false); }
void aclshmemx_udma_qp_quiet(unsigned, unsigned) { assert(false); }
void aclshmemx_roce_qp_put_nbi(uint8_t* dst, uint8_t* src, uint8_t*, unsigned bytes, unsigned peer,
                              unsigned qp, unsigned event) {
    assert(peer / 16U != localRank / 16U && event == 7U);
    offsetOf(src, bytes);
    ++heads[2*peer + qp];
    pending[2*peer + qp].push_back({offsetOf(dst, bytes), src, bytes});
}
template<aclshmemi_rdma_backend_t> unsigned aclshmemi_roce_poll_cq(unsigned peer, unsigned qp, unsigned head,
    AscendC::LocalTensor<uint64_t>&, AscendC::LocalTensor<uint32_t>&, unsigned event) {
    assert(event == 7U && head == heads[2*peer + qp]);
    if (cqError) return cqError;
    auto& writes = pending[2*peer + qp];
    // A ready flag must be submitted only after the earlier payload was
    // drained. Simulate the receiver observing a flag at its first arrival.
    for (auto& t : writes) {
        if (t.offset == 128U || t.offset == 256U) assert(writes.size() == 1U);
        std::memcpy(heaps[peer].data() + t.offset, t.source, t.bytes);
    }
    writes.clear();
    return 0;
}

#include "kernels/mega_moe_urma/common/mega_moe_transport.hpp"
using namespace MegaMoeUrma;

template<class F> void mustFail(F action) {
    try { action(); } catch (const std::runtime_error&) { return; }
    assert(false);
}
uint64_t value(unsigned peer, unsigned offset) {
    uint64_t n;
    std::memcpy(&n, heaps[peer].data() + offset, sizeof(n));
    return n;
}

int main() {
    for (uint32_t i = 0; i < 64; ++i) sq[i].head_addr = reinterpret_cast<uint64_t>(&heads[i]);
    for (unsigned world : {8U, 16U, 32U}) for (localRank = 0; localRank < world; ++localRank) {
        auto& heap = heaps[localRank];
        Arguments a{};
        a.symmetric = reinterpret_cast<uint64_t>(heap.data());
        a.config.world = world;
        a.config.rank = localRank;
        a.config.symmetricBytes = heap.size();
        a.config.windowedTransport = 1U;
        a.config.epochSourceOffset = 0U;
        for (auto epoch : {uint64_t(15), uint64_t(16), uint64_t(17)}) {
            a.config.epoch = epoch;
            std::memcpy(heap.data(), &epoch, sizeof(epoch));
            for (unsigned peer = 0; peer < world; ++peer) {
                if (peer == localRank) continue;
                const bool remote = world == 32U && peer / 16U != localRank / 16U;
                assert(RemoteHost(a, peer) == remote);
                const uint64_t row = 256U * localRank + peer + epoch;
                std::memcpy(heap.data() + 512, &row, sizeof(row));
                const bool empty = (localRank + peer) % 7U == 0;
                if (!empty) Put(a, 640, heap.data() + 512, sizeof(row), peer);
                if (remote) {
                    RocePut(a, 768, heap.data() + 512, sizeof(row), peer, kRouteQp);
                    RocePublish(a, 256, peer, kRouteQp);
                    assert(value(peer, 768) == row && value(peer, 256) == epoch);
                    assert(pending[2*peer + kPayloadQp].size() == (empty ? 0U : 1U));
                }
                PublishOrdered(a, 128, peer, epoch);
                Quiet(a, peer);
                assert(value(peer, 128) == epoch && (empty || value(peer, 640) == row));
            }
        }
    }
    localRank = 0;
    Arguments a{};
    a.symmetric = reinterpret_cast<uint64_t>(heaps[0].data());
    a.config.world = 32U;
    a.config.symmetricBytes = heaps[0].size();
    a.config.epoch = 18U;
    a.config.timeoutCycles = 10;
    mustFail([&] { WaitLocalDone(a, heaps[0].data() + 128U, 23); });
    mustFail([&] { RocePut(a, 4092, heaps[0].data(), 8, 23, kPayloadQp); });
    mustFail([&] { RocePut(a, 128, heaps[0].data() + 4092, 8, 23, kPayloadQp); });
    mustFail([&] { RocePut(a, 128, ub, 8, 23, kPayloadQp); });
    cqError = 1;
    const auto oldFlag = value(23, 128);
    mustFail([&] { RocePublish(a, 128, 23, kPayloadQp); });
    assert(pending[2*23 + kPayloadQp].empty() && value(23, 128) == oldFlag);
}
