// Copyright (c) 2026, Lu Lu
#pragma once

#include "device/gm2gm/engine/shmem_device_rdma.h"
#include "kernels/mega_moe_urma/common/mega_moe_compute_contract.hpp"

namespace MegaMoeUrma {

// The host enforces two contiguous groups of sixteen unique local NPUs.
__aicore__ inline bool RemoteHost(const Arguments& a, uint32_t peer)
{
    return a.config.world == 32U && a.config.rank / 16U != peer / 16U;
}

// Route exchange can overlap the peer's startup/Dispatch on a different
// AIV. Each (destination,QP) has exactly one producer and one CQ poller.
constexpr uint32_t kPayloadQp = 0U;
constexpr uint32_t kRouteQp = 1U;
constexpr uint32_t kRoceScratchBytes = 128U; // pinned HNS-1825 put/quiet contract
constexpr uint32_t kRoceScratchOffset = kAiv1PersistentUbBase - kRoceScratchBytes;
constexpr uint32_t kMteScratchBytes = 16U * 1024U;
constexpr uint32_t kMteScratchOffset = kAiv1PersistentUbBase - kMteScratchBytes;

__aicore__ inline __ubuf__ uint8_t* RoceScratch()
{
    AscendC::LocalTensor<uint8_t> buffer(AscendC::TPosition::VECCALC, kRoceScratchOffset, kRoceScratchBytes);
    return reinterpret_cast<__ubuf__ uint8_t*>(buffer.GetPhyAddr());
}

__aicore__ inline void RoceQuiet(const Arguments& a, uint32_t peer, uint32_t qp)
{
    // The public void quiet only logs a CQ error in debug builds. Check
    // the same pinned HNS CQ poll result before any ready/finished write.
    static_assert(ACLSHMEMI_K_RDMA_BACKEND == aclshmemi_rdma_backend_t::HNS_1825);
    auto* info = aclshmemi_qp_info_fetch();
    const uint64_t index = uint64_t(peer) * info->qp_num + qp;
    auto* sq = reinterpret_cast<__gm__ aclshmemi_rdma_sq_ctx*>(info->sq_ptr + index * sizeof(aclshmemi_rdma_sq_ctx));
    const uint32_t head = aclshmemi_hns_1825_read_u32_gm(sq->head_addr);
    AscendC::LocalTensor<uint32_t> ub32(AscendC::TPosition::VECCALC, kRoceScratchOffset, 8U);
    AscendC::LocalTensor<uint64_t> ub64(AscendC::TPosition::VECCALC, kRoceScratchOffset + 32U, 4U);
    const uint32_t status = aclshmemi_roce_poll_cq<ACLSHMEMI_K_RDMA_BACKEND>(peer, qp, head, ub64, ub32, 7U);
    if (status != 0U)
        aclshmemi_kernel_abort("MegaMoE RoCE CQ failed rank=%u peer=%u qp=%u epoch=%llu status=%u\n",
                               a.config.rank, peer, qp, static_cast<unsigned long long>(a.config.epoch), status);
}

__aicore__ inline void RocePut(const Arguments& a, uint64_t offset, __gm__ uint8_t* source, uint32_t bytes,
                               uint32_t peer, uint32_t qp)
{
    const uint64_t addr = reinterpret_cast<uint64_t>(source);
    const uint64_t capacity = a.config.symmetricBytes;
    if (addr < a.symmetric || offset > capacity || bytes > capacity - offset ||
        addr - a.symmetric > capacity || bytes > capacity - (addr - a.symmetric)) {
        aclshmemi_kernel_abort("MegaMoE RoCE transfer outside registered heap rank=%u peer=%u\n", a.config.rank, peer);
        return;
    }
    // The SDK translates the LOCAL symmetric destination to the peer MR. Never
    // pass a remote MTE mapping to RDMA. Sources remain registered until Quiet.
    aclshmemx_roce_qp_put_nbi(reinterpret_cast<__gm__ uint8_t*>(a.symmetric + offset), source, RoceScratch(),
                             bytes, peer, qp, 7U);
}

__aicore__ inline void RocePublish(const Arguments& a, uint64_t offset, uint32_t peer, uint32_t qp)
{
    // Complete all payload writes on this RC QP before writing the peer's exact
    // generation. Drain that write too, including on zero-row peer edges.
    RoceQuiet(a, peer, qp);
    RocePut(a, offset, reinterpret_cast<__gm__ uint8_t*>(a.symmetric + a.config.epochSourceOffset),
            sizeof(uint64_t), peer, qp);
    RoceQuiet(a, peer, qp);
}

enum PeerWaitPhase : uint32_t { kWaitRoutes = 1U, kWaitPeerProgress = 2U, kWaitFinished = 3U };

__aicore__ inline void CheckPeerDeadline(const Arguments& a, uint64_t since, uint32_t peer, PeerWaitPhase phase)
{
    if (a.config.world == 32U && AscendC::GetSystemCycle() - since >= a.config.timeoutCycles) {
        aclshmemi_kernel_abort("MegaMoE timeout rank=%u peer=%u epoch=%llu phase=%u\n", a.config.rank, peer,
                               static_cast<unsigned long long>(a.config.epoch), static_cast<uint32_t>(phase));
    }
}
}  // namespace MegaMoeUrma
