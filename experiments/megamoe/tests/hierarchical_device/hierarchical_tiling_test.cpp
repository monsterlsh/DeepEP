// Copyright (c) 2026, Lu Lu

#include <cassert>
#include <cstdint>
#include <stdexcept>
#include <string_view>

#include "../../csrc/roce_tiling.hpp"

namespace {

using namespace deepep::megamoe::roce;
using namespace HierarchicalMegaMoeImpl;

constexpr PlatformInfo platform{28, 56, 256U * 1024U, 28, 16U * 1024U * 1024U};

template <typename Callable>
void Rejects(Callable attempt)
{
    try {
        attempt();
    } catch (const std::invalid_argument&) {
        return;
    }
    assert(false);
}

void CheckCase(const char* mma, const char* activation, int topk, int bs, int timing)
{
    constexpr int hidden = 4096;
    constexpr int intermediate = 1536;
    const uint8_t mode = MmaMode(mma);
    const bool gelu = std::string_view(activation) == "gelu";
    const bool early = timing == 1;
    const auto tiling = MakeTiling(bs, hidden, intermediate, topk, 25, 32, bs, 0, platform, activation, mode, timing);

    assert(tiling.hiddenDim == (gelu ? intermediate : 2 * intermediate));
    assert(tiling.groupedMatmulMode == mode && tiling.topkWeightsPrefetch == timing);
    assert(IntermediateWidth(tiling) == intermediate);
    assert(tiling.maxOutputSize == static_cast<uint32_t>(bs * 32 * 25));
    const uint32_t tile_n = !gelu && std::string_view(mma) == "fp8xfp4" ? 128U : 256U;
    const uint32_t early_status_slots = ((bs + 7U) / 8U) * (intermediate / tile_n);
    assert(tiling.mGroupsPerWave != 0 && tiling.maxTilesPerExpert == (early ? early_status_slots : 0U));
    assert(tiling.roceScratchUbOffset + HIERARCHICAL_ROCE_UB_BYTES == platform.ub);

    const uint32_t expected_record = hidden + hidden / 32U + (early ? 128U : 0U);
    const uint32_t packets = (expected_record + 479U) / 480U;
    assert(QuantizedRecordBytes(tiling) == expected_record);
    assert(HierarchicalDispatchPacketCount(&tiling) == packets);
    assert(tiling.dispatchBufferConfig.copyBufferBytes == packets * 480U);

    HierarchicalMegaMoeArenaOffsets arena{};
    const auto heap = LayoutArena(32, 25, bs, topk, hidden, 16, 0, 1, early, arena);
    const auto row_bytes = static_cast<int64_t>(packets) * 512;
    assert(arena.combine - arena.quant >= bs * expected_record);
    assert(arena.packetSend - arena.relay == bs * row_bytes);
    assert(arena.gateway - arena.packetSend == bs * row_bytes);
    assert(arena.partialSend - arena.gateway >= CalcGatewayCombineBytes(bs, topk, hidden));
    assert(arena.partialRecv - arena.partialSend == bs * hidden * 4);
    assert(heap % (2 * 1024 * 1024) == 0 && heap <= 1LL << 30);

    for (int64_t region : {arena.counts, arena.rail, arena.control, arena.quant, arena.combine,
                           arena.relay, arena.packetSend, arena.gateway, arena.partialRecv}) {
        assert(region % 512 == 0);
    }
    const auto workspace = WorkspaceBytes(tiling, platform);
    const WorkspaceInfo layout(nullptr, &tiling);
    assert(workspace > layout.workspaceSize && workspace - layout.workspaceSize == 66LL * 1024 * 1024);
    const auto bytes_between = [](const uint8_t* first, const uint8_t* second) {
        return reinterpret_cast<uintptr_t>(second) - reinterpret_cast<uintptr_t>(first);
    };
    const uint64_t rows = tiling.maxOutputSize;
    assert(bytes_between(layout.dispatchRevDataPtr, layout.dispatchRevScalePtr) == rows * hidden);
    assert(bytes_between(layout.activationQuantDataPtr, layout.activationQuantScalePtr) ==
           rows * intermediate);
    assert(bytes_between(layout.gmm1MmadResPtr, layout.gmm2MmadResPtr) >= rows * tiling.hiddenDim * 2);
    assert(layout.gmm1TileStatusElementCount == (early ? (25 * early_status_slots + 1) * INT_CACHELINE : 0));

    const auto tail = MakeTiling(1, hidden, intermediate, topk, 25, 32, bs, 0, platform, activation, mode, timing);
    assert(tail.maxOutputSize == tiling.maxOutputSize);
    assert(WorkspaceBytes(tail, platform) == workspace);
}

}  // namespace

int main()
{
    static_assert(sizeof(HierarchicalMegaMoeTilingData) == 368);
    for (const auto* mma : {"fp8xfp4", "fp8xfp8"}) {
        for (const auto* activation : {"gelu", "swiglu"}) {
            for (int timing : {0, 1}) {
                for (int topk : {25, 32}) {
                    for (int bs : {1, 32, 64, 128, 256}) {
                        CheckCase(mma, activation, topk, bs, timing);
                    }
                }
            }
        }
    }
    Rejects([] { MmaMode("unsupported"); });
    Rejects([] { CheckArenaGeometry(32, 25, 257, 25, 4096, 0); });
    Rejects([] { CheckArenaGeometry(32, 25, 32, 33, 4096, 0); });
    Rejects([] { CheckArenaGeometry(16, 25, 32, 25, 4096, 0); });
    Rejects([] { CheckArenaGeometry(32, 25, 32, 25, 4096, 2); });
    Rejects([] { MakeTiling(32, 4096, 1536, 25, 25, 32, 31, 0, platform, "swiglu", MmaMode("fp8xfp4"), 0); });
    Rejects([] { MakeTiling(32, 4096, 1536, 25, 25, 32, 32, 1, platform, "gelu", MmaMode("fp8xfp4"), 1); });
    constexpr PlatformInfo low_ub{28, 56, 32U * 1024U, 28, 0};
    Rejects([&] { MakeTiling(256, 8192, 4096, 32, 128, 32, 256, 0, low_ub, "gelu", MmaMode("fp8xfp8"), 1); });
}
