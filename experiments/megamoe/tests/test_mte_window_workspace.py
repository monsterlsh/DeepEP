"""Compile the MTE host planner for the target model, including its short last window."""

from pathlib import Path
import shutil
import subprocess
import sys

import pytest


def test_mte_ep32_shape_windows(tmp_path):
    compiler = next((path for name in ("c++", "g++", "clang++") if (path := shutil.which(name))), None)
    if compiler is None:
        pytest.skip("The MTE layout probe requires a host C++17 compiler")
    csrc = Path(__file__).resolve().parents[1] / "csrc"
    math = tmp_path / "op_common/op_host/util/math_util.h"
    math.parent.mkdir(parents=True)
    math.write_text(
        "#pragma once\nnamespace Ops::Base {\n"
        "template<class T, class U> inline T CeilDiv(T a, U b) {return (a+b-1)/b;}\n"
        "template<class T, class U> inline T CeilAlign(T a, U b) {return CeilDiv(a,b)*b;}\n}\n",
        encoding="ascii",
    )
    tiling = tmp_path / "kernel_tiling/kernel_tiling.h"
    tiling.parent.mkdir()
    tiling.write_text("", encoding="ascii")
    source = tmp_path / "mte.cpp"
    native = (csrc / "forward.cpp").read_text(encoding="utf-8")
    # Pull in the production tiler while keeping the probe independent of Torch, NPU streams and SHMEM.
    helpers = native[native.index("template <typename T>") : native.index("PlatformInfo GetPlatformInfo")]
    make_tiling = native[native.index("MegaMoeTilingData MakeTiling") : native.index("void CheckTensor")]
    wave = (csrc / "kernels/mega_moe/mega_moe_wave_a8w4.hpp").read_text(encoding="utf-8")
    # Exercise the device storage decision, including which block's ready flag and prefix it reads.
    storage = wave[wave.index("    bool reuseWaveSlots_") : wave.index("    // 各任务流")]
    source.write_text(
        r"""
#include <algorithm>
#include <cassert>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
#include "kernels/mega_moe_common/common/mega_moe_workspace.hpp"
struct PlatformInfo { unsigned aic, aiv, ub, block_dim, system_workspace; };
"""
        + helpers
        + make_tiling
        + r"""
using namespace MegaMoeImpl;
#define __aicore__
#define __gm__
struct BlockWorkspaceContext { unsigned blockIdx, blockNum; };
int32_t *publishedPrefix;
uint8_t *publishedFlag;
unsigned readyWaits, prefixReads;
void WaitForMoeExpertTokenCountReady(uint8_t* flag, BlockWorkspaceContext owner, unsigned expert) {
    assert(flag == publishedFlag && owner.blockIdx == 0 && owner.blockNum == 28 && expert == 0);
    ++readyWaits;
}
namespace AscendC {
int32_t ReadGmByPassDCache(int32_t* p) {
    assert(readyWaits == 1 && p == publishedPrefix);
    ++prefixReads;
    return *p;
}
}
template<bool TopkWeightsPrefetch, bool FuseGateUp = false, int CombineQuantMode = COMBINE_NO_QUANT>
struct StoragePolicy {
    struct {
        const MegaMoeTilingData* tilingData;
        struct { uint8_t *flagSendCntCalToUpdParamsPtr, *cumsumInfoPtr; } workspaceInfo;
    } params_;
    struct { unsigned worldSize, moeExpertPerRank, activationFactor; } commonConfig_;
    unsigned sharedExpertNum_;
    BlockWorkspaceContext countWorkspace_;
"""
        + storage
        + r"""
};
int main() {
    using namespace MegaMoeImpl;
    // Six GMM1 N256 tiles per GELU M group and four rounds on 28 cores give 19 groups.
    const PlatformInfo platform{28, 56, 248*1024, 0, 0};
    for (int bs : {255, 256, 257, 4095, 4096}) {
        auto gelu = MakeTiling(bs, 4096, 1536, 25, 100, 8, bs, 0, true, false, 0, platform, 1, "gelu");
        assert(gelu.mGroupsPerWave == 19U);
        for (int early : {0, 1}) {
            auto swiglu = MakeTiling(bs, 4096, 3072, 25, 100, 8, bs, 0, true, false, 0, platform, early, "swiglu");
            assert(swiglu.mGroupsPerWave == (bs <= 256 ? 5U : 10U));
        }
    }
    // A six-row decode Combine ring ends at the table for H6144; prefill uses two rows.
    for (int bs : {256, 257, 2048}) for (int h : {4096, 6144, 6400, 8192}) for (int k : {25, 32}) {
        auto plan = MakeTiling(bs, h, 1536, k, 100, 8, bs, 0, true, false, 0, platform, 1, "gelu");
        assert(MteGeluLutFits(plan, platform.ub) == (bs > 256 || h <= 6144));
        assert(!MteGeluLutFits(plan, platform.ub - 1));  // Combine metadata no longer fits.
        plan.dispatchBufferConfig.routeItemsPerBatch = 65536;
        assert(!MteGeluLutFits(plan, platform.ub));  // The Dispatch index buffer reaches the table.
    }
    {
        auto plan = MakeTiling(1, 1024, 256, 1, 1, 8, 1, 0, true, false, 0, platform, 1, "gelu");
        const WorkspaceInfo original(nullptr, &plan);
        for (uint16_t words : {0, 640, 7928, 8192}) {
            plan.geluLutWords = words;
            const WorkspaceInfo sizes(nullptr, &plan);
            const int64_t padded = AlignUp<int64_t>(words * 4, 512);
            assert(sizes.workspaceSize == original.workspaceSize + padded);
            std::vector<uint8_t> memory(sizes.workspaceSize + 512, 0xcc);
            const WorkspaceInfo ws(memory.data(), &plan);
            assert(ws.geluLutPtr == (words ? memory.data() : nullptr));
            assert(ws.dispatchRevDataPtr == memory.data() + padded);
            std::fill_n(memory.data(), words * 4, 0xa1);
            std::fill_n(ws.dispatchRevDataPtr, original.workspaceSize, 0xb2);
            assert(std::all_of(memory.begin(), memory.begin() + words * 4, [](uint8_t v) {return v == 0xa1;}));
            assert(std::all_of(memory.begin() + sizes.workspaceSize, memory.end(), [](uint8_t v) {return v == 0xcc;}));
        }
    }
    auto t = MakeTiling(257, 4096, 1536, 25, 100, 8, 257, 0, true, false, 0, platform, 1, "gelu");
    const unsigned capacity = MteWorkspaceRowCapacity(t);
    assert(capacity == 3 * 19 * 256 && capacity < t.maxOutputSize);
    int32_t counts[800]{};
    publishedPrefix = &counts[799];
    uint8_t ready = 1;
    publishedFlag = &ready;
    // Simulate every physical block across launches, including ranks receiving no rows or a hot expert.
    for (unsigned rows : {0U, 6400U, capacity - 1, capacity, capacity + 1, t.maxOutputSize, 1U}) {
        *publishedPrefix = rows;
        for (unsigned block = 0; block < 28; ++block) {
            StoragePolicy<true> p{{&t, {&ready, reinterpret_cast<uint8_t*>(counts)}}, {8, 100, 1}, 0, {block, 28}};
            readyWaits = prefixReads = 0;
            p.SelectWaveStorage();
            assert(readyWaits == 1 && prefixReads == 1);
            assert(p.reuseWaveSlots_ == (rows > capacity));
            // Uneven expert counts can leave large gaps at slot boundaries. Both scheduler roles
            // must agree where a row lands when the wave wraps to slot zero.
            for (unsigned wave = 0; wave < 8; ++wave) {
                auto start = 137 * wave;
                auto map = p.WaveStorage(wave, start);
                auto expected = rows > capacity ? (wave % 3) * 19U * 256U : start;
                assert(MapWaveRow(map, start + 13) == expected + 13);
            }
        }
    }
    StoragePolicy<true> sw{{&t, {nullptr, nullptr}}, {8, 100, 2}, 0, {12, 28}};
    readyWaits = prefixReads = 0;
    sw.SelectWaveStorage();
    assert(sw.reuseWaveSlots_);
    sw.commonConfig_.activationFactor = 1;
    sw.sharedExpertNum_ = 1;
    sw.SelectWaveStorage();
    assert(sw.reuseWaveSlots_);
    sw.sharedExpertNum_ = 0;
    t.maxOutputSize = capacity;
    sw.SelectWaveStorage();
    assert(!sw.reuseWaveSlots_ && readyWaits == 0 && prefixReads == 0);
    for (int world : {8, 16, 32}) for (int topk : {25, 32}) for (bool w4 : {true, false}) {
        PeermemSizeParams params{256, topk, 4096, 800/world, world, 2, 1, true, false, TOPO_TYPE_MTE, true};
        const auto required = AlignUp<int64_t>(EXCEPTION_DUMP_REGION_SIZE + CalcPeermemLeastSize(params), 2LL << 20);
        assert(required < (1LL << 30));
        params.numMaxTokensPerRank = 39280;
        assert(CalcPeermemLeastSize(params) > (1LL << 30));
        int64_t previous = 0;
        for (int bs : {256, 112}) {
            auto t = MakeTiling(bs, 4096, 1536, topk, 800/world, world, 256, 0, w4, false, 0,
                                PlatformInfo{28, 56, 248*1024, 0, 0}, 1, "gelu");
            WorkspaceInfo info(nullptr, &t);
            assert(t.numMaxTokensPerRank == 256 && t.bs == bs);
            assert(t.maxOutputSize == 256 * world * std::min(topk, 800/world));
            assert(t.maxTilesPerExpert == unsigned(world * 6));
            assert(!UseDecodeRailRouting(&t));
            assert(info.workspaceSize > 0);
            if (previous) assert(info.workspaceSize == previous);
            previous = info.workspaceSize;
        }
    }
    // Sweep capacities on both sides of the transition to a larger compact-route batch.
    // Reconstruct the device SendAndQuantBuffInit layout, including both padded weight records.
    for (int topk : {25, 32}) {
        for (uint32_t ub = 160 * 1024; ub < 248 * 1024; ub += 32) {
            auto t = MakeTiling(2048, 1024, 256, topk, 25, 32, 2048, 0, true, false, 0,
                                PlatformInfo{28, 56, ub, 0, 0}, 1, "gelu");
            WorkspaceInfo info(nullptr, &t);
            auto reset = std::max(info.flagResetElementCount, info.gmm1TileStatusElementCount);
            auto resetBytes = 4 * AlignUp(std::min<int64_t>(CeilDiv<int64_t>(reset, 56), DISPATCH_RESET_BATCH),
                                          int64_t{INT32_PER_256B});
            uint32_t rowBytes = CalcQuantTokenScaleBytes(1024, 1, topk, true);
            assert(rowBytes == 1024 + 32 + 128);
            auto fixed = resetBytes + 2048 + 2 * rowBytes + 2 * 1024 * sizeof(uint16_t) +
                         AlignUp(CalcSendMaskExpertCapacity(&t, 56) * 4U, 32U);
            for (auto config : {t.sendMaskConfigForCoreWithExtraExpert, t.sendMaskConfigForCoreWithoutExtraExpert}) {
                auto end = fixed + 2 * config.routeItemsPerBatch * sizeof(int32_t) + config.bufferCount * config.bufferBytes;
                assert(end <= ub);
            }
        }
    }
}
""",
        encoding="utf-8",
    )
    executable = tmp_path / ("layout.exe" if sys.platform == "win32" else "layout")
    subprocess.run(
        [compiler, "-std=c++17", "-I", str(tmp_path), "-I", str(csrc), str(source), "-o", str(executable)],
        check=True,
    )
    subprocess.run([str(executable)], check=True, timeout=30)
