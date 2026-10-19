# Copyright (c) 2026, Lu Lu
# Modified by SimpleBright_Man 2026

"""Compile the production host/device policies without Torch, CANN or an NPU."""

import os
from pathlib import Path
import shutil
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[3]
CSRC = ROOT / "experiments/megamoe/csrc"


def compile_and_run(tmp_path, source):
    compiler = os.environ.get("CXX") or shutil.which("c++") or shutil.which("g++") or shutil.which("clang++")
    if not compiler:
        pytest.skip("A C++17 compiler is required; set CXX to its executable path")
    path = tmp_path / "policy.cpp"
    path.write_text(source, encoding="utf-8")
    executable = tmp_path / ("policy.exe" if os.name == "nt" else "policy")
    built = subprocess.run(
        [compiler, "-std=c++17", "-Wall", "-Wextra", "-Werror", "-I", str(CSRC), str(path), "-o", str(executable)],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert built.returncode == 0, built.stdout + built.stderr
    result = subprocess.run([str(executable)], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr


def test_send_route_buffers_stay_within_actual_ub(tmp_path):
    compile_and_run(
        tmp_path,
        r"""
#include <cassert>
#include <cstdint>
#include <initializer_list>
#include "send_route_policy.hpp"
using namespace deepep::megamoe;
int main() {
    // Previous 12288-item minimum overflowed the usable 248KiB UB.
    assert(SendRouteUbBytes(12288, 1, 59968, 2) == 259648);
    const auto fitted = FitSendRouteBatch(16384, 1, 59968, 2, 253952);
    assert(fitted == 11776);
    assert(SendRouteUbBytes(fitted, 1, 59968, 2) == 251328);
    // Smaller EP8 / E32 configuration also exposed the same under-budgeting.
    assert(SendRouteUbBytes(12288, 1, 56000, 2) > 253952);
    assert(FitSendRouteBatch(256, 1, 59968, 2, 60000) == 0);
    assert(FitSendRouteBatch(256, 0, 0, 2, 253952) == 0);
    for (uint32_t topk = 1; topk <= 32; ++topk) {
        for (uint32_t slots = 2; slots <= 6; ++slots) {
            for (uint32_t fixed : {0U, 2048U, 59968U, 81920U}) {
                for (uint32_t ub : {196608U, 253952U, 262144U}) {
                    for (uint32_t limit : {256U, 512U, 12288U, 16384U, 2097152U}) {
                        const uint32_t batch = FitSendRouteBatch(limit, topk, fixed, slots, ub);
                        assert(batch > 0 && batch % 256 == 0 && batch <= limit);
                        assert(SendRouteUbBytes(batch, topk, fixed, slots) <= ub);
                        if (batch < limit) {
                            assert(SendRouteUbBytes(batch + 256, topk, fixed, slots) > ub);
                        }
                        // An arbitrary route offset may straddle both boundary tokens.
                        for (uint32_t offset = 0; offset < topk; ++offset) {
                            const uint32_t touched = (offset + batch - 1) / topk + 1;
                            assert(batch / 8 + touched * sizeof(int32_t) <= SendRouteSlotBytes(batch, topk));
                        }
                    }
                }
            }
        }
    }
}
""",
    )


def test_gmm1_ub_subtiles_cover_rows_and_publish_once(tmp_path):
    compile_and_run(
        tmp_path,
        r"""
#include <array>
#include <cassert>
#include "kernels/mega_moe/common/mega_moe_a8w8_policy.hpp"
using namespace MegaMoeImpl;
static_assert(A8W8_GMM1_UB_INPUT_BYTES == 128U * 1024U);
// The old up buffer started at 128KiB: row 241 already exceeded usable UB.
static_assert(256U * 256U * 2U + 240U * 256U * 2U == 248U * 1024U);
static_assert(256U * 256U * 2U + 241U * 256U * 2U > 248U * 1024U);
int main() {
    for (uint32_t rows = 1; rows <= 256; ++rows) {
        std::array<uint32_t, 256> visits{};
        uint32_t publications = 0;
        uint32_t completed = 0;
        for (uint32_t offset = 0; offset < rows; offset += A8W8_GMM1_UB_TILE_M) {
            const auto count = Gmm1UbSubtileRows(rows, offset, A8W8_GMM1_UB_TILE_M);
            assert(count > 0 && count <= A8W8_GMM1_UB_TILE_M);
            const auto upEnd = (A8W8_GMM1_UB_TILE_M + count) * A8W8_GMM1_UB_TILE_N * 2U;
            assert(upEnd <= 248U * 1024U);
            for (uint32_t row = offset; row < offset + count; ++row) {
                assert(row < rows);
                ++visits[row];
            }
            completed += count;
            if (IsLastGmm1UbSubtile(rows, offset, count)) {
                assert(completed == rows);  // GMM2 cannot observe an incomplete M256 group.
                ++publications;
            } else {
                assert(completed < rows);
            }
        }
        assert(publications == 1 && completed == rows);
        for (uint32_t row = 0; row < rows; ++row) assert(visits[row] == 1);
        assert(Gmm1UbSubtileRows(rows, rows, A8W8_GMM1_UB_TILE_M) == 0);
        assert(!IsLastGmm1UbSubtile(rows, rows, 0));
    }
}
""",
    )
