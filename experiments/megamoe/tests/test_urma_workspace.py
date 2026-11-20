"""Compile the native workspace layout without a Torch or CANN runtime."""

from pathlib import Path
import shutil
import subprocess
import sys

import pytest


def test_registered_windows_and_local_scratch(tmp_path):
    compiler = next((path for name in ("c++", "g++", "clang++") if (path := shutil.which(name))), None)
    if compiler is None:
        pytest.skip("This test requires a host C++17 compiler")
    csrc = Path(__file__).resolve().parents[1] / "csrc"
    source = tmp_path / "urma_workspace.cpp"
    source.write_text(
        r"""
#include <cassert>
#include <vector>
#include "urma_workspace.hpp"

using namespace deepep::megamoe;
using namespace MegaMoeUrma;

uint64_t check(int world, int tokens, int k, int h = 4096, int i = 1536)
{
    const auto layout = MakeUrmaSymmetricLayout(world, tokens, k, h, UrmaCvRows(world, tokens), 1, true);
    const auto& c = layout.config;
    const uint64_t sent = uint64_t(tokens) * k, recv = world * sent;
    assert(c.sourceRowCapacity == sent);
    assert(c.receiveXOffset == 0 && c.receiveScaleOffset >= recv * h);
    assert(c.returnYOffset >= c.receiveScaleOffset + recv * c.dispatchScaleStride);
    assert(c.dispatchReadyOffset >= c.returnYOffset + sent * h * 2);
    const uint64_t flags = world * ((sent + 127) / 128) * kFlagBytes;
    assert(c.combineReadyOffset >= c.dispatchReadyOffset + flags);
    assert(c.barrierOffset >= c.combineReadyOffset + flags);
    assert(c.routeInboxOffset >= c.barrierOffset + world * 4 * kFlagBytes);
    assert(c.routeEntryOffset >= c.routeInboxOffset + world * kRouteMailboxBytes);
    assert(c.routeReadyOffset >= c.routeEntryOffset + world * kFlagBytes);
    const auto endFlags = c.routeReadyOffset + world * kFlagBytes;
    assert(c.symmetricBytes % (2 << 20) == 0 && c.symmetricBytes >= endFlags);
    if (world == 32) {
        assert(c.routeOutboxOffset >= endFlags);
        assert(c.epochSourceOffset >= c.routeOutboxOffset + world * kRouteMailboxBytes);
        assert(layout.expandedX >= c.epochSourceOffset + kFlagBytes);
        assert(layout.expandedScale >= layout.expandedX + sent * h);
        assert(layout.gmm2Output >= layout.expandedScale + sent * c.dispatchScaleStride);
        assert(layout.gmm2Output + recv * h * 2 <= c.symmetricBytes);
    }
    // Map local allocations above symmetric memory, without allocating GB of RAM.
    Arguments a{};
    a.config = c;
    a.config.localExperts = 800 / world;
    a.config.intermediate = i;
    a.symmetric = 8192;
    a.config.windowedTransport = 1;
    uint64_t localBase = a.symmetric + c.symmetricBytes;
    uint64_t scratch = 0;
    BindUrmaScratch(a, layout, tokens, 4096, [&](uint64_t bytes) {
        assert(bytes && bytes % 512 == 0);
        uint64_t address = localBase + scratch;
        scratch += bytes;
        return address;
    });
    for (auto address : {a.expandedX, a.expandedScale, a.gmm2Output})
        assert(address >= a.symmetric && (address < localBase) == (world == 32));
    assert(a.expertRoutes >= localBase && a.activation >= localBase && a.systemWorkspace >= localBase);
    const bool target = (k == 25 || k == 32) && h == 4096 && i == 1536;
    assert(ReserveUrmaGeluLookup(a) == target);
    assert(a.geluLookupWords == 0 && (a.geluLookup != 0) == target);
    if (target) {
        assert(a.geluLookup >= localBase && a.geluLookup + WindowGeluUb::tableBytes <= localBase + scratch);
        assert(WindowGeluUb::tableBytes == 32768);
        // Replay the preflight counting visitor and the full-batch fallback.
        uint64_t estimate = 0;
        Arguments planned = a;
        BindUrmaScratch(planned, layout, tokens, 4096, [&](uint64_t n) { estimate += n; return uint64_t{0}; });
        assert(estimate == scratch);
        planned.config.windowedTransport = 0;
        uint64_t withoutLut = 0;
        BindUrmaScratch(planned, layout, tokens, 4096, [&](uint64_t n) { withoutLut += n; return uint64_t{0}; });
        assert(estimate == withoutLut + WindowGeluUb::tableBytes);
    }
    const uint64_t payload = recv * (uint64_t(h) + c.dispatchScaleStride + uint64_t(i) + i / 32 + 2ULL*h + h / 32 + 32)
                           + sent * (3ULL*h + c.dispatchScaleStride + 800 * 4 + sizeof(int64_t));
    assert(c.symmetricBytes + scratch >= payload);
    return c.symmetricBytes + scratch;
}

int main()
{
    for (int world : {8, 16, 32}) {
        for (int k : {1, 25, 32}) {
            auto w64 = check(world, 64, k), w128 = check(world, 128, k), w256 = check(world, 256, k);
            assert(w64 < w128 && w128 < w256);
        }
    }
    assert(check(32, 256, 32, 8192, 16384) > (4ULL << 30));
    for (auto shape : {std::vector<int>{4, 256, 1}, {32, 257, 25}, {32, 0, 25}, {32, 128, 33}}) {
        try { MakeUrmaSymmetricLayout(shape[0], shape[1], shape[2], 4096, 128, 1, true); assert(false); }
        catch (const std::invalid_argument&) {}
    }
    const auto old = MakeUrmaSymmetricLayout(16, 16384, 8, 4096, 256, 0, false);
    assert(old.config.sourceRowCapacity == 131072U && old.config.dispatchScaleStride == 128);
    try { MakeUrmaSymmetricLayout(32, 128, 8, 4096, 128, 0, false); assert(false); }
    catch (const std::invalid_argument&) {}
}
""",
        encoding="utf-8",
    )
    executable = tmp_path / ("layout.exe" if sys.platform == "win32" else "layout")
    subprocess.run([compiler, "-std=c++17", "-I", str(csrc), str(source), "-o", str(executable)], check=True)
    subprocess.run([str(executable)], check=True, timeout=30)


def test_window_gelu_mailbox_and_ready(tmp_path):
    compiler = next((path for name in ("c++", "g++", "clang++") if (path := shutil.which(name))), None)
    if compiler is None:
        pytest.skip("The GELU mailbox probe requires a host C++17 compiler")
    tests = Path(__file__).resolve().parent
    csrc = tests.parent / "csrc"
    source = (csrc / "kernels/mega_moe_urma/stage/mega_moe_gmm.hpp").read_text(encoding="utf-8")
    begin = source.index("    template <bool Early, bool Lookup")
    end = source.index("\n    }", source.index("timer.End(Phase::Activation);", begin)) + len("\n    }")
    (tmp_path / "routed_activation.inc").write_text(source[begin:end], encoding="utf-8")
    executable = tmp_path / ("activation.exe" if sys.platform == "win32" else "activation")
    subprocess.run([compiler, "-std=c++17", "-I", str(csrc), "-I", str(tmp_path),
                    str(tests / "urma_window_gelu.cpp"), "-o", str(executable)], check=True)
    subprocess.run([str(executable)], check=True, timeout=30)
