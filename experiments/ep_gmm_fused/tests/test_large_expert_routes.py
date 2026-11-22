# Copyright (c) 2026, Lu Lu

"""Exercise public route inputs and export dispatch without an NPU session."""

import importlib.metadata
import importlib.util
from pathlib import Path
import shutil
import struct
import subprocess
import sys
from types import SimpleNamespace

import pytest


COMPONENT = Path(__file__).resolve().parents[1]


@pytest.fixture
def api(monkeypatch):
    torch = pytest.importorskip("torch")
    path = COMPONENT / "python/deep_ep_experimental/ep_gmm_fused"
    name = "ep_gmm_fused_route_test"
    version = importlib.metadata.version
    monkeypatch.setattr(importlib.metadata, "version", lambda project: "0.0" if project == "ascend-deepep" else version(project))
    spec = importlib.util.spec_from_file_location(name, path / "__init__.py", submodule_search_locations=[str(path)])
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    ops = importlib.import_module(f"{name}.ops")
    monkeypatch.setattr(ops, "get_udma_config", lambda: SimpleNamespace(world_size=8))
    yield torch, module
    for loaded in tuple(sys.modules):
        if loaded.startswith(name + "."):
            sys.modules.pop(loaded)


@pytest.mark.parametrize("local_experts", [64, 65, 100, 128])
@pytest.mark.parametrize("route_kind", ["list", "cpu", "device"])
@pytest.mark.parametrize("preallocated", [False, True])
def test_export_selects_descriptor_by_capacity(api, monkeypatch, local_experts, route_kind, preallocated):
    torch, package = api
    rank, ep_size = 5, 8
    expert_num = ep_size * local_experts
    rows = expert_num + 41
    local, incoming = package.build_rank_route_tables(rows, rank, ep_size, expert_num)
    if route_kind == "device":
        args = package.build_rank_route(rows, rank, ep_size, expert_num, "meta")
        local, incoming, explicit_received = args
        device = "meta"  # Exercise same-device tensor routing without data or D2H reads.
    else:
        explicit_received = None
        device = "cpu"
        if route_kind == "cpu":
            local = torch.tensor(local, dtype=torch.int64)
            incoming = torch.tensor(incoming, dtype=torch.int64).view(ep_size, local_experts)
    x = torch.empty((rows, 2), dtype=torch.bfloat16, device=device)
    weight = torch.empty((local_experts, 2, 3), dtype=x.dtype, device=device)
    out = torch.empty((rows, 3), dtype=x.dtype, device=device) if preallocated else None
    export = torch.empty_like(x) if preallocated else None

    launches = []

    def launch(x_arg, weight_arg, send, receive, world, experts, received, metadata, *outputs):
        assert x_arg is x and weight_arg is weight
        assert (world, experts, received) == (ep_size, expert_num, rows)
        for table in (send, receive):
            assert table.dtype == torch.int64 and table.device == x.device and table.is_contiguous()
            assert list(table.shape) == [expert_num]
        if route_kind != "device":
            expected_local, expected_incoming = package.build_rank_route_tables(rows, rank, ep_size, expert_num)
            assert send.tolist() == expected_local and receive.tolist() == expected_incoming
        if route_kind != "device" and local_experts == 64:
            assert metadata.dtype == torch.uint8 and metadata.numel() == 12880
            wire = bytes(metadata.tolist())
            assert struct.unpack_from("<IIII", wire) == (ep_size, local_experts, expert_num, 0)
            source_prefix, output_prefix, segment_rows, source_rank, expert = struct.unpack_from(
                "<QQIHH", wire, 16 + 8 * 8 + 64 * 8 + (expert_num - 1) * 24
            )
            assert (source_rank, expert) == (ep_size - 1, local_experts - 1)
            assert segment_rows == expected_incoming[-1]
            assert source_prefix == sum(expected_incoming[-local_experts:-1])
            assert output_prefix == received - segment_rows
        else:
            assert metadata is None
        assert len(outputs) == (2 if preallocated else 1)
        assert outputs[-1].shape == x.shape
        launches.append(True)
        return (outputs[0] if preallocated else torch.empty((received, 3), dtype=x.dtype, device=x.device), outputs[-1])

    monkeypatch.setattr(torch.ops, "deep_ep_ep_gmm_fused", SimpleNamespace(
        all_to_allv_grouped_mat_mul_udma_export=launch,
        all_to_allv_grouped_mat_mul_udma_out_export=launch,
    ))
    result, dispatched = package.alltoallv_gmm(
        x, weight, local, incoming, ep_size=ep_size, expert_num=expert_num, output_all_to_allv=True,
        received_rows=explicit_received, out=out, all_to_allv_out=export,
    )
    assert launches == [True] and result.shape == (rows, 3) and dispatched.shape == x.shape
    if preallocated:
        assert result is out and dispatched is export


def test_ep8_e800_receive_validation_and_inverse_output(api, monkeypatch):
    torch, package = api
    world, experts, rows = 8, 800, 845
    local, incoming = package.build_rank_route_tables(rows, 3, world, experts)
    x = torch.empty((rows, 2), dtype=torch.bfloat16)
    weight = torch.empty((100, 2, 3), dtype=x.dtype)

    def inverse(x, w, send, receive, ep, count, out):
        assert (ep, count) == (world, experts)
        assert (send.tolist(), receive.tolist()) == (local, incoming)
        assert out.shape == (rows, 3)
        return out

    monkeypatch.setattr(torch.ops, "deep_ep_ep_gmm_fused", SimpleNamespace(grouped_mat_mul_all_to_allv_udma_out=inverse))
    assert package.gmm_alltoallv(x, weight, local, incoming, ep_size=world, expert_num=experts).shape == (rows, 3)
    with pytest.raises(ValueError, match="received_rows does not match"):
        package.alltoallv_gmm(x, weight, local, incoming, ep_size=world, expert_num=experts,
                              received_rows=rows - 1, output_all_to_allv=True)
    with pytest.raises(ValueError, match="fit uint32"):
        package.alltoallv_gmm(x, weight, local, [1 << 32, *incoming[1:]], ep_size=world, expert_num=experts,
                              output_all_to_allv=True)
    with pytest.raises(ValueError, match="128"):
        package.build_global_route_matrix(rows, world, world * 129)
    for operator in (package.alltoallv_gmm, package.gmm_alltoallv):
        with pytest.raises(ValueError, match="128"):
            operator(x, weight, [], [], ep_size=world, expert_num=world * 129)


def test_native_compact_expert_schedulers(tmp_path):
    compiler = shutil.which("g++") or shutil.which("clang++")
    if compiler is None:
        pytest.skip("C++17 compiler required for CatCCOS scheduler tests")
    # The fixture supplies coordinate operations and core IDs only. Scheduler state,
    # route parsing, row prefixes, and specialization dispatch come from production.
    stub = tmp_path / "device_types.hpp"
    stub.write_text(
        r'''#pragma once
#include <cstdint>
#define CATLASS_HOST_DEVICE
#define CATLASS_DEVICE
#define __gm__
using GM_ADDR = uint8_t*;
namespace AscendC {
inline uint32_t GetBlockIdx() { return 0; }
inline uint32_t GetBlockNum() { return 8; }
}
template <typename T> T Min(T a, T b) { return a < b ? a : b; }
template <typename T> T Max(T a, T b) { return a > b ? a : b; }
inline uint32_t ClipSub(uint32_t a, uint32_t b) { return a >= b ? a - b : 0; }
inline uint32_t CeilDiv(uint32_t a, uint32_t b) { return a / b + (a % b != 0); }
namespace Catlass {
struct MatrixCoord {
    uint32_t r, c;
    uint32_t row() const { return r; }
    uint32_t column() const { return c; }
    MatrixCoord operator+(MatrixCoord b) const { return {r + b.r, c + b.c}; }
};
struct GemmCoord {
    uint32_t mr, nc, kc;
    uint32_t m() const { return mr; }
    uint32_t n() const { return nc; }
    uint32_t k() const { return kc; }
    MatrixCoord GetCoordMN() const { return {mr, nc}; }
};
inline GemmCoord Min(GemmCoord a, GemmCoord b) { return {::Min(a.m(), b.m()), ::Min(a.n(), b.n()), ::Min(a.k(), b.k())}; }
inline GemmCoord ClipSub(GemmCoord a, GemmCoord b) {
    return {::ClipSub(a.m(), b.m()), ::ClipSub(a.n(), b.n()), ::ClipSub(a.k(), b.k())};
}
inline GemmCoord CeilDiv(GemmCoord a, GemmCoord b) {
    return {::CeilDiv(a.m(), b.m()), ::CeilDiv(a.n(), b.n()), ::CeilDiv(a.k(), b.k())};
}
inline uint32_t Numel(GemmCoord a) { return a.m() * a.n() * a.k(); }
template <typename T> MatrixCoord MakeCoord(T a, T b) { return {a, b}; }
}
namespace Catccos {
struct DistGemmCoord {
    uint32_t mr, nc, kc, r;
    uint32_t m() const { return mr; }
    uint32_t n() const { return nc; }
    uint32_t rank() const { return r; }
    Catlass::MatrixCoord GetCoordMK() const { return {mr, kc}; }
    Catlass::MatrixCoord GetCoordMN() const { return {mr, nc}; }
    Catlass::GemmCoord GetCoordMNK() const { return {mr, nc, kc}; }
};
struct DistMatrixCoord { Catlass::MatrixCoord coord; uint32_t rank; };
template <uint32_t, uint32_t> struct MatrixSwizzle {
    static Catlass::MatrixCoord GetCoord(Catlass::MatrixCoord grid, uint32_t i) {
        return {i / grid.column(), i % grid.column()};
    }
};
}
namespace Catlass {
template <typename T> Catccos::DistGemmCoord MakeCoord(T a, T b, T c, T d) { return {a, b, c, d}; }
}
''',
        encoding="utf-8",
    )
    for header in ("catlass/catlass.hpp", "catlass/gemm_coord.hpp", "catlass/matrix_coord.hpp",
                   "catccos/catccos.hpp", "catccos/dist_coord.hpp"):
        path = tmp_path / header
        path.parent.mkdir(exist_ok=True)
        path.write_text('#include "device_types.hpp"\n', encoding="utf-8")

    source = tmp_path / "schedulers.cpp"
    source.write_text(
        r'''#include <cassert>
#include <initializer_list>
#include "expert_capacity.hpp"
#include "catccos/dgemm/moe_route_metadata.hpp"
#include "catccos/dgemm/block/block_scheduler_alltoallv_allgather.hpp"
#include "catccos/dgemm/block/block_scheduler_gmm_alltoallv_target_order.hpp"
using namespace Catccos::DGemm;
using namespace Catccos::DGemm::Block;
using namespace deepep::ep_gmm_fused;

template <uint32_t LocalExperts> struct Config {
    using MoeConstraints = Catccos::DGemm::MoeConstraints<1, 8, LocalExperts>;
    template <uint32_t C> using WithExpertCapacity = Config<C>;
};

int main() {
    static_assert(sizeof(MoeRouteMetadata) == 12880);
    using Large = MoeConstraints<1, 8, kMaxLocalExperts>;
    using Forward = BlockMmadSchedulerAllToAllVAllGather<Large, 1, true>;
    using ForwardWide = BlockMmadSchedulerAllToAllVAllGather<Large>;
    using Inverse = BlockMmadSchedulerGmmAllToAllVTargetOrder<Large, true>;
    using InverseWide = BlockMmadSchedulerGmmAllToAllVTargetOrder<Large>;
    static_assert(sizeof(Forward) < 18 * 1024 && sizeof(Inverse) < 26 * 1024);
    static_assert(sizeof(InverseWide) > 32 * 1024);
    auto pick = [](uint32_t local) {
        uint32_t chosen = 0;
        LaunchWithExpertCapacity(TypeTag<Config<kForwardSmallLocalExperts>>{}, local, [&](auto tag) {
            chosen = decltype(tag)::type::MoeConstraints::LOCAL_EXPERT_NUM_LIMIT;
        });
        return chosen;
    };
    assert(pick(64) == 64 && pick(65) == 128 && pick(100) == 128 && pick(128) == 128);

    for (uint32_t experts : {65U, 100U, 128U}) {
        constexpr uint32_t world = 8;
        int64_t counts[128 * world];
        for (uint32_t slot = 0; slot < experts * world; ++slot) counts[slot] = slot % 5 + 1;
        auto* route = reinterpret_cast<GM_ADDR>(counts);
        AllToAllVAllGatherProblemShape shape{{512, 4096, 8192}, world, 5, world, experts * world, route, route};
        Forward forward{shape, 64, {128, 256}};
        ForwardWide forwardWide{shape, 64, {128, 256}};
        Inverse inverse{shape, 32, {128, 256}};
        InverseWide inverseWide{shape, 32, {128, 256}};
        for (uint32_t step = 0; step < forward.GetCommLoops(); ++step) {
            forward.UpdateCommContext(step);
            forwardWide.UpdateCommContext(step);
            for (uint32_t expert = 0; expert < experts; ++expert) {
                for (uint32_t rank = 0; rank < world; ++rank) {
                    Catccos::DistGemmCoord pos{0, 256, 0, rank};
                    assert(forward.GetRemapperA(step, expert)(pos).coord.row() ==
                           forwardWide.GetRemapperA(step, expert)(pos).coord.row());
                    assert(forward.GetRemapperC(step, expert)(pos).row() ==
                           forwardWide.GetRemapperC(step, expert)(pos).row());
                }
            }
        }
        for (uint32_t step = 0; step < inverse.GetCommLoops(); ++step) {
            inverse.UpdateCommContext(step);
            inverseWide.UpdateCommContext(step);
            for (uint32_t expert = 0; expert < experts; ++expert) {
                for (uint32_t rank = 0; rank < world; ++rank) {
                    Catccos::DistGemmCoord pos{0, 256, 0, rank};
                    auto tail = inverse.GetRemapperA(step, expert).GetResidueShape(pos);
                    auto wideTail = inverseWide.GetRemapperA(step, expert).GetResidueShape(pos);
                    assert(tail.m() == counts[rank * experts + expert]);
                    assert(tail.m() == wideTail.m() && tail.n() == wideTail.n() && tail.k() == wideTail.k());
                    if (inverse.commContext.groupNumList[rank][expert] != 0) {
                        pos.mr = inverse.commContext.finishedGroupList[rank][expert] * 128;
                        assert(inverse.GetRemapperC(step, expert)(pos).coord.row() ==
                               inverseWide.GetRemapperC(step, expert)(pos).coord.row());
                    }
                }
            }
        }
    }

    int64_t wideCounts[128 * 8]{};
    wideCounts[0] = 1'500'000'000;
    wideCounts[128] = 1'000'000'000;
    wideCounts[127 + 7 * 128] = 2;
    auto* wideRoute = reinterpret_cast<GM_ADDR>(wideCounts);
    AllToAllVAllGatherProblemShape wideShape{{2'500'000'002U, 4096, 8192}, 8, 5, 8, 1024, wideRoute, wideRoute};
    Forward forward{wideShape, 64, {128, 256}};
    auto row = forward.GetRemapperC(0, 127)({0, 256, 0, 7}).row();
    assert(row == 2'500'000'000U);
    assert(uint64_t{row} * wideShape.n() > UINT32_MAX);
}
''',
        encoding="utf-8",
    )
    binary = tmp_path / ("schedulers.exe" if sys.platform == "win32" else "schedulers")
    result = subprocess.run(
        [compiler, "-std=c++17", "-Wall", "-Wextra", "-Werror", f"-I{tmp_path}", f"-I{COMPONENT / 'csrc'}",
         f"-I{COMPONENT / 'csrc/catccos_overlay/include'}", str(source), "-o", str(binary)],
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stderr
    result = subprocess.run([str(binary)], capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
