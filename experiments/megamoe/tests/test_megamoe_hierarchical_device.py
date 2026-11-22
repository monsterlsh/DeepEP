# Copyright (c) 2026, Lu Lu

"""Compile the hierarchical device protocol with host-side event and SHMEM doubles."""

from pathlib import Path
import shutil
import subprocess

import pytest


FIXTURES = Path(__file__).with_name("hierarchical_device")
KERNEL = FIXTURES.parent.parent / "csrc/kernels/mega_moe_hierarchical"


@pytest.fixture(scope="module")
def compiler():
    command = shutil.which("g++") or shutil.which("clang++")
    if command is None:
        pytest.skip("Hierarchical device tests require g++ or clang++")
    return command


def compile_and_run(compiler, source, binary, include_directories, flags):
    command = [compiler, "-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror", *flags]
    command += [f"-I{path}" for path in include_directories]
    result = subprocess.run([*command, str(source), "-o", str(binary)], capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    result = subprocess.run([str(binary)], capture_output=True, text=True, timeout=45)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("backend", ["HNS_1825", "XSCALE"])
def test_hierarchical_dispatch_and_qp_completion(compiler, tmp_path, backend):
    compile_and_run(
        compiler,
        FIXTURES / "hierarchical_dispatch_test.cpp",
        tmp_path / "dispatch.exe",
        [FIXTURES / "dispatch_stubs", FIXTURES / "roce_stubs", FIXTURES / "layout_stubs"],
        [f"-DACLSHMEMI_RDMA_K_BACKEND_{backend}=1"],
    )


def test_hns_ready_progress_tracks_cqes_and_completed_wqes(compiler, tmp_path):
    compile_and_run(
        compiler,
        FIXTURES / "roce_progress_test.cpp",
        tmp_path / "progress.exe",
        [FIXTURES / "roce_progress_stubs"],
        ["-DACLSHMEMI_RDMA_K_BACKEND_HNS_1825=1"],
    )


@pytest.mark.parametrize("hierarchical", [False, True], ids=["mte", "mte-roce"])
def test_shared_headers_use_device_namespace(compiler, tmp_path, hierarchical):
    flags = ["-DDEEPEP_MEGAMOE_MTE_ROCE=1"]
    if hierarchical:
        flags.append("-DDEEPEP_MEGAMOE_HIERARCHICAL_IMPL=1")
    compile_and_run(
        compiler,
        FIXTURES / "hierarchical_shared_headers_test.cpp",
        tmp_path / "shared_headers.exe",
        [KERNEL.parent.parent],
        flags,
    )


def device_function(source, name):
    start = source.index(f"__aicore__ inline void {name}(")
    opening = source.index("{", start)
    end, depth = opening + 1, 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return source[start:end]


def test_hierarchical_weighted_gateway_reduction_and_unpermute(compiler, tmp_path):
    dispatch = (KERNEL / "stage/mega_moe_hierarchical_dispatch.hpp").read_text(encoding="utf-8")
    fixture = (FIXTURES / "hierarchical_reduce_test.cpp").read_text(encoding="utf-8")
    source = tmp_path / "hierarchical_reduce.cpp"
    source.write_text(
        fixture.replace("/* PRODUCTION_WEIGHT_PACKER */", device_function(dispatch, "PackHierarchicalCombineWeights")),
        encoding="utf-8",
    )
    compile_and_run(
        compiler,
        source,
        tmp_path / "reduce.exe",
        [FIXTURES, FIXTURES / "layout_stubs", FIXTURES / "combine_vf_stubs"],
        ["-ffp-contract=off", "-fno-strict-aliasing"],
    )


def test_hierarchical_target_host_tiling_and_arena(compiler, tmp_path):
    compile_and_run(
        compiler,
        FIXTURES / "hierarchical_tiling_test.cpp",
        tmp_path / "tiling.exe",
        [FIXTURES / "layout_stubs"],
        [],
    )
