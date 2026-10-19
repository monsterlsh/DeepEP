"""Compile production GMM1 template declarations, without compiling CANN kernels.

Only SDK argument types are stand-ins. The epilogue template parameters, its
public tile constant, and both GMM1 overload declarations come from production
headers so a missing FP4-expansion parameter cannot be hidden by a copied test
signature. Device function bodies still require the real CANN/NPU build.
"""

import os
from pathlib import Path
import re
import shutil
import subprocess

import pytest


CSRC = Path(__file__).resolve().parents[3] / "experiments/megamoe/csrc"


def test_gmm1_overloads_accept_fp4_expansion(tmp_path):
    compiler = os.environ.get("CXX") or shutil.which("c++") or shutil.which("g++") or shutil.which("clang++")
    if not compiler:
        pytest.skip("A C++17 compiler is required; set CXX to its executable path")
    stage = (CSRC / "kernels/mega_moe/stage/mega_moe_gmm1_activation.hpp").read_text(encoding="utf-8")
    epilogue = (CSRC / "kernels/mega_moe/epilogue/block_epilogue_activation_mx_quant.hpp").read_text(encoding="utf-8")
    signatures = re.findall(
        r"template <typename ElementA, typename EpilogueElementA,[\s\S]*?"
        r"\n__aicore__ inline void RunGmm1Generic\([\s\S]*?\n\{",
        stage,
    )
    assert len(signatures) == 2, "Expected both production RunGmm1Generic overloads"
    class_head = re.search(r"template <typename DataTypeOut_,[\s\S]*?class BlockEpilogueActivationMxQuant \{", epilogue)
    tile_constant = re.search(r"static constexpr uint32_t TILE_M = TileM;", epilogue)
    assert class_head is not None and tile_constant is not None, "Missing production epilogue tile contract"
    source = r"""
#include <cstdint>
#include <type_traits>
#include <utility>
#define __aicore__
constexpr uint32_t L1_TILE_M_256 = 256, L1_TILE_N = 256;
namespace AscendC { template <typename... T> struct Shape {}; }
struct Params {};
struct GMMAddrInfo {};
struct BlockJobContext {};
struct Fp8 {};
struct Bf16 {};
struct Scale {};
"""
    source += class_head.group() + "\npublic:\n" + tile_constant.group() + "\n};\n"
    source += "\n".join(signature[:-1] + ";" for signature in signatures)
    source += r"""
template <bool Expanded, bool Prefetch, bool Interleaved> void Check() {
    using E = BlockEpilogueActivationMxQuant<Fp8, Bf16, 128, 256, Prefetch, Interleaved, Expanded>;
    static_assert(E::TILE_M == 128);
    using Shape = AscendC::Shape<int64_t, int64_t, int64_t, int64_t>;
    // Use the production wrapper's explicit arguments; expansion must be deduced.
    using WithJob = decltype(RunGmm1Generic<Fp8, Fp8, Fp8, Bf16, Scale, Scale,
        false, 256, 128, Prefetch, false, Interleaved, true>(
        std::declval<E&>(), std::declval<const Params&>(), std::declval<const Shape&>(),
        std::declval<const GMMAddrInfo&>(), std::declval<uint32_t&>(), std::declval<int32_t&>(),
        std::declval<const BlockJobContext&>(), uint32_t{}, uint32_t{}, std::declval<uint16_t&>()));
    using WithoutJob = decltype(RunGmm1Generic<Fp8, Fp8, Fp8, Bf16, Scale, Scale,
        false, 256, 128, Prefetch, false, Interleaved, true>(
        std::declval<E&>(), std::declval<const Params&>(), std::declval<const Shape&>(),
        std::declval<const GMMAddrInfo&>(), std::declval<uint32_t&>(), std::declval<int32_t&>(),
        uint32_t{}, uint32_t{}, std::declval<uint16_t&>()));
    static_assert(std::is_same_v<WithJob, void> && std::is_same_v<WithoutJob, void>);
}
"""
    for expanded in ("false", "true"):
        for prefetch in ("false", "true"):
            for interleaved in ("false", "true"):
                source += f"template void Check<{expanded}, {prefetch}, {interleaved}>();\n"
    path = tmp_path / "gmm1_template_contract.cpp"
    path.write_text(source, encoding="utf-8")
    result = subprocess.run(
        [compiler, "-std=c++17", "-Wall", "-Wextra", "-Werror", "-fsyntax-only", str(path)],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
