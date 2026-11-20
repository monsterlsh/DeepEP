"""Host regressions for weighted GELU rounding and the MTE BF16 factor table."""

from pathlib import Path
import shutil
import subprocess
import sys

import pytest


def test_weight_before_bf16_rounding(tmp_path):
    compiler = next((p for name in ("c++", "g++", "clang++") if (p := shutil.which(name))), None)
    if compiler is None:
        pytest.skip("The GELU register-order probe requires a C++17 host compiler")
    csrc = Path(__file__).resolve().parents[1] / "csrc/kernels"
    activation = (csrc / "mega_moe_common/epilogue/activation/activation_common.hpp").read_text(encoding="utf-8")
    helper = activation[activation.index("__simd_callee__ inline void ComputeGeluFactor") : activation.index(
        "template <typename InputType, bool FuseTopkWeight"
    )]
    # Run the device helper with scalar FP32 registers. The CANN Erf result below is the
    # FP32 value from its arch35 subsection polynomial for this particular lane.
    source = tmp_path / "gelu.cpp"
    source.write_text(
        r"""
#include <cassert>
#include <cstdint>
#include <cstring>
#define __simd_callee__
uint32_t bits(float x) { uint32_t b; std::memcpy(&b, &x, sizeof(b)); return b; }
namespace AscendC {
namespace Reg {
template <class T> using RegTensor = T;
using MaskReg = bool;
void Adds(float& dst, float a, float b, bool) { dst = a + b; }
void Muls(float& dst, float a, float b, bool) { dst = a * b; }
void Mul(float& dst, float a, float b, bool) { dst = a * b; }
}
namespace ErfAPI {
void ErfSubsectionCompute(float& dst, float scaled, bool) {
    assert(bits(scaled) == 0x3da8ff9fU);
    const uint32_t expected = 0x3dbe4330U;
    std::memcpy(&dst, &expected, sizeof(dst));
}
}
}
"""
        + helper
        + r"""
int main() {
    // All three lanes 142, 156, 184 of expert 8 for rank 6/token 10 share this GMM1 result.
    bool mask = true;
    float x = 0.11669921875F, weight = 0.002998440293595195F, y;
    WeightedGelu<true>(y, x, weight, mask);
    assert(bits(y) == 0x39487fffU);
    assert((bits(y) + 0x7fffU + ((bits(y) >> 16) & 1U)) >> 16 == 0x3948U);
    weight = -weight;
    WeightedGelu<true>(y, x, weight, mask);
    assert(bits(y) == 0xb9487fffU);
    WeightedGelu<false>(y, x, weight, mask);
    assert(bits(y) == 0x3d829a0cU);
}
""",
        encoding="utf-8",
    )
    executable = tmp_path / ("gelu.exe" if sys.platform == "win32" else "gelu")
    subprocess.run([compiler, "-std=c++17", str(source), "-o", str(executable)], check=True)
    subprocess.run([str(executable)], check=True, timeout=30)


def test_gelu_lookup_page_capacity(tmp_path):
    compiler = next((p for name in ("c++", "g++", "clang++") if (p := shutil.which(name))), None)
    if compiler is None:
        pytest.skip("The GELU table probe requires a C++17 host compiler")
    csrc = Path(__file__).resolve().parents[1] / "csrc"
    source = tmp_path / "pages.cpp"
    source.write_text(
        r"""
#include <cassert>
#include "gelu_lut_setup.hpp"

int main() {
    using deepep::megamoe::PackGeluLookupTable;
    std::vector<uint32_t> factors(65536);
    // 512 directory entries plus 60 distinct 128-word pages fill 32 KiB.
    for (unsigned i = 0; i < factors.size(); ++i) {
        const unsigned page = (i / 128) % 60, lane = i % 128;
        factors[i] = lane == 0 ? (page % 2) << 31 : 0x7fc00000U | (page << 7) | lane;
    }
    const auto packed = PackGeluLookupTable(factors);
    assert(packed.size() == 8192);
    for (unsigned i = 0; i < factors.size(); ++i) {
        assert(packed.at(packed.at(i / 128) + i % 128) == factors[i]);
    }
    // A single extra payload in a repeated page creates the 61st distinct page.
    factors.back() = 0x7f800001U;
    assert(PackGeluLookupTable(factors).empty());
    factors.assign(65536, 0U);
    assert(PackGeluLookupTable(factors).size() == 512 + 128);
    factors.pop_back();
    assert(PackGeluLookupTable(factors).empty());
}
""",
        encoding="utf-8",
    )
    executable = tmp_path / ("pages.exe" if sys.platform == "win32" else "pages")
    subprocess.run([compiler, "-std=c++17", "-I", str(csrc), str(source), "-o", str(executable)], check=True)
    subprocess.run([str(executable)], check=True, timeout=30)
