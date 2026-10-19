"""Compare shared native FP4 thresholds against an independent nearest-value oracle."""

import os
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]


def test_native_fp4_expansion_all_bf16_bit_patterns(tmp_path):
    compiler = os.environ.get("CXX") or shutil.which("c++") or shutil.which("g++") or shutil.which("clang++")
    if not compiler:
        pytest.skip("A C++17 compiler is required; set CXX to its executable path")
    source = tmp_path / "fp4_policy.cpp"
    source.write_text(
        r"""
#include <cstdint>
#include <cstdio>
#include "kernels/mega_moe/common/fp4_expanded_policy.hpp"
int main() {
    for (uint32_t bits = 0; bits < 65536U; ++bits) {
        auto value = MegaMoeImpl::Fp4Expanded::EncodeScaledBf16(static_cast<uint16_t>(bits));
        if (std::fwrite(&value, 1, 1, stdout) != 1) return 1;
    }
    for (uint32_t code = 0; code < 16U; ++code) {
        auto value = MegaMoeImpl::Fp4Expanded::ExpandE2M1(static_cast<uint8_t>(code));
        if (std::fwrite(&value, 1, 1, stdout) != 1) return 1;
    }
}
""",
        encoding="utf-8",
    )
    executable = tmp_path / ("fp4_policy.exe" if os.name == "nt" else "fp4_policy")
    result = subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-I",
            str(ROOT / "experiments/megamoe/csrc"),
            str(source),
            "-o",
            str(executable),
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    payload = subprocess.check_output([str(executable)], timeout=30)
    assert len(payload) == 65536 + 16
    actual = np.frombuffer(payload[:65536], dtype=np.uint8)
    bit_patterns = np.arange(65536, dtype=np.uint32)
    values = (bit_patterns << 16).view(np.float32)
    finite = np.isfinite(values)
    positive_fp4 = np.array([0, 0.5, 1, 1.5, 2, 3, 4, 6], dtype=np.float64)
    distances = np.abs(np.minimum(np.abs(values[finite]).astype(np.float64), 6)[:, None] - positive_fp4)
    nearest = distances.argmin(axis=1)
    ties = distances == distances.min(axis=1, keepdims=True)
    even_ties = ties & (np.arange(8) % 2 == 0)
    nearest = np.where(even_ties.any(axis=1), even_ties.argmax(axis=1), nearest)
    positive_fp8 = np.array([0x00, 0x30, 0x38, 0x3C, 0x40, 0x44, 0x48, 0x4C], dtype=np.uint8)
    expected = positive_fp8[nearest] | (np.signbit(values[finite]).astype(np.uint8) << 7)
    np.testing.assert_array_equal(actual[finite], expected)
    # Explicit signed zeros and representational edge cases.
    assert actual[0] == 0 and actual[0x8000] == 0x80
    assert actual[0x7F80] == 0x4C and actual[0xFF80] == 0xCC
    assert np.all((actual[np.isnan(values)] & 0x7F) == 0x7F)
    np.testing.assert_array_equal(
        np.frombuffer(payload[65536:], dtype=np.uint8),
        np.concatenate((positive_fp8, positive_fp8 | 0x80)),
    )
