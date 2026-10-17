# Copyright (c) 2026, Lu Lu
# Modified by SimpleBright_Man 2026

"""Benchmark mode preflight and bit-preserving weight upload, without NPU kernels."""

import sys
from types import SimpleNamespace

import numpy as np
import pytest

from deep_ep_experimental.megamoe.benchmark import device_weight_pair, parse_args
from deep_ep_experimental.megamoe.reference import fp8_decode


@pytest.mark.parametrize("mma_type", ["fp8xfp4", "fp8xfp8"])
@pytest.mark.parametrize("input_kind", ["fp8", "bf16"])
def test_benchmark_accepts_weight_and_input_types_independently(mma_type, input_kind, tmp_path):
    args, config = parse_args(
        [
            "--output-dir",
            str(tmp_path / "run"),
            "--mma-type",
            mma_type,
            "--input",
            input_kind,
            "--bs-values",
            "1,256,257",
        ]
    )
    assert args.mma_type == mma_type and args.input == input_kind
    assert config["bs_values"] == [1, 256, 257]
    assert not args.timer


@pytest.mark.parametrize(
    "options,message",
    [
        (["--timer"], "W8A8 does not support --timer"),
        (["--config", "deepseek_v4_pro_shared1"], "W8A8 requires num_shared_experts=0"),
    ],
)
def test_unsupported_w8_cases_fail_before_device_initialization(options, message, tmp_path, capsys):
    with pytest.raises(SystemExit) as error:
        parse_args(["--output-dir", str(tmp_path / "run"), "--mma-type", "fp8xfp8", *options])
    assert error.value.code == 2
    assert message in capsys.readouterr().err
    assert not (tmp_path / "run").exists()


def test_default_keeps_w4_shared_and_timer_configuration(tmp_path):
    args, _ = parse_args(["--output-dir", str(tmp_path), "--config", "deepseek_v4_pro_shared1", "--timer"])
    assert args.mma_type == "fp8xfp4" and args.timer


@pytest.mark.parametrize("mma_type", ["fp8xfp4", "fp8xfp8"])
def test_weight_upload_preserves_quantized_bits_and_scales(monkeypatch, mma_type):
    class Tensor:
        def __init__(self, data, dtype="uint8"):
            self.data, self.dtype = data, dtype

        def npu(self):
            return self

        def view(self, dtype):
            return Tensor(self.data, dtype)

        def to(self, *args, **kwargs):
            pytest.fail("Quantized code bytes must not be numerically cast")

    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(from_numpy=Tensor, float8_e4m3fn="e4m3", float8_e8m0fnu="e8m0"),
    )
    # E4M3 1, -2, 448, -0: numeric uint8->FP8 conversion would corrupt these codes.
    codes = np.tile(np.array([56, 192, 126, 128], dtype=np.uint8), 16).reshape(1, 1, 64)
    scales = np.array([[[[127, 128]]]], dtype=np.uint8)
    weight, scale = device_weight_pair((codes, scales), mma_type)
    np.testing.assert_array_equal(weight.data, codes)
    np.testing.assert_array_equal(scale.data, scales)
    assert weight.dtype == ("e4m3" if mma_type == "fp8xfp8" else "uint8")
    assert scale.dtype == "e8m0"
    if mma_type == "fp8xfp8":
        np.testing.assert_array_equal(fp8_decode(weight.data[0, 0, :4]), [1, -2, 448, -0.0])
