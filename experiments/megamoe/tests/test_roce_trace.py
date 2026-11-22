# Copyright (c) 2026, Lu Lu

from collections import defaultdict
from pathlib import Path
import re

import pytest

from deep_ep_experimental.megamoe import roce_trace as trace


def test_hierarchical_timer_layout_matches_device_header():
    header = Path(__file__).parents[1] / "csrc/kernels/mega_moe_hierarchical/common/ascend_timer_v2.hpp"
    source = header.read_text(encoding="utf-8")

    def names(enum):
        body = re.search(rf"enum {enum} \{{(.*?)\}};", source, re.S).group(1)
        return re.findall(r"^\s*([A-Z][A-Z_0-9]*)\s*(?:=\s*0)?\s*,", body, re.M)

    assert trace.FIXED_NAMES == names("FixedTiming")
    assert trace.DYNAMIC_NAMES == names("DynamicTimingType")
    assert trace.COUNT_DIAGNOSTIC_WORDS_PER_CORE == len(names("CountDiagnostic"))
    assert f"N_CORE_COUNT = {trace.CORE_COUNT};" in source
    assert f"MAX_DYNAMIC_ITER = {trace.MAX_DYNAMIC_ITER};" in source
    assert f"COUNT_SNAPSHOT_MAX_WORDS = {trace.COUNT_SNAPSHOT_WORDS_PER_CORE * 2};" in source


def test_roce_trace_decodes_rdma_completion_and_rejects_truncated_slots():
    raw = defaultdict(int)
    raw[0], raw[1] = 100, 1000
    core = 2
    name = "AIV_RDMA_QUIET_WAIT"
    family = trace.DYNAMIC_NAMES.index(name)
    slots = trace.COUNTS_OFFSET + core * trace.COUNTS_WORDS_PER_CORE + family
    task = family * trace.MAX_DYNAMIC_ITER
    slot = len(trace.FIXED_NAMES) + task
    timing = core * trace.TIMING_WORDS_PER_CORE + 2 * slot
    metadata = trace.METADATA_OFFSET + core * trace.METADATA_WORDS_PER_CORE + task * 2

    raw[slots] = 1
    raw[timing], raw[timing + 1] = 150, 200
    raw[metadata], raw[metadata + 1] = (17 << 32) | (8 << 8) | 3, (64 << 48) | (5 << 40) | (1 << 32) | 4
    raw[trace.WAVE_OFFSET + core * trace.WAVE_WORDS_PER_CORE + slot] = -1
    assert list(trace.iter_spans(raw)) == [
        (0, "KERNEL_TIMING", 100, 1000, {}),
        (2, name, 150, 200, dict(phase="combine", peer=16, row_count=8, row_start=64, qp=5,
                                 commit=True, bytes=4, wave=-1, slot=0)),
    ]
    raw[timing + 1] = 0
    with pytest.raises(ValueError, match="incomplete or invalid span"):
        list(trace.iter_spans(raw))
    raw[slots] = trace.MAX_DYNAMIC_ITER + 1
    with pytest.raises(ValueError, match="exceed timer capacity"):
        list(trace.iter_spans(raw))
