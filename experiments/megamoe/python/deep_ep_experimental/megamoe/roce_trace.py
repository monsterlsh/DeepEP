# Copyright (c) 2026, Lu Lu

"""Decode the HierarchicalAscendTimer device buffer for MTE+RoCE."""

from collections import Counter
import json

from .timer.protocol import CYCLE_TO_US


# The enum order and alignment are defined in mega_moe_hierarchical/common/ascend_timer_v2.hpp.
FIXED_NAMES = """
    KERNEL_TIMING AIV_INPUT_BUFFER_INIT AIV_QUANT AIV_ROUTE_MASK AIV_RESET
    AIV_INPUT_CROSS_RANK_SYNC AIV_EXPORT_SYNC AIV_FINAL_CROSS_RANK_WAIT AIV_UNPERMUTE
    AIC_WAIT_INPUT_PREP AIC_MOE_STARTUP AIV_MOE_STARTUP AIC_DISPATCH_BUFFER_INIT
    AIV_DISPATCH_BUFFER_INIT AIV_INPUT_PREP_BARRIER_WAIT AIV_COUNT_TABLE_PREPARE
    AIV_COUNT_METADATA_WAIT AIV_GATEWAY_PAYLOAD_CHECK AIV_COUNT_PREFIX_BUILD
    AIV_COUNT_TABLE_PUBLISH AIC_COUNT_READY_WAIT AIV_COUNT_READY_WAIT AIV_COMBINE_INIT
    AIV_HOST_COMBINE_BARRIER_WAIT AIV_GATEWAY_PAYLOAD_BROADCAST
""".split()
DYNAMIC_NAMES = """
    AIV_DISPATCH AIV_SWIGLU_WAIT AIV_SWIGLU AIV_COMBINE_WAIT AIV_COMBINE
    AIV_GMM1_PROLOGUE AIV_GMM2_PROLOGUE AIC_GMM1_WAIT AIC_GMM1 AIC_GMM2_WAIT AIC_GMM2
    AIC_SHARED_GMM1 AIV_SHARED_GMM1_PROLOGUE AIC_SHARED_GMM2 AIV_SHARED_GMM2_PROLOGUE
    AIV_SHARED_SWIGLU AIV_RDMA_POST AIV_RDMA_QUIET_WAIT AIV_RDMA_RECV_WAIT
    AIV_GATEWAY_PAYLOAD_WAIT AIV_COMBINE_PROGRESS AIV_COMBINE_DRAIN AIV_RELAY_STEP
    AIV_COMBINE_RUN AIV_COMBINE_REDUCE AIV_COMBINE_ASYNC AIV_COMBINE_READY
    AIV_COMBINE_SYNC AIV_PAYLOAD_ALL_LANES_WAIT
""".split()
CORE_COUNT = 96
MAX_DYNAMIC_ITER = 4096
COUNT_DIAGNOSTIC_WORDS_PER_CORE = 20
COUNT_SNAPSHOT_WORDS_PER_CORE = 4096 // 2  # Two int32 snapshot values per int64.


def _align(value, multiple):
    return (value + multiple - 1) // multiple * multiple


TIMING_SLOTS_PER_CORE = len(FIXED_NAMES) + len(DYNAMIC_NAMES) * MAX_DYNAMIC_ITER
TIMING_WORDS_PER_CORE = _align(TIMING_SLOTS_PER_CORE * 2, 16)
COUNTS_OFFSET = TIMING_WORDS_PER_CORE * CORE_COUNT
COUNTS_WORDS_PER_CORE = _align(len(DYNAMIC_NAMES), 16)
METADATA_OFFSET = COUNTS_OFFSET + COUNTS_WORDS_PER_CORE * CORE_COUNT
METADATA_WORDS_PER_CORE = len(DYNAMIC_NAMES) * MAX_DYNAMIC_ITER * 2
WAVE_OFFSET = METADATA_OFFSET + METADATA_WORDS_PER_CORE * CORE_COUNT
WAVE_WORDS_PER_CORE = _align(TIMING_SLOTS_PER_CORE, 16)
COUNT_SNAPSHOT_OFFSET = _align(
    WAVE_OFFSET + CORE_COUNT * (WAVE_WORDS_PER_CORE + COUNT_DIAGNOSTIC_WORDS_PER_CORE), 4,
)
BUFFER_WORDS = COUNT_SNAPSHOT_OFFSET + CORE_COUNT * COUNT_SNAPSHOT_WORDS_PER_CORE

RDMA_EVENTS = {"AIV_RDMA_POST", "AIV_RDMA_QUIET_WAIT", "AIV_RDMA_RECV_WAIT"}
RDMA_PHASES = {1: "dispatch_meta", 2: "dispatch_payload", 3: "combine"}
GMM_EVENTS = {"AIC_GMM1", "AIC_GMM2", "AIC_SHARED_GMM1", "AIC_SHARED_GMM2"}
WORD_MASK = (1 << 64) - 1
LOW32 = (1 << 32) - 1


def _task_args(name, word0, word1):
    if name in RDMA_EVENTS:
        phase = RDMA_PHASES.get(word0 & 0xFF)
        if phase is None:
            raise ValueError(f"{name} is missing a valid RDMA phase")
        return dict(
            phase=phase, peer=(word0 >> 32) - 1, row_count=(word0 >> 8) & 0xFFFFFF,
            row_start=word1 >> 48, qp=(word1 >> 40) & 0xFF,
            commit=bool(word1 & (1 << 32)), bytes=word1 & LOW32,
        )
    if name in GMM_EVENTS and word0 >> 32:
        return dict(expert=(word0 >> 32) - 1, row_start=word0 & LOW32,
                    n_tile=(word1 >> 32) - 1, row_count=word1 & LOW32)
    # Scheduler events use several distinct metadata formats. Preserve both words for offline use.
    return dict(metadata0=hex(word0), metadata1=hex(word1)) if word0 or word1 else {}


def _times(raw, core, slot, name):
    index = core * TIMING_WORDS_PER_CORE + slot * 2
    start, end = int(raw[index]), int(raw[index + 1])
    if start or end:
        if start <= 0 or end < start:
            raise ValueError(f"core={core} {name} slot={slot}: incomplete or invalid span ({start}, {end})")
        return start, end
    return None


def iter_spans(raw):
    """Yield device-cycle spans on each core of a single sampled rank."""
    for core in range(CORE_COUNT):
        for slot, name in enumerate(FIXED_NAMES):
            times = _times(raw, core, slot, name)
            if times:
                yield core, name, *times, {}
        for family, name in enumerate(DYNAMIC_NAMES):
            count = int(raw[COUNTS_OFFSET + core * COUNTS_WORDS_PER_CORE + family])
            if not 0 <= count <= MAX_DYNAMIC_ITER:
                raise ValueError(f"core={core} {name}: {count} events exceed timer capacity {MAX_DYNAMIC_ITER}")
            for index in range(count):
                task = family * MAX_DYNAMIC_ITER + index
                slot = len(FIXED_NAMES) + task
                times = _times(raw, core, slot, name)
                if times is None:
                    # Wait families can reserve empty slots to match the corresponding GMM tile IDs.
                    continue
                offset = METADATA_OFFSET + core * METADATA_WORDS_PER_CORE + task * 2
                args = _task_args(name, int(raw[offset]) & WORD_MASK, int(raw[offset + 1]) & WORD_MASK)
                args["wave"] = int(raw[WAVE_OFFSET + core * WAVE_WORDS_PER_CORE + slot])
                args["slot"] = index
                yield core, name, *times, args


def export_diagnostics(raw, directory, rank, bs, window_tokens):
    """Save the first-window raw buffer and a per-rank Chrome trace."""
    if raw.size != BUFFER_WORDS:
        raise ValueError(f"rank={rank}: MTE+RoCE timer has {raw.size} int64 elements, expected {BUFFER_WORDS}")
    raw.tofile(directory / "roce_timer.bin")
    starts = [int(raw[core * TIMING_WORDS_PER_CORE]) for core in range(CORE_COUNT)]
    if not any(starts):
        raise ValueError(f"rank={rank}: MTE+RoCE timer contains no kernel timing")
    base_cycle = min(start for start in starts if start > 0)

    counts = Counter()
    trace_path = directory / "roce_trace.json"
    temporary_path = trace_path.with_suffix(".json.tmp")
    try:
        with temporary_path.open("w", encoding="utf-8") as stream:
            stream.write('{"displayTimeUnit":"us","traceEvents":[\n')
            for core, name, start, end, args in iter_spans(raw):
                if counts:
                    stream.write(",\n")
                event = dict(name=name, ph="X", pid=rank, tid=core, ts=(start - base_cycle) / CYCLE_TO_US,
                             dur=(end - start) / CYCLE_TO_US, args=args)
                stream.write(json.dumps(event))
                counts[name] += 1
            stream.write("\n]}\n")
        temporary_path.replace(trace_path)
    finally:
        temporary_path.unlink(missing_ok=True)

    summary = dict(schema_version=1, backend="mte_roce", rank=rank, bs=bs, timer_scope="first_window_only",
                   timer_token_range_half_open=[0, window_tokens], timer_elements=BUFFER_WORDS,
                   device_cycles_per_us=CYCLE_TO_US, event_counts=dict(counts))
    (directory / "roce_timer.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"[TIMER] rank={rank} BS={bs} MTE+RoCE trace {trace_path}", flush=True)
