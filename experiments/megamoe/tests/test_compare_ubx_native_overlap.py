# Copyright (c) 2026, Lu Lu

from collections import defaultdict
import json
import struct

import pytest

from experiments.megamoe.scripts import compare_ubx_native_overlap as overlap


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def raw_events(arm):
    return [
        (93, "KERNEL_TIMING", 80, 300), (95, "KERNEL_TIMING", 80, 300),
        (93, "AIC_GMM1", 110, 160), (93, "AIC_GMM2", 175, 255 if arm == "gelu" else 225),
        (95, "AIV_DISPATCH", 100, 120), (95, "AIV_DISPATCH", 150, 170),
        (95, "AIV_COMBINE", 210, 230), (95, "AIV_COMBINE", 260, 280),
    ]


def mte_raw(arm):
    protocol = overlap.mte
    raw = defaultdict(int)
    for core, name, start, end in raw_events(arm):
        base = core * protocol.COUNTERS_PER_CORE
        if name == "KERNEL_TIMING":
            slot = 0
        else:
            family = protocol.DYNAMIC_NAMES.index(name)
            counter = protocol.N_TIMING_COUNTER + core * protocol.DYNAMIC_ITER_PER_CORE + family
            slot = len(protocol.FIXED_NAMES) + family * protocol.MAX_DYNAMIC_ITER + raw[counter]
            raw[counter] += 1
        raw[base + 2 * slot], raw[base + 2 * slot + 1] = start, end
    return raw


class SparseRoce(defaultdict):
    size = overlap.roce_trace.BUFFER_WORDS

    def tofile(self, path):
        # This fixture exercises the stock JSON exporter without writing a 437 MiB file.
        pass


def roce_raw(arm):
    protocol = overlap.roce_trace
    raw = SparseRoce(int)

    def event(core, name, start, end, metadata=None):
        base = core * protocol.TIMING_WORDS_PER_CORE
        slot = 0
        if name != "KERNEL_TIMING":
            family = protocol.DYNAMIC_NAMES.index(name)
            counter = protocol.COUNTS_OFFSET + core * protocol.COUNTS_WORDS_PER_CORE + family
            task = family * protocol.MAX_DYNAMIC_ITER + raw[counter]
            raw[counter] += 1
            slot = len(protocol.FIXED_NAMES) + task
            if metadata is not None:
                address = protocol.METADATA_OFFSET + core * protocol.METADATA_WORDS_PER_CORE + task * 2
                raw[address], raw[address + 1] = metadata
        raw[base + 2 * slot], raw[base + 2 * slot + 1] = start, end

    for core, name, start, end in raw_events(arm):
        if name == "AIV_COMBINE":
            for phase, begin, finish in ((2, start - 20, start + 4), (1, start, end), (3, end - 5, end)):
                event(core, overlap.ASYNC, begin, finish, ((1 << 32) | start, (1 << 32) | phase))
        else:
            event(core, name, start, end)
    event(94, "KERNEL_TIMING", 80, 300)
    event(94, "AIV_RDMA_POST", 290, 295, ((17 << 32) | 3, (1 << 40) | 4))
    return raw


def job(tmp_path, arm, backend, *, identity=False, precision="w4a8", pod=False, ep=None):
    world = ep or (32 if pod or backend == "mte_roce" else 16)
    transport = overlap.TRANSPORT[backend]
    local = min(world, 8 if pod else 16)
    mma_type, input_kind = {"w4a8": ("fp8xfp4", "fp8"), "w8a8": ("fp8xfp8", "fp8"),
                            "w4a4": ("fp4xfp4", "fp4")}[precision]
    root = tmp_path / arm / f"perf-{precision}"
    sample = list(range(0, world, local // 4 if pod else local))
    config = dict(bs_values=[80], seed=1234, iters=2, warmup=1, alignment=dict(rows=4, cols=4, repeats=1),
                  model=dict(activation=arm, topk_weights_type=int(arm == "gelu"),
                             mma_type=mma_type,
                             num_shared_experts=0, num_experts=800, num_topk=25, hidden=4096, intermediate_hidden=1536))
    report = dict(windows_per_forward=2, host_e2e=dict(
        scope="public_forward_through_device_synchronize", timer_pass=False, per_rank_rounds_us=[[100, 120]] * world,
        rank_max_rounds_us=[100, 120], rank_max_mean_us=110, global_input_tokens_per_second=80 * world * 1e6 / 110,
    ))
    for rank in range(world):
        ranked = root / f"rank_{rank}"
        manifest = dict(
            config=config, input=input_kind, backend=backend, ep_size=world, device_communication=transport,
            rank=rank, local_rank=rank % local, ranks_per_node=local, mode="perf", perf_scope="all",
            numerical_contract=overlap.CONTRACTS[arm], timer_scope="first_window_only", run_id=f"job-{arm}",
            build=dict(source_sha256="a" * 64, megamoe=dict(enabled=True, timer=True, mte_roce=True)), routing="balanced",
            paired_weights=False, timer_ranks=sample, window_tokens=64, windows_per_forward_by_bs={"80": 2},
        )
        if identity:
            manifest["device_identity"] = dict(host_boot_id=f"host-{rank // local}", physical_device=rank % local)
        write(ranked / "manifest.json", manifest)
        case = ranked / "bs_80"
        write(case / "resources.json", dict(prepared_bs=80, window_tokens=64, windows_per_forward=2,
                                           timer_token_range_half_open=[0, 64] if rank in sample else None,
                                           device_free_after_case_bytes=1))
        write(case / "profiling.json", report)
        if backend == "mte_roce" and rank in sample:
            overlap.roce_trace.export_diagnostics(roce_raw(arm), case, rank, 80, 64)
    return root


@pytest.mark.parametrize(("precision", "input_kind"), [("w4a8", "fp8"), ("w4a4", "fp4")])
def test_ep16_legacy_manifests_and_disjoint_phase_envelopes(tmp_path, monkeypatch, capsys, precision, input_kind):
    gelu, swiglu = (job(tmp_path, arm, "mte", precision=precision) for arm in ("gelu", "swiglu"))

    def raw(path, words, decode):
        assert path.relative_to(path.parents[2]).as_posix() == "bs_80/timer/rank_0_npu_time.bin"
        assert words == overlap.mte.TOTAL_BUFFER_SIZE
        return decode(mte_raw(path.parents[3].name))

    monkeypatch.setattr(overlap, "binary_spans", raw)
    output = tmp_path / "result.json"
    overlap.main(["--gelu", str(gelu), "--swiglu", str(swiglu), "--output", str(output)])
    result = overlap.read_json(output)
    console = capsys.readouterr().out
    assert "physical_mapping=unverified" in console
    assert "Full-batch timer-off host E2E  GELU 0.110 ms  SwiGLU 0.110 ms" in console
    assert result["schema"] == "ubx_native_first_window_overlap_v1"
    assert result["case"]["precision"] == precision
    assert result["case"]["input"] == input_kind
    assert result["case"]["timer_token_range_half_open"] == [0, 64]
    assert result["case"]["model"]["num_topk"] == 25
    assert result["arms"]["gelu"]["host_e2e"]["rank_max_mean_us"] == 110
    row, = result["arms"]["gelu"]["ranks"]
    assert row["combine_endpoint"] == "local_mte_drained_batch"
    assert row["phases"]["Dispatch"]["communication_ticks"] == 70
    assert row["phases"]["Combine"]["communication_ticks"] == 70
    assert row["phases"]["Total"]["communication_ticks"] == 140
    assert row["phases"]["Total"]["overlapping_ticks"] == 95
    assert row["recorded_spans"]["Total"]["communication_ticks"] == 80
    assert row["recorded_spans"]["Total"]["overlapping_ticks"] == 40
    assert result["gelu_minus_swiglu_pp"]["Total"] == pytest.approx(100 * 30 / 140)


def test_ep32_exported_trace_uses_only_local_async_lifetime(tmp_path):
    gelu, swiglu = (job(tmp_path, arm, "mte_roce", identity=True) for arm in ("gelu", "swiglu"))
    result = overlap.compare(gelu, swiglu)
    assert result["physical_mapping"] == "verified"
    assert result["case"]["timer_ranks"] == [0, 16]
    a = result["arms"]["gelu"]
    assert all(row["combine_endpoint"] == "local_mte_retired_row" and row["source"] == "exported_trace" for row in a["ranks"])
    assert a["phases"]["Dispatch"]["communication_us"] == pytest.approx(0.14)
    assert a["phases"]["Combine"]["communication_us"] == pytest.approx(0.14)
    assert a["phases"]["Total"]["communication_us"] == pytest.approx(0.28)
    assert a["phases"]["Total"]["overlap_rate"] == pytest.approx(95 / 140)
    assert a["recorded_spans"]["Total"]["overlap_rate"] == pytest.approx(40 / 80)

    path = swiglu / "rank_16" / "manifest.json"
    manifest = overlap.read_json(path)
    manifest["device_identity"]["physical_device"] = 99
    write(path, manifest)
    with pytest.raises(ValueError, match="devices"):
        overlap.compare(gelu, swiglu)


def test_950pod_first_window_uses_four_nodes_and_all_even_ranks(tmp_path, monkeypatch):
    a, b = (job(tmp_path, arm, "mte", identity=True, precision="w4a4", pod=True) for arm in ("gelu", "swiglu"))
    monkeypatch.setattr(overlap, "binary_spans", lambda path, size, decode: decode(mte_raw(path.parents[3].name)))
    path = tmp_path / "result.json"
    overlap.main(["--platform", "950pod", "--gelu", str(a), "--swiglu", str(b), "--output", str(path)])
    result = overlap.read_json(path)
    assert result["schema"] == "950pod_native_first_window_overlap_v1"
    assert result["physical_mapping"] == "verified"
    assert result["case"]["ep_size"] == 32 and result["case"]["ranks_per_node"] == 8
    assert result["case"]["timer_ranks"] == list(range(0, 32, 2))
    assert result["gelu_minus_swiglu_pp"]["Total"] == pytest.approx(100 * 30 / 140)
    manifest = a / "rank_0/manifest.json"
    data = overlap.read_json(manifest)
    data["timer_ranks"].remove(30)
    write(manifest, data)
    with pytest.raises(ValueError, match="four evenly spaced timer ranks"):
        overlap.compare(a, b, platform="950pod")


@pytest.mark.parametrize("platform, ep", [("ubx", 4), ("ubx", 8), ("950pod", 4), ("950pod", 8), ("950pod", 16)])
def test_small_ep_overlap_uses_manifest_topology(tmp_path, monkeypatch, platform, ep):
    a, b = (job(tmp_path, arm, "mte", identity=True, pod=platform == "950pod", ep=ep)
            for arm in ("gelu", "swiglu"))
    monkeypatch.setattr(overlap, "binary_spans", lambda path, size, decode: decode(mte_raw(path.parents[3].name)))
    report = overlap.compare(a, b, platform=platform)
    local = min(ep, 8 if platform == "950pod" else 16)
    assert (report["case"]["ep_size"], report["case"]["ranks_per_node"]) == (ep, local)
    assert report["physical_mapping"] == "verified"


def test_incomplete_selected_roce_trace_fails(tmp_path):
    gelu, swiglu = (job(tmp_path, arm, "mte_roce") for arm in ("gelu", "swiglu"))
    path = gelu / "rank_16" / "bs_80" / "roce_trace.json"
    trace = overlap.read_json(path)
    trace["traceEvents"].pop()
    write(path, trace)
    with pytest.raises(ValueError, match="gelu rank 16: decoded RoCE event counts"):
        overlap.compare(gelu, swiglu)
    path.unlink()
    with pytest.raises(ValueError, match="gelu rank 16: .*roce_trace.json"):
        overlap.compare(gelu, swiglu)


@pytest.mark.parametrize(("precision", "input_kind"), [("w4a4", "fp4"), ("w8a8", "fp8")])
def test_ep32_reports_native_precision_and_rejects_mixed_precision(tmp_path, precision, input_kind):
    gelu, swiglu = (job(tmp_path, arm, "mte_roce", precision=precision) for arm in ("gelu", "swiglu"))
    result = overlap.compare(gelu, swiglu)
    assert result["case"]["precision"] == precision
    assert result["case"]["input"] == input_kind
    assert result["case"]["timer_token_range_half_open"] == [0, 64]
    assert result["arms"]["gelu"]["phases"]["Total"]["overlap_rate"] == pytest.approx(95 / 140)
    assert all(row["combine_endpoint"] == "local_mte_retired_row" for row in result["arms"]["swiglu"]["ranks"])
    with pytest.raises(ValueError, match="A/B manifest mismatch: precision"):
        overlap.compare(gelu, job(tmp_path, "swiglu", "mte_roce"))


def test_ep16_rejects_a8w8_without_native_mte_timer(tmp_path):
    with pytest.raises(ValueError, match="mte native overlap supports"):
        overlap.read_job(job(tmp_path, "gelu", "mte", precision="w8a8"), "gelu")


def test_roce_binary_is_preferred_and_uses_export_counts(tmp_path, monkeypatch):
    gelu = overlap.read_job(job(tmp_path, "gelu", "mte_roce"), "gelu")
    path = gelu["cases"][0] / "roce_timer.bin"
    path.touch()

    def raw(actual, words, decode):
        assert actual == path and words == overlap.roce_trace.BUFFER_WORDS
        return decode(roce_raw("gelu"))

    monkeypatch.setattr(overlap, "binary_spans", raw)
    row = overlap.rank_phases(gelu, 0)
    assert row["source"] == "binary"
    assert row["phases"]["Total"]["communication_ticks"] == 140


def test_real_counter_offsets_reject_overflow_and_a_missing_attempted_record():
    raw = mte_raw("gelu")
    protocol = overlap.mte
    family = protocol.DYNAMIC_NAMES.index("AIV_COMBINE")
    raw[protocol.N_TIMING_COUNTER + 95 * protocol.DYNAMIC_ITER_PER_CORE + family] = protocol.MAX_DYNAMIC_ITER + 1
    with pytest.raises(ValueError, match="exceeds timer capacity"):
        list(overlap.mte_spans(raw))
    raw = roce_raw("gelu")
    protocol = overlap.roce_trace
    family = protocol.DYNAMIC_NAMES.index(overlap.ASYNC)
    raw[protocol.COUNTS_OFFSET + 95 * protocol.COUNTS_WORDS_PER_CORE + family] += 1
    with pytest.raises(ValueError, match="AIV_COMBINE_ASYNC: missing or invalid interval"):
        list(overlap.roce_binary_spans(raw))


def test_binary_reader_is_little_endian_and_requires_exact_size(tmp_path):
    path = tmp_path / "timer.bin"
    path.write_bytes(struct.pack("<2q", -1, 2**42))
    assert list(overlap.binary_spans(path, 2, lambda raw: (raw[0], raw[1]))) == [-1, 2**42]
    with pytest.raises(ValueError, match="invalid binary timer size"):
        list(overlap.binary_spans(path, 3, lambda raw: (raw[0], raw[1])))
