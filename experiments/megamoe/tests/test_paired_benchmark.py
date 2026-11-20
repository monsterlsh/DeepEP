from contextlib import nullcontext
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import paired_benchmark as paired
import stage_mte_windows as staging
from summarize_isolated_pairs import summarize


def test_pair_rank_max_before_ratio_and_throughput():
    rows = [[dict(gelu=2, swiglu=8), dict(gelu=10, swiglu=2)],
            [dict(gelu=4, swiglu=1), dict(gelu=1, swiglu=5)]]
    report = paired.paired_report(rows, 100)
    assert [r["gelu_over_swiglu"] for r in report["pairs"]] == [0.5, 2]
    assert report["median_pair_ratio"] == 1.25
    assert not report["passed_median_pair_ratio_le_one"]
    assert [r["order"] for r in report["pairs"]] == [("gelu", "swiglu"), ("swiglu", "gelu")]
    assert report["global_input_tokens_per_second"]["gelu"] == pytest.approx(2e8 / 7)
    assert report["median_ratio_by_first_arm"] == dict(gelu=0.5, swiglu=2)
    with pytest.raises(ValueError):
        paired.paired_report([[dict(gelu=0, swiglu=5)]], 100)


@pytest.mark.parametrize("bs,staged,expected", [
    (257, None, [257]), (39280, None, [256] * 153 + [112]),
    (39280, 512, [512] * 76 + [368]), (39280, 1024, [1024] * 38 + [368]),
    (39280, 2048, [2048] * 19 + [368]),
])
def test_original_plan_cache_regions_layout_and_tail(monkeypatch, bs, staged, expected):
    from deep_ep import _native

    recorded = []
    plans = []

    class Tensor:
        def __init__(self, rows, name, offset=0):
            self.rows, self.name, self.offset = rows, name, offset

        def __getitem__(self, s):
            first, end, _ = s.indices(self.rows)
            return Tensor(end - first, self.name, self.offset + first)

        def add_(self, _):
            pass

        def record_stream(self, stream):
            pass

    class Plan:
        workspace_bytes = 256
        def __init__(self, x, i, k, experts, world, capacity, a8w4, shared, local_world, ffts, address, size, **cfg):
            self.rows = x.rows
            plans.append((self.rows, capacity, address, size, cfg))

        def forward(self, x, scale, ids, weights, w1, w2, s1, s2, timer, sw1, sw2, ss1, ss2, out, generation, clamp):
            assert x.rows == self.rows == out.rows and x.offset == out.offset
            assert all(t.rows == x.rows for t in (ids, scale, weights))
            recorded.append(("native", x.offset, self.rows, generation, w1, w2))
            return out, Tensor(1, "counts")

    stream = SimpleNamespace(wait_stream=lambda _: None)
    npu = SimpleNamespace(synchronize=lambda: None, current_stream=lambda: stream, stream=lambda _: nullcontext())
    torch = SimpleNamespace(npu=npu, empty=lambda shape, **kwargs: Tensor(shape[0], "out"), bfloat16=object())
    native = SimpleNamespace(megamoe_workspace_address=lambda _: 2 << 20)
    monkeypatch.setattr(_native, "require_native", lambda _: native)
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "8")
    # Native bytes model makes both target arms exceed 1 GiB for the full logical batch.
    def heap(world, experts, n, k, h, **kwargs):
        return 2 << 30 if n > 2048 else max(1, n // 256) * (2 << 20)
    half_heap = heap(8, 100, expected[0], 25, 4096)

    class Buffer:
        workspace_bytes = 512
        def __init__(self, group, e, n, k, h, i, **kwargs):
            self._windowed = n > 2048
            self._urma_windowed = False
            self._native_capacity = n if not self._windowed else 256
            self.num_max_tokens_per_rank = n
            self.allocate_inputs()
            self._runtime = api._MegaRuntime(group, 2 << 20, 120, backend="mte")
            self._timeout, self._device = 120, "npu"

        def allocate_inputs(self):
            self.x, self.x_sf, self.topk_idx, self.topk_weights = (
                Tensor(self.num_max_tokens_per_rank, t) for t in ("x", "sc", "id", "prob"))

        @property
        def window_tokens(self):
            return self._native_capacity

        def prepare(self, n):
            assert n == bs

    def public(out, w1, w2, owner, activation):
        if activation == "swiglu":
            assert owner._plan.rows == bs and not owner._windowed and owner.topk_weights_type == 0
        recorded.append(("public", activation, w2))

    def factory(group, size, timeout, **kwargs):
        assert size == 2 * half_heap
        return SimpleNamespace(native=object(), stream=stream)

    api = SimpleNamespace(SymmBuffer=Buffer, _MegaRuntime=factory, fp8_fp4_mega_moe=public,
                          _extension=lambda: SimpleNamespace(buffer_size=heap, Plan=Plan, ffts_config=lambda: 0,
                                                            small_batch_max_tokens=256))
    w1 = {"gelu": ("gate", "s1"), "swiglu": SimpleNamespace(select=lambda fused: ("decode" if fused else "prefill", "s1"))}
    monkeypatch.setattr(paired, "prepare_tensors", lambda *args: (w1, ("shared_w2", "shared_s2")))
    model = dict(num_experts=800, num_topk=25, hidden=4096, intermediate_hidden=1536, num_shared_experts=0)
    group = SimpleNamespace(size=lambda: 8, barrier=lambda _: None)
    args = SimpleNamespace(bs=bs, routing="balanced", swiglu_weight_mode=0, stage_mte_window=staged)
    _, calls, windows, layout = paired.prepare_calls(args, {"model": model}, group, torch, api)
    assert windows["swiglu"] == windows["gelu"] == expected
    assert layout["weight_mode"] == {"gelu": 1, "swiglu": 0}
    assert [(row, address, cfg["windowed"]) for row, _, address, _, cfg in plans] == [
        (n, (2 << 20) + half_heap, False) for n in dict.fromkeys(expected)]
    for _ in range(2):
        calls["gelu"]()
        calls["swiglu"]()
    assert len(plans) == len(set(expected))
    if len(expected) == 1:
        assert [r[1] for r in recorded] == ["gelu", "swiglu"] * 2
    else:
        assert len(plans) == 2
        native = [r for r in recorded if r[0] == "native"]
        assert [(r[1], r[2]) for r in native] == [(i * expected[0], n) for i, n in enumerate(expected)] * 2
        assert [r[3] for r in native] == list(range(1, 2 * len(expected) + 1))
        assert {r[4:] for r in native} == {
            ("decode" if n <= 256 else "prefill", "shared_w2") for n in expected}


def test_isolated_aggregation_matches_inputs_and_topology(tmp_path):
    ref = dict(config={}, backend="urma", world=2, ranks_per_node=2, hosts_by_rank=["a", "a"],
               devices_by_rank=[0, 1], bs=17, routing="balanced", warmup_pairs=2, formal_pairs=2, cache="warm",
               input="fp8", routes="shared", weights="gate", torch="1", torch_npu="1", build={})
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    for pair in range(2):
        for arm in paired.order(pair):
            path = tmp_path / str(pair) / arm / "isolated.json"
            path.parent.mkdir(parents=True)
            scale = 2.0 if arm == "swiglu" else 1.0
            times = [[{arm: scale}, {arm: 6 * scale}], [{arm: 3 * scale}, {arm: 2 * scale}]]
            entry = dict(ref, isolated_arm=arm, windows={arm: [17]}, layout={"weight_mode": int(arm == "gelu")},
                         started_at=start.isoformat(), recorded_at=(start + timedelta(seconds=1)).isoformat(),
                         host_e2e=dict(per_rank=times))
            path.write_text(json.dumps(entry))
            start += timedelta(seconds=2)
    report = summarize(tmp_path, 2)
    assert report["host_e2e"]["median_pair_ratio"] == 0.5
    assert report["host_e2e"]["global_input_tokens_per_second"]["gelu"] == pytest.approx(34e6 / 4.5)
    path.write_text(json.dumps(dict(entry, routing="skewed")))
    with pytest.raises(ValueError, match="routing"):
        summarize(tmp_path, 2)


def test_window_sweep_compares_both_independent_optima_per_routing(tmp_path):
    windows = [256, 1024, 2048]
    assert staging.cases(windows) == [
        ("balanced", 256), ("skewed", 2048), ("balanced", 1024),
        ("skewed", 1024), ("balanced", 2048), ("skewed", 256)]
    for route, window in staging.cases(windows):
        smaller = window // 256
        values = dict(gelu=smaller, swiglu=12 / smaller)
        if route == "skewed":
            values = dict(gelu=values["swiglu"], swiglu=values["gelu"])
        sizes = paired.window_lengths(39280, window)
        entry = dict(config={}, backend="mte", world=8, ranks_per_node=8, hosts_by_rank=["a"]*8,
                     devices_by_rank=list(range(8)), bs=39280, warmup_pairs=2, formal_pairs=2, cache="warm",
                     input="fp8", routes="shared", weights="gate", torch="1", torch_npu="1", build={},
                     routing=route, windows={arm: sizes for arm in paired.ARMS},
                     layout=dict(staged_window=window, weight_mode={"gelu": 1, "swiglu": 0}),
                     host_e2e=paired.paired_report([[values, values]] * 8, 39280))
        path = tmp_path / f"{route}-w{window}" / "paired.json"
        path.parent.mkdir()
        path.write_text(json.dumps(entry))
    study = staging.summarize(tmp_path, windows)["by_routing"]
    assert study["balanced"]["best_window_by_arm"] == dict(gelu=256, swiglu=2048)
    assert study["skewed"]["best_window_by_arm"] == dict(gelu=2048, swiglu=256)
    assert study["balanced"]["fastest_gelu_over_fastest_swiglu"] == pytest.approx(1 / 1.5)
    path.write_text(json.dumps(dict(entry, hosts_by_rank=["unexpected"] * 8)))
    with pytest.raises(ValueError, match="topology"):
        staging.summarize(tmp_path, windows)
