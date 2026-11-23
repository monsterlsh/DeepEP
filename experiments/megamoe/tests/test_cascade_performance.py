import csv
from contextlib import nullcontext
import json
from types import SimpleNamespace

import pytest

from deep_ep_experimental.megamoe import cascade_performance
from deep_ep_experimental.megamoe.cascade_performance import read_forward_tasks


_COLUMNS = ("Name", "Device_id", "Stream ID", "Task ID", "Start Time(us)", "Duration(us)")


def _write_profile(directory, rows, *, missing_flow=False):
    events = []
    for sample in range(2):
        for edge, offset in (("start", 0), ("end", 10)):
            flow_id = 2 * sample + int(edge == "end")
            host_start = 300 + sample * 25 + offset
            dev_start = str(1795317091082100 + sample * 20 + offset)
            events.extend((
                {
                    "name": f"deep_ep_cascade_sample/{sample}/{edge}", "ph": "X",
                    "pid": 5, "tid": 5, "ts": host_start, "dur": 2,
                },
                {
                    "name": "torch_to_npu", "ph": "s", "cat": "async_npu",
                    "id": flow_id, "pid": 5, "tid": 5, "ts": host_start + 1,
                },
                {
                    "name": "EVENT RECORD", "ph": "X",
                    "pid": 8, "tid": 17, "ts": dev_start, "dur": 0.024,
                },
            ))
            if not (missing_flow and flow_id == 3):
                events.append({
                    "name": "torch_to_npu", "ph": "f", "cat": "async_npu",
                    "id": flow_id, "pid": 8, "tid": 17, "ts": dev_start,
                })
    (directory / "trace_view.json").write_text(json.dumps({"traceEvents": events}), encoding="utf-8")
    with (directory / "kernel_details.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(_COLUMNS)
        writer.writerows(rows)


@pytest.mark.parametrize("name", [
    "_Z15mega_moe_kernelILb0EE",
    "mega_moe_urma_kernel(MegaMoeUrma::Arguments)",
    "mega_moe_hierarchical_kernel<0, true, false>(void*)",
])
def test_device_samples_include_all_native_windows_without_flows(tmp_path, name):
    _write_profile(tmp_path, [
        ("guard", "3", "17", "1", "1795317091082080.010", "10"),
        (name, "3", "19", "2", "1795317091082101.001", "4"),
        ("aux", "3", "20", "3", "1795317091082102.002", "1"),
        (name, "3", "19", "4", "1795317091082106.002", "2"),
        ("flush", "3", "17", "5", "1795317091082113.010", "5"),
        (name, "3", "19", "6", "1795317091082121.001", "2"),
        (name, "3", "19", "7", "1795317091082126.002", "1"),
    ])

    trace, kernels, rounds = read_forward_tasks(tmp_path, 2, expected_fused_windows=2)

    assert trace.name == "trace_view.json"
    assert kernels.name == "kernel_details.csv"
    assert [row["kernel_sum_us"] for row in rounds] == [7, 3]
    assert [row["device_span_us"] for row in rounds] == [7.001, 6.001]
    assert [task["name"] for task in rounds[0]["tasks"]] == [name, "aux", name]
    assert rounds[0]["tasks"][0]["start_us"] == "1795317091082101.001"

    with pytest.raises(ValueError, match="expected 1 fused MegaMoE windows, found 2"):
        read_forward_tasks(tmp_path, 2, expected_fused_windows=1)


def test_profiler_rejects_missing_marker_flow(tmp_path):
    _write_profile(tmp_path, [], missing_flow=True)

    with pytest.raises(ValueError, match="must link to one device EVENT RECORD"):
        read_forward_tasks(tmp_path, 2)


def test_profiler_rejects_kernel_crossing_sample_boundary(tmp_path):
    _write_profile(tmp_path, [
        ("crossing", "3", "17", "1", "1795317091082109.99", "5"),
    ])

    with pytest.raises(ValueError, match="crosses Cascade sample 0"):
        read_forward_tasks(tmp_path, 2)


def test_comparison_discards_warmup_and_compares_mean_rank_max(monkeypatch, tmp_path):
    def fused():
        pass

    def cascade():
        pass

    def profile(forward, torch, directory, warmup, iters, guard):
        assert warmup == 1 and iters == 2
        directory.mkdir(parents=True)
        assert forward is (fused if directory.parent.name == "fused" else cascade)

    def read(directory, iterations, *, expected_fused_windows):
        is_fused = directory.parent.name == "fused"
        assert expected_fused_windows == (2 if is_fused else None)
        assert iterations == 3
        samples = (200, 4, 6) if is_fused else (100, 12, 18)
        records = [dict(iteration=i, kernel_sum_us=value, device_span_us=value) for i, value in enumerate(samples)]
        return directory / "trace.json", directory / "kernels.csv", records

    def all_gather(values, label):
        second_rank = dict(values, kernel_sum=[value * 2 for value in values["kernel_sum"]])
        return [values, second_rank]

    def host(forward, group, torch, iterations, bs):
        assert iterations == 2 and bs == 39280
        return {"rank_max_mean_us": 5 if forward is fused else 8}

    monkeypatch.setattr(cascade_performance, "_profile_forward", profile)
    monkeypatch.setattr(cascade_performance, "read_forward_tasks", read)
    monkeypatch.setattr(cascade_performance, "collective_check", lambda group, label, error: None)
    monkeypatch.setattr(cascade_performance, "measure_synchronized_forward", host)
    report = cascade_performance.measure_cascade_comparison(
        fused, cascade, SimpleNamespace(all_gather=all_gather), SimpleNamespace(inference_mode=nullcontext),
        {"warmup": 1, "iters": 2}, tmp_path, 39280, 1, SimpleNamespace(describe=dict),
        cascade_kind="prefill", expected_fused_windows=2,
    )

    assert report["fused"]["kernel_sum"]["rank_max_mean_us"] == 10
    assert report["prefill_cascade"]["kernel_sum"]["rank_max_mean_us"] == 30
    assert report["speedup"] == {"kernel_sum": 3, "device_span": 3, "host_e2e": 1.6}
