# Copyright (c) 2026, Lu Lu

from concurrent.futures import ThreadPoolExecutor
import json
from threading import Event
from types import SimpleNamespace

import pytest

from experiments.megamoe.scripts.ubx_accuracy_cache import AccuracyCache, atomic_write_json, file_lock
from experiments.megamoe.scripts.run_megamoe import accuracy_report_line
from experiments.megamoe.tests.accuracy_reports import direct_accuracy_row


def case(tmp_path, dual=False, sample=0):
    identity = dict(
        ep_size=2, bs=16, check_tokens=sample, dual_reference=dual, routing="balanced",
        config={"model": {"mma_type": "fp8xfp8", "hidden": 32}}, build={"revision": "a"}, golden="digest",
    )
    output = tmp_path / "run" / "accuracy-w8a8"
    if dual:
        identity["accuracy_cascade"] = "prefill"
    rows = []
    checked = min(sample or 16, 16)
    for rank in range(2):
        row = dict(finite=True, counts_equal=True, passed=True)
        if dual:
            row = direct_accuracy_row(rank, "prefill", 16, checked, 32)
        else:
            row.update(policy="cpu_mixed_error", rank=rank, checked_token_count=checked, check_tokens=sample,
                       total_tokens=16, cumulative_counts_equal=True)
        rows.append(row)
    for rank in range(2):
        directory = output / f"rank_{rank}"
        manifest = {key: identity[key] for key in ("config", "routing", "dual_reference", "ep_size", "build")}
        manifest.update(rank=rank, backend="urma", mode="accuracy", input="fp8")
        if dual:
            manifest["accuracy_cascade"] = identity["accuracy_cascade"]
        atomic_write_json(directory / "manifest.json", manifest)
        atomic_write_json(directory / "bs_16/accuracy.json", rows)
    return AccuracyCache(tmp_path, identity), output


@pytest.mark.parametrize(("dual", "sample"), [(False, 0), (True, 3), (False, 20)])
def test_reuse_only_after_successful_publication(tmp_path, dual, sample):
    cache, output = case(tmp_path, dual, sample)
    with cache.lock():
        assert cache.lookup() is None
        cache.publish(output)
        assert cache.lookup() == output
        for key, value in (("golden", "new digest"), ("build", {"revision": "b"}), ("dual_reference", not dual)):
            assert AccuracyCache(tmp_path, {**cache.identity, key: value}).lookup() is None
        cache.invalidate()
        assert cache.lookup() is None
        assert (output / "rank_1/bs_16/accuracy.json").is_file()


@pytest.mark.parametrize("missing", ["runtime", "golden"])
def test_missing_execution_identity_invalidates_legacy_passes(tmp_path, missing):
    initial, output = case(tmp_path)
    first = AccuracyCache(tmp_path, {**initial.identity, "runtime": "runtime-a"})
    second = AccuracyCache(tmp_path, {**initial.identity, "runtime": "runtime-b"})
    unknown = AccuracyCache(tmp_path, {**first.identity, missing: None})

    for cache in (first, second):
        cache.publish(output)
        cache.index.replace(cache.legacy_index)
    assert first.lookup() == second.lookup() == output

    with unknown.lock():
        assert unknown.lookup() is None
        unknown.invalidate()
    assert first.lookup() is None and second.lookup() is None

    with first.lock():
        first.publish(output)
    assert first.lookup() == output and second.lookup() is None


@pytest.mark.parametrize("damage", ["missing", "manifest", "changed_bytes", "rank_reports_disagree", "failed", "index"])
def test_partial_corrupt_or_failed_run_misses(tmp_path, damage):
    cache, output = case(tmp_path)
    cache.publish(output)
    report = output / "rank_1/bs_16/accuracy.json"
    if damage == "missing":
        report.unlink()
    elif damage == "manifest":
        manifest = output / "rank_1/manifest.json"
        atomic_write_json(manifest, {**json.loads(manifest.read_bytes()), "rank": 0})
    elif damage == "changed_bytes":
        with (output / "rank_1/manifest.json").open("ab") as stream:
            stream.write(b" ")
    elif damage == "index":
        cache.index.write_bytes(b'{')
    else:
        rows = json.loads(report.read_bytes())
        rows[1]["passed"] = False
        atomic_write_json(report, rows)
        if damage == "failed":
            atomic_write_json(output / "rank_0/bs_16/accuracy.json", rows)
    assert cache.lookup() is None
    if damage in ("manifest", "rank_reports_disagree", "failed", "missing"):
        with pytest.raises((ValueError, FileNotFoundError)):
            cache.publish(output)


def test_only_complete_json_replaces_a_published_record(tmp_path):
    path = tmp_path / "entry.json"
    atomic_write_json(path, {"valid": True})
    with pytest.raises(TypeError):
        atomic_write_json(path, {"invalid": object()})
    assert json.loads(path.read_bytes()) == {"valid": True}
    assert list(tmp_path.iterdir()) == [path]


@pytest.mark.parametrize("damage", ["old_policy", "missing_kind", "wrong_kind", "manifest", "rank", "checked_tokens",
                                     "total_tokens", "passed", "counts_equal", "cascade_counts_equal", "finite",
                                     "cumulative_counts_equal", "fused_vs_cpu", "fused_vs_cascade", "sampled_pair",
                                     "cpu_coverage", "missing_diagnostic", "missing_metrics", "threshold", "bad_verdict"])
def test_dual_cache_and_launcher_require_matching_direct_checks(tmp_path, damage):
    cache, output = case(tmp_path, dual=True, sample=3)
    args = SimpleNamespace(**cache.identity, backend="urma", node_rank=0, ranks_per_node=2)
    assert "prefill" in accuracy_report_line(output, args, "w8a8")
    cache.publish(output)
    other = AccuracyCache(tmp_path, dict(cache.identity, accuracy_cascade="decode"))
    assert cache.index != other.index and other.lookup() is None
    with pytest.raises(ValueError):
        other.publish(output)
    if damage == "manifest":
        manifest_path = output / "rank_1/manifest.json"
        manifest = json.loads(manifest_path.read_bytes())
        manifest.pop("accuracy_cascade")
        atomic_write_json(manifest_path, manifest)
    else:
        for report_path in output.glob("rank_*/bs_16/accuracy.json"):
            rows = json.loads(report_path.read_bytes())
            if damage == "old_policy":
                rows[1].update(policy="cascade_error_budget", triangle={"passed": True})
            elif damage == "missing_kind":
                rows[1].pop("reference_kind")
            elif damage == "wrong_kind":
                rows[1]["reference_kind"] = "decode"
            elif damage in ("fused_vs_cpu", "fused_vs_cascade"):
                rows[1][damage]["passed"] = False
            elif damage == "sampled_pair":
                rows[1]["fused_vs_cascade"]["elements"] = 3 * 32
            elif damage == "cpu_coverage":
                rows[1]["fused_vs_cpu"]["elements"] = 16 * 32
            elif damage == "missing_diagnostic":
                del rows[1]["cascade_vs_cpu"]
            elif damage == "missing_metrics":
                rows[1]["fused_vs_cpu"] = {"passed": True}
            elif damage == "threshold":
                rows[1]["fused_vs_cascade"]["max_diff"] = 0.5
            elif damage == "bad_verdict":
                rows[1]["fused_vs_cascade"]["max_mixed_error"] = 0.11
            else:
                rows[1][damage] = False
            atomic_write_json(report_path, rows)
    assert cache.lookup() is None
    with pytest.raises(ValueError):
        cache.publish(output)
    with pytest.raises(ValueError):
        accuracy_report_line(output, args, "w8a8")


def test_cascade_vs_cpu_failed_diagnostic_can_be_cached(tmp_path):
    cache, output = case(tmp_path, dual=True)
    for report_path in output.glob("rank_*/bs_16/accuracy.json"):
        rows = json.loads(report_path.read_bytes())
        rows[0]["cascade_vs_cpu"].update(passed=False, max_mixed_error=0.15, bad_elements=1)
        atomic_write_json(report_path, rows)
    cache.publish(output)
    assert cache.lookup() == output


@pytest.mark.parametrize("dual", [False, True])
def test_mte_cache_requires_matching_backend_and_input(tmp_path, dual):
    previous, output = case(tmp_path, dual=dual)
    identity = dict(previous.identity, backend="mte", input="fp8")
    cache = AccuracyCache(tmp_path, identity)
    with pytest.raises(ValueError, match="manifest does not match"):
        cache.publish(output)
    for manifest_path in output.glob("rank_*/manifest.json"):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest.update(backend="mte", input="fp8")
        if not dual:
            # Legacy single-reference MTE reports omitted the field.
            manifest.pop("dual_reference")
        atomic_write_json(manifest_path, manifest)
    cache.publish(output)
    assert cache.lookup() == output
    assert previous.lookup() is None
    manifest_path = output / "rank_0/manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    atomic_write_json(manifest_path, dict(manifest, input="bf16"))
    assert cache.lookup() is None


def test_same_entry_lock_serializes_workers(tmp_path):
    path = tmp_path / "nested/entry.lock"
    waiting = Event()
    entered = Event()

    def waiter():
        waiting.set()
        with file_lock(path):
            entered.set()

    with ThreadPoolExecutor(max_workers=1) as pool:
        with file_lock(path):
            future = pool.submit(waiter)
            assert waiting.wait(5)
            assert not entered.wait(0.15)
        future.result(timeout=5)
    assert entered.is_set()


def test_sampled_request_checks_each_ranks_hbm_measurement(tmp_path):
    unsampled, output = case(tmp_path)
    unsampled.publish(output)
    sampled = AccuracyCache(tmp_path, dict(unsampled.identity, profile_memory_peak=True))
    assert sampled.lookup() is None
    for manifest_path in output.glob("rank_*/manifest.json"):
        manifest = json.loads(manifest_path.read_bytes())
        atomic_write_json(manifest_path, dict(manifest, profile_memory_peak=True))
        path = manifest_path.parent / "bs_16/resources.json"
        resources = dict(device_memory_peak=dict(peak_used_bytes=0, sample_count=2),
                         device_memory_peak_group=dict(rank_count=2, max_peak_used_bytes=1))
        atomic_write_json(path, resources)
    sampled.publish(output)
    assert sampled.lookup() == output
    path = output / "rank_1/bs_16/resources.json"
    resources["device_memory_peak"]["peak_used_bytes"] = 1
    atomic_write_json(path, resources)
    assert sampled.lookup() is None
    path.unlink()
    assert sampled.lookup() is None
    with pytest.raises(FileNotFoundError):
        sampled.publish(output)
