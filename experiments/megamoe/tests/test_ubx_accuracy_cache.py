# Copyright (c) 2026, Lu Lu

from concurrent.futures import ThreadPoolExecutor
import json
from threading import Event

import pytest

from experiments.megamoe.scripts.ubx_accuracy_cache import AccuracyCache, atomic_write_json, file_lock


def case(tmp_path, dual=False, sample=0):
    identity = dict(
        ep_size=2, bs=16, check_tokens=sample, dual_reference=dual, routing="balanced",
        config={"model": {"mma_type": "fp8xfp8"}}, build={"revision": "a"}, golden="digest",
    )
    output = tmp_path / "run" / "accuracy-w8a8"
    rows = []
    checked = min(sample or 16, 16)
    for rank in range(2):
        row = dict(finite=True, counts_equal=True, passed=True)
        if dual:
            row.update(policy="cascade_error_budget", checked_tokens=checked, cascade_counts_equal=True,
                       triangle={"passed": True})
        else:
            row.update(policy="cpu_mixed_error", rank=rank, checked_token_count=checked, check_tokens=sample,
                       total_tokens=16, cumulative_counts_equal=True)
        rows.append(row)
    for rank in range(2):
        directory = output / f"rank_{rank}"
        manifest = {key: identity[key] for key in ("config", "routing", "dual_reference", "ep_size", "build")}
        manifest.update(rank=rank, backend="urma", mode="accuracy", input="fp8")
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


def test_mte_packed_fp4_cache_requires_matching_backend_and_input(tmp_path):
    previous, output = case(tmp_path)
    identity = dict(previous.identity, backend="mte", input="fp4")
    cache = AccuracyCache(tmp_path, identity)
    with pytest.raises(ValueError, match="manifest does not match"):
        cache.publish(output)
    for manifest_path in output.glob("rank_*/manifest.json"):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest.update(backend="mte", input="fp4")
        manifest.pop("dual_reference")
        atomic_write_json(manifest_path, manifest)
    cache.publish(output)
    assert cache.lookup() == output
    assert previous.lookup() is None


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
