# Copyright (c) 2026, Lu Lu

"""Reuse completed single-node UBX accuracy reports across launcher invocations."""

from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import tempfile


def cache_key(identity):
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def atomic_write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = tempfile.NamedTemporaryFile(dir=path.parent, delete=False, mode="w", encoding="utf-8")
    try:
        with temporary as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
        os.replace(temporary.name, path)
    finally:
        Path(temporary.name).unlink(missing_ok=True)


@contextmanager
def file_lock(path):
    """Serialize users of this cache entry. Closing the file releases the OS lock."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as stream:
        if os.name == "nt":
            import errno
            import msvcrt
            import time

            stream.seek(0)
            while True:
                try:
                    msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
                    break
                except OSError as exc:
                    if exc.errno != errno.EACCES:
                        raise
                    time.sleep(0.1)
        else:
            import fcntl

            fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        yield


def _check_report(reports, identity, path):
    world, bs, sample = (identity[name] for name in ("ep_size", "bs", "check_tokens"))
    dual = identity["dual_reference"]
    checked = min(sample or bs, bs)
    if not isinstance(reports, list) or len(reports) != world:
        raise ValueError(f"Incomplete group accuracy report: {path}")
    for rank, row in enumerate(reports):
        if not isinstance(row, dict) or any(row.get(key) is not True for key in ("passed", "counts_equal", "finite")):
            raise ValueError(f"Missing or failed accuracy result for rank {rank}: {path}")
        if dual:
            valid = (
                row.get("policy") == "cascade_error_budget"
                and row.get("checked_tokens") == checked
                and row.get("cascade_counts_equal") is True
                and isinstance(row.get("triangle"), dict)
                and row["triangle"].get("passed") is True
            )
        else:
            valid = (
                row.get("policy") == "cpu_mixed_error"
                and row.get("rank") == rank
                and row.get("checked_token_count") == checked
                and row.get("total_tokens") == bs
                and row.get("check_tokens") == sample
                and row.get("cumulative_counts_equal") is True
            )
        if not valid:
            raise ValueError(f"Accuracy result does not match the requested comparison for rank {rank}: {path}")


class AccuracyCache:
    """Index the latest pass for a request. Hold lock() across lookup, launch and publish."""

    def __init__(self, run_base, identity):
        self.base = Path(run_base).resolve()
        self.identity = identity
        self.identity_unknown = None in (identity.get("runtime", ""), identity.get("golden", ""))
        request = {name: value for name, value in identity.items() if name not in ("golden", "runtime")}
        directory = self.base / ".ubx-accuracy-cache"
        self.index = directory / (cache_key(request) + ".request.json")
        self.legacy_index = directory / (cache_key(identity) + ".json")

    def lock(self):
        return file_lock(self.index.with_suffix(".lock"))

    def invalidate(self):
        # Keep a tombstone so failed runs cannot fall back to older per-runtime indexes.
        atomic_write_json(self.index, None)

    def _reports_digest(self, output):
        expected = {name: self.identity[name] for name in ("ep_size", "routing", "config", "dual_reference")}
        expected.update(mode="accuracy", backend=self.identity.get("backend", "urma"), input=self.identity.get("input", "fp8"))
        if "build" in self.identity:
            expected["build"] = self.identity["build"]
        sampled = self.identity.get("profile_memory_peak") is True
        if sampled:
            expected["profile_memory_peak"] = True
        digest = hashlib.sha256()
        group_report = None
        for rank in range(expected["ep_size"]):
            directory = output / f"rank_{rank}"
            manifest_path = directory / "manifest.json"
            manifest_data = manifest_path.read_bytes()
            manifest = json.loads(manifest_data)
            if not isinstance(manifest, dict) or manifest.get("rank") != rank or any(
                manifest.get(name, False if name == "dual_reference" else None) != value for name, value in expected.items()
            ):
                raise ValueError(f"Accuracy manifest does not match this run: {manifest_path}")
            report_path = directory / f"bs_{self.identity['bs']}" / "accuracy.json"
            report_data = report_path.read_bytes()
            if rank == 0:
                _check_report(json.loads(report_data), self.identity, report_path)
                group_report = report_data
            elif report_data != group_report:
                raise ValueError(f"Ranks disagree on the group accuracy report: {report_path}")
            digest.update(hashlib.sha256(manifest_data).digest())
            if sampled:
                resources_path = report_path.with_name("resources.json")
                resources_data = resources_path.read_bytes()
                resources = json.loads(resources_data)
                peak = resources["device_memory_peak"]
                summary = resources["device_memory_peak_group"]
                local, maximum = peak["peak_used_bytes"], summary["max_peak_used_bytes"]
                if (type(local) is not int or type(maximum) is not int or not 0 <= local <= maximum
                        or peak["sample_count"] < 2 or summary["rank_count"] != expected["ep_size"]):
                    raise ValueError(f"Missing or invalid peak HBM measurement: {resources_path}")
                digest.update(hashlib.sha256(resources_data).digest())
        digest.update(hashlib.sha256(group_report).digest())
        return digest.hexdigest()

    def lookup(self):
        if self.identity_unknown:
            return None
        try:
            try:
                data = self.index.read_bytes()
            except FileNotFoundError:
                data = self.legacy_index.read_bytes()
            record = json.loads(data)
            if record is None or record["version"] != 1 or record["identity"] != self.identity:
                return None
            output = (self.base / record["output"]).resolve()
            if not output.is_relative_to(self.base) or self._reports_digest(output) != record["reports_sha256"]:
                return None
            return output
        except (OSError, ValueError, TypeError, KeyError):
            # An interrupted or manually cleaned run can be recomputed in a new directory.
            return None

    def publish(self, output_dir):
        if self.identity_unknown:
            raise ValueError("Cannot publish an accuracy result without Golden and runtime identities")
        output = Path(output_dir).resolve()
        relative = output.relative_to(self.base)
        record = dict(
            version=1, identity=self.identity, output=relative.as_posix(), reports_sha256=self._reports_digest(output)
        )
        atomic_write_json(self.index, record)
