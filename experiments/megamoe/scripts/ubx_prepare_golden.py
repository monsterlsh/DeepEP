# Copyright (c) 2026, Lu Lu

"""Prepare the installed MegaMoE CPU reference and identify its validated outputs."""

import argparse
import json
import os
from pathlib import Path
import sys


# The launcher sets PYTHONSAFEPATH. Only the cache helpers come from this checkout.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ubx_accuracy_cache import atomic_write_json, cache_key, file_lock  # noqa: E402

sys.path.pop(0)


def file_stamp(path):
    stat = path.stat()
    return [stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns]


def inspect_rank(golden, directory, spec, rank, records):
    import numpy as np

    path = golden.cache_path(directory, spec, rank)
    stamp = file_stamp(path)
    prior = records.get(str(rank))
    if isinstance(prior, dict) and prior.get("stamp") == stamp:
        sha = prior.get("sha256")
        if isinstance(sha, str) and len(sha) == 64:
            return prior
    # Validate old/changed files with precisely the checks used by the NPU benchmark.
    golden.load(directory, spec, rank)
    with np.load(path, allow_pickle=False) as saved:
        checksum = json.loads(str(saved["metadata"]))["sha256"]
    if file_stamp(path) != stamp:
        raise RuntimeError(f"Golden changed during validation: {path}")
    return {"stamp": stamp, "sha256": checksum}


def prepare(config, spec, ranks, directory, *, workers, chunk_tokens, force, reuse_only=False):
    from deep_ep_experimental.megamoe import golden

    cache = golden.cache_path(directory, spec, ranks[0]).parent
    cache.mkdir(parents=True, exist_ok=True)
    index = cache / ".ubx-validated-v1.json"
    with file_lock(cache / ".ubx-prepare.lock"):
        try:
            ledger = json.loads(index.read_text(encoding="utf-8"))
            records = ledger["ranks"]
            if not isinstance(records, dict) or ledger["checksum"] != cache_key(records):
                records = {}
        except (FileNotFoundError, ValueError, TypeError, KeyError):
            records = {}
        original = records.copy()
        reused, missing = [], []
        for rank in ranks:
            if not force:
                try:
                    records[str(rank)] = inspect_rank(golden, directory, spec, rank, records)
                    reused.append(rank)
                    continue
                except FileNotFoundError:
                    pass
                except ValueError as error:
                    print(f"[GOLDEN] invalid rank {rank}: {error}", flush=True)
            missing.append(rank)
        if reused:
            print(f"[GOLDEN] reuse ranks={reused} {cache}", flush=True)

        if missing and not reuse_only:
            # Bound full-check memory at one rank, as in the original EP16 launcher.
            checked = min(spec["check_tokens"] or spec["bs"], spec["bs"])
            batch = spec["bs"] // checked
            for start in range(0, len(missing), batch):
                group = missing[start : start + batch]
                outputs = golden.generate(config, spec, group, workers=workers, chunk_tokens=chunk_tokens)
                for rank in group:
                    golden.save(directory, spec, rank, *outputs.pop(rank))
                    records[str(rank)] = inspect_rank(golden, directory, spec, rank, {})
                    print(f"[GOLDEN] saved rank={rank} {golden.cache_path(directory, spec, rank)}", flush=True)

        if records != original:
            atomic_write_json(index, dict(ranks=records, checksum=cache_key(records)))
        if reuse_only and missing:
            return dict(config=config, identity=None, generated=[], reused=reused, missing=missing)
        checksums = {str(rank): records[str(rank)]["sha256"] for rank in ranks}
        identity = cache_key(dict(contract=spec, ranks=checksums))
        summary = dict(config=config, identity=identity, generated=missing, reused=reused)
        if reuse_only:
            summary["missing"] = []
        return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--ep-size", type=int, required=True)
    parser.add_argument("--ranks", required=True)
    parser.add_argument("--bs-values", type=int, required=True, metavar="BS")
    parser.add_argument("--golden-dir", type=Path, required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--routing", required=True)
    parser.add_argument("--check-tokens", type=int, required=True)
    parser.add_argument("--workers", type=int, required=True)
    parser.add_argument("--chunk-tokens", type=int, default=128)
    parser.add_argument("--summary", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--force", action="store_true")
    mode.add_argument("--reuse-only", action="store_true", help="Report missing or damaged ranks without generating Golden")
    args = parser.parse_args()
    if args.workers < 1 or args.chunk_tokens < 1:
        parser.error("workers and chunk-tokens must be positive")
    cpus = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "BLIS_NUM_THREADS"):
        os.environ[name] = str(max(1, cpus // args.workers))
    from deep_ep_experimental.megamoe import golden
    from deep_ep_experimental.megamoe.benchmark import load_config
    from deep_ep_experimental.megamoe.timer.__main__ import parse_ranks

    ranks = parse_ranks(args.ranks)
    if max(ranks) >= args.ep_size:
        parser.error("ranks must be within EP size")
    config = load_config(args.config, str(args.bs_values))
    spec = golden.contract(config, args.ep_size, args.bs_values, args.input, args.routing, args.check_tokens)
    summary = prepare(
        config, spec, ranks, args.golden_dir, workers=args.workers, chunk_tokens=args.chunk_tokens,
        force=args.force, reuse_only=args.reuse_only,
    )
    atomic_write_json(args.summary, summary)


if __name__ == "__main__":
    main()
