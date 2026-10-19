"""Legacy A8W8 UB diagnosis; stdlib-only orchestration, no implicit device imports."""

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile


HERE = Path(__file__).resolve().parent
BASELINE = "b6ab7636679e9dd87dbb1da66464a33a024b38e1"
KERNEL = "experiments/megamoe/csrc/kernels/mega_moe"
MODEL = dict(hidden=1024, intermediate_hidden=256, num_experts=8, num_topk=2, num_shared_experts=0)
EP = 8


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def git(repo, *args):
    env = {k: v for k, v in os.environ.items() if k not in ("LD_LIBRARY_PATH", "LD_PRELOAD")}
    return subprocess.check_output(["git", "-C", str(repo), *args], env=env, text=True, timeout=30).strip()


def inspect_layout(repo):
    paths = [f"{KERNEL}/{name}" for name in (
        "mega_moe_wave_a8w8.hpp", "stage/mega_moe_gmm1_activation.hpp",
        "catlass_gmm/mega_moe_mx_matmul.hpp", "mega_moe.cpp", "common/mega_moe_constants.hpp",
    )]
    source = {p: (repo / p).read_text(encoding="utf-8") for p in paths}
    wave, stage, adapter, entry, constants = source.values()
    compact = lambda s: re.sub(r"\s+", "", s)
    if ("CopyL0CToUBMode::NO_SPLIT" not in adapter or
            "tla::MakeLayout<C,Row>(rows,256U)" not in compact(adapter) or
            not re.search(r"L1_TILE_N\s*=\s*256U?\b", constants) or
            "COMBINE_NO_QUANT,false,false>" not in compact(entry)):
        raise ValueError("Unrecognized A8W8 N256/AIV0 layout or entry path; refuse to assume its addresses")
    old = ("EPILOGUE_TILE_M=TopkWeightsPrefetch?L1_TILE_M_128:L1_TILE_M_256;" in compact(wave)
           and "ubBufSize = config.tileM" in stage
           and "ubOffsetSecond = static_cast<int64_t>(ubBufSize) * sizeof(ElementC)" in stage)
    policy = repo / KERNEL / "common/mega_moe_a8w8_policy.hpp"
    new = (policy.is_file() and "subOffset += UbTileM" in stage and "!GMM1_INTERLEAVED" in wave)
    if old:
        ub_m, name = 256, "legacy_m256"
    elif new:
        policy_text = policy.read_text(encoding="utf-8")
        match = re.search(r"A8W8_GMM1_UB_TILE_M\s*=\s*(\d+)U", policy_text)
        if not match or int(match[1]) != 128:
            raise ValueError("Unrecognized UB subtile policy")
        ub_m, name = 128, "subtile_m128"
        source[str(policy.relative_to(repo))] = policy_text
    else:
        raise ValueError("Unrecognized GMM1 implementation; layout evidence is unavailable")
    return dict(kind=name, logical_tile_m=256, ub_tile_m=ub_m, tile_n=256,
                element_bytes=2, up_start_bytes=ub_m * 256 * 2,
                source_sha256={p: hashlib.sha256(s.encode()).hexdigest() for p, s in source.items()})


def capacity_cases(layout, batches, ub_bytes):
    cases = []
    for bs in batches:
        rows = EP * bs * MODEL["num_topk"] // MODEL["num_experts"]
        tile_rows = min(rows, layout["logical_tile_m"])
        piece_rows = min(tile_rows, layout["ub_tile_m"])
        end = layout["up_start_bytes"] + piece_rows * layout["tile_n"] * layout["element_bytes"]
        cases.append(dict(bs=bs, rows_per_expert=rows, logical_tiles_per_expert=math.ceil(rows / 256),
                          largest_tile_rows=tile_rows, largest_subtile_rows=piece_rows,
                          subtiles_in_largest_tile=math.ceil(tile_rows / layout["ub_tile_m"]),
                          gate_start_bytes=0, up_start_bytes=layout["up_start_bytes"],
                          input_end_exclusive_bytes=end, available_ub_bytes=ub_bytes,
                          exceeds_capacity=end > ub_bytes, excess_bytes=max(0, end - ub_bytes)))
    return cases


def cli_mode(repo):
    directory = repo / "experiments/megamoe/python/deep_ep_experimental/megamoe"
    texts = [(directory / name).read_text(encoding="utf-8") for name in ("benchmark.py", "golden.py")]
    for option, value in (("--mma-type", "fp8xfp8"), ("--weights", "fp8")):
        if all(f'"{option}"' in text for text in texts):
            return [option, value]
    raise ValueError("No compatible benchmark/Golden CLI")


def preflight(repo, output, sanitized):
    # Runs only in an isolated Linux subprocess after explicit device execution.
    from importlib.metadata import distributions, version
    from importlib.resources import files
    import torch
    import torch_npu
    import deep_ep
    import deep_ep_experimental.megamoe as megamoe
    import deep_ep_experimental.megamoe._C as native
    from deep_ep._native import require_native

    expected = (repo / "VERSION").read_text(encoding="utf-8").strip()
    installed = version("ascend-deepep")
    if installed != expected or len(list(distributions(name="ascend-deepep"))) != 1:
        raise RuntimeError(f"Installed wheel/metadata mismatch: {installed}, expected unique {expected}")
    for module in (deep_ep, megamoe):
        if Path(module.__file__).resolve().is_relative_to(repo):
            raise RuntimeError("Source checkout shadows the installed wheel")
    info = json.loads(files(deep_ep).joinpath("_build_info.json").read_text())
    spec = importlib.util.spec_from_file_location("ub_probe_build_support", repo / "build_support.py")
    support = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(support)
    if info.get("source_sha256") != support.source_fingerprint():
        raise RuntimeError("Wheel source fingerprint differs; rebuild this checkout")
    mega = info.get("megamoe", {})
    if (not mega.get("enabled") or mega.get("timer") or native.api_protocol != 4 or
            bool(mega.get("ub_probe_sanitizer")) != sanitized):
        raise RuntimeError("Wrong build variant: require MegaMoE, timer OFF, matching sanitizer marker")
    if torch.npu.device_count() != EP:
        raise RuntimeError("This probe requires exactly 8 visible NPUs")
    require_native("UB probe preflight")
    write_json(output, dict(version=installed, build=info, python=sys.executable,
                           torch=torch.__version__, torch_npu=torch_npu.__version__,
                           devices=[torch.npu.get_device_name(i) for i in range(EP)],
                           package=str(Path(megamoe.__file__).resolve())))


def validate_report(directory, bs, input_kind, perf=False):
    """A successful launcher alone is insufficient: check all rank reports/manifests."""
    report_name = "profiling.json" if perf else "accuracy.json"
    reference = None
    for rank in range(EP):
        rank_dir = directory / f"rank_{rank}"
        manifest = json.loads((rank_dir / "manifest.json").read_text())
        mma = manifest.get("mma_type", {"fp8": "fp8xfp8"}.get(manifest.get("weights")))
        if (manifest["rank"] != rank or manifest["ep_size"] != EP or mma != "fp8xfp8" or
                manifest["input"] != input_kind or manifest["routing"] != "balanced" or
                manifest["mode"] != ("perf" if perf else "accuracy") or
                manifest["config"]["bs_values"] != [bs] or manifest["config"]["model"] != MODEL):
            raise ValueError(f"Wrong manifest for rank {rank}")
        report = json.loads((rank_dir / f"bs_{bs}" / report_name).read_text())
        if rank == 0:
            reference = report
        elif report != reference:
            raise ValueError(f"Rank {rank} report differs from rank 0")
    if perf:
        samples = reference["per_rank_rounds_us"]
        if (len(samples) != EP or not samples[0] or any(len(row) != len(samples[0]) for row in samples) or
                any(not math.isfinite(v) or v <= 0 for row in samples for v in row) or
                any(not math.isfinite(reference[k]) or reference[k] <= 0 for k in ("mean_us", "rank_max_mean_us"))):
            raise ValueError("Invalid performance samples")
        return {k: reference[k] for k in ("mean_us", "rank_max_mean_us")}
    if (len(reference) != EP or {row["rank"] for row in reference} != set(range(EP)) or
            not all(row["passed"] is True and row["counts_equal"] is True and
                    row["cumulative_counts_equal"] is True and row["checked_token_count"] == bs and
                    row["total_tokens"] == bs for row in reference)):
        raise ValueError("Accuracy/count checks failed or some tokens/ranks were not checked")
    return dict(all_ranks_passed=True, checked_tokens_per_rank=bs)


def sanitizer_evidence(directory):
    findings, missing, other_errors = [], [], []
    for rank in range(EP):
        path = directory / f"rank_{rank}.log"
        if not path.is_file() or path.stat().st_size == 0:
            missing.append(rank)
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        # Keep each ERROR block separate: an unrelated UB mention is not proof.
        blocks = re.split(r"(?=^.*={3,}\s*ERROR:)", text, flags=re.MULTILINE)[1:]
        for block in blocks:
            excerpt = block[:12000]
            normalized = excerpt.lower().replace("_", "")
            target = "megamoe" in normalized
            ub_access = bool(re.search(r"illegal\s+(read|write)", excerpt, re.I) and
                             re.search(r"\bUB\b|Unified Buffer", excerpt, re.I))
            item = dict(rank=rank, path=str(path), excerpt=excerpt)
            (findings if target and ub_access else other_errors).append(item)
        if "ERROR" in text and not blocks:
            other_errors.append(dict(rank=rank, path=str(path), excerpt=text[-12000:]))
    return dict(ub_illegal_access_reported=bool(findings), findings=findings, other_errors=other_errors,
                missing_rank_logs=missing, coverage_verified=False,
                interpretation="A report can confirm an illegal UB access in MegaMoE; silence cannot prove safety. "
                               "Check the stack/PC to attribute it specifically to GMM1. "
                               "Ascend950 SIMD VF coverage and tool/SHMEM restrictions still apply.")


class Runner:
    def __init__(self, output, env, timeout):
        self.output, self.env, self.timeout = output, env, timeout
        self.commands = []

    def run(self, label, argv, *, env=None, timeout=None):
        log = self.output / "logs" / f"{label}.log"
        record = dict(label=label, argv=[str(x) for x in argv], log=str(log), timeout=False)
        self.commands.append(record)
        print(f"[{label}] {shlex.join(record['argv'])}\n  log: {log}", flush=True)
        process = None
        try:
            with log.open("w", encoding="utf-8") as stream:
                process = subprocess.Popen(record["argv"], cwd=self.output, env=env or self.env,
                                           stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
                try:
                    record["exit_code"] = process.wait(timeout=timeout or self.timeout)
                except subprocess.TimeoutExpired:
                    record["timeout"] = True
                    record["exit_code"] = 124
        except OSError as error:
            record.update(exit_code=127, error=str(error))
        finally:
            # Only terminate the process group created for this launch, never other jobs.
            if process is not None and (record.get("exit_code", 1) != 0 or process.poll() is None):
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        pass
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    process.wait()
                except ProcessLookupError:
                    pass
            write_json(self.output / "commands.json", self.commands)
        return record

    def require(self, label, argv, **kwargs):
        result = self.run(label, argv, **kwargs)
        if result["exit_code"]:
            raise RuntimeError(f"{label} failed (exit {result['exit_code']}); see {result['log']}")
        return result


def launch_command(arguments, sanitized=False, logs=None, sanitizer=None):
    command = [sys.executable, "-I", "-c", "from torch.distributed.run import main; main()",
               "--standalone", "--nnodes=1", "--nproc-per-node=8", "--max-restarts=0"]
    if sanitized:
        return command + ["--no-python", sys.executable, "-I", str(HERE / "ub_probe.py"),
                          "_rank", str(logs), str(sanitizer), *arguments]
    return command + ["--module", "deep_ep_experimental.megamoe.benchmark", *arguments]


def query_platform(runner):
    cann_value = runner.env.get("ASCEND_HOME_PATH")
    if not cann_value:
        raise RuntimeError("ASCEND_HOME_PATH is required to query target UB capacity")
    cann = Path(cann_value)
    compiler = runner.env.get("CXX") or shutil.which("c++") or shutil.which("g++")
    if not compiler:
        raise RuntimeError("A host C++ compiler is required for the CANN platform probe")
    binary = runner.output / "ub_platform"
    runner.require("platform_compile", [compiler, "-std=c++17", str(HERE / "ub_platform.cpp"),
                    f"-I{cann / 'include'}", f"-I{cann / 'include/experiment'}",
                    f"-I{cann / 'pkg_inc'}", f"-L{cann / 'lib64'}", f"-Wl,-rpath,{cann / 'lib64'}",
                    "-ltiling_api", "-lplatform", "-lascendcl", "-ldl", "-o", str(binary)], timeout=120)
    record = runner.require("platform_query", [str(binary)], timeout=60)
    text = Path(record["log"]).read_text()
    match = re.search(r"^UBPROBE_JSON=(.+)$", text, re.M)
    data = json.loads(match[1]) if match else {}
    if type(data.get("ub_bytes")) is not int or data["ub_bytes"] <= 0:
        raise RuntimeError("No valid platform capacity report; refuse to assume 248 KiB")
    data["source"] = "CANN PlatformAscendCManager::GetCoreMemSize(UB), visible device 0"
    return data


def run_devices(args, output, summary, layout):
    if sys.platform != "linux":
        raise RuntimeError("NPU execution requires Linux; use --static-only or --dry-run here")
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env.update(PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="1")
    env.setdefault("ASCEND_RT_VISIBLE_DEVICES", "0,1,2,3,4,5,6,7")
    env.setdefault("DEEPEP_SHMEM_ENDPOINT", "tcp://127.0.0.1:26733")
    for name in ("SHMEM_ROOT", "CATLASS_ROOT", "CONDA_PREFIX", "ASCEND_HOME_PATH"):
        if env.get(name):
            env[name] = str(Path(env[name]).resolve())
    shmem = Path(env.get("SHMEM_ROOT", args.repo / "third-party/shmem/install/shmem"))
    libs = [str(shmem / "lib"), env.get("LD_LIBRARY_PATH", "")]
    if env.get("CONDA_PREFIX"):
        libs.insert(0, str(Path(env["CONDA_PREFIX"]) / "lib"))
    env["LD_LIBRARY_PATH"] = ":".join(libs)
    runner = Runner(output, env, args.timeout)
    modes = cli_mode(args.repo)

    def build(sanitized):
        summary["installed_variant_after_run"] = "unknown_build_in_progress"
        build_env = dict(env, PYTHON=sys.executable, DEEPEP_BUILD_MEGAMOE="ON", DEEPEP_MEGAMOE_TIMER="OFF",
                         DEEPEP_UB_PROBE_SANITIZER="ON" if sanitized else "OFF")
        runner.require("build_sanitizer" if sanitized else "build_regular",
                       ["bash", str(args.repo / "experiments/megamoe/scripts/build.sh")],
                       env=build_env, timeout=args.build_timeout)
        summary["installed_variant_after_run"] = "sanitizer" if sanitized else "regular"

    def check(sanitized):
        label = "preflight_sanitizer" if sanitized else "preflight_regular"
        runner.require(label, [sys.executable, "-I", str(HERE / "ub_probe.py"), "_preflight",
                               str(args.repo), str(output / f"{label}.json"), str(int(sanitized))], timeout=120)

    if args.sanitize_only:
        if not shutil.which("mssanitizer"):
            raise RuntimeError("mssanitizer not found")
        if "DEEPEP_UB_PROBE_SANITIZER" not in (args.repo / "build_support.py").read_text(encoding="utf-8"):
            raise RuntimeError("Target checkout lacks diagnostic build support; use this probe branch")
        build(True)
    elif args.build:
        build(False)
    check(args.sanitize_only)
    summary["capacity"] = query_platform(runner)
    summary["capacity_cases"] = capacity_cases(layout, args.bs_values, summary["capacity"]["ub_bytes"])
    write_json(output / "capacity.json", dict(layout=layout, platform=summary["capacity"],
                                             cases=summary["capacity_cases"]))
    config = dict(schema_version=1, model=MODEL, bs_values=args.bs_values, warmup=0, iters=1, seed=20260922)
    accuracy_config, perf_config = output / "accuracy_config.json", output / "perf_config.json"
    write_json(accuracy_config, config)
    write_json(perf_config, dict(config, warmup=10, iters=20))
    healthy = True

    def device_case(bs, input_kind, stage, repeat=0, sanitizer=None):
        nonlocal healthy
        label = f"{stage}_{input_kind}_bs{bs}" + (f"_r{repeat}" if repeat else "")
        record = dict(stage=stage, bs=bs, input=input_kind, repeat=repeat, status="not_run")
        if stage == "perf":
            record["diagnostic_only_due_to_capacity_violation"] = next(
                c["exceeds_capacity"] for c in summary["capacity_cases"] if c["bs"] == bs)
        summary["runs"].append(record)
        if not healthy:
            record["reason"] = "Earlier device health check failed; stop device launches"
            return False
        perf = stage == "perf"
        if not perf and (stage != "sanitize" or args.sanitize_only):
            golden = runner.run(f"golden_{input_kind}_bs{bs}", [sys.executable, "-I", "-m",
                "deep_ep_experimental.megamoe.golden", "--config", str(accuracy_config), *modes,
                "--ep-size", "8", "--ranks", "0-7", "--bs-values", str(bs), "--input", input_kind,
                "--routing", "balanced", "--check-tokens", "0", "--golden-dir", str(output / "golden")])
            if golden["exit_code"]:
                record.update(status="golden_failed", command=golden)
                return False
        directory = output / label
        arguments = ["--config", str(perf_config if perf else accuracy_config), *modes,
                     "--mode", "perf" if perf else "accuracy", "--input", input_kind,
                     "--routing", "balanced", "--bs-values", str(bs), "--output-dir", str(directory)]
        if not perf:
            arguments += ["--check-tokens", "0", "--golden-dir", str(output / "golden")]
        sanitizer_logs = output / f"{label}_memcheck"
        if sanitizer:
            sanitizer_logs.mkdir()
        command = runner.run(label, launch_command(arguments, bool(sanitizer), sanitizer_logs, sanitizer))
        record.update(command=command, status="timeout" if command["timeout"] else "runtime_failed")
        if command["exit_code"] == 0:
            try:
                record["metrics"] = validate_report(directory, bs, input_kind, perf)
                record["status"] = "passed"
            except (OSError, ValueError, KeyError, TypeError, IndexError) as error:
                record.update(status="report_failed", reason=str(error))
        if sanitizer:
            evidence = sanitizer_evidence(sanitizer_logs)
            record["sanitizer"] = evidence
            if evidence["ub_illegal_access_reported"]:
                record["status"] = "ub_illegal_access_reported"
            elif evidence["other_errors"]:
                record["status"] = "sanitizer_error"
            elif evidence["missing_rank_logs"]:
                record["status"] = "sanitizer_incomplete"
            elif record["status"] == "passed":
                record["status"] = "no_ub_error_reported_coverage_unverified"
        if command["exit_code"]:
            health = runner.run(label + "_health", [sys.executable, "-I", str(HERE / "ub_probe.py"),
                                                      "_health"], timeout=60)
            healthy = health["exit_code"] == 0
            record["device_health"] = health
        write_json(output / "summary.json", summary)
        return record["status"] == "passed"

    regular_ok = {}
    if not args.sanitize_only:
        for bs in args.bs_values:
            for input_kind in args.inputs:
                regular_ok[bs, input_kind] = device_case(bs, input_kind, "accuracy")
    if args.perf:
        for bs in args.bs_values:
            for input_kind in args.inputs:
                for repeat in range(1, args.repeats + 1):
                    if regular_ok[bs, input_kind]:
                        device_case(bs, input_kind, "perf", repeat)
                    else:
                        summary["runs"].append(dict(stage="perf", bs=bs, input=input_kind, repeat=repeat,
                                                    status="not_run", reason="Accuracy did not pass"))
    if args.sanitize:
        if not healthy:
            summary["sanitizer_status"] = "not_run_device_unhealthy"
        else:
            sanitizer = shutil.which("mssanitizer")
            if not sanitizer:
                raise RuntimeError("mssanitizer not found; ordinary accuracy is not a UB memory check")
            summary["sanitizer_version_command"] = runner.require("sanitizer_version", [sanitizer, "--version"], timeout=30)
            # Sanitized and normal variants have separate build-cache identities.
            # End with the instrumented wheel installed; never use it for performance.
            if "DEEPEP_UB_PROBE_SANITIZER" not in (args.repo / "build_support.py").read_text(encoding="utf-8"):
                raise RuntimeError("Target checkout lacks diagnostic build support; use this probe branch")
            if not args.sanitize_only:
                build(True)
                check(True)
            for bs in args.bs_values:
                for input_kind in args.inputs:
                    failed_golden = any(r["stage"] == "accuracy" and r["bs"] == bs and r["input"] == input_kind
                                        and r["status"] in ("golden_failed", "not_run") for r in summary["runs"])
                    if failed_golden:
                        summary["runs"].append(dict(stage="sanitize", bs=bs, input=input_kind,
                                                    status="not_run", reason="No validated Golden / device unhealthy"))
                    else:
                        device_case(bs, input_kind, "sanitize", sanitizer=sanitizer)


def finish(output, summary):
    capacity_violation = any(c["exceeds_capacity"] for c in summary.get("capacity_cases", []))
    runs = summary["runs"]
    observed = any(r.get("sanitizer", {}).get("ub_illegal_access_reported") for r in runs)
    failed = any(r["status"] not in ("passed", "no_ub_error_reported_coverage_unverified") for r in runs)
    incomplete = bool(summary.get("error")) or (summary["sanitize_requested"] and
                 (not any(r["stage"] == "sanitize" for r in runs) or
                  any(r["status"] in ("sanitizer_incomplete", "not_run") for r in runs if r["stage"] == "sanitize")))
    summary.update(capacity_exceeded=capacity_violation, device_ub_illegal_access_reported=observed,
                   device_memory_safety_proven=False, finished_utc=datetime.now(timezone.utc).isoformat(),
                   exit_code=2 if incomplete else (1 if capacity_violation or observed or failed else 0))
    lines = [f"Source: {summary.get('branch')} {summary.get('commit')}",
             f"Layout: {summary.get('layout', {}).get('kind', 'unknown')}",
             f"Capacity evidence: {summary.get('capacity')}"]
    for case in summary.get("capacity_cases", []):
        lines.append(f"BS={case['bs']} expert_rows={case['rows_per_expert']} "
                     f"M256_tiles={case['logical_tiles_per_expert']} input_end={case['input_end_exclusive_bytes']} "
                     f"excess_bytes={case['excess_bytes']}")
    lines.extend(f"{r['stage']} BS={r['bs']} input={r['input']}: {r['status']} {r.get('metrics', '')}" for r in runs)
    if summary.get("error"):
        lines.append("ERROR: " + summary["error"])
    lines += [f"Device UB illegal access reported: {observed}",
              "No UB report does not prove safety. Static estimates are not live UB measurements.",
              f"Exit: {summary['exit_code']}; reports: {output}"]
    write_json(output / "summary.json", summary)
    (output / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)
    return summary["exit_code"]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=HERE.parents[1])
    parser.add_argument("--bs-values", default="64,120,121,128,4096")
    parser.add_argument("--inputs", default="fp8,bf16")
    parser.add_argument("--build", action="store_true", help="Build/install normal wheel before tests")
    parser.add_argument("--sanitize", action="store_true", help="After normal tests, rebuild/instrument and memcheck each rank")
    parser.add_argument("--sanitize-only", action="store_true", help="Build and run detection directly, without normal device launches")
    parser.add_argument("--perf", action="store_true", help="Measure normal wheel only after accuracy passes")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=1800, help="Timeout per Golden/device launch, seconds")
    parser.add_argument("--build-timeout", type=int, default=7200)
    parser.add_argument("--static-only", action="store_true", help="CPU address analysis only, no dependencies/devices")
    parser.add_argument("--ub-bytes", type=int, default=248 * 1024, help="Assumed capacity for static-only; device runs query CANN")
    parser.add_argument("--output-dir", type=Path, help="New directory outside source checkout")
    parser.add_argument("--dry-run", action="store_true", help="Print plan only; no output files/builds/imports")
    args = parser.parse_args(argv)
    if args.sanitize_only:
        args.sanitize = True
        if args.perf:
            parser.error("--sanitize-only cannot measure normal performance")
    try:
        args.bs_values = [int(x) for x in args.bs_values.split(",")]
        if not args.bs_values or len(set(args.bs_values)) != len(args.bs_values) or any(not 1 <= b <= 65536 for b in args.bs_values):
            raise ValueError()
    except ValueError:
        parser.error("BS values must be distinct integers in [1,65536]")
    args.inputs = args.inputs.split(",")
    if not args.inputs or len(set(args.inputs)) != len(args.inputs) or any(x not in ("fp8", "bf16") for x in args.inputs):
        parser.error("--inputs must be fp8, bf16, or fp8,bf16")
    if min(args.repeats, args.timeout, args.build_timeout, args.ub_bytes) <= 0:
        parser.error("Counts, timeouts and capacity must be positive")
    if args.static_only and (args.build or args.sanitize or args.perf):
        parser.error("--static-only cannot build or run device tests")
    args.repo = args.repo.resolve()
    if args.output_dir:
        args.output_dir = args.output_dir.resolve()
        if args.output_dir.is_relative_to(args.repo) or args.output_dir.is_relative_to(HERE.parents[1]):
            parser.error("Output must be outside both the target and probe checkouts")
        if args.output_dir.exists():
            parser.error("Output already exists; select a new directory")
    return args


def main(argv=None):
    args = parse_args(argv)
    if args.dry_run:
        phases = (["static_address_analysis"] if args.static_only else
                  ["instrumented_rebuild", "wheel_preflight", "CANN_UB_query", "Golden", "per_rank_memcheck"]
                  if args.sanitize_only else ["wheel_preflight", "CANN_UB_query", "Golden", "accuracy"] +
                  (["normal_wheel_performance"] if args.perf else []) +
                  (["instrumented_rebuild", "per_rank_memcheck"] if args.sanitize else []))
        print(json.dumps(dict(repo=str(args.repo), ep_size=EP, model=MODEL, bs_values=args.bs_values,
                              inputs=args.inputs, static_only=args.static_only, build=args.build,
                              phases=phases,
                              repeats=args.repeats, timeout=args.timeout, no_side_effects=True), indent=2))
        return 0
    output = args.output_dir or Path(tempfile.mkdtemp(prefix="deepep-ub-probe-"))
    if args.output_dir:
        output.mkdir(parents=True, exist_ok=False)
    (output / "logs").mkdir()
    summary = dict(baseline=BASELINE, source=str(args.repo), runs=[], static_only=args.static_only,
                   sanitize_requested=args.sanitize, started_utc=datetime.now(timezone.utc).isoformat(),
                   installed_variant_after_run="unchanged")
    try:
        summary.update(branch=git(args.repo, "branch", "--show-current"), commit=git(args.repo, "rev-parse", "HEAD"),
                       source_status=git(args.repo, "status", "--porcelain"),
                       version=(args.repo / "VERSION").read_text(encoding="utf-8").strip())
        layout = inspect_layout(args.repo)
        summary["layout"] = layout
        if args.static_only:
            summary["capacity"] = dict(ub_bytes=args.ub_bytes, source="explicit/default assumption; not queried from hardware")
            summary["capacity_cases"] = capacity_cases(layout, args.bs_values, args.ub_bytes)
            write_json(output / "capacity.json", dict(layout=layout, platform=summary["capacity"], cases=summary["capacity_cases"]))
        else:
            run_devices(args, output, summary, layout)
    except (Exception, KeyboardInterrupt) as error:
        summary["error"] = f"{type(error).__name__}: {error}"
    return finish(output, summary)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "_preflight":
        preflight(Path(sys.argv[2]), Path(sys.argv[3]), bool(int(sys.argv[4])))
    elif len(sys.argv) > 1 and sys.argv[1] == "_health":
        import torch
        import torch_npu
        for device in range(EP):
            torch.npu.set_device(device)
            value = torch.ones(1, device="npu") + 1
            torch.npu.synchronize()
            if value.item() != 2:
                raise RuntimeError(f"Device {device} health check failed")
    elif len(sys.argv) > 1 and sys.argv[1] == "_rank":
        logs, executable = Path(sys.argv[2]), sys.argv[3]
        rank = int(os.environ["RANK"])
        command = [executable, "--tool=memcheck", "--log-level=info", "--kernel-name=mega_moe_kernel",
                   "--log-file", str(logs / f"rank_{rank}.log"), "--", sys.executable, "-I", "-m",
                   "deep_ep_experimental.megamoe.benchmark", *sys.argv[4:]]
        os.execvpe(executable, command, dict(os.environ, PYTORCH_NO_NPU_MEMORY_CACHING="1"))
    else:
        raise SystemExit(main())
