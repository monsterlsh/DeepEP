"""Host-only regression checks; run with python -B -m unittest discover ..."""

import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("ub_probe", HERE / "ub_probe.py")
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)
REPO = HERE.parents[1]


class ProbeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="ub-probe-test-")
        self.root = Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)

    def test_actual_legacy_source_boundary_and_large_batch(self):
        layout = probe.inspect_layout(REPO)
        self.assertEqual(layout["kind"], "legacy_m256")
        cases = probe.capacity_cases(layout, [64, 120, 121, 128, 4096], 253952)
        self.assertEqual([c["input_end_exclusive_bytes"] for c in cases],
                         [196608, 253952, 254976, 262144, 262144])
        self.assertEqual([c["excess_bytes"] for c in cases], [0, 0, 1024, 8192, 8192])
        self.assertEqual(cases[-1]["rows_per_expert"], 8192)
        self.assertEqual(cases[-1]["logical_tiles_per_expert"], 32)

    def test_capacity_is_taken_from_platform_not_always_248k(self):
        layout = probe.inspect_layout(REPO)
        self.assertFalse(probe.capacity_cases(layout, [128], 262144)[0]["exceeds_capacity"])
        self.assertTrue(probe.capacity_cases(layout, [128], 196608)[0]["exceeds_capacity"])

    def test_m128_subtiles_use_smaller_input_addresses(self):
        layout = dict(logical_tile_m=256, ub_tile_m=128, tile_n=256, element_bytes=2, up_start_bytes=65536)
        case = probe.capacity_cases(layout, [4096], 253952)[0]
        self.assertEqual(case["input_end_exclusive_bytes"], 131072)
        self.assertEqual(case["logical_tiles_per_expert"], 32)
        self.assertEqual(case["subtiles_in_largest_tile"], 2)
        self.assertFalse(case["exceeds_capacity"])

    def test_unknown_transfer_layout_is_rejected(self):
        layout = probe.inspect_layout(REPO)
        for relative in layout["source_sha256"]:
            target = self.root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(REPO / relative, target)
        adapter = self.root / probe.KERNEL / "catlass_gmm/mega_moe_mx_matmul.hpp"
        adapter.write_text(adapter.read_text().replace("CopyL0CToUBMode::NO_SPLIT", "CopyL0CToUBMode::SPLIT"))
        with self.assertRaisesRegex(ValueError, "Unrecognized"):
            probe.inspect_layout(self.root)

    def make_reports(self, perf=False):
        if perf:
            data = dict(per_rank_rounds_us=[[10.0, 12.0] for _ in range(8)], mean_us=10.0, rank_max_mean_us=12.0)
        else:
            data = [dict(rank=r, passed=True, counts_equal=True, cumulative_counts_equal=True,
                         checked_token_count=128, total_tokens=128) for r in range(8)]
        for rank in range(8):
            directory = self.root / f"rank_{rank}"
            (directory / "bs_128").mkdir(parents=True)
            probe.write_json(directory / "manifest.json", dict(rank=rank, ep_size=8, weights="fp8", input="fp8",
                routing="balanced", mode="perf" if perf else "accuracy", config=dict(bs_values=[128], model=probe.MODEL)))
            probe.write_json(directory / "bs_128" / ("profiling.json" if perf else "accuracy.json"), data)
        return data

    def test_all_rank_accuracy_reports(self):
        self.make_reports()
        self.assertTrue(probe.validate_report(self.root, 128, "fp8")["all_ranks_passed"])

    def test_missing_rank_is_failure(self):
        self.make_reports()
        (self.root / "rank_7/bs_128/accuracy.json").unlink()
        with self.assertRaises(OSError):
            probe.validate_report(self.root, 128, "fp8")

    def test_numerical_or_count_failure_is_not_hidden(self):
        data = self.make_reports()
        data[3]["counts_equal"] = False
        for rank in range(8):
            probe.write_json(self.root / f"rank_{rank}/bs_128/accuracy.json", data)
        with self.assertRaisesRegex(ValueError, "Accuracy/count"):
            probe.validate_report(self.root, 128, "fp8")

    def test_mismatched_case_manifest_is_failure(self):
        self.make_reports()
        path = self.root / "rank_6/manifest.json"
        data = json.loads(path.read_text())
        data["config"]["bs_values"] = [4096]
        probe.write_json(path, data)
        with self.assertRaisesRegex(ValueError, "manifest"):
            probe.validate_report(self.root, 128, "fp8")

    def test_invalid_performance_samples_are_rejected(self):
        data = self.make_reports(perf=True)
        self.assertEqual(probe.validate_report(self.root, 128, "fp8", True)["rank_max_mean_us"], 12)
        data["per_rank_rounds_us"][2][1] = float("nan")
        for rank in range(8):
            probe.write_json(self.root / f"rank_{rank}/bs_128/profiling.json", data)
        with self.assertRaisesRegex(ValueError, "performance"):
            probe.validate_report(self.root, 128, "fp8", True)

    def sanitizer_logs(self, content):
        for rank in range(8):
            (self.root / f"rank_{rank}.log").write_text(content)

    def test_actual_ub_error_block_is_kept_as_evidence(self):
        self.sanitizer_logs("====== ERROR: illegal write of size 512 byte(s)\n"
                            "====== Access to 0x3e000 on UB\n====== in mega_moe_kernel on device 0\n")
        data = probe.sanitizer_evidence(self.root)
        self.assertTrue(data["ub_illegal_access_reported"])
        self.assertEqual(len(data["findings"]), 8)

    def test_gm_error_plus_unrelated_ub_text_is_not_ub_proof(self):
        self.sanitizer_logs("INFO: UB capacity 253952\n====== ERROR: illegal write of size 16 byte(s)\n"
                            "Access on GM in mega_moe_kernel\n====== ERROR: internal tool error on UB\n")
        data = probe.sanitizer_evidence(self.root)
        self.assertFalse(data["ub_illegal_access_reported"])
        self.assertTrue(data["other_errors"])

    def test_other_kernel_ub_error_is_not_attributed_to_megamoe(self):
        self.sanitizer_logs("====== ERROR: illegal read of size 32 byte(s)\nAccess on UB in another_kernel\n")
        self.assertFalse(probe.sanitizer_evidence(self.root)["ub_illegal_access_reported"])

    def test_sanitizer_silence_and_missing_logs_never_prove_safety(self):
        self.sanitizer_logs("INFO: no errors reported\n")
        data = probe.sanitizer_evidence(self.root)
        self.assertFalse(data["coverage_verified"])
        self.assertFalse(data["ub_illegal_access_reported"])
        (self.root / "rank_4.log").unlink()
        self.assertEqual(probe.sanitizer_evidence(self.root)["missing_rank_logs"], [4])

    def test_launcher_instruments_each_rank_not_only_torchrun_parent(self):
        args = ["--mode", "accuracy", "--weights", "fp8", "--bs-values", "128"]
        normal = probe.launch_command(args)
        instrumented = probe.launch_command(args, True, self.root, "/tools/mssanitizer")
        self.assertIn("--module", normal)
        self.assertNotIn("_rank", normal)
        self.assertIn("--no-python", instrumented)
        self.assertIn("_rank", instrumented)
        self.assertEqual(instrumented[-len(args):], args)

    def test_dry_run_has_no_files_or_device_imports(self):
        output = self.root / "unused"
        result = subprocess.run([sys.executable, "-I", str(HERE / "ub_probe.py"), "--dry-run", "--build",
                                 "--sanitize", "--perf", "--output-dir", str(output)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse(output.exists())
        phases = json.loads(result.stdout)["phases"]
        self.assertLess(phases.index("normal_wheel_performance"), phases.index("instrumented_rebuild"))

    def test_cli_rejects_invalid_bs_and_mixed_static_device_modes(self):
        for args in (["--bs-values", "0"], ["--bs-values", "128,128"], ["--bs-values", "65537"],
                     ["--static-only", "--sanitize"], ["--inputs", "fp4"]):
            with self.subTest(args=args), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                probe.parse_args(args)

    def test_existing_and_source_output_directories_rejected(self):
        for path in (self.root, REPO / "ub-output"):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                probe.parse_args(["--output-dir", str(path)])

    def test_summary_distinguishes_capacity_from_device_evidence(self):
        summary = dict(runs=[], sanitize_requested=False,
                       capacity_cases=[dict(bs=128, rows_per_expert=256, logical_tiles_per_expert=1,
                                            input_end_exclusive_bytes=262144, excess_bytes=8192, exceeds_capacity=True)])
        with contextlib.redirect_stdout(io.StringIO()):
            code = probe.finish(self.root, summary)
        data = json.loads((self.root / "summary.json").read_text())
        self.assertEqual(code, 1)
        self.assertTrue(data["capacity_exceeded"])
        self.assertFalse(data["device_ub_illegal_access_reported"])
        self.assertFalse(data["device_memory_safety_proven"])

    def test_setup_error_and_missing_sanitizer_are_incomplete(self):
        summary = dict(runs=[], sanitize_requested=True, error="tool missing")
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(probe.finish(self.root, summary), 2)
        self.assertIn("tool missing", (self.root / "summary.txt").read_text())

    def simulate_devices(self, health_ok, sanitizer_only=False):
        calls = []

        class FakeRunner:
            def __init__(inner, output, env, timeout):
                inner.output, inner.env = output, env

            def run(inner, label, argv, **kwargs):
                calls.append(label)
                result = dict(exit_code=0, timeout=False, log=str(inner.output / (label + ".log")))
                if label.endswith("_health"):
                    result["exit_code"] = 0 if health_ok else 1
                elif label == "accuracy_fp8_bs128":
                    result["exit_code"] = 1  # a device failure, not proof of UB overflow
                elif label.startswith(("accuracy_", "perf_", "sanitize_")):
                    bs = int(argv[argv.index("--bs-values") + 1])
                    directory = Path(argv[argv.index("--output-dir") + 1])
                    perf = label.startswith("perf_")
                    data = (dict(per_rank_rounds_us=[[10.0] for _ in range(8)], mean_us=10.0, rank_max_mean_us=10.0)
                            if perf else [dict(rank=r, passed=True, counts_equal=True, cumulative_counts_equal=True,
                                              checked_token_count=bs, total_tokens=bs) for r in range(8)])
                    for rank in range(8):
                        case = directory / f"rank_{rank}" / f"bs_{bs}"
                        case.mkdir(parents=True)
                        probe.write_json(case.parent / "manifest.json", dict(rank=rank, ep_size=8, weights="fp8",
                            input="fp8", routing="balanced", mode="perf" if perf else "accuracy",
                            config=dict(bs_values=[bs], model=probe.MODEL)))
                        probe.write_json(case / ("profiling.json" if perf else "accuracy.json"), data)
                        if label.startswith("sanitize_"):
                            (inner.output / f"{label}_memcheck" / f"rank_{rank}.log").write_text("INFO: completed\n")
                return result

            def require(inner, label, argv, **kwargs):
                result = inner.run(label, argv, **kwargs)
                if result["exit_code"]:
                    raise RuntimeError(label)
                return result

        mode = ["--sanitize-only"] if sanitizer_only else ["--build", "--sanitize", "--perf"]
        args = probe.parse_args([*mode, "--bs-values", "128,4096", "--inputs", "fp8"])
        summary = dict(runs=[], sanitize_requested=True)
        with mock.patch.object(probe.sys, "platform", "linux"), mock.patch.object(probe, "Runner", FakeRunner), \
                mock.patch.object(probe, "query_platform", return_value=dict(ub_bytes=253952)), \
                mock.patch.object(probe.shutil, "which", return_value="/tools/mssanitizer"):
            probe.run_devices(args, self.root, summary, probe.inspect_layout(REPO))
        return calls, summary

    def test_device_failure_continues_other_bs_and_keeps_perf_before_instrumentation(self):
        calls, summary = self.simulate_devices(True)
        self.assertIn("accuracy_fp8_bs4096", calls)
        self.assertNotIn("perf_fp8_bs128_r1", calls)
        self.assertIn("sanitize_fp8_bs128", calls)
        self.assertLess(calls.index("perf_fp8_bs4096_r3"), calls.index("build_sanitizer"))
        self.assertFalse(any(r.get("sanitizer", {}).get("ub_illegal_access_reported") for r in summary["runs"]))
        perf = [r for r in summary["runs"] if r["stage"] == "perf" and r["status"] == "passed"]
        self.assertTrue(all(r["diagnostic_only_due_to_capacity_violation"] for r in perf))

    def test_failed_health_check_stops_later_device_launches(self):
        calls, summary = self.simulate_devices(False)
        self.assertNotIn("accuracy_fp8_bs4096", calls)
        self.assertNotIn("build_sanitizer", calls)
        self.assertEqual(summary["sanitizer_status"], "not_run_device_unhealthy")

    def test_detection_only_never_launches_regular_kernel_or_performance(self):
        calls, summary = self.simulate_devices(True, sanitizer_only=True)
        self.assertNotIn("build_regular", calls)
        self.assertNotIn("preflight_regular", calls)
        self.assertFalse(any(label.startswith(("accuracy_", "perf_")) for label in calls))
        self.assertIn("golden_fp8_bs128", calls)
        self.assertIn("sanitize_fp8_bs4096", calls)
        self.assertEqual(calls.count("build_sanitizer"), 1)
        self.assertEqual(summary["installed_variant_after_run"], "sanitizer")


if __name__ == "__main__":
    unittest.main()
