# Copyright (c) 2026, Lu Lu

import json
import sys

import pytest

from experiments.megamoe.scripts import export_k01_results as export


def run(tmp_path, data):
    path = tmp_path / "accuracy.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return export.accuracy(path, dict(ep_size=len(data), dual_reference=True, accuracy_cascade="prefill"))[0]


def rank_report(rank, checked=export.BS):
    def pair(tokens, error, bad=0):
        return dict(elements=tokens * export.MODEL["hidden"], max_abs_error=error, bad_elements=bad, passed=True,
                    max_mixed_error=0.02 if bad else 0.005, diff_thd=0.01, pct_thd=0.05, max_diff=0.1, reason="")

    return dict(policy="direct_dual_reference", reference_kind="prefill", rank=rank, passed=True, finite=True,
                counts_equal=True, cascade_counts_equal=True, cumulative_counts_equal=True,
                checked_tokens=checked, total_tokens=export.BS,
                fused_vs_cpu=pair(checked, 0.01 + rank, 1), cascade_vs_cpu=pair(checked, 0.02 + rank),
                fused_vs_cascade=pair(export.BS, 0.03 + rank))


def test_exports_workbook_columns_and_both_verdicts(tmp_path, capsys):
    reports = [rank_report(0), rank_report(1)]
    reports[1]["cascade_vs_cpu"].update(passed=False, max_mixed_error=0.2, bad_elements=1)
    row = run(tmp_path, reports)
    assert row == [2, export.BS, 2*export.BS, "是", "是", "是", 1.01, 1.02, 1.03, 2, "无", None, "通过"]
    err = capsys.readouterr().err
    assert "Y 列（融合/CPU通过）=是" in err and "Z 列（融合/小算子通过）=是" in err


@pytest.mark.parametrize("pair,label", [("fused_vs_cpu", "Y 列（融合/CPU通过）"),
                                        ("fused_vs_cascade", "Z 列（融合/小算子通过）")])
def test_exports_direct_failure_even_if_old_pass_flag_is_true(tmp_path, capsys, pair, label):
    reports = [rank_report(0), rank_report(1)]
    reports[1][pair].update(passed=False, max_mixed_error=0.2, bad_elements=1)
    row = run(tmp_path, reports)
    assert row[0] == 1 and row[10] == "1" and row[-1] == "未通过"
    assert label + "=否" in capsys.readouterr().err


def test_export_sampled_and_partial_coverage(tmp_path):
    r = rank_report(0, checked=64)
    assert run(tmp_path, [r])[-1] == "部分验证"
    r["fused_vs_cascade"]["elements"] = r["fused_vs_cpu"]["elements"]
    with pytest.raises(ValueError, match="coverage"):
        run(tmp_path, [r])


def test_export_rejects_historical_budget_and_other_cascade(tmp_path):
    with pytest.raises(ValueError, match="legacy"):
        run(tmp_path, [dict(passed=True, policy="cascade_error_budget")])
    r = dict(rank_report(0), reference_kind="decode")
    with pytest.raises(ValueError, match="inconsistent"):
        run(tmp_path, [r])


def test_export_missing_error_cannot_pass_but_nonfinite_failure_is_exportable(tmp_path):
    r = rank_report(0)
    for name in ("fused_vs_cpu", "fused_vs_cascade"):
        r[name].update(passed=False, reason="non-finite actual or Golden", max_abs_error=None, max_mixed_error=None)
    with pytest.raises(ValueError, match="missing"):
        run(tmp_path, [r])
    r.update(passed=False, finite=False)
    tsv = run(tmp_path, [r])
    assert tsv[6] is None and tsv[8] is None and tsv[-1] == "未通过"


def write_case(root, model, ep, local, bs, kind, *, rank=0, routing="balanced", backend="mte"):
    directory = root / f"rank_{rank}"
    rows = [rank_report(r, checked=bs) for r in range(ep)]
    for row in rows:
        row.update(reference_kind=kind, total_tokens=bs)
        for key in ("fused_vs_cpu", "cascade_vs_cpu", "fused_vs_cascade"):
            row[key]["elements"] = bs * model["hidden"]
    report = directory / f"bs_{bs}" / "accuracy.json"
    report.parent.mkdir(parents=True)
    report.write_text(json.dumps(rows), encoding="utf-8")
    manifest = dict(ep_size=ep, rank=rank, ranks_per_node=local, backend=backend, mode="accuracy",
                    dual_reference=True, accuracy_cascade=kind, routing=routing, config=dict(model=model, bs_values=[bs]))
    (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return report


@pytest.mark.parametrize("case,machine,ep,local,bs,kind,model,subdir,routing,backend", [
    ("K01", "ubx", 16, 16, 4096, "prefill", {}, "accuracy-w4a8", "balanced", "mte"),
    ("K20", "ubx", 32, 16, 4096, "prefill", dict(mma_type="fp8xfp8"), "accuracy-w8a8", "balanced", "mte_roce"),
    ("K46", "950pod", 32, 8, 4096, "prefill", dict(num_topk=32), "accuracy", "skewed", "mte"),
    ("C03", "ubx", 8, 8, 96, "decode", dict(hidden=3584, intermediate_hidden=1280, num_topk=8,
                                              mma_type="fp8xfp8"), "accuracy-w8a8", "balanced", "mte"),
    ("R02", "950pod", 8, 8, 96, "prefill", dict(activation="swiglu", num_shared_experts=1, num_topk=6),
     "accuracy", "balanced", "mte"),
])
def test_accuracy_cli(tmp_path, monkeypatch, capsys, case, machine, ep, local, bs, kind, model, subdir, routing, backend):
    path = write_case(tmp_path / subdir, dict(export.MODEL, **model), ep, local, bs, kind, routing=routing, backend=backend)
    options = [] if case == "K01" else ["--case", case, "--machine", machine]
    monkeypatch.setattr(sys, "argv", ["export_k01_results.py", "accuracy", str(tmp_path), *options])
    export.main()
    out, err = capsys.readouterr()
    columns = out.rstrip("\n").split("\t")
    assert len(columns) == 13 and columns[:3] == [str(ep), str(bs), str(ep * bs)] and columns[-1] == "通过"
    assert f"{machine.upper()}-EP{ep}-{case}" in err and str(path) in err and f"{kind.capitalize()}Cascade" in err


def test_l0_select_arm_and_read_nonzero_rank_copy(tmp_path):
    shape = dict(export.MODEL, hidden=4096, intermediate_hidden=1536, num_experts=800)
    for arm in ("gelu", "swiglu"):
        model = dict(shape, activation=arm, topk_weights_type=int(arm == "gelu"))
        directory = tmp_path / f"{arm}-accuracy" / "accuracy"
        write_case(directory, model, 16, 8, 39280, "prefill")
        other = write_case(directory, model, 16, 8, 39280, "prefill", rank=8)
        for source in (directory.parent, other):
            assert export.checklist_accuracy(source, "950pod", "L04")[0][-1] == "通过"
    with pytest.raises(ValueError, match="multiple accuracy reports"):
        export.checklist_accuracy(tmp_path, "950pod", "L04")


def test_refuses_wrong_case_platform_reference_and_routing(tmp_path):
    report = write_case(tmp_path, export.MODEL, 16, 16, 4096, "prefill")
    with pytest.raises(ValueError, match="mma_type"):
        export.checklist_accuracy(report, "ubx", "K02")
    with pytest.raises(ValueError, match="topology"):
        export.checklist_accuracy(report, "950pod", "K37")
    shape = dict(export.MODEL, hidden=3584, intermediate_hidden=1280, num_topk=8, mma_type="fp8xfp8")
    report = write_case(tmp_path / "c", shape, 8, 8, 96, "prefill")
    with pytest.raises(ValueError, match="expected direct decode"):
        export.checklist_accuracy(report, "ubx", "C03")
    report = write_case(tmp_path / "s", export.MODEL, 16, 16, 4096, "prefill", routing="skewed")
    with pytest.raises(ValueError, match="routing"):
        export.checklist_accuracy(report, "ubx", "K01")
