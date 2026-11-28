# Copyright (c) 2026, Lu Lu

"""List the MegaMoE checklist and generate a run_megamoe.py JSON for a selected case."""

import argparse
from dataclasses import asdict, dataclass
from itertools import product
import json
from pathlib import Path


CONFIGS = Path(__file__).resolve().with_name("configs")
MODELS = CONFIGS.parents[1] / "python/deep_ep_experimental/megamoe/configs"
MMA = {"a8w4": "fp8xfp4", "a8w8": "fp8xfp8"}
MACHINES = ("ubx", "950pod")
EPS = (4, 8, 16, 32)


@dataclass(frozen=True)
class Case:
    case: str
    machine: str
    ep_size: int
    bs: int
    hidden: int
    intermediate: int
    experts: int
    topk: int
    activation: str
    weight_mode: int
    precision: str
    shared: int = 0
    groupsize: int | None = None
    model_preset: str | None = None

    @property
    def backend(self):
        return "mte_roce" if self.machine == "ubx" and self.ep_size == 32 else "mte"

    @property
    def nodes(self):
        local = 16 if self.machine == "ubx" else 8
        return max(1, self.ep_size // local)

    @property
    def accuracy_cascade(self):
        return "decode" if self.case.startswith("C") else "prefill"

    @property
    def limitation(self):
        if self.groupsize:
            return f"groupsize={self.groupsize} 缺少生产配置键和实际生效证据"
        if self.experts // self.ep_size > 128:
            return f"{self.experts} / EP{self.ep_size} = {self.experts // self.ep_size} 个本地专家，MTE 上限为 128"
        # EP32 950POD auto-MTE uses the full SwiGLU batch with A8 inputs. These two
        # payloads alone suffice to prove overflow, even before scales and metadata.
        if (self.machine, self.ep_size) == ("950pod", 32) and self.activation == "swiglu":
            minimum_heap = self.bs * self.hidden * (2 * self.topk + 1)
            if minimum_heap > 1 << 30:
                return f"950POD 原生 SwiGLU 整批对称区至少 {minimum_heap} 字节，超过 1 GiB；可预生成 CPU Golden"
        return None

    def phases(self):
        golden = ["golden"] if self.machine == "950pod" and self.case[0] in "RCK" else []
        if golden and self.case.startswith("K") and self.topk == 32:
            golden.append("golden-skewed")
        if self.limitation:
            return golden
        if self.case.startswith("L"):
            return ["full", "accuracy", "perf", "memory", "prefill"]
        phases = [*golden, "accuracy", "perf", "memory"]
        if self.precision != "a8w8" or self.backend == "mte_roce":
            phases.append("timer")
        if self.case.startswith("C"):
            phases.append("decode")
        if self.case.startswith("K"):
            phases.append("prefill")
            if self.topk == 32:
                phases.append("accuracy-skewed")
        return phases


def cases():
    """Keep the original EP32 rows and expand distinct workloads to EP4/8/16 on both machines."""
    for machine, ep in product(MACHINES, EPS):
        if machine == "950pod" or ep < 32:
            pro = ((96, 0), (96, 1), (4096, 0), (4096, 1))
            for i, (bs, shared) in enumerate(pro, 1):
                preset = "deepseek_v4_pro_shared1" if shared else "deepseek_v4_pro"
                yield Case(f"R{i:02}", machine, ep, bs, 7168, 3072, 384, 6, "swiglu", 0, "a8w4",
                           shared=shared, model_preset=preset)
            for i, bs in enumerate((8192, 12288, 16384), 5):
                yield Case(f"R{i:02}", machine, ep, bs, 5120, 2304, 384, 6, "swiglu", 0, "a8w4",
                           shared=1, model_preset="deepseek_v4_1")
        if ep < 32:
            for i, (activation, precision) in enumerate(
                    (("swiglu", "a8w4"), ("swiglu", "a8w8"), ("gelu", "a8w8")), 1):
                yield Case(f"C{i:02}", machine, ep, 96, 3584, 1280, 384, 8, activation, 0, precision)
        start = 37 if machine == "950pod" else (19 if ep == 32 else 1)
        arms = (("gelu", 0), ("gelu", 1), ("swiglu", 0))
        # Each original workload reserved three case numbers. Keep the published IDs.
        for group, (topk, (act, weight)) in enumerate(product((25, 32), arms)):
            for i, precision in enumerate(MMA, start + 3 * group):
                yield Case(f"K{i:02}", machine, ep, 4096, 7168, 3072, 384, topk, act, weight, precision)
        for i, precision in enumerate(MMA, 1 if machine == "ubx" else 4):
            yield Case(f"L{i:02}", machine, ep, 39280, 4096, 1536, 800, 25, "gelu", 1, precision)
        if machine == "ubx" or ep < 32:
            for group, (gs, bs) in enumerate(product((256, 512), (1024, 2048, 4096, 8192))):
                for i, precision in enumerate(MMA, 1 + 3 * group):
                    yield Case(f"D{i:02}", machine, ep, bs, 3584, 1280, 384, 8, "gelu", 1, precision, groupsize=gs)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def run_config(row, phase, *, node_rank=None, results=None, master_addr=None):
    suite = row.case.startswith("L")
    phase = phase or ("full" if suite else "accuracy")
    if phase not in row.phases():
        reason = row.limitation or f"可用阶段为 {', '.join(row.phases())}"
        raise ValueError(f"{row.case} {row.machine} EP{row.ep_size} 无法生成 {phase}：{reason}")
    if node_rank is not None and (suite or row.machine != "950pod" or not 0 <= node_rank < row.nodes):
        raise ValueError("--node-rank 仅用于 950POD 普通作业，须在该 EP 的节点范围内；L 组从主节点启动")
    if suite:
        job = read_json(CONFIGS / "l0" / f"{row.case}.json")
        job["mode"] = "both" if phase == "full" else phase
        if row.machine == "950pod":
            job["nodes"] = job["nodes"][:row.nodes]
        elif row.nodes == 1:
            job.pop("node_ip")
            job["master_addr"] = "127.0.0.1"
    else:
        preset = "950pod_target" if row.machine == "950pod" else f"ubx_ep{32 if row.nodes == 2 else 16}_target"
        job = read_json(CONFIGS / f"{preset}.json")
        job.pop("_comments", None)
        job.pop("comparison", None)
        if row.model_preset:
            job["config"] = read_json(MODELS / f"{row.model_preset}.json")
        job["config"]["model"].update(
            hidden=row.hidden, intermediate_hidden=row.intermediate, num_experts=row.experts, num_topk=row.topk,
            num_shared_experts=row.shared, activation=row.activation, topk_weights_type=row.weight_mode,
            mma_type=MMA[row.precision],
        )
        job["config"]["bs_values"] = [row.bs]
        accuracy = phase.startswith(("accuracy", "golden"))
        job.update(mode="accuracy" if accuracy else "perf", precision=row.precision, check_tokens=0,
                   dual_reference=accuracy, timer=phase == "timer",
                   routing="skewed" if phase.endswith("-skewed") else "balanced")
        if phase in ("decode", "prefill"):
            job[f"compare_{phase}_cascade"] = True
        if phase == "memory":
            job.update(profile_memory_peak=True, perf_scope="memory")
        if row.machine == "950pod":
            job["node_rank"] = node_rank or 0
            if row.nodes == 1:
                job["master_addr"] = "127.0.0.1"
            elif not results:
                raise ValueError("950POD 多机普通作业须用 --results 指定所有节点的同一共享结果目录")
            if phase.startswith("golden"):
                job["golden_only"] = True
        elif row.nodes == 1 and accuracy:
            job["force_npu"] = True
    job.update(ep_size=row.ep_size, backend=row.backend, accuracy_cascade=row.accuracy_cascade)
    if master_addr:
        job["master_addr"] = master_addr
        if suite and row.machine == "950pod":
            job["nodes"][0] = master_addr
        elif row.machine == "ubx" and row.nodes > 1:
            job["node_ip"][0] = master_addr
    if results:
        job["output_dir"] = results
    return job


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    listing = commands.add_parser("list", help="List configurations, available phases and known constraints as JSON")
    generate = commands.add_parser("generate", help="Write one run configuration for the selected phase")
    for command in (listing, generate):
        required = command is generate
        command.add_argument("--machine", choices=MACHINES, required=required)
        command.add_argument("--ep-size", type=int, choices=EPS, required=required)
        command.add_argument("--case", type=str.upper, required=required)
    generate.add_argument("--phase", help="See list for available phases. Defaults to accuracy, or full for L cases.")
    generate.add_argument("--output", type=Path, required=True, help="Destination run-config JSON")
    generate.add_argument("--node-rank", type=int, help="950POD non-suite jobs use one JSON per node, starting at 0")
    generate.add_argument("--master-addr", help="Override the preset's rendezvous IPv4")
    generate.add_argument("--results", help="Job output_dir (shared across nodes, required for 950POD multinode non-L jobs)")
    args = parser.parse_args(argv)
    selected = [r for r in cases() if (not args.machine or args.machine == r.machine)
                and (not args.ep_size or args.ep_size == r.ep_size) and (not args.case or args.case == r.case)]
    if not selected:
        parser.error("没有匹配的配置。UBX 的 K01–17 对应 EP4/8/16，K19–35 对应 EP32；950POD 使用 K37–53。K 组均跳过 3 的倍数。")
    if args.command == "list":
        print(json.dumps([dict(asdict(r), backend=r.backend, nodes=r.nodes, phases=r.phases(), limitation=r.limitation,
                               accuracy_cascade=r.accuracy_cascade,
                               performance_baseline="feat/roce-support" if r.ep_size < 32 else None)
                          for r in selected], indent=2, ensure_ascii=False))
        return
    try:
        config = run_config(selected[0], args.phase, node_rank=args.node_rank, results=args.results,
                            master_addr=args.master_addr)
    except ValueError as error:
        parser.error(str(error))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(config, indent=2, ensure_ascii=False) + "\n")
    except FileExistsError:
        parser.error(f"配置已存在，请为本次作业指定新的 --output：{args.output}")


if __name__ == "__main__":
    main()
