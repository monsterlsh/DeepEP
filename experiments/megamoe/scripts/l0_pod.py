# Copyright (c) 2026, Lu Lu

"""Launch the L-series 950POD jobs from node zero on a shared checkout."""

from ipaddress import IPv4Address
import json
from pathlib import Path
import shlex
from uuid import uuid4

if __package__:
    from . import run_megamoe as runner
    from .ubx_multinode import run_stage
else:
    import run_megamoe as runner
    from ubx_multinode import run_stage


def _settings(job, config_dir, nodes):
    if not isinstance(nodes, list) or not all(isinstance(node, str) for node in nodes):
        raise ValueError("950POD nodes must be a list of IPv4 addresses in node-rank order")
    hosts = [str(IPv4Address(node)) for node in nodes]
    args = runner.pod_settings(config_dir / "job.json", job)
    count = args.ep_size // args.ranks_per_node
    if (len(hosts) != count or len(set(hosts)) != count
            or hosts[0] != str(args.master_addr)):
        raise ValueError(f"950POD EP{args.ep_size} requires {count} distinct node(s) and nodes[0] must be master_addr")
    return args


def _supervise(name, commands, args, nodes):
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    scripts, logs = {}, {}
    for rank, command in enumerate(commands):
        # The same interpreter and CANN setup are required on every node. Loading
        # them here also makes the interpreter usable on a non-interactive SSH login.
        launch = ["bash", "-c", runner.POD_BOOTSTRAP, "950pod", args.cann_env, args.python,
                  str(Path(args.base_python).parent.parent / "lib"), *command]
        scripts[rank] = "\n".join((
            "set -e", f"cd -- {shlex.quote(str(runner.REPO_ROOT))}",
            "export DEEPEP_950POD_BOOTSTRAPPED=1 PYTHONNOUSERSITE=1 PYTHONSAFEPATH=1",
            "export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1",
            f"export PYTHON_EXEC={shlex.quote(args.python)}",
            f"export DEEPEP_SHMEM_ENDPOINT={shlex.quote(f'tcp://{args.master_addr}:{args.shmem_port}')}",
            "exec " + shlex.join(map(str, launch)),
        ))
        logs[rank] = out / f"{name}-node{rank}-launcher.log"
    run_stage(name, scripts, logs, ssh_targets=dict(enumerate(nodes[1:], start=1)))


def run_pod_job(job, config_dir, *, nodes, plan=False):
    """Prepare each node's Golden, then start all torchrun workers together."""
    args = _settings(job, config_dir, nodes)
    if plan:
        return
    config_dir.mkdir(parents=True, exist_ok=True)
    out = Path(args.output_dir)
    phases = ("golden", "accuracy") if args.mode == "accuracy" else ("perf",)
    if args.accuracy_stage:
        phases = ("golden",) if args.golden_only else ("accuracy",)
    for phase in phases:
        commands = []
        run_id = uuid4().hex
        for rank in range(len(nodes)):
            cfg = config_dir / f"{out.name}-{phase}-node{rank}.json"
            data = dict(job, node_rank=rank, rdzv_id=run_id, golden_only=phase == "golden",
                        skip_golden=phase == "accuracy")
            cfg.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            commands.append([runner.SCRIPT, "--run-config", cfg])
        _supervise(phase, commands, args, nodes)


def run_pod_paired(job, config_dir, *, nodes, plan=False):
    """Run the co-resident A8W4 pair with latency profiling or a memory-only diagnostic."""
    args = _settings(job, config_dir, nodes)
    model = args.spec["model"]
    if (args.mode != "perf" or args.precision != "w4a8" or args.cascade_stage
            or model["activation"] != "gelu" or model["topk_weights_type"] != 1 or model["num_shared_experts"]):
        raise ValueError("The 950POD paired benchmark requires the routed A8W4 GELU early-weighted perf model")
    if plan:
        return
    config_dir.mkdir(parents=True, exist_ok=True)
    out = Path(args.output_dir)
    model_path = config_dir / f"{out.name}-model.json"
    spec = dict(args.spec, model=dict(model, mma_type="fp8xfp4"))
    model_path.write_text(json.dumps(spec, indent=2) + "\n", encoding="utf-8")
    commands = []
    run_id = uuid4().hex
    for rank in range(len(nodes)):
        commands.append([
            "-I", "-m", "torch.distributed.run", f"--nnodes={len(nodes)}", f"--nproc-per-node={args.ranks_per_node}",
            f"--node-rank={rank}",
            f"--master-addr={args.master_addr}", f"--master-port={args.master_port}", f"--rdzv-id={run_id}", "--max-restarts=0",
            runner.SCRIPT.with_name("paired_benchmark.py"), "--config", model_path, "--bs", str(args.bs),
            "--routing", args.routing, "--pairs", "8", "--warmup", "2", "--output-dir", out,
            "--memory-only" if args.perf_scope == "memory" else "--profile",
            *runner.memory_option(args.profile_memory_peak),
        ])
    _supervise("paired", commands, args, nodes)
