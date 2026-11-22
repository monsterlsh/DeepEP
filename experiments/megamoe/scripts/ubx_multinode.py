# Copyright (c) 2026, Lu Lu

"""Run one UBX stage on the local host and an SSH peer."""

from contextlib import ExitStack
import shlex
import subprocess
import sys
import time


# The supervisor owns a process group for the stage, and its stdin stays open
# until the launcher has seen the stage finish. EOF also cancels a remote stage
# when its SSH channel closes, even though SSH is running without a terminal.
_SUPERVISOR = r"""
set -m
child=
watcher=
cleanup() {
    trap - EXIT
    trap '' HUP INT TERM
    if [[ -n $watcher ]]; then
        kill "$watcher" 2>/dev/null || true
        wait "$watcher" 2>/dev/null || true
    fi
    if [[ -n $child ]] && kill -TERM -- "-$child" 2>/dev/null; then
        # Allow torchrun to stop workers that it put in separate sessions.
        for _ in {1..140}; do
            kill -0 -- "-$child" 2>/dev/null || break
            sleep 0.25
        done
        kill -KILL -- "-$child" 2>/dev/null || true
        wait "$child" 2>/dev/null || true
    fi
}
trap cleanup EXIT
trap 'exit 129' HUP
trap 'exit 130' INT
trap 'exit 143' TERM
bash -c "$1" </dev/null &
child=$!
supervisor=$BASHPID
( IFS= read -r _ || kill -TERM "$supervisor" 2>/dev/null ) <&0 &
watcher=$!
wait "$child"
"""


def node_script(work_dir, env, environment_script, argv):
    """Load the runtime on the current node before executing its worker."""
    lines = ["set -e", f"cd -- {shlex.quote(str(work_dir))}"]
    lines.extend(f"export {shlex.quote(f'{name}={value}')}" for name, value in env.items())
    lines.extend([
        'if [[ -z ${PYTHON:-} || -z ${UBX_CANN_LOADED:-} || '
        '( -n ${UBX_CANN_SETUP:-} && $UBX_CANN_SETUP != "$UBX_CANN_LOADED" ) ]]; then',
        f"    source {shlex.quote(str(environment_script))} --preserve-python",
        "fi",
        'exec "$PYTHON" ' + shlex.join(map(str, argv)),
    ])
    return "\n".join(lines)


def _node_command(rank, script, ssh_target, ssh_port):
    command = ["bash", "-c", _SUPERVISOR, "ubx-stage", script]
    if rank == 0:
        return command
    return [
        "ssh", "-T", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15",
        "-o", "ServerAliveInterval=10", "-o", "ServerAliveCountMax=3",
        "-p", str(ssh_port), "--", ssh_target, shlex.join(command),
    ]


def _stop_nodes(processes):
    # Close both lifelines before waiting for either node to finish shutdown.
    for process in processes:
        process.stdin.close()
    deadline = time.monotonic() + 45
    for process in processes:
        try:
            process.wait(timeout=max(0, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def run_stage(stage, scripts, log_paths, *, ssh_target, ssh_port=22):
    """Run a stage on ranks 0 and 1, returning only if both succeed."""
    with ExitStack() as stack:
        # Open both logs before starting any work. Log setup failure is a stage
        # failure, and child exit status is independent of log collection.
        logs = {rank: stack.enter_context(log_paths[rank].open("wb")) for rank in (0, 1)}
        pending = {}
        try:
            for rank in (0, 1):
                command = _node_command(rank, scripts[rank], ssh_target, ssh_port)
                print(f"[UBX] 开始 {stage} node{rank}，日志 {log_paths[rank]}", flush=True)
                pending[rank] = subprocess.Popen(
                    command, stdin=subprocess.PIPE, stdout=logs[rank], stderr=subprocess.STDOUT,
                )
            while pending:
                for rank, process in list(pending.items()):
                    status = process.poll()
                    if status is None:
                        continue
                    if status:
                        print(
                            f"[UBX] {stage} node{rank} 退出码 {status}，日志 {log_paths[rank]}",
                            file=sys.stderr, flush=True,
                        )
                        raise subprocess.CalledProcessError(status, process.args)
                    process.stdin.close()
                    del pending[rank]
                    print(f"[UBX] 完成 {stage} node{rank}", flush=True)
                if pending:
                    time.sleep(0.1)
        finally:
            _stop_nodes(list(pending.values()))
