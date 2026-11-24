# Copyright (c) 2026, Lu Lu

"""Check the installed UBX wheel from each node's actual worker interpreter."""

from importlib import metadata, util
from importlib.machinery import EXTENSION_SUFFIXES
import json
from pathlib import Path
import socket
import sys
import sysconfig


def _wheel_sha256(dist):
    try:
        direct = json.loads(dist.read_text("direct_url.json") or "{}").get("archive_info", {})
        digest = direct.get("hashes", {}).get("sha256")
        if digest:
            return digest
        algorithm, _, digest = direct.get("hash", "").partition("=")
        if algorithm == "sha256":
            return digest
    except (OSError, ValueError, AttributeError):
        pass  # Optional pip metadata must not mask the native import failure.
    return "unrecorded"


def check_wheel_packages():
    # An executable .pth can prepend a checkout even under python -I. Look up
    # the wheel in this interpreter's install directories independently of sys.path.
    sites = {sysconfig.get_path(key) for key in ("purelib", "platlib")}
    dist = next(metadata.distributions(name="ascend-deepep", path=sorted(sites)), None)
    if dist is None:
        try:
            dist = metadata.distribution("ascend-deepep")
        except metadata.PackageNotFoundError:
            pass
    owned = set(map(str, dist.files or ())) if dist else set()
    if dist:
        print(f"[UBX] ascend-deepep={dist.version} install={dist.locate_file('')} "
              f"wheel_sha256={_wheel_sha256(dist)}", flush=True)
    for package, component in (("deep_ep", ""), ("deep_ep_experimental", "megamoe")):
        spec = util.find_spec(package)
        origin = Path(spec.origin).resolve() if spec and spec.origin else None
        print(f"[UBX] {package}={origin}", flush=True)
        if origin is None:
            raise ImportError(f"{sys.executable} cannot find {package}; install the node0 MegaMoE wheel in this interpreter")
        member = f"{package}/__init__.py"
        if member in owned:
            installed = Path(dist.locate_file(member)).resolve()
            if origin != installed:
                raise ImportError(f"{package} resolves to {origin}, but this interpreter's wheel owns {installed}. "
                                  "Check editable installs, .pth files and sitecustomize in this Python environment.")
        directory = origin.parent / component
        if not any((directory / ("_C" + suffix)).is_file() for suffix in EXTENSION_SUFFIXES):
            candidates = [p.name + (" (broken symlink)" if p.is_symlink() and not p.exists() else "")
                          for p in sorted(directory.glob("_C.*"))]
            raise ImportError(f"No compatible native extension in {directory}; expected "
                              f"_C{EXTENSION_SUFFIXES[0]} (or another supported suffix); found {candidates}. "
                              "Install the matching node0 wheel with this interpreter's -m pip.")


def main():
    print(f"[UBX] {socket.gethostname()} python={sys.executable} abi={sys.implementation.cache_tag}", flush=True)
    import torch
    import torch_npu  # noqa: F401

    check_wheel_packages()
    from deep_ep._machine import board_id, physical_device
    from deep_ep_experimental.megamoe.api import _extension

    local, require_w8a8 = int(sys.argv[1]), sys.argv[2] == "1"
    backend = sys.argv[3] if len(sys.argv) > 3 else "urma"
    require_timer = len(sys.argv) > 4 and sys.argv[4] == "1"
    count = torch.npu.device_count()
    if count < local:
        sys.exit(f"UBX 每机需要 {local} 张可见 NPU，当前只有 {count} 张。")
    torch.npu.set_device(0)
    board = board_id(physical_device())
    if board != 0x44:
        sys.exit(f"可见设备 0 的板型为 0x{board:02x}，UBX 要求 0x44。")
    extension = _extension()
    if require_w8a8 and not getattr(extension, "urma_w8a8_enabled", False):
        sys.exit("当前 MegaMoE wheel 未启用 UBX URMA W8A8。")
    if backend == "mte_roce" and not getattr(extension, "mte_roce_enabled", False):
        sys.exit("当前 MegaMoE wheel 未启用 EP32 MTE+RoCE，请在两台主机安装同版本的新版 wheel。")
    if backend == "mte_roce" and not getattr(extension, "mte_roce_mx_enabled", False):
        sys.exit("当前 MegaMoE wheel 未启用新版 MTE+RoCE 矩阵精度与分块支持。")
    if backend == "mte_roce" and require_timer and not getattr(extension, "mte_roce_timer_enabled", False):
        sys.exit("MTE+RoCE 设备 timer 需要两台节点安装支持此功能并启用 DEEPEP_MEGAMOE_TIMER 的 wheel。")
    print(f"[UBX] MegaMoE wheel: {extension.__file__}", flush=True)


if __name__ == "__main__":
    main()
