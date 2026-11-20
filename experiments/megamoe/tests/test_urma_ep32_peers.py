"""Exercise both production peer state machines on one physical communication core."""

from pathlib import Path
import shutil
import subprocess

import pytest


def test_two_peer_progress_with_delayed_start_and_empty_edges(tmp_path):
    compiler = next((path for name in ("c++", "g++", "clang++") if (path := shutil.which(name))), None)
    if not compiler:
        pytest.skip("A host C++17 compiler is required for the peer progress check")
    csrc = Path(__file__).resolve().parents[1] / "csrc"
    header = (csrc / "kernels/mega_moe_urma/stage/mega_moe_dispatch_combine.hpp").read_text(encoding="utf-8")
    start = header.index("class CooperativePeer {")
    peer_scheduler = header[start:header.rindex("}  // namespace MegaMoeUrma")]
    transport = (csrc / "kernels/mega_moe_urma/common/mega_moe_roce.hpp").read_text(encoding="utf-8")
    deadline = transport[transport.index("enum PeerWaitPhase"):transport.rindex("}  // namespace MegaMoeUrma")]
    (tmp_path / "ep32_peer_scheduler.inc").write_text(deadline + peer_scheduler, encoding="utf-8")
    test = Path(__file__).with_name("urma_ep32_peers.cpp")
    executable = tmp_path / "peers.exe"
    subprocess.run(
        [compiler, "-std=c++17", f"-I{csrc}", f"-I{tmp_path}", str(test), "-o", str(executable)],
        check=True,
        timeout=60,
    )
    subprocess.run([str(executable)], check=True, timeout=10)
