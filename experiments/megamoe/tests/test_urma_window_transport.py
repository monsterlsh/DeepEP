"""Exercise the device window transport on local memory with asynchronous SDK doubles."""

from pathlib import Path
import shutil
import subprocess
import sys

import pytest


def test_window_transport(tmp_path):
    compiler = next((path for name in ("c++", "g++", "clang++") if (path := shutil.which(name))), None)
    if compiler is None:
        pytest.skip("A C++17 compiler is required for the transport check")
    tests = Path(__file__).resolve().parent
    for header in (
        "kernel_operator.h", "shmem.h", "device/gm2gm/engine/shmem_device_rdma.h",
        "device/gm2gm/engine/shmem_device_mte.h",
    ):
        target = tmp_path / header
        target.parent.mkdir(parents=True, exist_ok=True)
        target.touch()
    binary = tmp_path / ("window_transport.exe" if sys.platform == "win32" else "window_transport")
    subprocess.run([compiler, "-std=c++17", "-Wall", "-Wextra", "-Werror", f"-I{tmp_path}",
                    f"-I{tests.parent / 'csrc'}", str(tests / "urma_window_transport.cpp"),
                    "-o", str(binary)], check=True, timeout=60)
    subprocess.run([binary], check=True, timeout=10)


def test_window_session(tmp_path):
    compiler = next((path for name in ("c++", "g++", "clang++") if (path := shutil.which(name))), None)
    if compiler is None:
        pytest.skip("A C++17 compiler is required for the session check")
    repo = Path(__file__).resolve().parents[3]
    csrc = repo / "experiments/megamoe/csrc"
    (tmp_path / "dlfcn.h").write_text(
        "#pragma once\n#define RTLD_DEFAULT nullptr\nvoid* test_dlsym(void*, const char*);\n#define dlsym test_dlsym\n"
    )
    # Only the independent LL platform query needs a double. No window may
    # reach it even when the process enables DEEPEP_LL_BACKEND=rdma for LL.
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    (runtime / "ll_core_config.hpp").write_text(
        "#pragma once\n#include <cstdlib>\nnamespace deepep { struct Cores { unsigned active; }; "
        "inline Cores query_ll_core_config() { std::abort(); } }\n"
    )
    extra = []
    if sys.platform == "win32":
        compat = tmp_path / "env.hpp"
        compat.write_text(
            "#include <cstdlib>\n#define setenv(name, value, overwrite) _putenv_s(name, value)\n"
            '#define unsetenv(name) _putenv_s(name, "")\n'
        )
        extra = ["-include", str(compat)]
    binary = tmp_path / ("window_session.exe" if sys.platform == "win32" else "window_session")
    subprocess.run([compiler, "-std=c++17", "-Wall", "-Wextra", "-Werror", "-DDEEPEP_BUILD_MEGAMOE",
                    "-DDEEPEP_LL_RDMA", *extra, f"-I{tmp_path}", f"-I{repo / 'tests/runtime/stubs'}",
                    f"-I{repo / 'csrc'}", f"-I{csrc}", str(repo / "csrc/runtime/session.cpp"),
                    str(csrc / "runtime/megamoe_session.cpp"), str(repo / "tests/runtime/session_test.cpp"),
                    "-o", str(binary)], check=True, timeout=60)
    for size in (8, 16, 32):
        for suffix in ("", "-missing"):
            subprocess.run([binary, f"window-{size}{suffix}"], check=True, timeout=10)
