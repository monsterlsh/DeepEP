# Copyright (c) 2026, Lu Lu
"""Inspect two UBX nodes and optionally set the SHMEM HNS_1825 SQ doorbell GID index."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import hashlib
import ipaddress
from itertools import product
import json
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import shutil
import socket
import stat
import subprocess
import sys
import tempfile


HEADER = "src/device/gm2gm/engine/rdma_backends/rdma_device_backend_hns_1825.hpp"
FIELD = re.compile(rb"(\bconstexpr\s+uint32_t\s+ACLSHMEMI_HNS_1825_SQ_DB_SGID_IDX\s*=\s*)([0-9]+)(\s*;)")
VENDOR_ROW = re.compile(r"(\S+)\s+(\S+)\s+(\d+)\s+(\d+)\s+([0-9a-fA-F:%.]+)\s+"
                        r"(RoCE\s+v[12]|IB/RoCE v1)\s+(?:(\S+)\s+)?(up|down)", re.I)


def address(gid):
    ip = ipaddress.ip_address(gid)
    return (ip.ipv4_mapped or ip) if ip.version == 6 else ip


def usable_interface(row):
    if row["type"].lower() != "roce v2" or not row["netdev"] or not row["cidr"] or not 0 <= row["index"] <= 127:
        return None
    ip = address(row["gid"])
    interface = ipaddress.ip_interface(row["cidr"])
    if (ip != interface.ip or ip.is_unspecified or ip.is_link_local or ip.is_loopback or ip.is_multicast):
        return None
    return interface


def node_indices(node):
    ports = []
    for port, rows in node["ports"].items():
        valid = {}
        for row in rows:
            iface = usable_interface(row)
            if iface:
                if row["index"] in valid:
                    raise ValueError(f"{port}: duplicate GID index {row['index']}")
                valid[row["index"]] = iface
        if not valid:
            raise ValueError(f"{port}: no usable RoCE v2 GID matching a local netdev address")
        ports.append(valid)
    if not ports:
        raise ValueError("both nodes need at least one active fabric port")
    return {idx: [p[idx] for p in ports] for idx in sorted(set.intersection(*(set(p) for p in ports)))}


def choose_index(nodes, requested=None):
    """Report asymmetric matches too, but update only for a common index usable on all ports."""
    if len(nodes) != 2:
        raise ValueError("expected exactly two nodes")
    left, right = map(node_indices, nodes)
    pairs = []
    for i, j in product(left, right):
        fabrics = ((left[i], right[j]), (right[j], left[i]))
        if all(any(a.version == b.version and a.ip != b.ip and a.ip in b.network and b.ip in a.network
                   for b in peers) for group, peers in fabrics for a in group):
            pairs.append((i, j))
    candidates = [i for i, j in pairs if i == j]
    if pairs and not candidates:
        raise ValueError(f"on-subnet candidates (83/86): {', '.join(f'{i}/{j}' for i, j in pairs)}; "
                         "no common SGID index for a single wheel. Both source headers are unchanged.")
    if requested is not None and requested in candidates:
        return requested
    if requested is None and len(candidates) == 1:
        return candidates[0]
    raise ValueError(f"eligible common indices: {candidates}; verify the fabric ports and use --index "
                     "only to select among eligible indices")


def optional_text(path):
    try:
        return path.read_text().strip()
    except OSError:
        return ""


def read_sysfs():
    rows = []
    for port in sorted(Path("/sys/class/infiniband").glob("*/ports/*")):
        active = optional_text(port / "state").split(":")[-1].strip() == "ACTIVE"
        for f in (port / "gids").glob("[0-9]*"):
            gid = optional_text(f)
            if gid:
                rows.append(dict(port=f"{port.parents[1].name}/{port.name}", active=active, index=int(f.name),
                                 gid=gid, type=optional_text(port / "gid_attrs/types" / f.name),
                                 netdev=optional_text(port / "gid_attrs/ndevs" / f.name)))
    return rows


def read_vendor(output):
    rows = []
    for line in output.splitlines():
        match = VENDOR_ROW.fullmatch(line.strip())
        if match:
            dev, netdev, port, index, gid, kind, _, link = match.groups()
            rows.append(dict(port=f"{dev}/{port}", active=link.lower() == "up", index=int(index),
                             gid=gid, type=kind, netdev=netdev))
        elif re.match(r"^\s*hrn5_\S+\s", line):
            raise ValueError(f"unrecognized hiroce5 row: {line.strip()}")
    return rows


def inventory(port_names):
    rows = []
    source = "hiroce5 gids"
    if shutil.which("hiroce5"):
        result = subprocess.run(["hiroce5", "gids"], capture_output=True, text=True, timeout=20)
        if result.returncode == 0:
            rows = read_vendor(result.stdout)
    if not rows:
        rows = read_sysfs()
        source = "Linux RDMA sysfs"
    all_ports = sorted({r["port"] for r in rows})
    selected = set(port_names or (r["port"] for r in rows if r["active"] and r["port"].startswith("hrn5_")))
    if not selected or selected.difference(all_ports):
        raise ValueError(f"{source}: check HNS_1825 port names; visible ports: {all_ports}, requested: {port_names}")
    addresses = {}
    cmd = subprocess.run(["ip", "-j", "addr", "show"], text=True, capture_output=True, check=True, timeout=20)
    for dev in json.loads(cmd.stdout):
        for item in dev.get("addr_info", []):
            ip = ipaddress.ip_address(item["local"])
            addresses[dev["ifname"], ip] = f"{ip}/{item['prefixlen']}"
    ports = {port: [] for port in sorted(selected)}
    for r in rows:
        if r["port"] in ports and r["active"] and not address(r["gid"]).is_unspecified:
            r["cidr"] = addresses.get((r["netdev"], address(r["gid"])))
            ports[r["port"]].append(r)
    return dict(host=socket.gethostname(), boot=Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
                source=source, ports=ports)


def header_values(data):
    matches = list(FIELD.finditer(data))
    if len(matches) != 1:
        raise ValueError("expected exactly one HNS_1825 SQ_DB_SGID_IDX constant in the source header")
    return matches[0], hashlib.sha256(data).hexdigest()


def patched(data, match, index):
    if type(index) is not int or not 0 <= index <= 127:
        raise ValueError("SGID index must fit in the 7-bit doorbell field")
    return data[:match.start(2)] + str(index).encode() + data[match.end(2):]


@contextmanager
def _directory_lock(directory):
    # Keep the SHMEM checkout free of lock files; its build fingerprints untracked inputs.
    import fcntl
    fd = os.open(directory, os.O_RDONLY)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        os.close(fd)


def change_header(path, before_sha, index, desired_sha=None):
    """Replace only the numeric literal. Accept a shared checkout already at the exact target hash."""
    with _directory_lock(path.parent):
        if path.is_symlink():
            raise ValueError(f"refusing to replace symlink {path}")
        data = path.read_bytes()
        match, current = header_values(data)
        if current == desired_sha and int(match[2]) == index:
            return current
        if current != before_sha:
            raise ValueError(f"{path}: source changed since inspection")
        updated = patched(data, match, index)
        digest = hashlib.sha256(updated).hexdigest()
        if desired_sha is not None and desired_sha != digest:
            raise ValueError("target hash mismatch")
        if updated != data:
            tmp = None
            try:
                with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as f:
                    tmp = Path(f.name)
                    f.write(updated)
                    f.flush()
                    os.fsync(f.fileno())
                os.chmod(tmp, stat.S_IMODE(path.stat().st_mode))
                os.replace(tmp, path)
            finally:
                if tmp is not None:
                    tmp.unlink(missing_ok=True)
        return digest


def worker(request):
    action = request["action"]
    if action == "inspect":
        return inventory(request["ports"])
    source = Path(request["source"])
    path = source / HEADER
    if not (source / "scripts/build.sh").is_file() or path.is_symlink():
        raise ValueError(f"expected a SHMEM source checkout at {source}")
    if action == "set":
        return change_header(path, request["before"], request["index"], request["after"])
    if action == "plan":
        data = path.read_bytes()
        match, digest = header_values(data)
        target = patched(data, match, request["index"])
        return dict(path=str(path), index=int(match[2]), before=digest, after=hashlib.sha256(target).hexdigest())
    raise ValueError("unknown worker request")


def on_node(host, request):
    if host == "local":
        return worker(request)
    if not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.@:\[\]-]*", host):
        raise ValueError(f"invalid SSH destination: {host}")
    cmd = shlex.join(["python3", "-", "--worker", json.dumps(request)])
    result = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", host, cmd],
                            input=Path(__file__).read_bytes(), capture_output=True, timeout=60)
    if result.returncode:
        raise RuntimeError(f"{host}: {result.stderr.decode(errors='replace').strip()}")
    return json.loads(result.stdout)


def show(host, node):
    print(f"\n{host} ({node['host']})  [{node['source']}]")
    print("RDMA_PORT        IDX  TYPE          NET_DEV          LOCAL_IP/PREFIX          GID")
    for port, rows in node["ports"].items():
        if not rows:
            print(f"{port}: no active nonzero rows")
        for r in sorted(rows, key=lambda row: row["index"]):
            print(f"{port:16} {r['index']:3}  {r['type']:12}  {r['netdev']:15}  "
                  f"{r['cidr'] or '-':23}  {r['gid']}")


def apply_plans(hosts, source, index, plans):
    attempted = []
    try:
        for host, plan in zip(hosts, plans):
            if plan["before"] != plan["after"]:
                attempted.append((host, plan))
            on_node(host, dict(action="set", source=source, before=plan["before"], after=plan["after"], index=index))
        for host, plan in zip(hosts, plans):
            check = on_node(host, dict(action="plan", source=source, index=index))
            if check["before"] != plan["after"]:
                raise ValueError(f"{host}: final header verification failed")
    except (Exception, KeyboardInterrupt):
        for host, plan in reversed(attempted):
            try:
                on_node(host, dict(action="set", source=source, before=plan["after"],
                                   after=plan["before"], index=plan["index"]))
            except Exception as error:
                print(f"Rollback unconfirmed on {host}: {plan['path']} originally index={plan['index']} "
                      f"sha256={plan['before']}: {error}", file=sys.stderr)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nodes", nargs=2, required=True, metavar=("NODE83", "NODE86"),
                        help="SSH aliases or user@address; use 'local' for the current Linux node")
    parser.add_argument("--repo", default="/home/l00911701/DeepEP", type=PurePosixPath, help="same DeepEP directory on both nodes")
    parser.add_argument("--shmem-src", type=PurePosixPath, default=PurePosixPath("third-party/shmem"),
                        help="SHMEM source path, absolute or relative to --repo")
    parser.add_argument("--ports83", nargs="+", help="fabric ports on node 83, e.g. hrn5_0/1 hrn5_1/1")
    parser.add_argument("--ports86", nargs="+", help="fabric ports on node 86 (default: all active hrn5_* ports)")
    parser.add_argument("--index", type=int, help="select a verified index if multiple indices qualify")
    parser.add_argument("--apply", action="store_true", help="write and verify the selected index on both nodes")
    args = parser.parse_args()
    if not args.repo.is_absolute():
        parser.error("--repo must be an absolute Linux path")
    source = str(args.repo / args.shmem_src)
    with ThreadPoolExecutor(max_workers=2) as pool:
        jobs = [pool.submit(on_node, host, dict(action="inspect", ports=ports))
                for host, ports in zip(args.nodes, (args.ports83, args.ports86))]
        nodes = [job.result() for job in jobs]
    for host, node in zip(args.nodes, nodes):
        show(host, node)
    if nodes[0]["boot"] == nodes[1]["boot"]:
        raise ValueError("both destinations resolve to the same booted host")
    index = choose_index(nodes, args.index)
    print(f"\nSelected common RoCE v2 SGID index: {index}")
    plans = [on_node(host, dict(action="plan", source=source, index=index)) for host in args.nodes]
    for host, plan in zip(args.nodes, plans):
        print(f"{host}: {plan['path']}  {plan['index']} -> {index}")
    if plans[0]["after"] != plans[1]["after"]:
        raise ValueError("the two target headers would differ; check the SHMEM checkouts used by both nodes")
    if args.apply:
        apply_plans(args.nodes, source, index, plans)
        print(f"Verified both source headers: index={index} sha256={plans[0]['after']}")
    else:
        print("Inspection complete. Use --apply to update the source headers.")


if __name__ == "__main__":
    try:
        if len(sys.argv) == 3 and sys.argv[1] == "--worker":
            print(json.dumps(worker(json.loads(sys.argv[2]))))
        else:
            main()
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
        sys.exit(f"SGID: {exc}")
