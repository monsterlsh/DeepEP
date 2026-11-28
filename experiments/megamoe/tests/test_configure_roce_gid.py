"""Exercise two-node GID selection and guarded edits to the SHMEM header."""

import hashlib
from contextlib import nullcontext
from ipaddress import IPv6Address
import os

import pytest

from experiments.megamoe.scripts import configure_roce_gid
from experiments.megamoe.scripts.configure_roce_gid import change_header, choose_index


def test_hiroce5_driver_format_with_and_without_ipv4_column():
    from experiments.megamoe.scripts.configure_roce_gid import read_vendor

    table = """ib_dev net_dev port idx gid type ip link
hrn5_0 ens100f0 1 0 fe80:0000:0000:0000:6466:66ff:feaa:047b RoCE v2 up
hrn5_0 ens100f0 1 3 0000:0000:0000:0000:0000:ffff:1e1e:1e65 RoCE v2 30.30.30.101 up"""
    rows = read_vendor(table)
    assert [r["index"] for r in rows] == [0, 3]
    assert rows[1]["port"] == "hrn5_0/1" and rows[1]["netdev"] == "ens100f0" and rows[1]["active"]


@pytest.mark.parametrize("shared_tree,interrupted", [(False, False), (True, False), (False, True)])
def test_apply_two_nodes_and_recover_lost_ssh_ack(tmp_path, monkeypatch, shared_tree, interrupted):
    from contextlib import nullcontext
    from experiments.megamoe.scripts import configure_roce_gid as script

    before = b"constexpr uint32_t ACLSHMEMI_HNS_1825_SQ_DB_SGID_IDX = 1;\n"
    roots = {"83": tmp_path / "node83", "86": tmp_path / ("node83" if shared_tree else "node86")}
    for root in set(roots.values()):
        (root / "scripts").mkdir(parents=True)
        (root / "scripts/build.sh").touch()
        (root / script.HEADER).parent.mkdir(parents=True)
        (root / script.HEADER).write_bytes(before)

    if script.os.name == "nt":
        monkeypatch.setattr(script, "_directory_lock", lambda path: nullcontext())

    def remote(host, request):
        result = script.worker(dict(request, source=str(roots[host])))
        if interrupted and host == "86" and request["action"] == "set" and request["index"] == 3:
            raise OSError("lost SSH acknowledgement")
        return result

    monkeypatch.setattr(script, "on_node", remote)
    plans = [remote(host, dict(action="plan", index=3)) for host in roots]
    if interrupted:
        with pytest.raises(OSError, match="lost SSH"):
            script.apply_plans(roots, "unused", 3, plans)
    else:
        script.apply_plans(roots, "unused", 3, plans)
    expected = before if interrupted else before.replace(b"= 1;", b"= 3;")
    assert all((root / script.HEADER).read_bytes() == expected for root in roots.values())


def row(index, address, netdev="eth0"):
    return {
        "index": index,
        "gid": str(IPv6Address(f"::ffff:{address}")),
        "type": "RoCE v2",
        "netdev": netdev,
        "cidr": f"{address}/24",
    }


@pytest.fixture
def nodes():
    return [
        {"ports": {
            "hrn5_0/1": [row(3, f"192.0.2.{host}")],
            "hrn5_1/1": [row(3, f"198.51.100.{host}", "eth1")],
        }}
        for host in (1, 2)
    ]


def test_selects_only_index_shared_by_every_port_on_both_nodes(nodes):
    for host, node in enumerate(nodes, start=1):
        node["ports"]["hrn5_0/1"].append(row(5, f"192.0.2.{host + 2}"))
    nodes[0]["ports"]["hrn5_1/1"].append(row(5, "198.51.100.3", "eth1"))
    first_row = nodes[0]["ports"]["hrn5_0/1"][0]
    first_row["gid"] = IPv6Address(first_row["gid"]).exploded

    assert choose_index(nodes) == 3
    with pytest.raises(ValueError):
        choose_index(nodes, requested=5)


def test_ambiguous_indices_require_an_explicit_common_index(nodes):
    for node in nodes:
        for rows in node["ports"].values():
            rows.append(dict(rows[0], index=7))

    with pytest.raises(ValueError):
        choose_index(nodes)
    assert choose_index(nodes, requested=7) == 7
    with pytest.raises(ValueError):
        choose_index(nodes, requested=6)


@pytest.mark.parametrize(("gid", "cidr"), [
    ("::", "::/64"),
    ("fe80::{host}", "fe80::{host}/64"),
    ("::ffff:169.254.1.{host}", "169.254.1.{host}/16"),
    ("::ffff:127.0.0.{host}", "127.0.0.{host}/8"),
    ("::ffff:192.0.2.{host}", None),
])
def test_rejects_unusable_gid_even_when_requested(gid, cidr):
    nodes = []
    for host in (1, 2):
        item = row(3, f"192.0.2.{host}")
        item.update(gid=gid.format(host=host), cidr=cidr and cidr.format(host=host))
        nodes.append({"ports": {"hrn5_0/1": [item]}})
    with pytest.raises(ValueError):
        choose_index(nodes, requested=3)


def test_rejects_disjoint_indices(nodes):
    nodes[1]["ports"]["hrn5_0/1"][0]["index"] = 9
    with pytest.raises(ValueError):
        choose_index(nodes)


def test_rejects_common_index_on_unreachable_peer_subnet(nodes):
    nodes[1]["ports"]["hrn5_1/1"] = [row(3, "203.0.113.2", "eth1")]
    with pytest.raises(ValueError):
        choose_index(nodes, requested=3)


def test_rejects_missing_selected_port(nodes):
    nodes[1]["ports"]["hrn5_1/1"] = []
    with pytest.raises(ValueError):
        choose_index(nodes)


def test_reports_asymmetric_only_indices_on_the_same_subnet():
    nodes = [
        {"ports": {"hrn5_0/1": [row(3, "192.0.2.1")]}},
        {"ports": {"hrn5_0/1": [row(1, "192.0.2.2")]}},
    ]
    with pytest.raises(ValueError, match="3/1"):
        choose_index(nodes)


ORIGINAL = (
    b"// Local SHMEM settings\r\n"
    b"    constexpr uint32_t ACLSHMEMI_HNS_1825_SQ_DB_SGID_IDX = 1; // SQ\r\n"
    b"    constexpr uint32_t UNRELATED = 1;\r\n"
)
TARGET = ORIGINAL.replace(b"SGID_IDX = 1;", b"SGID_IDX = 13;")


def sha(data):
    return hashlib.sha256(data).hexdigest()


@pytest.fixture
def header(tmp_path, monkeypatch):
    if os.name == "nt":
        monkeypatch.setattr(configure_roce_gid, "_directory_lock", lambda path: nullcontext())
    path = tmp_path / "rdma_device_backend_hns_1825.hpp"
    path.write_bytes(ORIGINAL)
    return path


def test_same_index_does_not_rewrite_header(header):
    os.utime(header, (1000000000, 1000000000))
    before = header.stat()
    change_header(header, sha(ORIGINAL), 1)
    assert header.read_bytes() == ORIGINAL
    assert header.stat().st_mtime_ns == before.st_mtime_ns


def test_changes_only_literal_and_resumes_from_exact_target(header):
    change_header(header, sha(ORIGINAL), 13, sha(TARGET))
    assert header.read_bytes() == TARGET
    os.utime(header, (1000000000, 1000000000))
    modified = header.stat().st_mtime_ns
    change_header(header, sha(ORIGINAL), 13, sha(TARGET))
    assert header.stat().st_mtime_ns == modified
    assert header.read_bytes() == TARGET


@pytest.mark.parametrize("current", [ORIGINAL, TARGET])
def test_stale_source_hash_preserves_competing_edit(header, current):
    competing = current + b"// A local change\r\n"
    header.write_bytes(competing)
    with pytest.raises(ValueError):
        change_header(header, sha(ORIGINAL), 13, sha(TARGET))
    assert header.read_bytes() == competing
