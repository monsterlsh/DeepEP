"""Elastic HT shape limits and EP8 prefill capacity; host-only, no NPU."""

import pytest

from deep_ep.backends.mesh_clos._elastic_layout import (
    MAX_TOPK,
    MAX_TOKENS_PER_RANK,
    PAGE_BYTES,
    combine_layout,
    dispatch_layout,
    elastic_size_hint,
    workspace_layout,
)

# Target prefill configuration: 800 experts on EP8, top-k 25, H=4096.
TARGET = {"world": 8, "tokens": 39280, "topk": 25, "experts": 800, "hidden": 4096}


def next_pow2(value):
    return 1 << (value - 1).bit_length()


def _target(world=None):
    values = dict(TARGET)
    if world is not None:
        values["world"] = world
    values["receive_rows"] = next_pow2(values["world"] * values["tokens"])
    values["route_rows"] = values["tokens"] * values["topk"]
    return values


@pytest.mark.parametrize("world", [8, 16, 32])
def test_target_shape_is_accepted_by_every_layout(world):
    v = _target(world)
    workspace_layout(v["world"], v["tokens"], v["topk"], v["experts"])
    dispatch_layout(
        v["world"], v["tokens"], v["topk"], v["experts"], v["receive_rows"], hidden=v["hidden"]
    )
    combine_layout(
        v["world"], v["tokens"], v["topk"], v["receive_rows"], v["route_rows"], hidden=v["hidden"]
    )


def test_token_and_topk_limits_are_the_published_bounds():
    workspace_layout(8, MAX_TOKENS_PER_RANK, MAX_TOPK, 800)
    with pytest.raises(ValueError):
        workspace_layout(8, MAX_TOKENS_PER_RANK + 1, 25, 800)
    with pytest.raises(ValueError):
        workspace_layout(8, 39280, MAX_TOPK + 1, 800)


def test_experts_must_be_positive_and_divisible_by_ep():
    with pytest.raises(ValueError):
        workspace_layout(8, 39280, 25, 801)
    with pytest.raises(ValueError):
        workspace_layout(8, 39280, 25, 4)


def test_receive_rows_are_bounded_by_all_routes():
    v = _target()
    all_routes = v["world"] * v["tokens"] * v["topk"]
    with pytest.raises(ValueError):
        dispatch_layout(v["world"], v["tokens"], v["topk"], v["experts"], all_routes + 1)
    with pytest.raises(ValueError):
        combine_layout(v["world"], v["tokens"], v["topk"], all_routes + 1, v["receive_rows"])


def test_size_hint_reserves_token_stride_and_average_route_rows():
    v = _target()
    hint = elastic_size_hint(
        v["world"], v["tokens"], v["topk"], 1024, v["receive_rows"], v["route_rows"], False, v["hidden"]
    )
    dispatch = dispatch_layout(
        v["world"], v["tokens"], v["topk"], 1024, v["receive_rows"], hidden=v["hidden"]
    )["workspace_bytes"]
    combine = combine_layout(
        v["world"], v["tokens"], v["topk"], v["receive_rows"], v["route_rows"], hidden=v["hidden"]
    )["workspace_bytes"]
    half = (max(dispatch, combine) + PAGE_BYTES - 1) // PAGE_BYTES * PAGE_BYTES
    assert hint == 2 * half
    assert hint % PAGE_BYTES == 0
    # The former min(world, 4) * tokens * topk row estimate would exceed this.
    assert hint < 2**36


def test_buffer_size_hint_matches_the_layout_budget(monkeypatch):
    import importlib

    elastic = importlib.import_module("deep_ep.backends.mesh_clos.elastic")
    monkeypatch.setattr(elastic, "require_native", lambda *_: None)
    v = _target()
    hint = elastic.ElasticBuffer.get_buffer_size_hint(
        None, v["tokens"], v["hidden"], v["topk"], world_size=v["world"]
    )
    assert hint == elastic_size_hint(
        v["world"], v["tokens"], v["topk"], 1024, v["receive_rows"], v["route_rows"], False, v["hidden"]
    )
    # topk=1 keeps the power-of-two receive stride inside the route bound.
    assert elastic.ElasticBuffer.get_buffer_size_hint(None, 100, 4096, 1, world_size=8) > 0
