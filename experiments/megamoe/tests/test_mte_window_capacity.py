"""Exercise per-rank MTE window negotiation with a native sizing double."""
from types import SimpleNamespace

import pytest

from deep_ep_experimental.megamoe import api

WORLD, LOCAL, TOKENS, K, H, I = 8, 100, 39280, 25, 4096, 1536


def make_ext(eligible=(2048,1024,512,256)):
    requests = []

    def heap(world, experts, width, k, h, *, backend, topk_weights_type, mma_type, windowed):
        assert backend == "mte" and not windowed and topk_weights_type == 1
        return width * (1 << 16) if width in eligible else 2 << 30

    def plan(world, experts, rows, width, k, h, i, activation, mma_type):
        assert 0 < rows <= width and activation == "gelu"
        requests.append((rows, width, mma_type))
        return width * 8000 + rows * 400

    return SimpleNamespace(buffer_size=heap, mte_plan_bytes=plan), requests


def select(monkeypatch, ext, free, *, peers=None, tokens=TOKENS, mma_type="fp8xfp4"):
    npu = SimpleNamespace(current_device=lambda: 0, mem_get_info=lambda _: (free,free))
    monkeypatch.setattr(api, "npu_api", lambda: npu)
    def gather(group, stage, value):
        assert stage == "MTE window HBM budget"
        return [value, *(peers or [])]
    monkeypatch.setattr(api, "gather_control", gather)
    return api._mte_window_capacity(object(), ext, WORLD, LOCAL, tokens, K, H, I, mma_type)


def minimum_for(width, mma_type="fp8xfp4", tokens=TOKENS):
    logical = tokens * (3*H + H//32 + 20*K)
    total_weights = 2*LOCAL*H*I
    weights = total_weights//(2 if mma_type=="fp8xfp4" else 1) + total_weights//32
    heap = width * (1 << 16)
    tail = tokens % width or min(tokens - width,width - 1)
    plans = width * 8000 + width*400 + width*8000 + tail*400
    return logical + weights + 3*LOCAL*4 + heap + plans


@pytest.mark.parametrize("mma_type",["fp8xfp4","fp8xfp8"])
@pytest.mark.parametrize("width",[256,512,1024,2048])
def test_local_budget_boundary(monkeypatch,mma_type,width):
    ext,requests = make_ext()
    free = minimum_for(width,mma_type)
    assert select(monkeypatch,ext,free,mma_type=mma_type)==width
    if width==256:
        assert select(monkeypatch,ext,0,mma_type=mma_type)==256
        assert not any(cap == 256 for rows,cap,kind in requests)
    else:
        assert (width,width,mma_type) in requests
        assert (TOKENS%width,width,mma_type) in requests
        assert select(monkeypatch,ext,free-1,mma_type=mma_type)==width//2


def test_heap_and_collective_intersection(monkeypatch):
    ext,_ = make_ext(eligible=(2048,512,256))
    assert select(monkeypatch,ext,32<<30,peers=[(None,(1024,512))])==512
    with pytest.raises(ValueError,match="no common MTE window"):
        select(monkeypatch,ext,32<<30,peers=[(None,(1024,))])
    with pytest.raises(ValueError,match="rank=1: out of memory"):
        select(monkeypatch,ext,32<<30,peers=[("out of memory",())])


def test_configured_max_and_shorter_calls(monkeypatch):
    ext,requests = make_ext()
    monkeypatch.setattr(api,"_MTE_MAX_WINDOW_TOKENS",1024)
    assert select(monkeypatch,ext,32<<30,tokens=8192)==1024
    assert (1023,1024,"fp8xfp4") in requests


def test_older_native_module_preserves_256(monkeypatch):
    ext,_ = make_ext()
    del ext.mte_plan_bytes
    assert select(monkeypatch,ext,0)==256


def test_heap_budget_precedes_plan_sizing(monkeypatch):
    ext,requests = make_ext(eligible=())
    with pytest.raises(ValueError,match="no candidate fits the MTE symmetric heap"):
        select(monkeypatch,ext,32<<30)
    assert not requests
