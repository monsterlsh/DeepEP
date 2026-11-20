from contextlib import nullcontext
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import paired_benchmark as paired


class Rows:
    def __init__(self, rows):
        self.rows = rows

    def __getitem__(self, key):
        return Rows(self.rows[key])

    def record_stream(self, stream):
        pass

    def add_(self, other):
        return self


@pytest.mark.parametrize("width",[256,1024,2048])
@pytest.mark.parametrize("staged",[False,True])
def test_original_single_heap_swiglu_logical_inputs(monkeypatch,width,staged):
    calls, plans, reserved, weights, windows = [], [], [], [], []
    bs = 39280
    whole = 2 << 30
    # W2048 paired symmetric regions would exceed the 1 GiB limit.
    def heap(world, experts, tokens, k, h, **kwargs):
        return whole if tokens==bs else (270 << 20)*(tokens//1024 + 1)
    stream = SimpleNamespace(wait_stream=lambda _: None)
    torch = SimpleNamespace(bfloat16=None, empty=lambda shape, **kwargs: Rows(range(shape[0])),
        npu=SimpleNamespace(current_stream=lambda:stream,stream=lambda _:nullcontext(),synchronize=lambda:None))
    class NativePlan:
        workspace_bytes=123
        def __init__(self,rows):
            self.rows=rows
            plans.append(rows)
        def forward(self,x,sc,ids,weights,w1,w2,s1,s2,t,sh1,sh2,ss1,ss2,out,generation,clamp):
            assert self.rows==len(x.rows)==len(out.rows)
            assert x.rows==sc.rows==ids.rows==weights.rows==out.rows
            calls.append((generation,self.rows,x.rows.start,w1))
            return out,Rows(range(4))
    class Buffer:
        def __init__(self,group,e,n,k,h,i,*,activation,topk_weights_type,ranks_per_node):
            assert n==width and activation=="swiglu" and topk_weights_type==0
            self._native_capacity,self.num_max_tokens_per_rank=n,n
            self._device=0
            self.allocate_inputs()
            self._runtime=SimpleNamespace(stream=stream)
            self.symmetric_buffer_bytes=heap(group.size(),e//group.size(),n,k,h)
            reserved.append(self.symmetric_buffer_bytes)
        def allocate_inputs(self):
            assert self.num_max_tokens_per_rank==bs
            self.x=self.x_sf=self.topk_idx=self.topk_weights=Rows(range(bs))
        def _create_plan(self,rows):
            assert rows<=self._native_capacity
            return NativePlan(rows)
    def choose(group,ext,world,local,tokens,k,h,i,mode):
        windows.append(tokens)
        assert tokens==bs and mode=="fp8xfp4"
        return width
    api=SimpleNamespace(_mte_window_capacity=choose,_extension=lambda:SimpleNamespace(
            buffer_size=heap,small_batch_max_tokens=256),SymmBuffer=Buffer)
    def prepared(config,buffer,group,routing,torch,api,arms):
        assert arms==("swiglu",) and buffer.num_max_tokens_per_rank==bs
        return {"swiglu": SimpleNamespace(select=lambda decode: ("decode" if decode else "prefill","s1"))},("w2","s2")
    monkeypatch.setattr(paired,"prepare_tensors",prepared)
    monkeypatch.setenv("LOCAL_WORLD_SIZE","8")
    opts=SimpleNamespace(bs=bs,isolated_arm="swiglu",swiglu_weight_mode=0,stage_mte_window=width if staged else None,routing="balanced")
    model=dict(hidden=4096,intermediate_hidden=1536,num_topk=32,num_experts=800,num_shared_experts=0)
    group=SimpleNamespace(size=lambda:8,barrier=lambda _:None)
    buffer,arms,shapes,metadata=paired.isolated_call(opts,dict(model=model),"mte",group,torch,api)
    expected=paired.window_lengths(bs,width)
    assert metadata["window_tokens"]==width and metadata["full_symmetric_bytes"]==whole
    assert max(reserved)<(1<<30) and len(reserved)==1
    assert set(arms)=={"swiglu"} and shapes=={"swiglu":expected}
    assert plans==list(dict.fromkeys(expected))
    assert windows==([] if staged else [bs])
    arms["swiglu"]()
    arms["swiglu"]()
    assert len(plans)==2
    assert [entry[0] for entry in calls]==list(range(1,1+2*len(expected)))
    assert [(r,s) for _,r,s,_ in calls]==[(r,j*width) for j,r in enumerate(expected)]*2
    assert all(layout==("decode" if rows<=256 else "prefill") for _,rows,_,layout in calls)




def test_isolated_pair_rejects_mismatched_fallback_windows(tmp_path):
    from datetime import datetime, timedelta, timezone
    import json
    from summarize_isolated_pairs import summarize
    now = datetime(2026,1,1,tzinfo=timezone.utc)
    spec = dict(config={}, backend="mte", world=1, ranks_per_node=1, hosts_by_rank=["a"], devices_by_rank=[0],
        bs=39280, routing="balanced", warmup_pairs=2,formal_pairs=2,cache="warm",input="fp8",routes="shared",
        weights="gate",torch="1",torch_npu="1",build={})
    for index, arm in enumerate(paired.order(0)):
        path = tmp_path/"0"/arm/"isolated.json"
        path.parent.mkdir(parents=True)
        start = now + timedelta(seconds=2*index)
        implementation = "original_native_swiglu_cached_windows_isolated" if arm=="swiglu" else "public_forward_isolated_process"
        record = dict(spec, isolated_arm=arm, started_at=start.isoformat(),recorded_at=(start+timedelta(seconds=1)).isoformat(),
            windows={arm:paired.window_lengths(39280,1024 if arm=="swiglu" else 2048)},layout=dict(implementation=implementation),
            host_e2e=dict(per_rank=[[{arm:1},{arm:2}]]))
        path.write_text(json.dumps(record))
    with pytest.raises(ValueError,match="same native window sequence"):
        summarize(tmp_path,1)
