# Copyright (c) 2026, Lu Lu
# Modified by SimpleBright_Man 2026

"""Execute public wrappers using CPU storage doubles; this does not simulate the NPU kernel."""

import contextlib
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from deep_ep_experimental.megamoe import api
from deep_ep_experimental.megamoe.validation import check_routing, collective_check


class Device(str):
    @property
    def type(self):
        return self.split(":")[0]


DTYPES = dict(
    uint8=np.uint8,
    int8=np.int8,
    int32=np.int32,
    int64=np.int64,
    float32=np.float32,
    bfloat16=np.float32,
    float8_e4m3fn=np.uint8,
    float8_e8m0fnu=np.uint8,
)


class Tensor:
    def __init__(self, data, dtype, device="npu:0"):
        self.data, self.dtype, self.device = np.asarray(data, dtype=DTYPES[dtype]), dtype, Device(device)
        self.requires_grad = False
        self.streams = []
        self.npu_format = 2
        self._version = 0

    @property
    def shape(self):
        return self.data.shape

    @property
    def ndim(self):
        return self.data.ndim

    def __getitem__(self, index):
        return Tensor(self.data[index], self.dtype, self.device)

    def view(self, dtype):
        return Tensor(self.data.view(DTYPES[dtype]), dtype, self.device)

    def reshape(self, shape):
        return Tensor(self.data.reshape(shape), self.dtype, self.device)

    def transpose(self, first, second):
        return Tensor(self.data.swapaxes(first, second), self.dtype, self.device)

    def contiguous(self):
        return Tensor(np.ascontiguousarray(self.data), self.dtype, self.device)

    def clone(self):
        return Tensor(self.data.copy(), self.dtype, self.device)

    def is_contiguous(self):
        return self.data.flags.c_contiguous

    def untyped_storage(self):
        base = self.data
        while isinstance(base.base, np.ndarray):
            base = base.base
        return SimpleNamespace(data_ptr=lambda: base.ctypes.data)

    def record_stream(self, stream):
        self.streams.append(stream)

    def cpu(self):
        return self

    def tolist(self):
        return self.data.tolist()

    def add_(self, other):
        self.data += other.data
        self._version += 1
        return self


@pytest.fixture
def backend(monkeypatch):
    events = []

    class Group:
        def __init__(self, size=2):
            self._size = size

        def size(self):
            return self._size

    def gather(output, value, **kwargs):
        output[:] = [value] * len(output)

    dist = SimpleNamespace(ProcessGroup=Group, get_world_size=lambda group: group.size(), all_gather_object=gather)
    torch = SimpleNamespace(
        **{name: name for name in DTYPES},
        device=Device,
        empty=lambda shape, dtype, device: Tensor(np.empty(shape, dtype=DTYPES[dtype]), dtype, device),
        zeros=lambda shape, dtype, device: Tensor(np.zeros(shape), dtype, device),
        empty_like=lambda t, dtype: Tensor(np.empty(t.shape, dtype=DTYPES[dtype]), dtype, t.device),
        distributed=dist,
        inference_mode=lambda _: contextlib.nullcontext(),
    )
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "torch.distributed", dist)

    def format_cast(t, acl_format, *, customize_dtype=None, input_dtype=None):
        # 检查实际传入的协议，不能把所有 format_cast 参数都静默忽略。
        assert t.dtype == "uint8" and acl_format == 29
        typed = customize_dtype is not None or input_dtype is not None
        if typed:
            assert customize_dtype == "float8_e4m3fn" and input_dtype == "float4_e2m1fn_x2"
        events.append(("format_cast", t))
        t = Tensor(t.data.copy(), t.dtype, t.device)
        t.npu_format = 50 if typed else 51
        return t

    monkeypatch.setitem(
        sys.modules,
        "torch_npu",
        SimpleNamespace(
            get_npu_format=lambda t: t.npu_format, npu_format_cast=format_cast, float4_e2m1fn_x2="float4_e2m1fn_x2"
        ),
    )
    stream = SimpleNamespace(
        wait_stream=lambda _: events.append("wait"), synchronize=lambda: events.append("synchronize")
    )
    npu = SimpleNamespace(
        current_device=lambda: 0, current_stream=lambda _: stream, stream=lambda _: contextlib.nullcontext(),
        mem_get_info=lambda _: (96 << 30, 96 << 30),
    )
    monkeypatch.setattr(api, "npu_api", lambda: npu)
    monkeypatch.setattr(api, "check_window_routes",
                        lambda ids, weights, e: check_routing(ids.tolist(), weights.tolist(), e, ids.shape[1]))
    monkeypatch.setattr(api, "require_native", lambda _: SimpleNamespace(megamoe_workspace_address=lambda _: 4096))

    class Runtime:
        def __init__(self, group, capacity, timeout, *, backend="mte", windowed=False):
            self.device, self.stream, self.native = 0, stream, SimpleNamespace(capacity=capacity)
            self.backend = backend
            self.closed = False

        def ensure_alive(self):
            if self.closed:
                raise RuntimeError("Buffer runtime has been destroyed")

        def close(self):
            events.append("close")
            self.closed = True

    monkeypatch.setattr(api, "_MegaRuntime", Runtime)

    class Plan:
        def __init__(
            self,
            example,
            intermediate,
            topk,
            experts,
            world,
            max_tokens,
            is_a8w4,
            shared_experts,
            ranks_per_node,
            ffts,
            address,
            capacity,
            backend="mte",
            topk_weights_type=0,
            activation="swiglu",
            mma_type="fp8xfp4",
            windowed=False,
            timeout_seconds=120,
        ):
            self.bs, self.workspace_bytes = example.shape[0], 4096
            self.max_tokens = max_tokens
            self.windowed = windowed
            self.backend = backend
            self.topk_weights_type = topk_weights_type
            self.activation = activation
            self.mma_type = mma_type
            self.local_experts = experts
            events.append(("prepare", self.bs))
            events.append(("plan_backend", backend))

        def forward(
            self,
            x,
            scales,
            ids,
            route_weights,
            w1,
            w2,
            s1,
            s2,
            timer,
            shared_w1,
            shared_w2,
            shared_s1,
            shared_s2,
            out,
            generation=0,
            activation_clamp=None,
        ):
            args = (
                x,
                scales,
                ids,
                route_weights,
                w1,
                w2,
                s1,
                s2,
                timer,
                shared_w1,
                shared_w2,
                shared_s1,
                shared_s2,
                out,
                generation,
            )
            events.append(("launch", args))
            events.append(("activation_clamp", activation_clamp))
            output = out if out is not None else Tensor(np.zeros(x.shape), "bfloat16")
            if self.windowed:
                assert 0 < x.shape[0] <= self.max_tokens and out is not None
                output.data[:] = x.data[:, :1]
                counts = np.bincount(ids.data.ravel(), minlength=self.local_experts)[: self.local_experts]
            else:
                assert self.bs == x.shape[0]
                output.data.fill(3)
                counts = np.arange(self.local_experts) + 2
            if self.backend == "urma" and timer is not None:
                timer.data[-1] = generation
            return output, Tensor(counts, "int32")

    def buffer_size(*args, backend="mte", topk_weights_type=0, mma_type="fp8xfp4", windowed=False):
        events.append(("buffer_size", args, backend, windowed))
        return 2 << 20

    def validate_timer(timer, backend):
        events.append(("validate_timer", timer, backend))
        if timer.dtype != "int64" or timer.shape != (65536 if backend == "urma" else 64,):
            raise ValueError("Invalid native timer shape or dtype")

    ext = SimpleNamespace(
        buffer_size=buffer_size,
        urma_window_bytes=lambda world, experts, tokens, topk, hidden, intermediate: (2 << 20, tokens << 20),
        Plan=Plan,
        ffts_config=lambda: 123,
        timer_enabled=True,
        timer_numel=64,
        allocate_timer=lambda example, backend="mte": Tensor(np.zeros(65536 if backend == "urma" else 64), "int64"),
        validate_timer=validate_timer,
        small_batch_max_tokens=256,
        urma_ep8_window_tokens=256,
        urma_windowed_swiglu_enabled=True,
    )
    monkeypatch.setattr(api, "_extension", lambda: ext)
    buffer = api.SymmBuffer(Group(), 4, 17, 2, 1024, 256)
    buffer.topk_idx.data[:] = [0, 1]
    buffer.topk_weights.data[:] = [0.4, 0.6]
    yield SimpleNamespace(buffer=buffer, events=events, ext=ext, dist=dist, group_type=Group)
    buffer.destroy()


def weights():
    prepared = (
        (Tensor(np.zeros((2, 512, 512)), "uint8"), Tensor(np.full((2, 512, 16, 2), 127), "float8_e8m0fnu")),
        (Tensor(np.zeros((2, 1024, 128)), "uint8"), Tensor(np.full((2, 1024, 4, 2), 127), "float8_e8m0fnu")),
    )
    for weight, _ in prepared:
        weight.npu_format = 50
    return prepared


@pytest.mark.parametrize("bf16", [False, True])
def test_public_workflow_output_stats_timer_and_preparation_reuse(backend, bf16):
    b = backend.buffer
    assert b.x.shape == (17, 1024) and b.x_sf.shape == (17, 32)
    original = b.x
    assert b.allocate_inputs() is b and b.x is original
    x = Tensor(np.ones((17, 1024)), "bfloat16") if bf16 else None
    b.validate_inputs(17, x=x)
    b.prepare(17)
    b.prepare(17)
    y, stats = Tensor(np.zeros((17, 1024)), "bfloat16"), Tensor([7, 11], "int32")
    assert api.fp8_fp4_mega_moe(y, *weights(), b, cumulative_local_expert_recv_stats=stats, x=x) is None
    np.testing.assert_array_equal(y.data, 3)
    np.testing.assert_array_equal(stats.data, [9, 14])
    timer = b.allocate_timer()
    assert not timer.data.any()
    api.fp8_fp4_mega_moe(y, *weights(), b, cumulative_local_expert_recv_stats=stats, x=x, timer=timer)
    np.testing.assert_array_equal(stats.data, [11, 17])
    launches = [event[1] for event in backend.events if isinstance(event, tuple) and event[0] == "launch"]
    assert launches[0][8] is None and launches[1][8] is timer
    assert (launches[0][1] is None) == bf16
    assert launches[0][-2] is y and launches[1][-2] is y
    assert (
        sum(event == ("prepare", 17) for event in backend.events if isinstance(event, tuple) and event[0] == "prepare")
        == 1
    )
    assert stats.streams == [b._runtime.stream, b._runtime.stream]
    assert b.workspace_bytes == 4096 and b.symmetric_buffer_bytes == 2 << 20
    b.prepare(1)
    assert b._plan.bs == 1 and b._plan_bs == 1


def test_output_and_stats_aliases_rejected_before_launch(backend):
    b = backend.buffer
    y = Tensor(np.zeros((17, 1024)), "bfloat16")
    with pytest.raises(ValueError, match="alias"):
        api.fp8_fp4_mega_moe(y, *weights(), b, cumulative_local_expert_recv_stats=b.topk_idx[0])
    with pytest.raises(ValueError, match="overlap"):
        api.fp8_fp4_mega_moe(y, *weights(), b, x=y)
    assert not any(isinstance(event, tuple) and event[0] == "launch" for event in backend.events)


@pytest.mark.parametrize("bad", [-1, 4, 0])
def test_collective_routing_preflight_rejects_bad_ids(backend, bad):
    backend.buffer.topk_idx.data[0, 1] = bad
    with pytest.raises(ValueError, match="rank=0.*token=0"):
        backend.buffer.validate_inputs(17)


def test_collective_remote_error_and_bs_mismatch(backend):
    def mismatched(output, value, **kwargs):
        output[:] = [value, (None, 96)]

    backend.dist.all_gather_object = mismatched
    with pytest.raises(ValueError, match="ranks disagree.*rank=1: 96"):
        collective_check(backend.buffer.group, "prepare", None, 17)

    def failed(output, value, **kwargs):
        output[:] = [value, ("out of memory", 17)]

    backend.dist.all_gather_object = failed
    with pytest.raises(ValueError, match="rank=1: out of memory"):
        collective_check(backend.buffer.group, "prepare", None, 17)


def test_timer_disabled_and_destroyed_buffer_fail_explicitly(backend):
    backend.ext.timer_enabled = False
    with pytest.raises(RuntimeError, match="DEEPEP_MEGAMOE_TIMER"):
        backend.buffer.allocate_timer()
    backend.buffer.destroy()
    with pytest.raises(RuntimeError, match="destroyed"):
        api.fp8_fp4_mega_moe(Tensor(np.zeros((17, 1024)), "bfloat16"), *weights(), backend.buffer)


@pytest.mark.parametrize("dtype", ["int8", "uint8"])
def test_fp4_and_packed_e8m0_are_reinterpreted_without_numeric_cast(backend, dtype):
    raw = np.arange(512 * 64, dtype=np.uint8).reshape(1, 512, 64)
    # Distinct row-dependent bytes expose both weight and scale column mismatches.
    row_ids = np.arange(512, dtype=np.uint16)
    raw[0, :, 0] = row_ids % 256
    raw[0, :, 1] = row_ids // 256
    sf = np.empty((1, 512, 4), dtype=np.uint8)
    sf[0, :, 0] = row_ids % 256
    sf[0, :, 1] = row_ids // 256
    sf[0, :, 2:] = np.array([129, 140], dtype=np.uint8)
    pair = Tensor(raw.view(DTYPES[dtype]), dtype), Tensor(sf.view(np.int32), "int32")
    order = np.array([*range(128), *range(256, 384), *range(128, 256), *range(384, 512)])
    prepared = api.transform_weights_for_mega_moe(pair, pair)
    for idx, (weight, scale) in enumerate(prepared):
        rows = order if idx == 0 else np.arange(512)
        np.testing.assert_array_equal(weight.data, raw[:, rows])
        np.testing.assert_array_equal(scale.data, sf[:, rows].reshape(1, 512, 2, 2))
        assert weight.dtype == "uint8" and scale.dtype == "float8_e8m0fnu"
        assert weight.npu_format == 50
    assert sum(isinstance(event, tuple) and event[0] == "format_cast" for event in backend.events) == 3
    planar_weight, planar_scale = prepared[0].select(False)
    np.testing.assert_array_equal(planar_weight.data, raw)
    np.testing.assert_array_equal(planar_scale.data, sf.reshape(1, 512, 2, 2))


def test_shared_fp4_preparation_retains_planar_weights_and_scales(backend):
    raw = np.arange(512 * 64, dtype=np.uint8).reshape(1, 512, 64)
    sf = np.arange(512 * 4, dtype=np.uint8).reshape(1, 512, 2, 2)
    pair = Tensor(raw, "uint8"), Tensor(sf, "float8_e8m0fnu")
    for weight, scale in api.transform_weights_for_mega_moe(pair, pair, shared=True):
        np.testing.assert_array_equal(weight.data, raw)
        np.testing.assert_array_equal(scale.data, sf)


def test_a4w4_preparation_casts_planar_bytes_and_marks_generic_nz(backend, monkeypatch):
    def cast_bytes(weight, acl_format):
        assert weight.dtype == "uint8" and acl_format == 29
        result = Tensor(weight.data.copy(), weight.dtype, weight.device)
        result.npu_format = 29
        return result

    monkeypatch.setattr(sys.modules["torch_npu"], "npu_format_cast", cast_bytes)
    weights, scales = np.arange(512 * 64, dtype=np.uint8).reshape(1, 512, 64), np.empty((1, 512, 2, 2), dtype=np.uint8)
    weights[:, :, 0] = np.arange(512, dtype=np.uint16) % 251
    scales[:] = np.arange(512, dtype=np.uint16)[None, :, None, None] % 251
    raw = Tensor(weights, "uint8"), Tensor(scales, "float8_e8m0fnu")
    for pair in api.transform_weights_for_mega_moe(raw, raw, mma_type="fp4xfp4"):
        np.testing.assert_array_equal(pair[0].data, weights)
        np.testing.assert_array_equal(pair[1].data, scales)
        api._check_a4w4_weight_format(pair, "routed")
        with pytest.raises(ValueError, match="got format=29"):
            api._check_a4w4_weight_format(tuple(pair), "routed")


@pytest.mark.parametrize("bf16", [False, True])
def test_routed_layout_switches_with_actual_bs_without_hot_path_cast(backend, bf16, request):
    b = api.SymmBuffer(backend.group_type(), 4, 4096, 2, 1024, 256)
    request.addfinalizer(b.destroy)
    b.x.data.fill(0)
    b.x_sf.data.fill(0)
    b.topk_idx.data[:] = [0, 1]
    b.topk_weights.data.fill(1)
    raw = weights()
    for weight, _ in raw:
        weight.npu_format = 2
    raw[0][0].data[:, :, 0] = np.arange(512, dtype=np.uint16) % 251
    w1, w2 = api.transform_weights_for_mega_moe(*raw)
    # Preparing weights must not share writable storage with the caller's raw scales.
    assert w1.select(False)[1].untyped_storage().data_ptr() != raw[0][1].untyped_storage().data_ptr()
    casts = sum(isinstance(e, tuple) and e[0] == "format_cast" for e in backend.events)
    for bs in (96, 256, 257, 4096, 96):
        output = Tensor(np.zeros((bs, 1024)), "bfloat16")
        x = Tensor(np.ones((bs, 1024)), "bfloat16") if bf16 else None
        api.fp8_fp4_mega_moe(output, w1, w2, b, x=x)
        launch = [e[1] for e in backend.events if isinstance(e, tuple) and e[0] == "launch"][-1]
        selected = w1.select(bs <= 256)
        assert launch[4] is selected[0] and launch[6] is selected[1]
        assert b._plan.bs == bs  # actual BS, not max capacity or per-expert M
    assert sum(isinstance(e, tuple) and e[0] == "format_cast" for e in backend.events) == casts
    with pytest.raises(ValueError, match="keep its returned l1 pair intact"):
        api.fp8_fp4_mega_moe(Tensor(np.zeros((257, 1024)), "bfloat16"), tuple(w1), w2, b)


@pytest.mark.parametrize("layout", [True, False])
@pytest.mark.parametrize("index", [0, 1])
def test_prepared_layout_mutation_is_not_silently_cached(backend, layout, index):
    raw = weights()
    for weight, _ in raw:
        weight.npu_format = 2
    w1, _ = api.transform_weights_for_mega_moe(*raw)
    tensor = w1.select(layout)[index]
    tensor.add_(Tensor(np.ones(tensor.shape), tensor.dtype))
    # Even requesting the other layout must fail rather than using an out-of-date copy.
    with pytest.raises(ValueError, match="modified in-place"):
        w1.select(not layout)


def test_fp4_transform_rejects_double_preparation(backend):
    pair = Tensor(np.zeros((1, 256, 64)), "uint8"), Tensor(np.ones((1, 256, 2, 2)), "float8_e8m0fnu")
    pair[0].npu_format = 50
    with pytest.raises(ValueError, match="already converted NZ"):
        api.transform_weights_for_mega_moe(pair, pair)


def test_routed_transform_rejects_incomplete_gate_up_pair(backend):
    pair = Tensor(np.zeros((1, 128, 64)), "uint8"), Tensor(np.ones((1, 128, 2, 2)), "float8_e8m0fnu")
    with pytest.raises(ValueError, match="divisible by 128"):
        api.transform_weights_for_mega_moe(pair, pair)


def test_fp8_weights_remain_nd_without_fp4_format_cast(backend):
    pair = (Tensor(np.zeros((1, 128, 128)), "float8_e4m3fn"), Tensor(np.full((1, 128, 2, 2), 127), "float8_e8m0fnu"))
    for weight, scale in api.transform_weights_for_mega_moe(pair, pair):
        assert weight is pair[0]
        np.testing.assert_array_equal(scale.data, pair[1].data)
    assert not any(isinstance(event, tuple) and event[0] == "format_cast" for event in backend.events)


@pytest.mark.parametrize(("backend_name", "shared"), [("urma", False), ("mte_roce", True)])
def test_fp8_swiglu_interleaves_weight_and_scale_rows_once(backend, backend_name, shared):
    rows = np.repeat(np.arange(4, dtype=np.uint8), 128)
    first = (Tensor(np.broadcast_to(rows[None, :, None], (1, 512, 64)).copy(), "float8_e4m3fn"),
             Tensor(np.broadcast_to((rows + 120)[None, :, None, None], (1, 512, 1, 2)).copy(), "float8_e8m0fnu"))
    second = (Tensor(np.zeros((1, 64, 256)), "float8_e4m3fn"),
              Tensor(np.full((1, 64, 4, 2), 127), "float8_e8m0fnu"))
    packed, gmm2 = api.transform_weights_for_mega_moe(first, second, backend=backend_name, shared=shared)
    assert packed[0].npu_format == 2 and packed[0].is_contiguous() and packed[1].is_contiguous()
    np.testing.assert_array_equal(packed[0].data[0, ::128, 0], [0, 2, 1, 3])
    np.testing.assert_array_equal(packed[1].data[0, ::128, 0, 0], [120, 122, 121, 123])
    np.testing.assert_array_equal(first[0].data[0, :, 0], rows)
    assert gmm2[0] is second[0]
    with pytest.raises(ValueError, match="already packed"):
        api.transform_weights_for_mega_moe(packed, gmm2, backend=backend_name, shared=shared)


@pytest.mark.parametrize("weight_format", [29, 50])
def test_prepared_fp4_nz_variants_reach_launch(backend, weight_format):
    w1, w2 = weights()
    w1[0].npu_format = w2[0].npu_format = weight_format
    api.fp8_fp4_mega_moe(Tensor(np.zeros((17, 1024)), "bfloat16"), w1, w2, backend.buffer)
    assert any(isinstance(event, tuple) and event[0] == "launch" for event in backend.events)


@pytest.mark.parametrize("name", ["l1_weights", "l2_weights", "shared_l1_weights", "shared_l2_weights"])
@pytest.mark.parametrize("weight_format", [2, 4, 51, 52, 53, 54])
def test_invalid_fp4_layout_rejected_with_tensor_context_before_launch(backend, name, weight_format):
    b = backend.buffer
    b.num_shared_experts = 2
    pairs = dict(zip(("l1_weights", "l2_weights", "shared_l1_weights", "shared_l2_weights"), (*weights(), *weights())))
    pairs[name][0].npu_format = weight_format
    with pytest.raises(ValueError, match=rf"{name}:.*got format={weight_format}, dtype=uint8, shape="):
        api.fp8_fp4_mega_moe(Tensor(np.zeros((17, 1024)), "bfloat16"), sym_buffer=b, **pairs)
    assert not any(isinstance(event, tuple) and event[0] == "launch" for event in backend.events)


def test_format_cast_returning_nd_fails_at_preparation(backend, monkeypatch):
    monkeypatch.setattr(sys.modules["torch_npu"], "npu_format_cast", lambda t, *a, **k: t)
    pair = (Tensor(np.zeros((1, 256, 64)), "uint8"), Tensor(np.full((1, 256, 2, 2), 127), "float8_e8m0fnu"))
    with pytest.raises(ValueError, match="l1_weights:.*got format=2"):
        api.transform_weights_for_mega_moe(pair, pair)


def test_store_group_public_configuration_and_prepare_without_process_group(backend, monkeypatch):
    from deep_ep._store import StoreGroup

    stages = []
    group = StoreGroup(None, 0, 2, prefix="api-test")

    def gather(value, stage):
        stages.append(stage)
        return [value, value]

    group.all_gather = gather

    def forbidden(*args, **kwargs):
        pytest.fail("Store API accessed a ProcessGroup")

    monkeypatch.setattr(backend.dist, "get_world_size", forbidden)
    monkeypatch.setattr(backend.dist, "all_gather_object", forbidden)
    buffer = api.SymmBuffer(group, 4, 17, 2, 1024, 256)
    buffer.topk_idx.data[:] = [0, 1]
    buffer.topk_weights.data[:] = [0.4, 0.6]
    buffer.validate_inputs(17)
    buffer.prepare(17)
    buffer.destroy()
    assert stages == [
        "check:MegaMoE product selection",
        "check:configuration",
        "check:input allocation",
        "check:input preflight",
        "check:prepare",
        "check:prepare BS=17",
    ]


def test_non_finite_route_weights_and_bad_capacity_fail_before_launch(backend):
    backend.buffer.topk_weights.data[2, 0] = np.nan
    with pytest.raises(ValueError, match="token=2.*finite"):
        backend.buffer.validate_inputs(17)
    with pytest.raises(ValueError, match="prepare.*BS"):
        backend.buffer.prepare(18)
    assert not any(isinstance(event, tuple) and event[0] == "launch" for event in backend.events)


def test_plan_allocation_failure_is_collective_and_retryable(backend):
    plan_type = backend.ext.Plan

    def fail(*args, **kwargs):
        raise RuntimeError("workspace allocation failed")

    backend.ext.Plan = fail
    with pytest.raises(ValueError, match="rank=0: workspace allocation failed"):
        backend.buffer.prepare(17)
    assert backend.buffer._plan is None and backend.buffer._plan_bs is None
    backend.ext.Plan = plan_type
    backend.buffer.prepare(17)
    assert backend.buffer._plan_bs == 17


@pytest.mark.parametrize("board,expected", [(0x44, "urma"), (7, "mte"), (3, "mte"), (None, "mte"), (0x6C, "mte")])
def test_megamoe_product_selection(monkeypatch, board, expected):
    from deep_ep_experimental.megamoe import backend as selector
    from deep_ep import _machine

    calls = []
    monkeypatch.setattr(_machine, "physical_device", lambda: 17)
    monkeypatch.setattr(_machine, "board_id", lambda physical: board)
    monkeypatch.setattr(selector, "collective_check", lambda group, stage, error, value: calls.append(value))
    assert selector.select_backend(object()) == expected
    assert calls == [expected]


def urma_buffer(backend, monkeypatch, *, shared=0):
    from deep_ep_experimental.megamoe import backend as selector

    monkeypatch.setattr(selector, "select_backend", lambda group: "urma")
    # Also cover a module-level imported selector without coupling the test to
    # whether api.py imports the helper locally or at module initialization.
    monkeypatch.setattr(api, "select_backend", lambda group: "urma", raising=False)
    b = api.SymmBuffer(backend.group_type(16), 32, 2048, 2, 256, 256, num_shared_experts=shared)
    b.topk_idx.data[:] = [0, 1]
    b.topk_weights.data[:] = [0.4, 0.6]
    raw = (
        (Tensor(np.zeros((2, 512, 128)), "uint8"), Tensor(np.full((2, 512, 4, 2), 127), "float8_e8m0fnu")),
        (Tensor(np.zeros((2, 256, 128)), "uint8"), Tensor(np.full((2, 256, 4, 2), 127), "float8_e8m0fnu")),
    )
    return b, api.transform_weights_for_mega_moe(*raw)


def test_urma_uses_unified_plan_and_generation_survives_bs_changes(backend, monkeypatch):
    b, (w1, w2) = urma_buffer(backend, monkeypatch)

    def forbidden_readback(*args, **kwargs):
        pytest.fail("Normal forward must not read routing tensors on Host")

    monkeypatch.setattr(Tensor, "cpu", forbidden_readback)
    stats = Tensor([7, 11], "int32")
    generations = []
    plans = []
    casts_before = sum(isinstance(e, tuple) and e[0] == "format_cast" for e in backend.events)
    try:
        assert not hasattr(b, "_urma")
        for bs in (17, 1024, 1025, 2048, 17):
            y = Tensor(np.zeros((bs, 256)), "bfloat16")
            assert api.fp8_fp4_mega_moe(y, w1, w2, b, cumulative_local_expert_recv_stats=stats) is None
            launch = [e[1] for e in backend.events if isinstance(e, tuple) and e[0] == "launch"][-1]
            assert launch[-2] is y
            assert launch[4] is w1.select(True)[0]  # URMA uses fused layout even above the MTE threshold.
            assert isinstance(b._plan, backend.ext.Plan)
            assert b._plan.backend == "urma"
            generations.append(launch[-1])
            plans.append(b._plan)
            np.testing.assert_array_equal(y.data, 3)
        assert generations == [1, 2, 3, 4, 5]
        assert len({id(plan) for plan in plans}) > 1  # generation belongs to the owner, not a replaced plan.
        np.testing.assert_array_equal(stats.data, [17, 26])
        assert stats.streams == [b._runtime.stream] * 5
        assert sum(isinstance(e, tuple) and e[0] == "format_cast" for e in backend.events) == casts_before
    finally:
        b.destroy()


def test_urma_shared_shapes_reach_same_forward_without_extra_host_interface(backend, monkeypatch):
    import inspect

    b, weights = urma_buffer(backend, monkeypatch, shared=1)
    # Independent shared I=512; routed I=256. Native Plan is responsible for
    # inferring and preparing shared resources from these existing operands.
    sw1 = Tensor(np.zeros((1, 1024, 256)), "float8_e4m3fn")
    ss1 = Tensor(np.full((1, 1024, 4, 2), 127), "float8_e8m0fnu")
    sw2 = Tensor(np.zeros((1, 256, 512)), "float8_e4m3fn")
    ss2 = Tensor(np.full((1, 256, 8, 2), 127), "float8_e8m0fnu")
    y = Tensor(np.zeros((17, 256)), "bfloat16")
    try:
        assert "shared_intermediate_hidden" not in inspect.signature(api.SymmBuffer).parameters
        api.fp8_fp4_mega_moe(y, *weights, b, shared_l1_weights=(sw1, ss1), shared_l2_weights=(sw2, ss2))
        launch = [e[1] for e in backend.events if isinstance(e, tuple) and e[0] == "launch"][-1]
        assert launch[9:13] == (sw1, sw2, ss1, ss2)
        assert b._plan.backend == "urma"
        assert b._plan.workspace_bytes == b.workspace_bytes
    finally:
        b.destroy()


def test_urma_timer_uses_unified_forward_and_plan_diagnostics(backend, monkeypatch):
    b, weights = urma_buffer(backend, monkeypatch)
    try:
        timer = b.allocate_timer()
        assert timer.data.size > backend.ext.timer_numel
        y = Tensor(np.zeros((17, 256)), "bfloat16")
        api.fp8_fp4_mega_moe(y, *weights, b, timer=timer)
        launch = [e[1] for e in backend.events if isinstance(e, tuple) and e[0] == "launch"][-1]
        assert launch[8] is timer
        assert timer.data[-1] == launch[-1]
        assert not hasattr(b._plan, "diagnostic_timer")
        assert not hasattr(b._plan, "cv_rows")
        assert not hasattr(b, "_urma")
    finally:
        b.destroy()


def test_urma_plan_allocation_failure_poisoned_without_caching_shape(backend, monkeypatch):
    b, weights = urma_buffer(backend, monkeypatch)
    original_plan = backend.ext.Plan
    attempts = []

    def fail(*args, **kwargs):
        attempts.append((args, kwargs))
        raise RuntimeError("URMA workspace allocation failed")

    backend.ext.Plan = fail
    y = Tensor(np.zeros((17, 256)), "bfloat16")
    try:
        with pytest.raises(RuntimeError, match="URMA workspace allocation failed"):
            api.fp8_fp4_mega_moe(y, *weights, b)
        assert b._failure == "URMA workspace allocation failed"
        assert b._plan is None and b._plan_bs is None
        assert len(attempts) == 1
        backend.ext.Plan = original_plan
        with pytest.raises(RuntimeError, match="buffer failed.*restart"):
            api.fp8_fp4_mega_moe(y, *weights, b)
        assert b._plan is None and b._plan_bs is None
        assert len(attempts) == 1
        assert not any(isinstance(e, tuple) and e[0] == "launch" for e in backend.events)
    finally:
        backend.ext.Plan = original_plan
        b.destroy()


def test_activation_clamp_reaches_native_forward(backend):
    y = Tensor(np.zeros((17, 1024)), "bfloat16")
    api.fp8_fp4_mega_moe(y, *weights(), backend.buffer, activation_clamp=0.25)
    assert ("activation_clamp", 0.25) in backend.events
    with pytest.raises(ValueError, match="activation_clamp"):
        api.fp8_fp4_mega_moe(y, *weights(), backend.buffer, activation_clamp=-1)


@pytest.mark.parametrize("mode", [0, 1])
def test_route_weight_mode_reaches_plan(backend, mode):
    b = api.SymmBuffer(backend.group_type(), 4, 17, 2, 1024, 256, topk_weights_type=mode)
    try:
        b.prepare(17)
        assert b._plan.topk_weights_type == mode
    finally:
        b.destroy()


@pytest.mark.parametrize("mode", [-1, 2, True, 1.0])
def test_invalid_route_weight_mode(backend, mode):
    with pytest.raises((ValueError, TypeError), match="topk_weights_type"):
        api.SymmBuffer(backend.group_type(), 4, 17, 2, 1024, 256, topk_weights_type=mode)


def gelu_weights():
    _, second = weights()
    first = (
        Tensor(np.zeros((2, 256, 512)), "uint8"),
        Tensor(np.full((2, 256, 16, 2), 127), "float8_e8m0fnu"),
    )
    first[0].npu_format = 50
    return first, second


def test_gelu_single_projection_and_forced_prequant_mode_reach_native(backend):
    b = api.SymmBuffer(backend.group_type(), 4, 17, 2, 1024, 256, activation="gelu")
    try:
        assert b.topk_weights_type == 1
        b.prepare(17)
        assert b._plan.activation == "gelu" and b._plan.topk_weights_type == 1
        y = Tensor(np.zeros((17, 1024)), "bfloat16")
        api.fp8_fp4_mega_moe(y, *gelu_weights(), b, activation="gelu")
        assert any(isinstance(event, tuple) and event[0] == "launch" for event in backend.events)
        with pytest.raises(ValueError, match="must match"):
            api.fp8_fp4_mega_moe(y, *gelu_weights(), b)
        with pytest.raises(ValueError, match="activation_clamp"):
            api.fp8_fp4_mega_moe(y, *gelu_weights(), b, activation="gelu", activation_clamp=0.2)
    finally:
        b.destroy()


def test_gelu_weight_transform_keeps_one_planar_projection(backend):
    raw = (
        Tensor(np.zeros((1, 256, 64)), "uint8"),
        Tensor(np.full((1, 256, 2, 2), 127), "float8_e8m0fnu"),
    )
    transformed = api.transform_weights_for_mega_moe(raw, raw, activation="gelu")
    assert not isinstance(transformed[0], api._RoutedWeightPair)
    assert sum(isinstance(event, tuple) and event[0] == "format_cast" for event in backend.events) == 2
    for weight, _ in transformed:
        assert weight.npu_format == 50


@pytest.mark.parametrize("activation", ["swiglu", "gelu"])
def test_a8w8_activation_modes_reach_native(backend, activation):
    b = api.SymmBuffer(backend.group_type(), 4, 17, 2, 1024, 256, activation=activation, mma_type="fp8xfp8")
    try:
        first_width = 512 if activation == "swiglu" else 256
        first = (
            Tensor(np.zeros((2, first_width, 1024)), "float8_e4m3fn"),
            Tensor(np.full((2, first_width, 16, 2), 127), "float8_e8m0fnu"),
        )
        second = (
            Tensor(np.zeros((2, 1024, 256)), "float8_e4m3fn"),
            Tensor(np.full((2, 1024, 4, 2), 127), "float8_e8m0fnu"),
        )
        b.topk_idx.data[:] = [0, 1]
        b.topk_weights.data[:] = [0.4, 0.6]
        y = Tensor(np.zeros((17, 1024)), "bfloat16")
        assert not b.timer_enabled
        with pytest.raises(RuntimeError, match="windowed URMA W8A8"):
            b.allocate_timer()
        with pytest.raises(RuntimeError, match="windowed URMA W8A8"):
            api.fp8_fp4_mega_moe(y, first, second, b, activation=activation, timer=backend.buffer.allocate_timer())
        assert not native_launches(backend)
        api.fp8_fp4_mega_moe(y, first, second, b, activation=activation)
        assert b._plan.mma_type == "fp8xfp8" and b._plan.activation == activation
        assert b._plan.topk_weights_type == (1 if activation == "gelu" else 0)
        assert any(isinstance(event, tuple) and event[0] == "launch" for event in backend.events)
    finally:
        b.destroy()


def test_a4w4_rejects_unsupported_topology(backend):
    with pytest.raises(ValueError, match="16 ranks on one host"):
        api.SymmBuffer(backend.group_type(), 4, 17, 2, 1024, 256, mma_type="fp4xfp4")


@pytest.mark.parametrize("activation", ["gelu", "swiglu"])
def test_pod_a4w4_target_uses_fp4_mte_full_and_tail_plans(backend, monkeypatch, activation):
    gather_control = api.gather_control

    def gather(group, stage, value):
        if stage == "MTE A4W4 host layout":
            return [f"host-{rank // 8}" for rank in range(group.size())]
        return gather_control(group, stage, value)

    monkeypatch.setattr(api, "gather_control", gather)
    monkeypatch.setattr(backend.ext, "mte_a4w4_enabled", True, raising=False)
    monkeypatch.setattr(backend.ext, "mte_plan_bytes", lambda *args, **kwargs: 4096, raising=False)
    torch = sys.modules["torch"]

    def shape_only(shape, dtype, device):
        return Tensor(np.broadcast_to(np.zeros((), dtype=DTYPES[dtype]), shape), dtype, device)

    monkeypatch.setattr(torch, "empty", shape_only)
    monkeypatch.setattr(torch, "empty_like", lambda tensor, dtype: shape_only(tensor.shape, dtype, tensor.device))
    b = api.SymmBuffer(backend.group_type(32), 800, 39280, 25, 4096, 1536, ranks_per_node=8,
                       backend="mte", mma_type="fp4xfp4", activation=activation)
    try:
        assert b.x.shape == (39280, 2048) and b.x.dtype == "uint8"
        assert b.ranks_per_node == 8 and b.window_tokens == 256 and b.num_forward_windows(39280) == 154
        assert not b.timer_enabled
        monkeypatch.setattr(backend.ext, "mte_a4w4_timer_enabled", True, raising=False)
        assert b.timer_enabled
        b.prepare(39280)
        for plan, rows in ((b._plan, 256), (b._tail_plan, 112)):
            assert plan.bs == rows and plan.max_tokens == 256 and plan.local_experts == 25
            assert plan.mma_type == "fp4xfp4" and plan.backend == "mte"
            assert plan.topk_weights_type == int(activation == "gelu")
    finally:
        b.destroy()


def test_pod_a4w4_rejects_noncontiguous_hosts(backend, monkeypatch):
    gather_control = api.gather_control

    def gather(group, stage, value):
        if stage == "MTE A4W4 host layout":
            return [f"host-{rank % 4}" for rank in range(group.size())]
        return gather_control(group, stage, value)

    monkeypatch.setattr(api, "gather_control", gather)
    with pytest.raises(ValueError, match="four contiguous 8-rank hosts"):
        api.SymmBuffer(backend.group_type(32), 800, 39280, 25, 4096, 1536, ranks_per_node=8,
                       backend="mte", mma_type="fp4xfp4")


@pytest.mark.parametrize("transport,world", [("mte", 16), ("mte_roce", 32)])
@pytest.mark.parametrize("mma", ["fp8xfp4", "fp8xfp8", "fp4xfp4"])
@pytest.mark.parametrize("activation,weight_mode", [("gelu", 1), ("swiglu", 0)])
def test_ubx_mx_window_and_tail_forward(backend, monkeypatch, transport, world, mma, activation, weight_mode):
    ext = backend.ext
    def scratch(_world, _local, _rows, capacity, *args, **kwargs):
        if capacity > 16:
            raise RuntimeError("tiling exceeds UB")
        return 4096

    for name in ("mte_a4w4_enabled", "mte_roce_mx_enabled"):
        monkeypatch.setattr(ext, name, True, raising=False)
    for name in ("roce_plan_bytes", "mte_plan_bytes"):
        monkeypatch.setattr(ext, name, scratch, raising=False)
    monkeypatch.setattr(ext, "roce_buffer_size", ext.buffer_size, raising=False)
    def roce_plan(example, intermediate, k, experts, ep, width, shared, ffts, address, capacity, **kwargs):
        return ext.Plan(example, intermediate, k, experts, ep, width, mma == "fp8xfp4", shared,
                        16, ffts, address, capacity, backend="mte_roce", **kwargs)
    monkeypatch.setattr(ext, "RocePlan", roce_plan, raising=False)
    local, tokens, topk, h, i = 2, 35, 32, 1024, 256
    b = api.SymmBuffer(backend.group_type(world), local * world, tokens, topk, h, i, ranks_per_node=16,
                       backend=transport, mma_type=mma, activation=activation)
    try:
        cols = i if activation == "gelu" else 2 * i
        def raw(n, k):
            wtype = "float8_e4m3fn" if mma == "fp8xfp8" else "uint8"
            w = Tensor(np.zeros((local, n, k if mma == "fp8xfp8" else k // 2)), wtype)
            w.npu_format = 2
            return w, Tensor(np.full((local, n, k // 64, 2), 127), "float8_e8m0fnu")
        wpairs = raw(cols, h), raw(h, i)
        weights = api.transform_weights_for_mega_moe(*wpairs, activation, backend=transport, mma_type=mma)
        assert b.window_tokens == 16 and b.num_forward_windows(tokens) == 3
        assert b.x.shape == (tokens, h // (2 if mma == "fp4xfp4" else 1))
        b.topk_idx.data[:] = np.arange(topk)
        b.topk_weights.data[:] = np.linspace(-0.5, 0.5, topk)
        y = Tensor(np.zeros((tokens, h)), "bfloat16")
        stats = Tensor(np.zeros(local), "int32")
        if mma == "fp4xfp4":
            assert all(pair[0].npu_format == 51 for pair in weights)
            wrong = api.transform_weights_for_mega_moe(*wpairs, activation, backend=transport)
            with pytest.raises(ValueError, match="A4W4 requires NZ with C0=64 FP4"):
                api.fp8_fp4_mega_moe(y, *wrong, b, activation=activation)
            assert not native_launches(backend)
        timer = None
        if transport == "mte_roce":
            assert not b.timer_enabled
            monkeypatch.setattr(ext, "mte_roce_timer_enabled", True, raising=False)
            timer = b.allocate_timer()
        elif mma == "fp4xfp4":
            assert not b.timer_enabled
            monkeypatch.setattr(ext, "mte_a4w4_timer_enabled", True, raising=False)
            timer = b.allocate_timer()
        api.fp8_fp4_mega_moe(y, *weights, b, activation=activation, cumulative_local_expert_recv_stats=stats, timer=timer)
        launches = native_launches(backend)
        assert [(launch[0].shape[0], launch[-1]) for launch in launches] == [(16, 1), (16, 2), (3, 3)]
        assert all(launch[8] is expected for launch, expected in zip(launches, (timer, None, None)))
        if timer is not None:
            assert ("validate_timer", timer, transport) in backend.events
        assert b._plan.topk_weights_type == b._tail_plan.topk_weights_type == weight_mode
        assert b._plan.mma_type == b._tail_plan.mma_type == mma
        np.testing.assert_array_equal(stats.data, [6, 9])
        assert np.all(y.data == 3)
    finally:
        b.destroy()


@pytest.mark.parametrize("world", [8, 16, 32])
@pytest.mark.parametrize("mma_type", ["fp8xfp4", "fp8xfp8"])
def test_mte_target_topology_topk25_reaches_plan_for_a8_modes(backend, world, mma_type):
    b = api.SymmBuffer(
        backend.group_type(world),
        800,
        39280,
        25,
        4096,
        1536,
        activation="gelu",
        mma_type=mma_type,
        ranks_per_node=min(world, 16),
    )
    try:
        assert b.x.shape == (39280, 4096) and b.topk_idx.shape == (39280, 25)
        b.prepare(39280)
        assert b._plan.local_experts == 800 // world
        assert b._plan.topk_weights_type == 1 and b._plan.mma_type == mma_type
    finally:
        b.destroy()


def test_mte_topk32_boundary_and_invalid_activation_precision(backend):
    b = api.SymmBuffer(backend.group_type(), 64, 17, 32, 1024, 256, activation="gelu")
    assert b.topk_weights_type == 1
    b.destroy()
    with pytest.raises(ValueError, match="num_topk"):
        api.SymmBuffer(backend.group_type(), 64, 17, 33, 1024, 256)
    with pytest.raises(ValueError, match="activation"):
        api.SymmBuffer(backend.group_type(), 4, 17, 2, 1024, 256, activation="tanh")
    with pytest.raises(ValueError, match="mma_type"):
        api.SymmBuffer(backend.group_type(), 4, 17, 2, 1024, 256, mma_type="bf16xbf16")


def test_mte_roce_routes_to_the_hierarchical_native_plan(backend, monkeypatch):
    group = backend.group_type(32)
    options = dict(ranks_per_node=16, backend="mte_roce")
    with pytest.raises(ValueError, match=r"install a MegaMoE wheel with MTE\+RoCE MX support"):
        api.SymmBuffer(group, 64, 17, 2, 1024, 256, **options)
    with pytest.raises(ValueError, match="two 16-rank hosts"):
        api.SymmBuffer(backend.group_type(16), 64, 17, 2, 1024, 256, **options)

    monkeypatch.setattr(backend.ext, "mte_roce_mx_enabled", True, raising=False)
    monkeypatch.setattr(backend.ext, "roce_plan_bytes", lambda *args, **kwargs: 4096, raising=False)

    def roce_buffer_size(world, experts, tokens, topk, hidden, **kwargs):
        assert (world, experts, topk, hidden) == (32, 2, 2, 1024) and tokens <= 17
        return 4 << 20

    def roce_plan(example, intermediate, topk, experts, world, max_tokens, shared, ffts, address, capacity, **kwargs):
        assert (intermediate, topk, experts, world, max_tokens, shared, ffts, address, capacity) == (
            256, 2, 2, 32, 17, 0, 123, 4096, 4 << 20
        )
        return backend.ext.Plan(
            example, intermediate, topk, experts, world, max_tokens, True, shared, 16, ffts,
            address, capacity, backend="mte_roce", **kwargs,
        )

    monkeypatch.setattr(backend.ext, "roce_buffer_size", roce_buffer_size, raising=False)
    monkeypatch.setattr(backend.ext, "RocePlan", roce_plan, raising=False)
    b = api.SymmBuffer(group, 64, 17, 2, 1024, 256, **options)
    try:
        assert b._runtime.backend == "mte_roce" and b.num_forward_windows(17) == 1
        assert not b.timer_enabled
        b.topk_idx.data[:] = [0, 1]
        b.topk_weights.data[:] = [0.4, 0.6]
        for bs in (17, 5, 5):
            b.prepare(bs)
            y = Tensor(np.zeros((bs, 1024)), "bfloat16")
            api.fp8_fp4_mega_moe(y, *weights(), b, activation_clamp=10)
            assert np.all(y.data == 3)
        assert [entry[1] for entry in backend.events if isinstance(entry, tuple) and entry[0] == "prepare"] == [17, 5]
        assert [launch[-1] for launch in native_launches(backend)] == [1, 2, 3]
    finally:
        b.destroy()


@pytest.mark.parametrize("shared_dtype", ["uint8", "float8_e4m3fn"])
def test_mte_roce_checks_all_shared_tensors_collectively_before_launch(backend, monkeypatch, shared_dtype):
    from deep_ep._store import StoreGroup

    group = StoreGroup(None, 0, 32, prefix="roce-shared-inputs")
    remote_error = None
    remote_precision_mismatch = False

    def gather(value, stage):
        peers = [value] * group.size()
        if stage == "check:window inputs":
            if remote_error:
                peers[-1] = (remote_error, None)
            elif remote_precision_mismatch:
                peers[-1] = (None, (*value[1][:-1], not value[1][-1]))
        return peers

    group.all_gather = gather
    ext = backend.ext
    monkeypatch.setattr(ext, "mte_roce_mx_enabled", True, raising=False)
    monkeypatch.setattr(ext, "roce_plan_bytes", lambda *args, **kwargs: 4096, raising=False)
    monkeypatch.setattr(ext, "roce_buffer_size", ext.buffer_size, raising=False)

    def roce_plan(example, intermediate, k, experts, ep, width, shared, ffts, address, capacity, **kwargs):
        return ext.Plan(example, intermediate, k, experts, ep, width, True, shared, 16, ffts,
                        address, capacity, backend="mte_roce", **kwargs)

    monkeypatch.setattr(ext, "RocePlan", roce_plan, raising=False)
    b = api.SymmBuffer(group, 64, 17, 2, 1024, 256, num_shared_experts=1, ranks_per_node=16, backend="mte_roce")

    def raw_pair(n, k):
        packing = 2 if shared_dtype == "uint8" else 1
        return (Tensor(np.zeros((1, n, k // packing)), shared_dtype),
                Tensor(np.full((1, n, k // 64, 2), 127), "float8_e8m0fnu"))

    first, second = api.transform_weights_for_mega_moe(
        raw_pair(512, 1024), raw_pair(1024, 256), shared=True, backend="mte_roce",
    )
    good = (first[0], second[0], first[1], second[1])
    routed = weights()
    y = Tensor(np.zeros((17, 1024)), "bfloat16")

    def invoke(tensors):
        api.fp8_fp4_mega_moe(
            y, *routed, b,
            shared_l1_weights=type(first)((tensors[0], tensors[2])),
            shared_l2_weights=(tensors[1], tensors[3]),
        )

    try:
        assert b.num_forward_windows(17) == 1
        b.topk_idx.data[:] = [0, 1]
        b.topk_weights.data[:] = [0.4, 0.6]
        fields = ("shared_l1_weights", "shared_l2_weights", "shared_l1_scales", "shared_l2_scales")
        for index, name in enumerate(fields):
            tensor = good[index]
            for bad in (tensor[..., :-1], Tensor(tensor.data, "int8"),
                        Tensor(tensor.data, tensor.dtype, "npu:1"), tensor[..., ::-1]):
                incorrect = list(good)
                incorrect[index] = bad
                with pytest.raises(ValueError, match=f"window inputs.*rank=0: {name}"):
                    invoke(incorrect)
        with pytest.raises(ValueError, match="window inputs.*rank=0.*all four shared"):
            invoke((*good[:3], None))
        remote_error = "invalid shared scale layout on peer"
        with pytest.raises(ValueError, match="window inputs.*rank=31: invalid shared scale layout on peer"):
            invoke(good)
        remote_error = None
        remote_precision_mismatch = True
        with pytest.raises(ValueError, match="window inputs.*ranks disagree.*rank=31"):
            invoke(good)
        assert not native_launches(backend)
        assert b._plan is None and b._generation == 0 and b._failure is None

        remote_precision_mismatch = False
        invoke(good)
        assert native_launches(backend)[0][9:13] == good
        assert np.all(y.data == 3)
    finally:
        b.destroy()


def test_urma_gelu_uses_single_planar_projection_within_existing_limits(backend, monkeypatch):
    from deep_ep_experimental.megamoe import backend as selector

    monkeypatch.setattr(selector, "select_backend", lambda group: "urma")
    b = api.SymmBuffer(backend.group_type(16), 32, 2048, 2, 256, 256, activation="gelu")
    raw = (
        (Tensor(np.zeros((2, 256, 128)), "uint8"), Tensor(np.full((2, 256, 4, 2), 127), "float8_e8m0fnu")),
        (Tensor(np.zeros((2, 256, 128)), "uint8"), Tensor(np.full((2, 256, 4, 2), 127), "float8_e8m0fnu")),
    )
    prepared = api.transform_weights_for_mega_moe(*raw, activation="gelu")
    try:
        b.topk_idx.data[:] = [0, 1]
        b.topk_weights.data[:] = [0.4, 0.6]
        y = Tensor(np.zeros((17, 256)), "bfloat16")
        api.fp8_fp4_mega_moe(y, *prepared, b, activation="gelu")
        assert b._plan.backend == "urma" and b._plan.topk_weights_type == 1
    finally:
        b.destroy()


@pytest.mark.parametrize("world", [16, 32])
@pytest.mark.parametrize("activation", ["gelu", "swiglu"])
def test_urma_w8a8_uses_timer_only_on_first_window(backend, monkeypatch, world, activation):
    from deep_ep_experimental.megamoe import backend as selector

    monkeypatch.setattr(selector, "select_backend", lambda group: "urma")
    monkeypatch.setattr(api, "_host_boot_id", lambda: "ab" * 16)
    group = backend.group_type(world)
    model = dict(activation=activation, mma_type="fp8xfp8")
    local = 800 // world
    with pytest.raises(ValueError, match="install a MegaMoE wheel with URMA W8A8 support"):
        api.SymmBuffer(group, 800, 257, 25, 256, 256, **model)

    def window_bytes(size, experts, tokens, topk, hidden, intermediate, mma_type):
        assert (size, experts, topk, hidden, intermediate, mma_type) == (world, local, 25, 256, 256, "fp8xfp8")
        return 2 << 20, tokens << 20

    monkeypatch.setattr(backend.ext, "urma_w8a8_enabled", True, raising=False)
    monkeypatch.setattr(backend.ext, "urma_window_bytes", window_bytes)
    if activation == "swiglu":
        with pytest.raises(ValueError, match="windowed URMA fp8xfp8 SwiGLU support"):
            api.SymmBuffer(group, 800, 257, 25, 256, 256, **model)
        monkeypatch.setattr(backend.ext, "urma_w8a8_swiglu_enabled", True, raising=False)
    b = api.SymmBuffer(group, 800, 257, 25, 256, 256, **model)
    try:
        with monkeypatch.context() as uninstrumented_build:
            uninstrumented_build.setattr(backend.ext, "timer_enabled", False)
            assert not b.timer_enabled
            with pytest.raises(RuntimeError, match="DEEPEP_MEGAMOE_TIMER=ON"):
                b.allocate_timer()
        assert b.timer_enabled
        timer = b.allocate_timer()
        assert not timer.data.any()

        raw = tuple(
            (Tensor(np.zeros((local, columns, 256)), "float8_e4m3fn"),
             Tensor(np.full((local, columns, 4, 2), 127), "float8_e8m0fnu"))
            for columns in (512 if activation == "swiglu" else 256, 256)
        )
        weights = api.transform_weights_for_mega_moe(*raw, activation=activation, backend="urma")
        b.topk_idx.data[:] = np.arange(25)
        b.topk_weights.data.fill(1.0 / 25)
        b.x.data[:] = (np.arange(257) % 17)[:, None]
        y = Tensor(np.full((257, 256), -1), "bfloat16")
        stats = Tensor(np.full(local, 7), "int32")
        if activation == "swiglu":
            with pytest.raises(ValueError, match="Prepare URMA FP8 SwiGLU GMM1"):
                api.fp8_fp4_mega_moe(y, *raw, b, activation=activation)
        assert api.fp8_fp4_mega_moe(
            y, *weights, b, activation=activation, cumulative_local_expert_recv_stats=stats, timer=timer,
        ) is None
        assert b._plan.windowed and b._plan.mma_type == "fp8xfp8"
        assert b._plan.topk_weights_type == int(activation == "gelu")
        launches = native_launches(backend)
        assert [launch[0].shape[0] for launch in launches] == [256, 1]
        assert [launch[8] for launch in launches] == [timer, None]
        assert backend.events.count(("validate_timer", timer, "urma")) == 1
        assert timer.data[-1] == launches[0][-1]
        assert all(launch[4] is weights[0][0] and launch[5] is weights[1][0] for launch in launches)
        assert all(np.shares_memory(launch[-2].data, y.data) for launch in launches)
        np.testing.assert_array_equal(y.data, b.x.data)
        np.testing.assert_array_equal(stats.data, np.where(np.arange(local) < 25, 264, 7))
    finally:
        b.destroy()


def test_urma_unsafe_target_limits_remain_explicit(backend, monkeypatch):
    from deep_ep_experimental.megamoe import backend as selector

    monkeypatch.setattr(selector, "select_backend", lambda group: "urma")
    for world, kwargs in (
        (8, dict(num_experts=32, num_max_tokens_per_rank=17, num_topk=9)),
        (16, dict(num_experts=32, num_max_tokens_per_rank=16385, num_topk=8, topk_weights_type=1)),
        (32, dict(num_experts=64, num_max_tokens_per_rank=17, num_topk=8, num_shared_experts=1)),
    ):
        with pytest.raises(ValueError):
            api.SymmBuffer(backend.group_type(world), hidden=256, intermediate_hidden=256, **kwargs)


def windowed_urma_buffer(backend, monkeypatch, capacity, topk, *, group=None, activation="gelu", window=None):
    from deep_ep_experimental.megamoe import backend as selector

    monkeypatch.setattr(selector, "select_backend", lambda group: "urma")
    monkeypatch.setattr(api, "_host_boot_id", lambda: "ab" * 16)

    def validate_routes(indices, weights, num_experts):
        check_routing(indices.data.tolist(), weights.data.tolist(), num_experts, indices.shape[1])

    monkeypatch.setattr(api, "check_window_routes", validate_routes)
    if group is None:
        group = backend.group_type(8)
    b = api.SymmBuffer(group, 64, capacity, topk, 256, 256, activation=activation, urma_window_tokens=window)
    rows = np.arange(capacity, dtype=np.int32)[:, None]
    slots = np.arange(topk, dtype=np.int32)[None, :]
    b.topk_idx.data[:] = (rows + 17 * slots) % b.num_experts
    b.topk_weights.data[:] = (1 + rows + slots) / 1024
    b.x.data[:] = rows % 251
    b.x_sf.data[:] = (rows + 101) % 251
    local = b.num_experts // group.size()
    columns = 512 if activation == "swiglu" else 256
    raw = (
        (Tensor(np.zeros((local, columns, 128)), "uint8"), Tensor(np.full((local, columns, 4, 2), 127), "float8_e8m0fnu")),
        (Tensor(np.zeros((local, 256, 128)), "uint8"), Tensor(np.full((local, 256, 4, 2), 126), "float8_e8m0fnu")),
    )
    return b, api.transform_weights_for_mega_moe(*raw, activation=activation)


def native_launches(backend):
    return [e[1] for e in backend.events if isinstance(e, tuple) and e[0] == "launch"]


@pytest.mark.parametrize("explicit", [False, True])
@pytest.mark.parametrize(
    "capacity,bs,topk,lengths",
    [
        (33, 1, 1, [1]),
        (512, 255, 9, [255]),
        (256, 256, 25, [256]),
        (1024, 257, 32, [256, 1]),
        (640, 529, 8, [256, 256, 17]),
    ],
)
def test_windowed_urma_launches_preserve_routes_and_results(
    backend, monkeypatch, explicit, capacity, bs, topk, lengths
):
    b, weights = windowed_urma_buffer(backend, monkeypatch, capacity, topk)
    local = b.num_experts // b.group.size()
    stats = Tensor(np.arange(local), "int32")
    expected_ids = b.topk_idx.data[:bs].copy()
    expected_weights = b.topk_weights.data[:bs].copy()
    expected_counts = np.array([(expected_ids == e).sum() for e in range(local)])
    timer = b.allocate_timer()
    native_capacity = min(capacity, 256)

    try:
        assert ("buffer_size", (8, local, native_capacity, topk, 256), "urma", True) in backend.events
        assert b.num_forward_windows(bs) == len(lengths)
        b.prepare(bs)
        plan = b._plan
        assert plan.windowed and plan.bs == plan.max_tokens == native_capacity and plan.topk_weights_type == 1

        if explicit:
            x = b.x[:bs], b.x_sf[:bs].reshape((bs, b.hidden // 64, 2))
            y, counts = api.mega_moe(x, b.topk_idx[:bs], b.topk_weights[:bs], *weights, b, timer=timer)
            np.testing.assert_array_equal(counts.data, expected_counts)
        else:
            y = Tensor(np.full((bs, b.hidden), -1), "bfloat16")
            assert api.fp8_fp4_mega_moe(
                y, *weights, b, activation="gelu", timer=timer, cumulative_local_expert_recv_stats=stats
            ) is None
            np.testing.assert_array_equal(stats.data, np.arange(local) + expected_counts)
            assert stats.streams == [b._runtime.stream]

        launches = native_launches(backend)
        assert [x[0].shape[0] for x in launches] == lengths
        assert [x[-1] for x in launches] == list(range(1, len(lengths) + 1))
        assert [x[8] for x in launches] == [timer, *([None] * (len(lengths) - 1))]
        assert backend.events.count(("validate_timer", timer, "urma")) == 1
        assert timer.data[-1] == 1
        np.testing.assert_array_equal(np.concatenate([x[0].data for x in launches]), b.x.data[:bs])
        np.testing.assert_array_equal(np.concatenate([x[1].data for x in launches]), b.x_sf.data[:bs])
        np.testing.assert_array_equal(np.concatenate([x[2].data for x in launches]), expected_ids)
        np.testing.assert_array_equal(np.concatenate([x[3].data for x in launches]), expected_weights)
        np.testing.assert_array_equal(y.data, np.broadcast_to(b.x.data[:bs, :1], y.shape))
        assert all(np.shares_memory(x[-2].data, y.data) for x in launches)

        b.prepare(1)
        b.prepare(capacity)
        assert b._plan is plan
        y1 = Tensor(np.full((1, b.hidden), -1), "bfloat16")
        api.fp8_fp4_mega_moe(y1, *weights, b, activation="gelu")
        assert native_launches(backend)[-1][-1] == len(lengths) + 1
        assert backend.events.count(("prepare", native_capacity)) == 1
    finally:
        b.destroy()


def test_windowed_urma_tail_validation_and_remote_error_allow_retry(backend, monkeypatch):
    from deep_ep._store import StoreGroup

    group = StoreGroup(None, 0, 8, prefix="window-inputs-test")
    remote_error = None
    mismatched_batch = False

    def gather(value, stage):
        peers = [value] * group.size()
        if stage == "check:window inputs":
            if remote_error is not None:
                peers[3] = (remote_error, value[1])
            if mismatched_batch:
                rows, *remaining = value[1]
                peers[3] = (None, (rows - 1, *remaining))
        return peers

    group.all_gather = gather
    b, weights = windowed_urma_buffer(backend, monkeypatch, 257, 32, group=group)
    y = Tensor(np.zeros((257, b.hidden)), "bfloat16")
    try:
        valid_id = b.topk_idx.data[-1, -1]
        b.topk_idx.data[-1, -1] = b.topk_idx.data[-1, 0]
        with pytest.raises(ValueError, match="window inputs.*rank=0.*token=256.*duplicate"):
            api.fp8_fp4_mega_moe(y, *weights, b, activation="gelu")
        b.topk_idx.data[-1, -1] = valid_id

        remote_error = "bad input shape on peer"
        with pytest.raises(ValueError, match="window inputs.*rank=3: bad input shape on peer"):
            api.fp8_fp4_mega_moe(y, *weights, b, activation="gelu")
        remote_error = None
        mismatched_batch = True
        with pytest.raises(ValueError, match="window inputs.*ranks disagree.*rank=3: \\(256,"):
            api.fp8_fp4_mega_moe(y, *weights, b, activation="gelu")
        assert b._plan is None and b._generation == 0 and b._failure is None
        assert not native_launches(backend)

        mismatched_batch = False
        api.fp8_fp4_mega_moe(y, *weights, b, activation="gelu")
        assert [x[-1] for x in native_launches(backend)] == [1, 2]
    finally:
        b.destroy()


@pytest.mark.parametrize("explicit_prepare", [False, True])
def test_windowed_urma_remote_plan_failure_discards_local_plan_for_retry(backend, monkeypatch, explicit_prepare):
    from deep_ep._store import StoreGroup

    group = StoreGroup(None, 0, 8, prefix="window-plan-test")
    remote_failure = True

    def gather(value, stage):
        peers = [value] * group.size()
        if stage == "check:window plan" and remote_failure:
            peers[1] = ("peer allocation failed", value[1])
        return peers

    group.all_gather = gather
    b, weights = windowed_urma_buffer(backend, monkeypatch, 257, 25, group=group)
    y = Tensor(np.zeros((257, b.hidden)), "bfloat16")
    try:
        with pytest.raises(ValueError, match="rank=1: peer allocation failed"):
            if explicit_prepare:
                b.prepare(257)
            else:
                api.fp8_fp4_mega_moe(y, *weights, b, activation="gelu")
        assert b._plan is None and b._plan_bs is None and b._generation == 0 and b._failure is None
        assert not native_launches(backend)

        remote_failure = False
        api.fp8_fp4_mega_moe(y, *weights, b, activation="gelu")
        assert [x[-1] for x in native_launches(backend)] == [1, 2]
        assert backend.events.count(("prepare", 256)) == 2
    finally:
        b.destroy()


def test_windowed_urma_timer_validation_precedes_launch(backend, monkeypatch):
    b, weights = windowed_urma_buffer(backend, monkeypatch, 257, 25)
    y = Tensor(np.zeros((257, b.hidden)), "bfloat16")
    try:
        wrong_timer = Tensor(np.zeros(64), "int64")
        with pytest.raises(ValueError, match="window inputs.*Invalid native timer"):
            api.fp8_fp4_mega_moe(y, *weights, b, activation="gelu", timer=wrong_timer)
        assert b._plan is None and b._generation == 0 and b._failure is None
        assert not native_launches(backend)
    finally:
        b.destroy()


def test_windowed_urma_failed_launch_poisoning_and_stats(backend, monkeypatch):
    b, weights = windowed_urma_buffer(backend, monkeypatch, 513, 25)
    y = Tensor(np.zeros((513, b.hidden)), "bfloat16")
    stats = Tensor(np.zeros(b.num_experts // b.group.size()), "int32")
    try:
        b.prepare(513)
        forward = b._plan.forward

        def fail_second_window(*args):
            if args[-2] == 2:
                raise RuntimeError("device launch failed")
            return forward(*args)

        monkeypatch.setattr(b._plan, "forward", fail_second_window)
        with pytest.raises(RuntimeError, match="device launch failed"):
            api.fp8_fp4_mega_moe(y, *weights, b, activation="gelu", cumulative_local_expert_recv_stats=stats)
        assert b._generation == 2
        assert [x[-1] for x in native_launches(backend)] == [1]
        assert stats._version == 0
        with pytest.raises(RuntimeError, match="buffer failed.*restart"):
            api.fp8_fp4_mega_moe(y, *weights, b, activation="gelu")
        assert [x[-1] for x in native_launches(backend)] == [1]
    finally:
        b.destroy()


@pytest.mark.parametrize(
    "topk,options,match",
    [
        (33, {}, "num_topk"),
        (32, {"ranks_per_node": 4}, "requires 8 ranks per host"),
    ],
)
def test_windowed_urma_invalid_configuration_rejected(backend, monkeypatch, topk, options, match):
    from deep_ep_experimental.megamoe import backend as selector

    monkeypatch.setattr(selector, "select_backend", lambda group: "urma")
    monkeypatch.setattr(api, "_host_boot_id", lambda: "ab" * 16)
    with pytest.raises(ValueError, match=match):
        api.SymmBuffer(backend.group_type(8), 64, 257, topk, 256, 256, activation="gelu", **options)
    assert sum(event[0] == "buffer_size" for event in backend.events if isinstance(event, tuple)) == 1


def test_windowed_urma_gelu_supports_finalize_weighting(backend, monkeypatch):
    from deep_ep_experimental.megamoe import backend as selector

    monkeypatch.setattr(selector, "select_backend", lambda group: "urma")
    monkeypatch.setattr(api, "_host_boot_id", lambda: "ab" * 16)
    b = api.SymmBuffer(
        backend.group_type(16), 32, 2048, 2, 256, 256, activation="gelu", topk_weights_type=0
    )
    raw = (
        (Tensor(np.zeros((2, 256, 128)), "uint8"), Tensor(np.full((2, 256, 4, 2), 127), "float8_e8m0fnu")),
        (Tensor(np.zeros((2, 256, 128)), "uint8"), Tensor(np.full((2, 256, 4, 2), 127), "float8_e8m0fnu")),
    )
    prepared = api.transform_weights_for_mega_moe(*raw, activation="gelu")
    try:
        b.topk_idx.data[:] = [0, 1]
        b.topk_weights.data[:] = [0.4, 0.6]
        y = Tensor(np.zeros((17, 256)), "bfloat16")
        api.fp8_fp4_mega_moe(y, *prepared, b, activation="gelu")
        assert b._plan.windowed and b._plan.topk_weights_type == 0
    finally:
        b.destroy()


def test_windowed_urma_rejects_different_hosts_even_with_ranks_per_node_8(backend, monkeypatch):
    from deep_ep_experimental.megamoe import backend as selector

    monkeypatch.setattr(selector, "select_backend", lambda group: "urma")
    monkeypatch.setattr(api, "_host_boot_id", lambda: "ab" * 16)

    def gather(output, value, **kwargs):
        error, config = value
        output[:] = [value] * len(output)
        output[1] = (error, (*config[:-1], "cd" * 16))

    monkeypatch.setattr(backend.dist, "all_gather_object", gather)
    with pytest.raises(ValueError, match="configuration.*ranks disagree"):
        api.SymmBuffer(backend.group_type(8), 64, 257, 32, 256, 256, activation="gelu", ranks_per_node=8)
    assert sum(event[0] == "buffer_size" for event in backend.events if isinstance(event, tuple)) == 1


@pytest.mark.parametrize(("world", "activation"), [(8, "gelu"), (16, "gelu"), (32, "gelu"), (16, "swiglu"), (32, "swiglu")])
@pytest.mark.parametrize("k", [25, 32])
def test_windowed_urma_target_prepares_256_rows_for_39280_token_capacity(backend, monkeypatch, world, k, activation):
    from deep_ep_experimental.megamoe import backend as selector

    monkeypatch.setattr(selector, "select_backend", lambda group: "urma")
    monkeypatch.setattr(api, "_host_boot_id", lambda: "ab" * 16)
    torch = sys.modules["torch"]

    def shape_only(shape, dtype, device):
        return Tensor(np.broadcast_to(np.zeros((), dtype=DTYPES[dtype]), shape), dtype, device)

    monkeypatch.setattr(torch, "empty", shape_only)
    monkeypatch.setattr(torch, "empty_like", lambda t, dtype: shape_only(t.shape, dtype, t.device))
    b = api.SymmBuffer(backend.group_type(world), 800, 39280, k, 4096, 1536, activation=activation)
    try:
        assert b.x.shape == (39280, 4096) and b.topk_idx.shape == (39280, k)
        assert b.num_forward_windows(39280) == 154
        assert b.ranks_per_node == min(world, 16)
        assert ("buffer_size", (world, 800 // world, 256, k, 4096), "urma", True) in backend.events
        b.prepare(39280)
        assert b._plan.windowed and b._plan.bs == b._plan.max_tokens == 256
        assert b._plan.local_experts == 800 // world and b._plan.topk_weights_type == int(activation == "gelu")
        assert not native_launches(backend)
    finally:
        b.destroy()


@pytest.mark.parametrize("world", [16, 32])
def test_windowed_urma_swiglu_uses_fixed_windows_and_first_window_timer(backend, monkeypatch, world):
    b, weights = windowed_urma_buffer(
        backend, monkeypatch, 129, 25, group=backend.group_type(world), activation="swiglu", window=64,
    )
    try:
        timer = b.allocate_timer()
        out = Tensor(np.zeros((129, 256)), "bfloat16")
        api.fp8_fp4_mega_moe(out, *weights, b, activation="swiglu", timer=timer)
        launches = native_launches(backend)
        assert b._plan.windowed and b._plan.topk_weights_type == 0
        assert [x[0].shape[0] for x in launches] == [64, 64, 1]
        assert [x[8] for x in launches] == [timer, None, None]
    finally:
        b.destroy()


@pytest.mark.parametrize("local_limit,peer_limit,expected", [(130, 256, 128), (270, 64, 64), (1, 256, None)])
def test_collective_window_memory_budget(backend, monkeypatch, local_limit, peer_limit, expected):
    from deep_ep._store import StoreGroup
    from deep_ep_experimental.megamoe import backend as selector

    monkeypatch.setattr(selector, "select_backend", lambda group: "urma")
    monkeypatch.setattr(api.npu_api(), "mem_get_info", lambda _: (local_limit << 20, 300 << 20))
    monkeypatch.setattr(backend.ext, "urma_window_bytes", lambda w, e, t, k, h, i: (t << 20, 0))
    group = StoreGroup(None, 0, 32, prefix="budget")
    group.all_gather = lambda value, stage: [value] * 31 + [
        (None, peer_limit) if stage == "window HBM budget" else value
    ]
    if expected is None:
        with pytest.raises(ValueError, match="rank=0.*insufficient free HBM"):
            api.SymmBuffer(group, 64, 257, 25, 256, 256, activation="gelu")
        return
    b = api.SymmBuffer(group, 64, 257, 25, 256, 256, activation="gelu")
    try:
        assert b.window_tokens == expected
        assert b.num_forward_windows(257) == 1 + 256 // expected
    finally:
        b.destroy()


@pytest.mark.parametrize("margin,window", [(-1, 64), (0, 128)])
def test_urma_w4a8_swiglu_reserves_both_weight_and_scale_layouts(backend, monkeypatch, margin, window):
    from deep_ep_experimental.megamoe import backend as selector

    tokens, topk, h, i, local_experts = 257, 25, 256, 256, 2
    projection = local_experts * h * i
    resident = tokens * (3 * h + h // 32 + 20 * topk) + 5 * (projection // 2 + projection // 32)
    budget = resident + (130 << 20) + margin
    monkeypatch.setattr(selector, "select_backend", lambda group: "urma")
    monkeypatch.setattr(api.npu_api(), "mem_get_info", lambda _: (budget, 300 << 20))
    b = api.SymmBuffer(backend.group_type(32), 32 * local_experts, tokens, topk, h, i, activation="swiglu")
    try:
        assert b.window_tokens == window
    finally:
        b.destroy()


@pytest.mark.parametrize("bf16", [True, False])
def test_mte_full_heap_overflow_launches_exact_window_and_tail_plans(backend, monkeypatch, bf16):
    monkeypatch.setattr(backend.ext, "buffer_size", lambda world, e, t, k, h, **kwargs: (2 << 30) if t > 256 else 2 << 20)
    b = api.SymmBuffer(backend.group_type(8), 32, 513, 25, 1024, 256, activation="gelu", ranks_per_node=8)
    local = 4
    raw = (
        (Tensor(np.zeros((local, 256, 512)), "uint8"), Tensor(np.full((local, 256, 16, 2), 127), "float8_e8m0fnu")),
        (Tensor(np.zeros((local, 1024, 128)), "uint8"), Tensor(np.full((local, 1024, 4, 2), 127), "float8_e8m0fnu")),
    )
    weights = api.transform_weights_for_mega_moe(*raw, activation="gelu")
    b.topk_idx.data[:] = np.arange(25)
    b.topk_weights.data[:] = 1
    try:
        y = Tensor(np.zeros((513, 1024)), "bfloat16")
        b.prepare(513)
        assert b._plan.bs == 256 and b._tail_plan.bs == 1 and b.workspace_bytes == 8192
        assert not b._plan.windowed and not b._tail_plan.windowed
        api.fp8_fp4_mega_moe(y, *weights, b, activation="gelu", x=Tensor(y.data.copy(), "bfloat16") if bf16 else None)
        assert [x[0].shape[0] for x in native_launches(backend)] == [256, 256, 1]
        assert all((x[1] is None) == bf16 for x in native_launches(backend))
    finally:
        b.destroy()


def test_shared_expert_activation_policy_matches_gmm2_tile_counts(tmp_path):
    compiler = next((path for name in ("c++", "g++", "clang++") if (path := shutil.which(name))), None)
    if compiler is None:
        pytest.skip("A host C++17 compiler is required for the native policy regression")
    csrc = Path(__file__).resolve().parents[1] / "csrc"
    source = tmp_path / "shared_policy.cpp"
    executable = tmp_path / ("shared_policy.exe" if sys.platform == "win32" else "shared_policy")
    source.write_text(
        r'''#include "kernels/mega_moe_common/common/mega_moe_shared_policy.hpp"
#include <cstdint>
#include <iostream>

uint32_t Tiles(uint32_t hidden, uint32_t tileN)
{
    return (hidden + tileN - 1U) / tileN;
}

int main()
{
    const uint32_t batches[] = {1U, 17U, 256U, 257U};
    const uint32_t hiddenSizes[] = {1024U, 4096U};
    for (uint32_t bs : batches) {
        for (uint32_t hidden : hiddenSizes) {
            const auto geluProducer = MegaMoeImpl::SelectSharedExpertPolicyForActivation(true, true, 1U, bs);
            const auto geluConsumer = MegaMoeImpl::SelectSharedExpertPolicyForActivation(true, true, 1U, bs);
            if (geluProducer.gmm2TileN != 256U || geluConsumer.gmm2TileN != geluProducer.gmm2TileN ||
                Tiles(hidden, geluProducer.gmm2TileN) != Tiles(hidden, geluConsumer.gmm2TileN)) {
                std::cerr << "GELU producer/consumer mismatch at BS=" << bs << " H=" << hidden << '\n';
                return 1;
            }
            const auto swigluProducer = MegaMoeImpl::SelectSharedExpertPolicyForActivation(true, false, 1U, bs);
            const auto swigluConsumer = MegaMoeImpl::SelectSharedExpertPolicyForActivation(true, false, 1U, bs);
            const uint32_t expectedN = bs <= 256U ? 224U : 256U;
            if (swigluProducer.gmm2TileN != expectedN || swigluConsumer.gmm2TileN != expectedN ||
                Tiles(hidden, swigluProducer.gmm2TileN) != Tiles(hidden, swigluConsumer.gmm2TileN)) {
                std::cerr << "SwiGLU policy regression at BS=" << bs << " H=" << hidden << '\n';
                return 2;
            }
        }
    }
    return 0;
}
''',
        encoding="utf-8",
    )
    subprocess.run(
        [compiler, "-std=c++17", "-I", str(csrc), str(source), "-o", str(executable)],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run([str(executable)], check=True, capture_output=True, text=True)
    for relative in (
        "kernels/mega_moe_common/common/mega_moe_gmm_common.hpp",
        "kernels/mega_moe/stage/mega_moe_unpermute.hpp",
    ):
        assert "SelectSharedExpertPolicyForActivation(" in (csrc / relative).read_text(encoding="utf-8")


def test_mte_wave_window_mapping_and_target_capacity(tmp_path):
    compiler = next((path for name in ("c++", "g++", "clang++") if (path := shutil.which(name))), None)
    if compiler is None:
        pytest.skip("A host C++17 compiler is required for the native wave-window regression")
    csrc = Path(__file__).resolve().parents[1] / "csrc"
    source = tmp_path / "wave_window.cpp"
    executable = tmp_path / ("wave_window.exe" if sys.platform == "win32" else "wave_window")
    source.write_text(
        r'''#include "kernels/mega_moe_common/common/mega_moe_wave_window.hpp"
#include <cstdint>
#include <iostream>
#include <vector>

struct Range {
    uint64_t storageBegin;
    uint64_t storageEnd;
};

bool Overlaps(Range a, Range b)
{
    return a.storageBegin < b.storageEnd && b.storageBegin < a.storageEnd;
}

int main()
{
    if (MegaMoeImpl::WaveWindowSlotCount(true, true) != 3U ||
        MegaMoeImpl::WaveWindowSlotCount(true, false) != 2U ||
        MegaMoeImpl::WaveWindowSlotCount(false, true) != 1U) {
        return 6;
    }
    const uint32_t lengths[] = {1U, 255U, 256U, 257U, 1024U, 17U, 1024U, 31U, 256U, 511U};
    const uint64_t maxRows = 100000U;
    for (uint32_t slots : {2U, 3U}) {
        const uint32_t groups = 4U;
        const uint64_t storageRows = MegaMoeImpl::WaveWindowRowCapacity(maxRows, groups, slots);
        if (storageRows != slots * 1024U) return 1;
        std::vector<Range> ranges;
        uint64_t logical = 0U;
        for (uint32_t wave = 0U; wave < 8U; ++wave) {
            const uint32_t length = lengths[wave];
            const auto mapping = MegaMoeImpl::MakeWaveRowMapping(maxRows, storageRows, groups, slots, wave, logical);
            const uint64_t begin = MegaMoeImpl::MapWaveRow(mapping, logical);
            const uint64_t end = MegaMoeImpl::MapWaveRow(mapping, logical + length - 1U) + 1U;
            if (begin != (wave % slots) * 1024U || end > storageRows || end - begin != length) {
                std::cerr << "invalid wave mapping at wave=" << wave << " slots=" << slots << '\n';
                return 2;
            }
            ranges.push_back({begin, end});
            logical += length;
        }
        for (uint32_t wave = 1U; wave < ranges.size(); ++wave) {
            const uint32_t previous = slots == 2U ? wave : wave - 1U;
            for (uint32_t other = wave; other < ranges.size() && other <= wave + 1U; ++other) {
                if (previous != other && Overlaps(ranges[previous], ranges[other])) {
                    std::cerr << "in-flight overlap at wave=" << wave << " slots=" << slots << '\n';
                    return 3;
                }
            }
            if (slots == 3U && wave > 0U && wave + 1U < ranges.size() && Overlaps(ranges[wave], ranges[wave + 1U])) {
                return 3;
            }
        }
    }
    const auto identity = MegaMoeImpl::MakeWaveRowMapping(1000U, 1000U, 4U, 3U, 9U, 777U);
    if (identity.logicalStart != 0U || identity.storageStart != 0U || MegaMoeImpl::MapWaveRow(identity, 777U) != 777U) {
        return 4;
    }
    constexpr uint64_t bytesPerRow = 4096U + 256U + 1536U + 48U + 32U + 3072U + 8192U;
    for (uint32_t ep : {8U, 16U, 32U}) {
        const uint64_t logicalRows = static_cast<uint64_t>(39280U) * ep * 25U;
        const uint64_t fullBytes = logicalRows * bytesPerRow;
        const uint64_t a8w4Rows = MegaMoeImpl::WaveWindowRowCapacity(logicalRows, 16U, 3U);
        const uint64_t a8w8Rows = MegaMoeImpl::WaveWindowRowCapacity(logicalRows, 16U, 2U);
        if (fullBytes < (100ULL << 30) || a8w4Rows != 12288U || a8w8Rows != 8192U ||
            a8w4Rows * bytesPerRow >= (256ULL << 20) || a8w8Rows * bytesPerRow >= (256ULL << 20)) {
            std::cerr << "target capacity regression at EP=" << ep << '\n';
            return 5;
        }
    }
    return 0;
}
''',
        encoding="utf-8",
    )
    subprocess.run(
        [compiler, "-std=c++17", "-I", str(csrc), str(source), "-o", str(executable)],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run([str(executable)], check=True, capture_output=True, text=True)
    expected = {
        "kernels/mega_moe_common/common/mega_moe_workspace.hpp": "MteWorkspaceRowCapacity",
        "kernels/mega_moe/stage/mega_moe_token_dispatch.hpp": "MapWaveRow",
        "kernels/mega_moe_common/stage/mega_moe_gmm1_activation.hpp": "ExpertStorageRow",
        "kernels/mega_moe/stage/mega_moe_gmm2_combine.hpp": "ExpertStorageRow",
        "kernels/mega_moe/mega_moe_wave_a8w4.hpp": "SyncAll<false>",
        "kernels/mega_moe/mega_moe_wave_a8w8.hpp": "SyncAll<false>",
    }
    for relative, marker in expected.items():
        assert marker in (csrc / relative).read_text(encoding="utf-8")
