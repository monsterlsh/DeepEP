# Copyright (c) 2026, Lu Lu
# Modified by SimpleBright_Man 2026

"""Thin inference-only API; all algorithm selection remains in the native layer."""

from __future__ import annotations

from functools import lru_cache
import importlib
import threading
from typing import TYPE_CHECKING

from deep_ep._runtime import _MEGAMOE_PROFILES, _host_boot_id, integer, warn_unclosed
from ._runtime import _MegaRuntime
from deep_ep._native import npu_api, require_native
from deep_ep._store import StoreGroup
from .validation import check_routing, check_window_routes, collective_check, gather_control
from .numerics import normalize_clamp

if TYPE_CHECKING:
    from torch import Tensor
    from torch.distributed import ProcessGroup

    WeightPair = tuple[Tensor, Tensor]


_TIMER_REQUIREMENTS = "Timer supports W4A8, MTE A4W4 and windowed URMA W8A8 in builds with DEEPEP_MEGAMOE_TIMER=ON"


@lru_cache(maxsize=1)
def _extension():
    core = require_native("MegaMoE")
    if not hasattr(core, "create_megamoe_runtime"):
        raise RuntimeError("Install a wheel built with DEEPEP_BUILD_MEGAMOE=ON")
    try:
        extension = importlib.import_module(__package__ + "._C")
    except ImportError as error:
        raise ImportError("Cannot load MegaMoE Beta: check matching Torch, torch_npu, CANN and SHMEM") from error
    if getattr(extension, "api_protocol", None) != 13:
        raise RuntimeError("MegaMoE Python/native versions differ; rebuild and reinstall the complete Beta wheel")
    return extension


def _pair(value, name):
    if not isinstance(value, tuple) or len(value) != 2:
        raise TypeError(f"{name} must be (quantized_data, E8M0_scales)")
    return value


def _urma_window_capacity(
    group, ext, world, local_experts, tokens, topk, hidden, intermediate, mma_type, *, activation, window_tokens,
):
    error, selected = None, 0
    try:
        free, _ = npu_api().mem_get_info(npu_api().current_device())
        # Public FP8 input/scales, routing IDs/weights and BF16 output.
        resident = tokens * (3 * hidden + hidden // 32 + 8 * topk)
        # Budget both gate/up layouts and their scales. FP4 preparation retains
        # both, and FP8 preparation can hold the raw W1 alongside the interleaved one.
        projections = 5 if activation == "swiglu" else 2
        elements = projections * local_experts * hidden * intermediate
        resident += (elements // 2 if mma_type == "fp8xfp4" else elements) + elements // 32
        # sort() produces int32 values and int64 indices during route preflight.
        resident += tokens * topk * 12
        sizes = (window_tokens,) if window_tokens is not None else (256, 128, 64)
        for candidate in dict.fromkeys(min(tokens, n) for n in sizes):
            args = (world, local_experts, candidate, topk, hidden, intermediate)
            if mma_type == "fp8xfp8":
                args += (mma_type,)
            # Early weighting is a conservative bound for Finalize weighting, which
            # omits the route weights in Dispatch and does not allocate the GELU lookup.
            heap, scratch = ext.urma_window_bytes(*args)
            if resident + heap + scratch <= free and heap <= _MEGAMOE_PROFILES["urma"].maximum_capacity:
                selected = candidate
                break
        if not selected:
            window = f"requested W{min(tokens, window_tokens)}" if window_tokens is not None else "smallest window"
            raise ValueError(f"insufficient free HBM for the {window} ({free} bytes available)")
    except Exception as exc:
        error = str(exc) or type(exc).__name__
    peers = gather_control(group, "window HBM budget", (error, selected))
    failures = [f"rank={rank}: {message}" for rank, (message, _) in enumerate(peers) if message]
    if failures:
        raise ValueError("MegaMoE window HBM budget failed: " + "; ".join(failures))
    return min(capacity for _, capacity in peers)


# A build-time performance choice. Smaller candidates preserve the MTE SHMEM
# layout and permit individual ranks to accommodate their available memory.
_MTE_MAX_WINDOW_TOKENS = 2048


def _ubx_mte_capacity(group, ext, backend, world, local_experts, tokens, topk, hidden,
                      intermediate, activation, mma_type, weight_mode, shared):
    """Select one physical window that fits the kernel tiling and HBM on every rank."""
    eligible, error = [], None
    try:
        roce = backend == "mte_roce"
        plan_size = ext.roce_plan_bytes if roce else ext.mte_plan_bytes
        plan_options = dict(activation=activation, mma_type=mma_type, topk_weights_type=weight_mode, shared_experts=shared)
        heap_size = ext.roce_buffer_size if roce else ext.buffer_size
        heap_options = dict(topk_weights_type=weight_mode, mma_type=mma_type)
        if not roce:
            heap_options["backend"] = backend
        free = npu_api().mem_get_info(npu_api().current_device())[0]
        profile = _MEGAMOE_PROFILES[backend]
        packed_x = mma_type == "fp4xfp4"
        packed_w = mma_type != "fp8xfp8"
        resident = tokens * (hidden // (2 if packed_x else 1) + hidden // 32 + hidden * 2 + topk * 20)
        # W4A8 SwiGLU retains planar and interleaved copies of its gate/up projections.
        w1_matrices = 1 if activation == "gelu" else (4 if mma_type == "fp8xfp4" else 2)
        weight_elements = local_experts * hidden * intermediate * (w1_matrices + 1)
        resident += weight_elements // (2 if packed_w else 1) + weight_elements // 32
        resident += shared * 3 * hidden * intermediate * 33 // 32
        rejected = "insufficient free HBM or symmetric heap capacity"
        for width in dict.fromkeys(min(tokens, n) for n in (256, 128, 64, 32, 16, 8, 4, 2, 1)):
            shape = (world, local_experts, width, topk, hidden)
            try:
                heap = heap_size(*shape, **heap_options)
                if heap > profile.maximum_capacity:
                    continue
                plan_rows = [width]
                if tokens > width and width > 1:
                    # Both plans retain scratch and share the same fixed-capacity symmetric arena.
                    plan_rows.append(tokens % width or width - 1)
                plan_bytes = sum(plan_size(world, local_experts, rows, width, topk, hidden, intermediate,
                                           **plan_options) for rows in plan_rows)
            except (RuntimeError, ValueError) as exc:
                # A smaller window may fit the route histogram or the per-core UB buffers.
                rejected = str(exc)
                continue
            if resident + heap + plan_bytes <= free:
                eligible.append(width)
        if not eligible:
            raise ValueError(f"no native window fits {backend} ({free} bytes available): {rejected}")
    except Exception as exc:
        error = str(exc) or type(exc).__name__
    peers = gather_control(group, "MTE window capacity", (error, tuple(eligible)))
    failures = [f"rank={r}: {message}" for r, (message, _) in enumerate(peers) if message]
    if failures:
        raise ValueError("MegaMoE window selection failed: " + "; ".join(failures))
    common = set(eligible).intersection(*(widths for _, widths in peers))
    if not common:
        raise ValueError(f"No common {backend} window capacity across ranks: {peers}")
    return max(common)


def _mte_window_capacity(group, ext, world, local_experts, tokens, topk, hidden, intermediate, mma_type):
    """Agree before inputs, plans and SHMEM are allocated on a public GELU S0 buffer."""
    eligible, error = [], None
    try:
        limit = _MEGAMOE_PROFILES["mte"].maximum_capacity
        query = getattr(ext, "mte_plan_bytes", None)
        # Protocol 13 wheels prior to the additive sizing query retain their original W256.
        candidates = (2048, 1024, 512, 256) if query is not None else (256,)
        # Buffered FP8 input/scale, BF16 output, route IDs and weights. The full-batch
        # route preflight sorts int32 keys and materializes their int64 permutation.
        logical = tokens * (3 * hidden + hidden // 32 + (8 + 12) * topk)
        # Both local single-projection GELU weights have shape H*I per expert.
        # Each matrix also holds one E8M0 scale per group of 32 weights.
        elements = 2 * local_experts * hidden * intermediate
        weights = (elements // 2 if mma_type == "fp8xfp4" else elements) + elements // 32
        # A windowed forward can hold the first counts, previous window counts and
        # newly returned counts simultaneously during the assignment.
        counts = 3 * local_experts * 4
        free = npu_api().mem_get_info(npu_api().current_device())[0] if query is not None else None
        for width in dict.fromkeys(min(tokens, n) for n in candidates if n <= _MTE_MAX_WINDOW_TOKENS):
            heap = ext.buffer_size(world, local_experts, width, topk, hidden, backend="mte",
                                   topk_weights_type=1, mma_type=mma_type, windowed=False)
            if heap > limit:
                continue
            # At W256 the allocator remains authoritative. We cannot tell if the caller's
            # weights/output were already charged against free, and allocator-cached HBM
            # can be reusable. This heuristic must only opt in a larger window.
            if query is not None and width > min(tokens, 256):
                def scratch(rows):
                    return query(world, local_experts, rows, width, topk, hidden, intermediate, "gelu", mma_type)
                plan = scratch(width)
                if tokens > width and width > 1:
                    # Single public buffers also support calls shorter than their configured
                    # logical maximum. A divisible maximum can still have a tail on a later call.
                    tail = tokens % width or min(tokens - width, width - 1)
                    plan += scratch(tail)
                required = logical + weights + counts + heap + plan
                if required > free:
                    continue
            eligible.append(width)
        if not eligible:
            raise ValueError(f"no candidate fits the MTE symmetric heap limit ({limit} bytes) and available HBM")
    except Exception as exc:
        error = str(exc) or type(exc).__name__
    # Full and short tail plans may have different footprints between candidate widths.
    # Exchanging all admissible widths also accommodates ranks on an older protocol-13 build.
    peers = gather_control(group, "MTE window HBM budget", (error, tuple(eligible)))
    failures = [f"rank={rank}: {message}" for rank, (message, _) in enumerate(peers) if message]
    if failures:
        raise ValueError("MegaMoE MTE window HBM budget failed: " + "; ".join(failures))
    common = set(eligible).intersection(*(widths for _, widths in peers))
    if not common:
        raise ValueError(f"MegaMoE ranks have no common MTE window capacity: {peers}")
    return max(common)



def _check_fp4_weight_format(weight, name):
    import torch_npu

    actual = torch_npu.get_npu_format(weight)
    # 请求 NZ(29) 不等于返回码必为29：torch_npu 将 FP4 C0=32 映射成 uint8 C0=16(50)。
    # 不能放行所有 NZ 子格式，uint8 C0=32(51) 并非本路径的打包布局。
    if actual not in (29, 50):
        raise ValueError(
            f"{name}: packed FP4 weights require NZ format 29 or 50 (NZ_C0_16), "
            f"got format={actual}, dtype={weight.dtype}, shape={tuple(weight.shape)}; "
            "call transform_weights_for_mega_moe on raw ND weights once"
        )


def _check_a4w4_weight_format(pair, name):
    import torch_npu

    weight, _ = pair
    actual = torch_npu.get_npu_format(weight)
    # Plain UINT8 format-cast gives 32 physical bytes per C0. Some runtimes
    # label this NZ(29), which is ambiguous without the preparation marker.
    if actual != 51 and not (actual == 29 and isinstance(pair, _NativeFp4WeightPair)):
        raise ValueError(
            f"{name}: A4W4 requires NZ with C0=64 FP4 values (32 packed bytes), got format={actual}; "
            "prepare packed ND with transform_weights_for_mega_moe(..., mma_type='fp4xfp4')"
        )


class SymmBuffer:
    """Collective resource owner. All ranks construct/call/destroy in the same order.

    num_experts is global; intermediate_hidden is the post-activation width.
    This Beta supports nonempty, same-BS calls and one in-flight operation per buffer.
    """

    def __init__(
        self,
        group: ProcessGroup | StoreGroup,
        num_experts: int,
        num_max_tokens_per_rank: int,
        num_topk: int,
        hidden: int,
        intermediate_hidden: int,
        num_shared_experts: int = 0,
        mma_type: str = "fp8xfp4",
        activation: str = "swiglu",
        *,
        ranks_per_node: int = 0,
        timeout: int = 120,
        topk_weights_type: int | None = None,
        urma_window_tokens: int | None = None,
        backend: str | None = None,
    ):
        import torch
        import torch.distributed as dist

        if not isinstance(group, (StoreGroup, dist.ProcessGroup)):
            raise TypeError("Pass a StoreGroup or an initialized ProcessGroup explicitly")
        size = group.size() if isinstance(group, StoreGroup) else dist.get_world_size(group)
        from .backend import select_backend

        self._backend_name = select_backend(group) if backend is None else backend
        urma = self._backend_name == "urma"
        roce = self._backend_name == "mte_roce"
        local_mte = self._backend_name == "mte" and (backend == "mte" or mma_type == "fp4xfp4")
        pod_a4w4_mte = local_mte and mma_type == "fp4xfp4" and size == 32 and ranks_per_node == 8
        if topk_weights_type is None:
            topk_weights_type = int(activation == "gelu")
        urma_swiglu = (
            urma and size in (16, 32) and activation == "swiglu"
            and num_shared_experts == 0 and topk_weights_type == 0
        )
        urma_w8a8 = (
            urma and size in (16, 32) and mma_type == "fp8xfp8"
            and num_shared_experts == 0 and (activation == "gelu" or urma_swiglu)
        )
        self._urma_windowed = self._windowed = urma_swiglu_windowed = False
        error = None
        try:
            if backend not in (None, "mte", "mte_roce"):
                raise ValueError("Explicit MegaMoE backend must be 'mte' or 'mte_roce'")
            integer("num_experts", num_experts, 1, 16384)
            integer("num_max_tokens_per_rank", num_max_tokens_per_rank, 1, 65536)
            integer("num_topk", num_topk, 1, min(32, num_experts))
            if urma_window_tokens is not None:
                integer("urma_window_tokens", urma_window_tokens, 1, 256)
            urma_swiglu_windowed = (
                urma_swiglu and (urma_w8a8 or (
                    mma_type == "fp8xfp4"
                    and (size == 32 or num_topk > 8 or num_max_tokens_per_rank > 16384 or urma_window_tokens is not None)
                ))
            )
            self._urma_windowed = (
                urma and size in (8, 16, 32) and activation == "gelu" and mma_type == "fp8xfp4"
                and num_shared_experts == 0
            ) or urma_w8a8 or urma_swiglu_windowed
            self._windowed = self._urma_windowed
            legacy_urma = urma and not self._windowed
            if legacy_urma:
                integer("num_max_tokens_per_rank", num_max_tokens_per_rank, 1, 16384)
                integer("num_topk", num_topk, 1, min(8, num_experts))
            integer("hidden", hidden, 256 if urma else 1024, 8192)
            integer("intermediate_hidden", intermediate_hidden, 256, 16384 if urma else 4096)
            integer("num_shared_experts", num_shared_experts, 0, 1 if urma else 8)
            integer("world_size", size, 2, 128)
            integer("ranks_per_node", ranks_per_node, 0, size)
            integer("timeout", timeout, 1, 300)
            integer("topk_weights_type", topk_weights_type, 0, 1)
            if roce:
                if size != 32 or ranks_per_node != 16:
                    raise ValueError("MTE+RoCE requires EP32 on two 16-rank hosts")
                if num_shared_experts and (mma_type, activation) != ("fp8xfp4", "swiglu"):
                    raise ValueError("MTE+RoCE shared experts require W4A8 SwiGLU")
            if local_mte and not pod_a4w4_mte:
                if size != 16 or ranks_per_node not in (0, 16):
                    if mma_type == "fp4xfp4":
                        raise ValueError("MTE A4W4 requires 16 ranks on one host or EP32 on four 8-rank hosts")
                    raise ValueError("UBX local MTE requires all 16 ranks on one host")
                ranks_per_node = 16
            if self._urma_windowed:
                local_ranks = min(size, 16)
                if ranks_per_node not in (0, local_ranks):
                    raise ValueError(f"Windowed URMA EP{size} requires {local_ranks} ranks per host")
                ranks_per_node = local_ranks
            h_align, i_align = (256, 256) if urma else (64, 128)
            if hidden % h_align or intermediate_hidden % i_align:
                raise ValueError(f"hidden must be divisible by {h_align}; intermediate_hidden by {i_align}")
            if activation not in {"swiglu", "gelu"}:
                raise ValueError("activation must be 'swiglu' or 'gelu'")
            if mma_type == "fp4xfp4" and not (local_mte or roce):
                raise ValueError("A4W4 requires native MTE or MTE+RoCE")
            if mma_type not in {"fp8xfp4", "fp8xfp8", "fp4xfp4"}:
                raise ValueError("mma_type must be fp8xfp4, fp8xfp8 or fp4xfp4")
            if urma and size not in (4, 8, 16, 32):
                raise ValueError("URMA supports EP4/EP8/EP16/EP32")
            if urma and mma_type == "fp8xfp8" and not urma_w8a8:
                raise ValueError("URMA W8A8 requires EP16/EP32 with zero shared experts and GELU or Finalize-weighted SwiGLU")
            if urma and size == 32 and not self._urma_windowed:
                raise ValueError("URMA EP32 requires windowed GELU or SwiGLU with Finalize weighting and zero shared experts")
            if urma_window_tokens is not None and not self._urma_windowed:
                raise ValueError("urma_window_tokens requires a windowed URMA configuration")
            if mma_type != "fp8xfp4" and num_shared_experts:
                raise ValueError("Shared experts currently require FP8xFP4 weights")
            if num_experts % size or num_experts // size > 128:
                raise ValueError("Experts must divide EP, with at most 128 local experts")
            if ranks_per_node and size % ranks_per_node:
                raise ValueError("ranks_per_node must divide EP (contiguous node-major ranks)")
        except (TypeError, ValueError) as failure:
            error = str(failure)
        # 在进入 SHMEM 之前共同校验布局，不能各 rank 各自选择通信偏移。
        config = (
            num_experts,
            num_max_tokens_per_rank,
            num_topk,
            hidden,
            intermediate_hidden,
            num_shared_experts,
            mma_type,
            activation,
            ranks_per_node,
            timeout,
            topk_weights_type,
            urma_window_tokens,
            backend,
            _host_boot_id() if (local_mte and not pod_a4w4_mte) or (self._urma_windowed and size <= 16) else None,
        )
        collective_check(group, "configuration", error, config)
        if pod_a4w4_mte:
            hosts = gather_control(group, "MTE A4W4 host layout", _host_boot_id())
            node_hosts = hosts[::ranks_per_node]
            if not all(node_hosts) or len(set(node_hosts)) != 4 or hosts != [
                host for host in node_hosts for _ in range(ranks_per_node)
            ]:
                raise ValueError("MTE A4W4 EP32 requires four contiguous 8-rank hosts")
        ext = _extension()
        if roce:
            error = None if getattr(ext, "mte_roce_mx_enabled", False) else "install a MegaMoE wheel with MTE+RoCE MX support"
            collective_check(group, "MTE+RoCE availability", error)
        if local_mte and mma_type == "fp4xfp4":
            error = None if getattr(ext, "mte_a4w4_enabled", False) else "install a MegaMoE wheel with MTE A4W4 support"
            collective_check(group, "MTE A4W4 availability", error)
        if urma_w8a8:
            error = None if getattr(ext, "urma_w8a8_enabled", False) else "install a MegaMoE wheel with URMA W8A8 support"
            collective_check(group, "URMA W8A8 availability", error)
        if urma_swiglu_windowed:
            feature = "urma_w8a8_swiglu_enabled" if urma_w8a8 else "urma_windowed_swiglu_enabled"
            error = None if getattr(ext, feature, False) else (
                f"install a MegaMoE wheel with windowed URMA {mma_type} SwiGLU support"
            )
            collective_check(group, "Windowed URMA SwiGLU availability", error)

        def heap_bytes(tokens):
            if roce:
                return int(ext.roce_buffer_size(size, num_experts // size, tokens, num_topk, hidden,
                                                topk_weights_type=topk_weights_type, mma_type=mma_type))
            return int(ext.buffer_size(
                size,
                num_experts // size,
                tokens,
                num_topk,
                hidden,
                backend=self._backend_name,
                topk_weights_type=topk_weights_type,
                mma_type=mma_type,
                windowed=self._urma_windowed,
            ))

        self._native_capacity = num_max_tokens_per_rank
        if roce or local_mte:
            self._native_capacity = _ubx_mte_capacity(
                group, ext, self._backend_name, size, num_experts // size, num_max_tokens_per_rank, num_topk,
                hidden, intermediate_hidden, activation, mma_type, topk_weights_type, num_shared_experts,
            )
            self._windowed = self._native_capacity < num_max_tokens_per_rank
        elif self._urma_windowed:
            self._native_capacity = _urma_window_capacity(
                group, ext, size, num_experts // size, num_max_tokens_per_rank, num_topk, hidden, intermediate_hidden,
                mma_type, activation=activation, window_tokens=urma_window_tokens,
            )
        capacity = heap_bytes(self._native_capacity)
        if self._backend_name == "mte" and not local_mte and activation == "gelu" and not num_shared_experts and capacity > _MEGAMOE_PROFILES["mte"].maximum_capacity:
            self._windowed = True
            self._native_capacity = _mte_window_capacity(
                group, ext, size, num_experts // size, num_max_tokens_per_rank, num_topk, hidden, intermediate_hidden, mma_type
            )
            capacity = heap_bytes(self._native_capacity)
        self._lock = threading.Lock()
        self._device = torch.device(f"npu:{npu_api().current_device()}")
        self._plan = self._plan_bs = None
        self._tail_plan = self._tail_bs = None
        self._timeout = timeout
        self._generation = 0
        self._failure = None
        self._world_size = size
        self.group, self.num_experts = group, num_experts
        self.num_max_tokens_per_rank, self.num_topk = num_max_tokens_per_rank, num_topk
        self.hidden, self.intermediate_hidden = hidden, intermediate_hidden
        self.num_shared_experts, self.mma_type = num_shared_experts, mma_type
        self.activation = activation
        self.ranks_per_node = ranks_per_node
        self.topk_weights_type = topk_weights_type
        self.x = self.x_sf = self.topk_idx = self.topk_weights = None
        error = None
        try:
            self.allocate_inputs()
        except Exception as failure:
            error = str(failure)
        # 输入分配失败先通知所有 rank，再进入 SHMEM，避免其他 rank 已在建链等待。
        collective_check(group, "input allocation", error)
        self._runtime = _MegaRuntime(group, capacity, timeout, backend=self._backend_name, windowed=self._urma_windowed)

    def allocate_inputs(self):
        """Legacy idempotent helper. Inputs are already allocated by the constructor."""
        import torch

        if hasattr(self, "_runtime"):
            self._runtime.ensure_alive()
        if self.x is not None:
            return self
        input_dtype = torch.uint8 if self.mma_type == "fp4xfp4" else torch.float8_e4m3fn
        input_width = self.hidden // (2 if self.mma_type == "fp4xfp4" else 1)
        self.x = torch.empty((self.num_max_tokens_per_rank, input_width), dtype=input_dtype, device=self._device)
        self.x_sf = torch.empty(
            (self.num_max_tokens_per_rank, self.hidden // 32), dtype=torch.float8_e8m0fnu, device=self._device
        )
        self.topk_idx = torch.empty(
            (self.num_max_tokens_per_rank, self.num_topk), dtype=torch.int32, device=self._device
        )
        self.topk_weights = torch.empty_like(self.topk_idx, dtype=torch.float32)
        return self

    def _prepare(self, bs):
        integer("BS", bs, 1, self.num_max_tokens_per_rank)
        logical_bs = bs
        if self._windowed:
            bs = self._native_capacity if self._urma_windowed else min(bs, self._native_capacity)
        if self._plan_bs == bs:
            self._prepare_tail(logical_bs)
            return
        if self._backend_name in ("urma", "mte_roce"):
            # Resource changes retire prior network sources before reusing the
            # fixed control regions. Exchange shape metadata, never route data.
            error = None
            try:
                self._runtime.stream.synchronize()
            except Exception as failure:
                error = str(failure) or type(failure).__name__
            collective_check(self.group, "plan configuration", error, (bs, self._backend_name))
        # Publish a new window plan only after every rank has reserved its scratch.
        self._plan = self._plan_bs = None
        if self._windowed or self._backend_name == "mte_roce":
            plan, error = None, None
            try:
                plan = self._create_plan(bs)
            except Exception as failure:
                error = str(failure) or type(failure).__name__
            collective_check(self.group, "window plan" if self._windowed else "MTE+RoCE plan", error)
        else:
            plan = self._create_plan(bs)
        self._plan, self._plan_bs = plan, bs
        self._prepare_tail(logical_bs)

    def _prepare_tail(self, bs):
        if not self._windowed or self._urma_windowed or bs <= self._native_capacity:
            return
        tail = bs % self._native_capacity
        if not tail or self._tail_bs == tail:
            return
        self._tail_plan = self._tail_bs = None
        plan, error = None, None
        try:
            plan = self._create_plan(tail)
        except Exception as exc:
            error = str(exc) or type(exc).__name__
        collective_check(self.group, "MTE tail plan", error)
        self._tail_plan, self._tail_bs = plan, tail

    def _create_plan(self, bs):
        runtime = self._runtime
        address = require_native("MegaMoE").megamoe_workspace_address(runtime.native)
        ext = _extension()
        if self._backend_name == "mte_roce":
            return ext.RocePlan(
                self.x[:bs], self.intermediate_hidden, self.num_topk, self.num_experts // self._world_size,
                self._world_size, self._native_capacity, self.num_shared_experts, ext.ffts_config(),
                address, runtime.native.capacity, activation=self.activation, mma_type=self.mma_type,
                topk_weights_type=self.topk_weights_type,
            )
        return ext.Plan(
            self.x[:bs],
            self.intermediate_hidden,
            self.num_topk,
            self.num_experts // self._world_size,
            self._world_size,
            self._native_capacity,
            self.mma_type == "fp8xfp4",
            self.num_shared_experts,
            self.ranks_per_node,
            ext.ffts_config(),
            address,
            runtime.native.capacity,
            backend=self._backend_name,
            topk_weights_type=self.topk_weights_type,
            activation=self.activation,
            mma_type=self.mma_type,
            windowed=self._urma_windowed,
            timeout_seconds=self._timeout,
        )

    def num_forward_windows(self, bs: int) -> int:
        """Number of native launches for one public forward at this batch size."""
        integer("BS", bs, 1, self.num_max_tokens_per_rank)
        if not self._windowed:
            return 1
        return (bs + self._native_capacity - 1) // self._native_capacity

    def prepare(self, bs: int) -> SymmBuffer:
        """Collectively prepare/reuse one BS before timing; all ranks must call together."""
        error = None
        with self._lock:
            try:
                self._runtime.ensure_alive()
                integer("BS", bs, 1, self.num_max_tokens_per_rank)
            except Exception as failure:
                error = str(failure)
            collective_check(self.group, "prepare", error, bs)
            try:
                runtime = self._runtime
                runtime.stream.wait_stream(npu_api().current_stream(runtime.device))
                with npu_api().stream(runtime.stream):
                    self._prepare(bs)
            except Exception as failure:
                error = str(failure)
            collective_check(self.group, f"prepare BS={bs}", error)
        return self

    @property
    def window_tokens(self) -> int:
        return self._native_capacity

    @property
    def workspace_bytes(self) -> int:
        """Prepared local scratch only; excludes symmetric heap, inputs and weights."""
        return sum(plan.workspace_bytes for plan in (self._plan, self._tail_plan) if plan is not None)

    @property
    def symmetric_buffer_bytes(self) -> int:
        return self._runtime.native.capacity

    @property
    def timer_enabled(self) -> bool:
        ext = _extension()
        if self._backend_name == "mte_roce":
            return bool(getattr(ext, "mte_roce_timer_enabled", False))
        if self.mma_type == "fp4xfp4":
            return self._backend_name == "mte" and bool(ext.timer_enabled and getattr(ext, "mte_a4w4_timer_enabled", False))
        supports_timer = self.mma_type == "fp8xfp4" or (self.mma_type == "fp8xfp8" and self._urma_windowed)
        return bool(ext.timer_enabled and supports_timer)

    def allocate_timer(self) -> Tensor:
        """Fresh zeroed device trace; pass only to a separate diagnostic invocation."""
        self._runtime.ensure_alive()
        if not self.timer_enabled:
            if self._backend_name == "mte_roce":
                raise RuntimeError("MTE+RoCE device timer requires a compatible wheel built with DEEPEP_MEGAMOE_TIMER=ON")
            raise RuntimeError(_TIMER_REQUIREMENTS)
        return _extension().allocate_timer(self.x, backend=self._backend_name)

    def validate_inputs(self, bs: int, *, x: Tensor | None = None) -> None:
        """Collective, synchronizing routing scan. Run after filling inputs, outside timing."""
        error = None
        try:
            import torch

            self._runtime.ensure_alive()
            integer("BS", bs, 1, self.num_max_tokens_per_rank)
            for name, tensor, dtype, shape in (
                ("topk_idx", self.topk_idx, torch.int32, (self.num_max_tokens_per_rank, self.num_topk)),
                ("topk_weights", self.topk_weights, torch.float32, (self.num_max_tokens_per_rank, self.num_topk)),
            ):
                if tensor.device != self._device or tensor.dtype != dtype or tuple(tensor.shape) != shape:
                    raise ValueError(f"{name} must retain its buffer dtype/device/shape")
            if x is None:
                input_dtype = torch.uint8 if self.mma_type == "fp4xfp4" else torch.float8_e4m3fn
                input_width = self.hidden // (2 if self.mma_type == "fp4xfp4" else 1)
                for name, tensor, dtype, shape in (
                    ("x", self.x, input_dtype, (self.num_max_tokens_per_rank, input_width)),
                    ("x_sf", self.x_sf, torch.float8_e8m0fnu, (self.num_max_tokens_per_rank, self.hidden // 32)),
                ):
                    if (
                        tensor.device != self._device
                        or tensor.dtype != dtype
                        or tuple(tensor.shape) != shape
                        or not tensor.is_contiguous()
                    ):
                        raise ValueError(f"{name} must retain its buffer dtype/device/contiguous shape")
            if x is not None and (
                x.dtype != torch.bfloat16 or x.device != self._device or tuple(x.shape) != (bs, self.hidden)
            ):
                raise ValueError("x override must be BF16 [BS,H] on the buffer device")
            check_routing(
                self.topk_idx[:bs].cpu().tolist(),
                self.topk_weights[:bs].cpu().tolist(),
                self.num_experts,
                self.num_topk,
            )
        except Exception as failure:
            error = str(failure)
        collective_check(self.group, "input preflight", error, (bs, x is not None))

    def destroy(self):
        with self._lock:
            self._runtime.close()
            self._plan = self._plan_bs = None
            self._tail_plan = self._tail_bs = None
            self.x = self.x_sf = self.topk_idx = self.topk_weights = None

    def __del__(self):
        warn_unclosed(self)


def _interleave_gate_up_128(tensor):
    """Permute the N axis of raw [E, 2I, ...] storage without numeric conversion."""
    experts, columns, *tail = tensor.shape
    return (
        tensor.reshape((experts, 2, columns // 256, 128, *tail))
        .transpose(1, 2)
        .contiguous()
        .reshape((experts, columns, *tail))
    )


class _RoutedWeightPair(tuple):
    """Tuple-compatible immutable prepared weights; both layouts are built off the hot path."""

    def __new__(cls, fused, planar):
        value = super().__new__(cls, fused)
        value._planar = planar
        value._versions = tuple(t._version for t in (*fused, *planar))
        return value

    def select(self, fused):
        if tuple(t._version for t in (*self, *self._planar)) != self._versions:
            raise ValueError("Prepared GMM1 weights/scales were modified in-place; transform raw ND weights again")
        return tuple(self) if fused else self._planar


class _UrmaMxSwiGluWeightPair(tuple):
    """Contiguous ND FP8 weights and scales packed in gate128/up128 pairs."""


class _RoceFp8SharedWeightPair(tuple):
    """FP8 shared GMM1 prepared in the MTE+RoCE fused layout."""


class _NativeFp4WeightPair(tuple):
    """Planar weights prepared in byte NZ, with 64 logical FP4 values per C0."""


def transform_weights_for_mega_moe(
    l1_weights: WeightPair, l2_weights: WeightPair, activation: str = "swiglu", *,
    shared: bool = False, backend: str | None = None, mma_type: str | None = None,
) -> tuple[WeightPair, WeightPair]:
    """Prepare raw Ascend weights once, without quantizing again.

    Raw inputs use [gate | up]. Routed FP4 GMM1 prepares both planar Prefill and
    [gate128, up128] Decode layouts. Keep the returned l1 pair intact and immutable;
    the API selects its layout by actual BS without a timed conversion. Pass shared=True for
    shared experts. MTE+RoCE FP8 shared GMM1 also uses the interleaved layout and
    requires backend="mte_roce" when preparing weights. For routed URMA FP8 SwiGLU, pass
    backend="urma" to pack the same gate128/up128 tiles in ND, and keep the
    returned l1 pair intact. Other FP8 weights keep their original layout.
    For native A4W4, pass mma_type="fp4xfp4" to prepare both projections in
    planar NZ with 32 packed bytes per C0. Keep the returned pairs intact.
    This is not the CUDA UTCCP transform. Old prepared FP4 routed weights must
    be regenerated from raw ND inputs; do not transform prepared NZ twice.
    """
    import torch
    import torch_npu

    if activation not in ("swiglu", "gelu"):
        raise ValueError("activation must be 'swiglu' or 'gelu'")
    if backend not in (None, "mte", "urma", "mte_roce"):
        raise ValueError("backend must be 'mte', 'urma' or 'mte_roce'")
    if mma_type not in (None, "fp8xfp4", "fp8xfp8", "fp4xfp4"):
        raise ValueError("mma_type must be fp8xfp4, fp8xfp8 or fp4xfp4")
    native_fp4 = mma_type == "fp4xfp4"
    if native_fp4 and shared:
        raise ValueError("Native A4W4 preparation requires routed FP4 weights")
    transformed = []
    for name, value in (("l1_weights", l1_weights), ("l2_weights", l2_weights)):
        if isinstance(value, (_UrmaMxSwiGluWeightPair, _RoceFp8SharedWeightPair)):
            raise ValueError(f"{name}: prepare from raw planar FP8 weights, not already packed gate/up weights")
        if isinstance(value, _NativeFp4WeightPair):
            raise ValueError(f"{name}: prepare A4W4 from raw packed ND weights")
        weight, scale = _pair(value, name)
        if weight.device.type != "npu" or scale.device != weight.device:
            raise ValueError("Move weights and scales to the same NPU before preparing")
        if weight.ndim != 3:
            raise ValueError(f"{name}: weights must be [E,N,K] or packed [E,N,K/2]")
        if weight.dtype in {torch.int8, torch.uint8} and torch_npu.get_npu_format(weight) in (29, 50, 51):
            raise ValueError(f"{name}: prepare from raw ND weights, not already converted NZ weights")
        if weight.dtype == torch.int8:
            weight = weight.view(torch.uint8)  # 重解释 FP4 的位模式，不能做数值类型转换。
        if weight.dtype not in {torch.uint8, torch.float8_e4m3fn}:
            raise TypeError("Weights must be packed FP4 int8/uint8 or FP8 E4M3")
        k = weight.shape[-1] * (2 if weight.dtype == torch.uint8 else 1)
        expected = (*weight.shape[:2], k // 64, 2)
        if k % 64:
            raise ValueError(f"{name}: K must be divisible by 64")
        if scale.dtype == torch.int32 and tuple(scale.shape) == (*weight.shape[:2], k // 128) and k % 128 == 0:
            # 仅接受逻辑行优先、尚未做 CUDA UTCCP 重排的 packed E8M0。
            scale = scale.contiguous().view(torch.uint8).view(torch.float8_e8m0fnu).reshape(expected)
        if scale.dtype != torch.float8_e8m0fnu or tuple(scale.shape) != expected:
            raise ValueError(f"{name}: expected E8M0 scales {expected} or raw packed int32 [E,N,K/128]")
        scale = scale.contiguous()
        if native_fp4:
            if weight.dtype != torch.uint8 or torch_npu.get_npu_format(weight) != 2:
                raise ValueError(f"{name}: A4W4 requires packed FP4 ND weights")
            # Cast packed bytes as UINT8, without the FP4 input-dtype override
            # that selects the smaller C0 used by the A8W4 dequant prologue.
            pair = _NativeFp4WeightPair((torch_npu.npu_format_cast(weight.contiguous(), 29), scale))
            _check_a4w4_weight_format(pair, name)
            transformed.append(pair)
            continue
        routed_swiglu = name == "l1_weights" and not shared and activation == "swiglu"
        if weight.dtype == torch.uint8:
            if routed_swiglu:
                if weight.shape[1] % 256:
                    raise ValueError("Routed FP4 GMM1 requires gate/up widths divisible by 128")
                # 两种布局一次准备；独立存储并启用版本计数，禁止原地修改后两份权重失配。
                with torch.inference_mode(False):
                    planar_weight = torch_npu.npu_format_cast(
                        weight, 29, customize_dtype=torch.float8_e4m3fn, input_dtype=torch_npu.float4_e2m1fn_x2
                    )
                    planar_scale = scale.view(torch.uint8).clone().view(torch.float8_e8m0fnu)
                    weight = _interleave_gate_up_128(weight)
                    scale = _interleave_gate_up_128(scale.view(torch.uint8)).view(torch.float8_e8m0fnu)
                    weight = torch_npu.npu_format_cast(
                        weight, 29, customize_dtype=torch.float8_e4m3fn, input_dtype=torch_npu.float4_e2m1fn_x2
                    )
                    _check_fp4_weight_format(planar_weight, name)
                    _check_fp4_weight_format(weight, name)
                    transformed.append(_RoutedWeightPair((weight, scale), (planar_weight, planar_scale)))
                continue
            # uint8 只是两个 FP4 的存储载体；FP8 布局参数不能代替输入的 FP4 类型声明。
            weight = torch_npu.npu_format_cast(
                weight,
                29,
                customize_dtype=torch.float8_e4m3fn,
                input_dtype=torch_npu.float4_e2m1fn_x2,
            )
            _check_fp4_weight_format(weight, name)
        elif shared and backend == "mte_roce" and name == "l1_weights" and activation == "swiglu":
            if weight.shape[1] % 256 or torch_npu.get_npu_format(weight) != 2:
                raise ValueError("MTE+RoCE FP8 shared GMM1 requires ND weights and aligned gate/up widths")
            weight = _interleave_gate_up_128(weight.view(torch.uint8)).view(torch.float8_e4m3fn)
            scale = _interleave_gate_up_128(scale.view(torch.uint8)).view(torch.float8_e8m0fnu)
            transformed.append(_RoceFp8SharedWeightPair((weight, scale)))
            continue
        elif routed_swiglu and backend == "urma":
            if weight.shape[1] % 256 or torch_npu.get_npu_format(weight) != 2:
                raise ValueError("URMA FP8 SwiGLU GMM1 requires ND weights and gate/up widths divisible by 128")
            weight = _interleave_gate_up_128(weight.view(torch.uint8)).view(torch.float8_e4m3fn)
            scale = _interleave_gate_up_128(scale.view(torch.uint8)).view(torch.float8_e8m0fnu)
            transformed.append(_UrmaMxSwiGluWeightPair((weight, scale)))
            continue
        transformed.append((weight, scale))
    return tuple(transformed)


def mega_moe(
    x,
    topk_idx,
    topk_weights,
    l1_weights,
    l2_weights,
    sym_buffer,
    *,
    shared_l1_weights=None,
    shared_l2_weights=None,
    out=None,
    timer=None,
    activation_clamp=None,
):
    """Explicit-input convenience API; prefer fp8_fp4_mega_moe for DeepGEMM-style use.

    x: BF16 [B,H] or (FP8 E4M3 [B,H], E8M0 [B,H/32]).
    Prequantized codes are copied into the original communication record, never dequantized/requantized.
    Routed work uses the prefill scheduler; W4 shared strategy switches at B=256.
    No backward or graph guarantee yet.
    """
    return _run(
        x,
        topk_idx,
        topk_weights,
        l1_weights,
        l2_weights,
        sym_buffer,
        shared_l1_weights=shared_l1_weights,
        shared_l2_weights=shared_l2_weights,
        out=out,
        timer=timer,
        activation_clamp=activation_clamp,
    )


def _check_roce_shared_weights(buffer, shared, l1_pair):
    import torch

    if any(tensor is None for tensor in shared):
        raise ValueError("MTE+RoCE requires all four shared weights and scales together")
    dtype = shared[0].dtype
    if dtype not in (torch.uint8, torch.float8_e4m3fn):
        raise ValueError("shared_l1_weights: MTE+RoCE shared weights must be packed FP4 or FP8 E4M3")
    fp8 = dtype == torch.float8_e4m3fn
    packing = 1 if fp8 else 2
    experts, h, i = buffer.num_shared_experts, buffer.hidden, buffer.intermediate_hidden
    if not experts:
        raise ValueError("Shared expert count does not match the buffer configuration")
    expected = (
        ("shared_l1_weights", (experts, 2 * i, h // packing), dtype),
        ("shared_l2_weights", (experts, h, i // packing), dtype),
        ("shared_l1_scales", (experts, 2 * i, (h + 63) // 64, 2), torch.float8_e8m0fnu),
        ("shared_l2_scales", (experts, h, (i + 63) // 64, 2), torch.float8_e8m0fnu),
    )
    for tensor, (name, shape, expected_dtype) in zip(shared, expected):
        if (
            tensor.device != buffer._device
            or tensor.shape != shape
            or tensor.dtype != expected_dtype
            or not tensor.is_contiguous()
        ):
            raise ValueError(f"{name}: MTE+RoCE requires contiguous {expected_dtype} {shape} on {buffer._device}")
    if fp8:
        import torch_npu

        if not isinstance(l1_pair, _RoceFp8SharedWeightPair):
            raise ValueError("Prepare MTE+RoCE FP8 shared GMM1 with shared=True and backend='mte_roce'")
        if any(torch_npu.get_npu_format(weight) != 2 for weight in shared[:2]):
            raise ValueError("MTE+RoCE shared weights require contiguous ND FP8 E4M3")
    else:
        _check_fp4_weight_format(shared[0], "shared_l1_weights")
        _check_fp4_weight_format(shared[1], "shared_l2_weights")


def _forward_inputs(
    x, topk_idx, topk_weights, l1_weights, l2_weights, sym_buffer,
    shared_l1_weights, shared_l2_weights, out, timer, cumulative_stats, activation_clamp, buffered_options,
):
    import torch

    if buffered_options is not None:
        recipe, activation, fast_math = buffered_options
        y = out
        if recipe != (1, 1, 32) or fast_math is not True:
            raise NotImplementedError("Beta supports recipe=(1,1,32) and default math only")
        if activation not in ("swiglu", "gelu"):
            raise ValueError("activation must be 'swiglu' or 'gelu'")
        if activation != sym_buffer.activation:
            raise ValueError("activation must match the SymmBuffer configuration")
        if y.ndim != 2 or y.dtype != torch.bfloat16:
            raise ValueError("y must be BF16 [BS,H]")
        bs = y.shape[0]
        if x is not None and (x.dtype != torch.bfloat16 or tuple(x.shape) != (bs, sym_buffer.hidden)):
            raise ValueError("x override must be BF16 [BS,H]; omit x for buffered FP8 input")
        integer("BS", bs, 1, sym_buffer.num_max_tokens_per_rank)
        if x is None:
            x = (sym_buffer.x[:bs], sym_buffer.x_sf[:bs])
        topk_idx, topk_weights = sym_buffer.topk_idx[:bs], sym_buffer.topk_weights[:bs]
    data, scales = _pair(x, "x") if isinstance(x, tuple) else (x, None)
    w1, s1 = _pair(l1_weights, "l1_weights")
    w2, s2 = _pair(l2_weights, "l2_weights")
    shared = (None, None, None, None)
    if (shared_l1_weights is None) != (shared_l2_weights is None):
        raise ValueError("Supply both shared weight pairs, or neither")
    if shared_l1_weights is not None:
        sw1, ss1 = _pair(shared_l1_weights, "shared_l1_weights")
        sw2, ss2 = _pair(shared_l2_weights, "shared_l2_weights")
        shared = (sw1, sw2, ss1, ss2)
    b = sym_buffer
    roce = b._backend_name == "mte_roce"
    if isinstance(l1_weights, _RoceFp8SharedWeightPair) or (
        isinstance(shared_l1_weights, _RoceFp8SharedWeightPair) and not roce
    ):
        raise ValueError("Prepared MTE+RoCE FP8 GMM1 weights require an MTE+RoCE shared expert")
    if timer is not None and not b.timer_enabled:
        if roce:
            raise RuntimeError("MTE+RoCE device timer requires a compatible wheel built with DEEPEP_MEGAMOE_TIMER=ON")
        raise RuntimeError(_TIMER_REQUIREMENTS)
    if b.activation == "gelu" and activation_clamp is not None:
        raise ValueError("activation_clamp is only defined for SwiGLU")
    clamp = normalize_clamp(activation_clamp)
    urma = b._backend_name == "urma"
    if urma and scales is None:
        raise TypeError("URMA input requires (FP8 E4M3, E8M0 scales)")
    packed_fp4 = b.mma_type == "fp4xfp4" and scales is not None
    expected_width = b.hidden // 2 if packed_fp4 else b.hidden
    if data.ndim != 2 or not 0 < data.shape[0] <= b.num_max_tokens_per_rank or data.shape[1] != expected_width:
        raise ValueError("x must be nonempty [BS,H] (or packed [BS,H/2]) within the buffer capacity")
    if data.device != b._device:
        raise ValueError("x must be on the buffer's creating device")
    expected_x_dtype = torch.uint8 if packed_fp4 else (torch.float8_e4m3fn if scales is not None else torch.bfloat16)
    if data.dtype != expected_x_dtype:
        raise TypeError("x dtype does not match the buffer activation precision")
    expected = (b.num_experts // b.group.size(), b.intermediate_hidden * (1 if b.activation == "gelu" else 2))
    if tuple(w1.shape[:2]) != expected or tuple(topk_idx.shape) != (data.shape[0], b.num_topk):
        raise ValueError("Weight/routing shapes do not match the buffer configuration")
    if roce and shared_l1_weights is not None:
        _check_roce_shared_weights(b, shared, shared_l1_weights)
    elif (shared[0].shape[0] if shared[0] is not None else 0) != b.num_shared_experts:
        raise ValueError("Shared expert count does not match the buffer configuration")
    expected_dtype = torch.uint8 if b.mma_type in {"fp8xfp4", "fp4xfp4"} else torch.float8_e4m3fn
    if w1.dtype != expected_dtype or w2.dtype != expected_dtype:
        raise TypeError("Weight dtype differs from buffer mma_type")
    native_fp4 = b.mma_type == "fp4xfp4"
    if native_fp4:
        _check_a4w4_weight_format(l1_weights, "l1_weights")
        _check_a4w4_weight_format(l2_weights, "l2_weights")
    elif isinstance(l1_weights, _NativeFp4WeightPair) or isinstance(l2_weights, _NativeFp4WeightPair):
        raise ValueError("Prepared A4W4 weights require an A4W4 buffer")
    requires_packed_mx = urma and b.mma_type == "fp8xfp8" and b.activation == "swiglu"
    if requires_packed_mx and not isinstance(l1_weights, _UrmaMxSwiGluWeightPair):
        raise ValueError("Prepare URMA FP8 SwiGLU GMM1 with transform_weights_for_mega_moe(..., backend='urma')")
    if not requires_packed_mx and isinstance(l1_weights, _UrmaMxSwiGluWeightPair):
        raise ValueError("Packed FP8 SwiGLU GMM1 weights require the URMA FP8 SwiGLU buffer")
    prepared_inputs = ()
    if w1.dtype == torch.uint8 and not native_fp4:
        fused = b.activation == "swiglu" and b.mma_type == "fp8xfp4" and (
            urma or min(data.shape[0], b.window_tokens) <= _extension().small_batch_max_tokens
        )
        if isinstance(l1_weights, _RoutedWeightPair):
            if b.activation != "swiglu":
                raise ValueError("GELU requires single-projection GMM1 weights prepared with activation='gelu'")
            prepared_inputs = (*l1_weights, *l1_weights._planar)
            w1, s1 = l1_weights.select(fused)
        elif b.activation == "swiglu" and b.mma_type == "fp8xfp4" and (urma or not fused):
            raise ValueError(
                "Prefill requires both prepared GMM1 layouts; call transform_weights_for_mega_moe "
                "on raw ND weights and keep its returned l1 pair intact"
            )
    if any(t is not None and t.requires_grad for t in (data, scales, topk_weights, w1, s1, w2, s2, out, *shared)):
        raise NotImplementedError("MegaMoE Beta is inference-only")
    if out is not None and any(
        out.untyped_storage().data_ptr() == tensor.untyped_storage().data_ptr()
        for tensor in (data, scales, topk_idx, topk_weights, w1, w2, s1, s2, timer, *shared, *prepared_inputs)
        if tensor is not None
    ):
        raise ValueError("out must not overlap any input or timer storage")
    if topk_idx.dtype != torch.int32:
        raise TypeError("topk_idx must be int32; convert explicitly before timing")
    if w1.dtype == torch.uint8 and not native_fp4:
        _check_fp4_weight_format(w1, "l1_weights")
        _check_fp4_weight_format(w2, "l2_weights")
        if not roce:
            for name, weight in zip(("shared_l1_weights", "shared_l2_weights"), shared[:2]):
                if weight is None:
                    continue
                if urma:
                    import torch_npu

                    if (
                        weight.dtype != torch.float8_e4m3fn
                        or not weight.is_contiguous()
                        or torch_npu.get_npu_format(weight) != 2
                    ):
                        raise ValueError("URMA shared weights require contiguous ND FP8 E4M3")
                else:
                    _check_fp4_weight_format(weight, name)
    elif not native_fp4 and (urma or roce):
        import torch_npu

        for weight in (w1, w2):
            if not weight.is_contiguous() or torch_npu.get_npu_format(weight) != 2:
                raise ValueError("UBX W8A8 routed weights require contiguous ND FP8 E4M3")
    if cumulative_stats is not None:
        stats = cumulative_stats
        if (
            stats.dtype != torch.int32
            or tuple(stats.shape) != (b.num_experts // b.group.size(),)
            or stats.device != b._device
            or not stats.is_contiguous()
        ):
            raise ValueError("Cumulative stats must be contiguous local-expert int32 on the buffer device")
        if any(
            stats.untyped_storage().data_ptr() == t.untyped_storage().data_ptr()
            for t in (data, scales, topk_idx, topk_weights, w1, w2, s1, s2, out, timer, *shared, *prepared_inputs)
            if t is not None
        ):
            raise ValueError("Cumulative stats must not alias inputs, output or timer")
    if b._windowed or roce:
        bs, h, i = data.shape[0], b.hidden, b.intermediate_hidden
        local = b.num_experts // b._world_size
        w1_columns = 2 * i if b.activation == "swiglu" else i
        tensors = [
            (topk_idx, (bs, b.num_topk), torch.int32),
            (topk_weights, (bs, b.num_topk), torch.float32),
            (data, (bs, expected_width), expected_x_dtype),
            (w1, (local, w1_columns, h // (2 if w1.dtype == torch.uint8 else 1)), expected_dtype),
            (w2, (local, h, i // (2 if w1.dtype == torch.uint8 else 1)), expected_dtype),
            (s1, (local, w1_columns, h // 64, 2), torch.float8_e8m0fnu),
            (s2, (local, h, i // 64, 2), torch.float8_e8m0fnu),
        ]
        if scales is not None:
            if scales.ndim == 3 and tuple(scales.shape) == (bs, h // 64, 2) and scales.is_contiguous():
                scales = scales.reshape((bs, h // 32))
            tensors.append((scales, (bs, h // 32), torch.float8_e8m0fnu))
        if out is not None:
            tensors.append((out, (bs, h), torch.bfloat16))
        for tensor, shape, dtype in tensors:
            if (
                tensor.device != b._device
                or tuple(tensor.shape) != shape
                or tensor.dtype != dtype
                or not tensor.is_contiguous()
            ):
                raise ValueError(f"Windowed {b._backend_name.upper()} requires contiguous {dtype} {shape} on {b._device}")
        if timer is not None:
            if timer.device != b._device:
                raise ValueError("timer must be on the buffer device")
            _extension().validate_timer(timer, b._backend_name)
            if any(
                timer.untyped_storage().data_ptr() == tensor.untyped_storage().data_ptr()
                for tensor in (data, scales, topk_idx, topk_weights, w1, w2, s1, s2, *shared, out)
                if tensor is not None
            ):
                raise ValueError("timer must not share storage with inputs, weights or output")
        check_window_routes(topk_idx, topk_weights, b.num_experts)
    return data, scales, topk_idx, topk_weights, (w1, w2, s1, s2), shared, clamp


def _run(
    x,
    topk_idx,
    topk_weights,
    l1_weights,
    l2_weights,
    sym_buffer,
    *,
    shared_l1_weights=None,
    shared_l2_weights=None,
    out=None,
    timer=None,
    cumulative_stats=None,
    activation_clamp=None,
    buffered_options=None,
):
    import torch

    b = sym_buffer
    if not isinstance(b, SymmBuffer):
        raise TypeError("sym_buffer must be this installation's MegaMoE SymmBuffer")
    runtime = b._runtime
    with b._lock:
        runtime.ensure_alive()
        check_collectively = b._windowed or b._backend_name == "mte_roce"
        if b._failure is not None:
            raise RuntimeError(f"MegaMoE buffer failed; restart the distributed job: {b._failure}")
        error = None
        signature = None
        try:
            data, scales, ids, weights, matrix, shared, clamp = _forward_inputs(
                x, topk_idx, topk_weights, l1_weights, l2_weights, b, shared_l1_weights,
                shared_l2_weights, out, timer, cumulative_stats, activation_clamp, buffered_options
            )
            bs = data.shape[0]
            if check_collectively:
                signature = (bs, b._generation, scales is not None, clamp)
                if b._backend_name == "mte_roce":
                    # The kernel's wave size depends on the shared weight precision.
                    signature += (shared[0] is not None and shared[0].dtype == torch.float8_e4m3fn,)
                if b._urma_windowed and b._generation + b.num_forward_windows(bs) >= 1 << 31:
                    raise ValueError("URMA generation exhausted; recreate the buffer")
                if b._backend_name == "mte_roce" and b._generation + b.num_forward_windows(bs) >= (1 << 31) - 1:
                    raise ValueError("MTE+RoCE sequence exhausted; recreate the buffer")
                if out is None:
                    out = torch.empty((bs, b.hidden), dtype=torch.bfloat16, device=b._device)
        except Exception as exc:
            if not check_collectively:
                raise
            error = str(exc) or type(exc).__name__
        if check_collectively:
            collective_check(b.group, "window inputs", error, signature)
        caller = npu_api().current_stream(runtime.device)
        runtime.stream.wait_stream(caller)
        with npu_api().stream(runtime.stream):
            start_generation = b._generation
            try:
                b._prepare(bs)
                if b.num_forward_windows(bs) == 1:
                    b._generation += 1
                    result = b._plan.forward(
                        data, scales, ids, weights, *matrix, timer, *shared, out, b._generation, clamp
                    )
                else:
                    counts = None
                    for first in range(0, bs, b._native_capacity):
                        rows = slice(first, min(first + b._native_capacity, bs))
                        b._generation += 1
                        plan = b._plan if b._urma_windowed or rows.stop - first == b._plan_bs else b._tail_plan
                        _, window_counts = plan.forward(
                            data[rows], scales[rows] if scales is not None else None, ids[rows], weights[rows], *matrix,
                            timer if first == 0 else None, *shared, out[rows], b._generation, clamp,
                        )
                        if counts is None:
                            counts = window_counts
                        else:
                            counts.add_(window_counts)
                    result = out, counts
            except Exception as exc:
                # Window plan failures are collective and retryable. Once a launch
                # starts, a peer may already be waiting on its generation.
                launched = b._generation != start_generation
                if (b._backend_name == "urma" and not b._windowed) or (
                    launched and (b._windowed or b._backend_name == "mte_roce")
                ):
                    b._failure = str(exc)
                    if isinstance(b.group, StoreGroup):
                        b.group.abort(b._failure)
                raise
            if cumulative_stats is not None:
                cumulative_stats.add_(result[1])
                cumulative_stats.record_stream(runtime.stream)
        caller.wait_stream(runtime.stream)
        for tensor in result:
            tensor.record_stream(caller)
        return result


def fp8_fp4_mega_moe(
    y: Tensor,
    l1_weights: WeightPair,
    l2_weights: WeightPair,
    sym_buffer: SymmBuffer,
    shared_l1_weights: WeightPair | None = None,
    shared_l2_weights: WeightPair | None = None,
    cumulative_local_expert_recv_stats: Tensor | None = None,
    recipe: tuple[int, int, int] = (1, 1, 32),
    activation: str = "swiglu",
    activation_clamp: float | None = None,
    fast_math: bool = True,
    *,
    x: Tensor | None = None,
    timer: Tensor | None = None,
) -> None:
    """Write BF16 y in place, return None; accumulate stats if supplied.

    Positional parameters follow DeepGEMM. Ascend-only keyword extensions:
    x is optional BF16 [BS,H] (quantized internally, NOT BF16 GEMM);
    timer is a fresh buffer.allocate_timer() for a separate diagnostic call.
    Routed weight placement follows sym_buffer.topk_weights_type; see README.
    """
    _run(
        x,
        None,
        None,
        l1_weights,
        l2_weights,
        sym_buffer,
        shared_l1_weights=shared_l1_weights,
        shared_l2_weights=shared_l2_weights,
        out=y,
        timer=timer,
        cumulative_stats=cumulative_local_expert_recv_stats,
        activation_clamp=activation_clamp,
        buffered_options=(recipe, activation, fast_math),
    )
