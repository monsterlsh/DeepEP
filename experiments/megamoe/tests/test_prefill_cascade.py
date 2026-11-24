from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from types import SimpleNamespace

import pytest

from deep_ep_experimental.megamoe import prefill_cascade


@pytest.mark.parametrize("early", [0, 1])
@pytest.mark.parametrize("empty_owner", [False, True])
def test_prefill_routes_payload_and_weight_through_uneven_two_rank_splits(monkeypatch, early, empty_owner):
    torch = pytest.importorskip("torch")
    # A tile bisects an expert, and the final source window has one token.
    monkeypatch.setattr(prefill_cascade, "FFN_ROW_TILE", 2)
    monkeypatch.setattr(prefill_cascade, "PREFILL_TOKEN_WINDOW", 2)
    barrier = Barrier(2, timeout=10)
    inputs = [None, None]

    def exchange(rank):
        def all_to_all_single(out, tensor, *, output_split_sizes, input_split_sizes):
            sizes = input_split_sizes if input_split_sizes is not None else [tensor.shape[0] // 2] * 2
            inputs[rank] = tensor.split(sizes)
            barrier.wait()
            received = torch.cat([inputs[source][rank] for source in (0, 1)])
            assert output_split_sizes is None or [inputs[source][rank].shape[0] for source in (0, 1)] == output_split_sizes
            out.copy_(received)
            barrier.wait()

        return all_to_all_single

    def init_routing(data, ids, *, expert_num, **_):
        # The CANN row map uses token + slot * BS, even though IDs are [BS, K].
        indices = ids.T.flatten().argsort(stable=True)
        order = indices.argsort().to(torch.int32)
        counts = ids.flatten().bincount(minlength=expert_num)
        return data.index_select(0, indices.remainder(ids.shape[0])), order, counts, None

    def re_routing(data, counts):
        expert_ids = torch.arange(counts.shape[1]).repeat(counts.shape[0])
        expanded = expert_ids.repeat_interleave(counts.flatten())
        order = expanded.argsort(stable=True)
        return data.index_select(0, order), None, order.to(torch.int32), counts.sum(dim=0)

    def finalize(data, *, scales, expanded_src_to_dst_row, **_):
        expanded = data.index_select(0, expanded_src_to_dst_row.long())
        expanded = expanded.reshape(scales.shape[1], scales.shape[0], data.shape[1]).transpose(0, 1)
        return (expanded * scales[:, :, None]).sum(dim=1)

    def run_rank(rank, data, scale, ids, weights):
        cascade = object.__new__(prefill_cascade.PrefillCascade)
        cascade.torch = torch
        cascade.dist = SimpleNamespace(all_to_all_single=exchange(rank))
        cascade.npu = SimpleNamespace(
            npu_moe_init_routing_v2=init_routing, npu_moe_re_routing=re_routing, npu_moe_finalize_routing=finalize,
        )
        cascade.world, cascade.topk, cascade.experts, cascade.local_experts, cascade.hidden = 2, 2, 4, 2, 4
        cascade.topk_weights_type, cascade.weights = early, None

        def ffn(rows, scales, counts, _weights, routing_weights):
            assert rows.shape[0] <= 2
            assert int(counts.sum()) == rows.shape[0]
            experts = torch.arange(rank * 2, rank * 2 + 2).repeat_interleave(counts)
            values = rows[:, 0] + scales[:, 0, 0] + experts * 2
            if routing_weights is not None:
                values = values * routing_weights
            return values[:, None].expand(-1, cascade.hidden).to(torch.bfloat16)

        cascade._ffn = ffn
        cascade._normalize_scale = lambda scales, *shape: scales
        # Use packed byte payloads like FP4 plus an independent scale payload.
        data = torch.tensor(data, dtype=torch.uint8)[:, None].repeat(1, 2)
        scale = torch.tensor(scale, dtype=torch.uint8)[:, None, None].repeat(1, 1, 2)
        ids = torch.tensor(ids, dtype=torch.int32)
        if empty_owner:
            ids = ids.remainder(2) + 2
        weights = torch.tensor(weights, dtype=torch.float32)
        actual = cascade.forward(None, ids, weights, input_kind="fp4", quantized=(data, scale))
        expected = ((data[:, 0, None] + scale[:, 0, :] + ids * 2) * weights).sum(dim=1)
        assert torch.equal(actual, expected[:, None].expand(-1, 4).to(torch.bfloat16))

    with ThreadPoolExecutor(2) as pool:
        rank0 = pool.submit(
            run_rank, 0, [11, 23, 5], [3, 9, 1], [[0, 2], [1, 2], [1, 3]], [[0.5, 1], [0.5, -0.25], [1, 2]],
        )
        rank1 = pool.submit(
            run_rank, 1, [7, 15, 23], [5, 7, 9], [[2, 3], [0, 2], [0, 1]], [[-1, 0.5], [2, 0.5], [1, 0.5]],
        )
        rank0.result()
        rank1.result()
