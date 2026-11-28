"""Prefill performance reference with rank-local AllToAll routing."""

from .cascade import CascadeReference


PREFILL_TOKEN_WINDOW = 4096
FFN_ROW_TILE = 32768


class PrefillCascade(CascadeReference):
    """Run the moe_test AllToAll prefill routing pattern with a routed FFN.

    Each source window expands only its own routes. The expert owner computes
    in bounded row tiles, preserving the grouping within each GMM tile.
    """

    def _exchange(self, tensor, send_sizes=None, recv_sizes=None):
        if self.world == 1:
            return tensor
        send = tensor.contiguous()
        if tensor.element_size() == 1:
            # Preserve FP8 and E8M0 codes through HCCL.
            send = send.view(self.torch.int8)
        if recv_sizes is None:
            received = self.torch.empty_like(send)
        else:
            received = send.new_empty((sum(recv_sizes), *send.shape[1:]))
        self.dist.all_to_all_single(received, send, output_split_sizes=recv_sizes, input_split_sizes=send_sizes)
        return received.view(tensor.dtype)

    def _select_codes(self, codes, indices):
        # NPU IndexSelect need only support an ordinary integer payload.
        return codes.view(self.torch.int8).index_select(0, indices).view(codes.dtype)

    def _activation_quantize(self, first, route_weights=None):
        if self.activation != "gelu":
            return super()._activation_quantize(first, route_weights)
        activated = self.torch.nn.functional.gelu(first.float(), approximate="none")
        if route_weights is not None:
            activated.mul_(route_weights[:, None])
        return self._quantize(activated.to(self.torch.bfloat16))

    def _run_expert_tiles(self, data, scale, counts, weights):
        rows = data.shape[0]
        if rows <= FFN_ROW_TILE:
            return self._ffn(data, scale, counts, self.weights, weights)
        output = data.new_empty((rows, self.hidden), dtype=self.torch.bfloat16)
        ends = counts.cumsum(0)
        starts = ends - counts
        for start in range(0, rows, FFN_ROW_TILE):
            end = min(start + FFN_ROW_TILE, rows)
            tile_counts = ends.clamp(start, end) - starts.clamp(start, end)
            tile_weights = None if weights is None else weights[start:end]
            output[start:end].copy_(
                self._ffn(data[start:end], scale[start:end], tile_counts, self.weights, tile_weights)
            )
        return output

    def _forward_window(self, data, scale, ids, weights):
        torch, npu = self.torch, self.npu
        expanded, row_idx, counts, _ = npu.npu_moe_init_routing_v2(
            data.contiguous().view(torch.bfloat16),
            ids,
            scale=None,
            active_num=ids.numel(),
            expert_num=self.experts,
            expert_tokens_num_type=1,
            expert_tokens_num_flag=True,
            quant_mode=-1,
            active_expert_range=[0, self.experts],
        )
        counts = counts.to(torch.int64)
        owner_counts = self._exchange(counts).reshape(self.world, self.local_experts)
        # HCCL AllToAllV takes host split arrays. Count exchange and this sync
        # are part of the measured forward, as is the payload communication.
        recv_sizes, send_sizes = torch.stack(
            (owner_counts.sum(dim=1), counts.reshape(self.world, self.local_experts).sum(dim=1))
        ).cpu().tolist()

        # InitRouting and FinalizeRouting index source slots as token + slot * BS.
        source_order = row_idx.reshape(-1).float().argsort()
        source_scales = self._select_codes(scale, source_order.remainder(data.shape[0]))
        owner_data = self._exchange(expanded.view(data.dtype), send_sizes, recv_sizes)
        del expanded
        owner_scales = self._exchange(source_scales, send_sizes, recv_sizes)
        del source_scales
        owner_weights = None
        if self.topk_weights_type:
            source_weights = weights.T.reshape(-1).index_select(0, source_order)
            owner_weights = self._exchange(source_weights, send_sizes, recv_sizes)

        if owner_data.shape[0] == 0:
            owner_output = owner_data.new_empty((0, self.hidden), dtype=torch.bfloat16)
            # Every source sent zero rows for every local expert.
            local_counts = owner_counts[0]
        elif self.world == 1:
            local_counts = counts
            owner_output = self._run_expert_tiles(owner_data, owner_scales, local_counts, owner_weights)
        else:
            # ReRouting only permutes rows. A BF16 view preserves the FP8 codes.
            ordered, _, owner_order, local_counts = npu.npu_moe_re_routing(
                owner_data.view(torch.bfloat16), owner_counts,
            )
            del owner_data
            owner_order = owner_order.reshape(-1).to(torch.int64)
            ordered_scales = self._select_codes(owner_scales, owner_order)
            del owner_scales
            if owner_weights is not None:
                owner_weights = owner_weights.index_select(0, owner_order)
            local_counts = local_counts.to(torch.int64)
            ordered_output = self._run_expert_tiles(
                ordered.view(data.dtype), ordered_scales, local_counts, owner_weights,
            )
            del ordered, ordered_scales
            owner_output = ordered_output.index_select(0, owner_order.float().argsort())
            del ordered_output

        source_output = self._exchange(owner_output, recv_sizes, send_sizes)
        del owner_output
        routed = npu.npu_moe_finalize_routing(
            source_output.float(),
            skip1=None,
            skip2=None,
            bias=None,
            scales=torch.ones_like(weights) if self.topk_weights_type else weights,
            expanded_src_to_dst_row=row_idx,
            export_for_source_row=None,
            drop_pad_mode=2,
        ).reshape(data.shape[0], self.hidden)
        return self._finalize_output(routed, data, scale), local_counts

    def forward(self, x, ids, weights, *, input_kind, quantized=None):
        """Return one rank's BF16 outputs for a large prefill forward."""
        return self._forward(x, ids, weights, input_kind, quantized, with_counts=False)

    def forward_with_counts(self, x, ids, weights, *, input_kind, quantized=None):
        """Return BF16 outputs and int64 receive counts for this rank's routed experts."""
        return self._forward(x, ids, weights, input_kind, quantized, with_counts=True)

    def _forward(self, x, ids, weights, input_kind, quantized, *, with_counts):
        if input_kind == "bf16":
            data, scale = self._quantize(x)
        elif input_kind == "fp8":
            data, scale = self._fp8_input(quantized)
            scale = self._normalize_scale(scale, data.shape[0], self.hidden)
        else:
            raise ValueError("Prefill input_kind must be bf16 or fp8")
        bs = data.shape[0]
        if bs <= PREFILL_TOKEN_WINDOW:
            output, counts = self._forward_window(data, scale, ids, weights)
            return (output, counts) if with_counts else output
        output = data.new_empty((bs, self.hidden), dtype=self.torch.bfloat16)
        counts = None
        for start in range(0, bs, PREFILL_TOKEN_WINDOW):
            rows = slice(start, start + PREFILL_TOKEN_WINDOW)
            window_output, window_counts = self._forward_window(data[rows], scale[rows], ids[rows], weights[rows])
            output[rows].copy_(window_output)
            del window_output
            if with_counts:
                if counts is None:
                    counts = window_counts
                else:
                    counts.add_(window_counts)
        return (output, counts) if with_counts else output
