"""Performance reference for the Dispatch/Combine V2 Decode path."""

from .cascade import CascadeReference


# ScalarType and runtime codes used by the moe_test Decode operator bindings.
_FP8_E4M3 = 24
_E8M0 = 293


def validate_decode_case(config):
    model = config["model"]
    activation = model.get("activation", "swiglu")
    if (
        activation not in ("swiglu", "gelu")
        or model.get("topk_weights_type", int(activation == "gelu")) != 0
        or model.get("activation_clamp") is not None
        or model["num_shared_experts"] != 0
    ):
        raise ValueError("Decode comparison requires unclamped SwiGLU or GELU, Finalize weights and zero shared experts")
    mma_type = model.get("mma_type", "fp8xfp4")
    if mma_type not in ("fp8xfp4", "fp8xfp8"):
        raise ValueError("Decode comparison supports A8W4 and A8W8")
    if activation == "gelu" and mma_type != "fp8xfp8":
        raise ValueError("Fused GELU Decode requires A8W8 with matching MX operands")
    # Match moe_test --stage auto so a regression run stays on the Decode workload.
    if max(config["bs_values"]) > 512 or model["num_topk"] > 16 or model["num_experts"] > 1024:
        raise ValueError("Decode comparison requires per-rank BS <= 512, TopK <= 16 and experts <= 1024")


def load_fused_gelu_op():
    try:
        from cann_ops_transformer.ops import grouped_matmul_activation_quant
    except ImportError as error:
        raise RuntimeError(
            "GELU Decode requires cann_ops_transformer.ops.grouped_matmul_activation_quant "
            "from a compatible CANN ops-transformer installation"
        ) from error
    return grouped_matmul_activation_quant


class DecodeCascade(CascadeReference):
    """Use MegaMoE's raw weights with the Decode communication and fused FFN operators."""

    def __init__(self, model, raw, *, host_group):
        super().__init__(model, raw, host_group=host_group)
        pg = self.dist.group.WORLD._get_backend(self.torch.device("npu"))
        comm = pg.get_hccl_comm_name(self.rank, init_comm=True)
        if not comm:
            raise RuntimeError("Decode cascade could not initialize its HCCL communication domain")
        self.communication = dict(
            group_ep=comm,
            ep_world_size=self.world,
            ep_rank_id=self.rank,
            tp_world_size=1,
            moe_expert_num=self.experts,
            expert_shard_type=0,
            shared_expert_num=1,
            shared_expert_rank_num=0,
        )
        self.gmm1, self.gmm1_options = self._prepare_gmm1()

    def _prepare_gmm1(self):
        weight, scale = self.weights[0]
        if self.activation == "gelu":
            weight = weight.view(self.torch.uint8).contiguous().view(self.fp8)
            nz_weight = self.npu.npu_format_cast(weight, 29, customize_dtype=self.fp8)
            options = dict(
                weight=[nz_weight],
                weight_scale=[scale.contiguous().view(self.e8m0)],
                activation_type="gelu_tanh",
                group_list_type=1,
                quant_mode="mx",
                y_dtype=self.fp8,
                weight_scale_dtype=_E8M0,
                x_scale_dtype=_E8M0,
            )
            return load_fused_gelu_op(), options
        options = dict(
            weight=[weight],
            weight_scale=[scale.view(self.e8m0)],
            dequant_mode=2,
            dequant_dtype=self.torch.float32,
            quant_mode=2,
            quant_dtype=_FP8_E4M3,
            group_list_type=1,
            weight_scale_dtype=self.e8m0_operator,
            x_scale_dtype=self.e8m0_operator,
        )
        if self.mma_type != "fp8xfp8":
            options["weight_dtype"] = self.fp4
        return self.npu.npu_grouped_matmul_swiglu_quant_v2, options

    def forward(self, x, ids, weights, *, input_kind, quantized=None):
        """Run one Decode forward and return the local BF16 output."""
        if input_kind == "bf16":
            data, scale = self._quantize(x)
        elif input_kind == "fp8":
            data, scale = self._fp8_input(quantized)
        else:
            raise ValueError("Decode input_kind must be bf16 or fp8")
        npu = self.npu
        global_bs = ids.shape[0] * self.world
        expanded, expanded_scale, assist, counts, ep_counts, tp_counts, _ = npu.npu_moe_distribute_dispatch_v2(
            x=data,
            scales=scale.flatten(1),
            x_dtype=_FP8_E4M3,
            scales_dtype=_E8M0,
            expert_ids=ids,
            x_active_mask=None,
            quant_mode=0,
            global_bs=global_bs,
            y_dtype=_FP8_E4M3,
            **self.communication,
        )
        intermediate, intermediate_scale = self.gmm1(
            x=expanded,
            x_scale=expanded_scale.reshape(expanded.shape[0], self.hidden // 64, 2),
            group_list=counts,
            **self.gmm1_options,
        )
        output = self._gmm(intermediate, intermediate_scale, counts, self.weights[1])
        return npu.npu_moe_distribute_combine_v2(
            expand_x=output,
            expert_ids=ids,
            assist_info_for_combine=assist,
            ep_send_counts=ep_counts,
            expert_scales=weights,
            shared_expert_x=None,
            x_active_mask=None,
            tp_send_counts=tp_counts,
            global_bs=global_bs,
            comm_quant_mode=0,
            **self.communication,
        )
