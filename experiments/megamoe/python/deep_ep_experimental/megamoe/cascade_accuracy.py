# Copyright (c) 2026, Lu Lu

"""Run only the NPU operator cascade and compare each rank to its CPU Golden."""

from contextlib import nullcontext
import json


def run_cascade_accuracy(args, config, group, rank_dir, sample_peak):
    import torch
    from .benchmark import accuracy, copy_quantized_input, inputs, routed_weights
    from .cascade import CascadeReference, init_process_group
    from .device_memory import DeviceMemorySampler, combine_device_memory_samples, summarize_device_memory_peaks
    from .reference import make_weights

    model = config["model"]
    world, rank = group.size(), group.rank()
    hidden = model["hidden"]
    mma = model.get("mma_type", "fp8xfp4")

    def sampler():
        return DeviceMemorySampler(torch.npu, torch.npu.current_device()) if sample_peak else nullcontext()

    with sampler() as setup:
        owned = init_process_group(group)
        shared = None
        if model["num_shared_experts"]:
            shared = make_weights(model["num_shared_experts"], hidden,
                                  model.get("shared_intermediate_hidden", model["intermediate_hidden"]),
                                  config["seed"] + 100000, model.get("activation", "swiglu"))
        reference = CascadeReference(model, routed_weights(config, world, rank), shared, host_group=group)
        del shared
    for bs in config["bs_values"]:
        case = rank_dir / f"bs_{bs}"
        case.mkdir()
        with sampler() as measurement:
            x, ids, weights = inputs(config, rank, bs, args.routing, world=world)
            activation, quantized = None, None
            if args.input == "bf16":
                activation = torch.from_numpy(x).to(device="npu", dtype=torch.bfloat16)
            else:
                packed = mma == "fp4xfp4"
                data = torch.empty((bs, hidden // 2 if packed else hidden), device="npu",
                                   dtype=torch.uint8 if packed else torch.float8_e4m3fn)
                scales = torch.empty((bs, hidden // 32), device="npu", dtype=torch.float8_e8m0fnu)
                copy_quantized_input(x, mma, data, scales)
                quantized = (data, scales)
                del data, scales
            del x
            output, counts = reference.forward(activation, torch.from_numpy(ids).npu(), torch.from_numpy(weights).npu(),
                                               input_kind=args.input, quantized=quantized)
            torch.npu.synchronize()
        if sample_peak:
            peak = combine_device_memory_samples([setup.report(), measurement.report()])
            peak["scope"] = "npu_cascade"
            peers = group.all_gather(peak, f"cascade device memory BS={bs}")
            resources = dict(device_memory_peak=peak, device_memory_peak_group=summarize_device_memory_peaks(peers))
            (case / "resources.json").write_text(json.dumps(resources, indent=2), encoding="utf-8")
        reports = accuracy(config, rank, world, bs, output, counts, args.check_tokens, args.routing,
                           group=group, golden_dir=args.golden_dir, input_kind=args.input, implementation="cascade")
        path = case / "accuracy.json"
        path.write_text(json.dumps(reports, indent=2), encoding="utf-8")
        if not all(report["passed"] for report in reports):
            raise AssertionError(f"Cascade vs CPU Golden failed: {path}")
        if rank == 0:
            print(f"[ACCURACY] Cascade vs CPU Golden: {world} ranks passed, {path}", flush=True)
        del output, counts, activation, quantized, ids, weights
    # On a rank failure the benchmark aborts the host group and torchrun stops its peers.
    if owned:
        torch.distributed.destroy_process_group()
