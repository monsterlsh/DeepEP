#!/usr/bin/env python3
"""Check every packed FP4 byte using the installed wheel on CPU or one NPU."""

import argparse
from contextlib import nullcontext
from importlib.metadata import version
import warnings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "npu"), default="npu")
    parser.add_argument("--strict-warning", action="store_true", help="Fail on the internal-format fallback warning")
    args = parser.parse_args()
    if args.strict_warning:
        warnings.filterwarnings(
            "error", message=r"Cannot create tensor with internal format while allow_internal_format=False.*"
        )

    import torch
    from deep_ep_experimental.megamoe import api

    print("ascend-deepep:", version("ascend-deepep"), "api:", api.__file__, flush=True)
    print("torch:", torch.__version__, "device:", args.device, flush=True)
    worker = caller = None
    if args.device == "npu":
        import torch_npu

        # This standalone process explicitly exercises the reported setting.
        torch.npu.config.allow_internal_format = False
        torch.npu.set_device(0)
        print("torch_npu:", torch_npu.__version__, "allow_internal_format:", False, flush=True)
        caller, worker = torch.npu.current_stream(), torch.npu.Stream()

    # Independent numerical reference, including negative zero; no production LUT.
    values = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0])
    codes = torch.arange(256, dtype=torch.int64)
    expected = torch.stack((values[codes % 16], values[codes // 16]), dim=-1).to(torch.float8_e4m3fn).view(torch.uint8)
    cases = 0
    for rows, columns in ((1, 256), (16, 16), (256, 1)):
        for offset in (0, 1):
            original = codes.to(torch.uint8).reshape(rows, columns)
            source = original.to(args.device)
            storage = torch.full((rows + 2, columns * 2), 0xA5, dtype=torch.uint8, device=args.device)
            output = storage[offset : offset + rows].view(torch.float8_e4m3fn)
            pointer = output.data_ptr()
            if worker is not None:
                worker.wait_stream(caller)
            with torch.npu.stream(worker) if worker is not None else nullcontext():
                returned = api._expand_fp4_input(source, output)
                if worker is not None:
                    source.record_stream(worker)
            if worker is not None:
                caller.wait_stream(worker)
            assert returned is output and output.data_ptr() == pointer, "Output buffer was replaced"
            expected_storage = torch.full((rows + 2, columns * 2), 0xA5, dtype=torch.uint8)
            expected_storage[offset : offset + rows].copy_(expected.reshape(rows, columns * 2))
            assert torch.equal(storage.cpu(), expected_storage), "Expanded bytes or adjacent rows differ"
            assert torch.equal(source.cpu(), original), "Packed input was modified"
            cases += 1
    print(f"PASS: {cases} cases; all 256 packed bytes, signed zero, output alias and buffer guards", flush=True)


if __name__ == "__main__":
    main()
