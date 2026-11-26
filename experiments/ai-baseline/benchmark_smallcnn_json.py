import argparse
import json
from pathlib import Path

from furiosa.torch import config

parser = argparse.ArgumentParser()
parser.add_argument("--output", type=Path, required=True)
parser.add_argument(
    "--model",
    choices=["smallcnn", "resnet18_chunked", "attention_core"],
    default="smallcnn",
)
args = parser.parse_args()

with config.cpu_fallback(False):
    import torch
    import furiosa.torch

    import sys

    repository_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repository_root))

    from models.project_models import build_model_pair, make_input

    if args.model == "attention_core":
        from experiments.attention.attention_core import AttentionCore

        reference = AttentionCore().eval()
        model = AttentionCore().eval()
    else:
        reference, model = build_model_pair(args.model, seed=42)
    device = torch.device("furiosa:0")
    model = model.to(device)

    import time

    compiled = torch.compile(
        model,
        backend=furiosa.torch.backend,
        mode="default"
    )

    def benchmark(model_fn, x, warmup=10, runs=50):
        times = []

        with torch.inference_mode():
            for _ in range(warmup):
                model_fn(*x).detach().cpu()

            for _ in range(runs):
                start = time.perf_counter()
                model_fn(*x).detach().cpu()
                times.append((time.perf_counter() - start) * 1000)

        return times

    print("Batch,Eager_ms,Compile_ms,Speedup", flush=True)

    results = []

    if args.model == "attention_core":
        configurations = [(1, 64), (1, 128)]
    elif args.model == "smallcnn":
        configurations = [(batch, None) for batch in [1, 4, 8, 16]]
    else:
        configurations = [(1, None)]

    for batch, sequence_length in configurations:
        configuration = {}
        if args.model == "attention_core":
            shape = (batch, 4, sequence_length, 64)
            generator = torch.Generator().manual_seed(43)
            x_cpu = tuple(
                torch.randn(shape, generator=generator, dtype=torch.bfloat16)
                for _ in range(3)
            )
            configuration = {
                "sequence_length": sequence_length,
                "heads": 4,
                "head_dimension": 64,
            }
            print(f"Attention input shape: {shape}", flush=True)
        else:
            x_cpu = (make_input(args.model, batch=batch, seed=43),)
        with torch.inference_mode():
            reference_output = reference(*x_cpu).float()
        x = tuple(value.to(device) for value in x_cpu)


        with torch.inference_mode():
            start = time.perf_counter()
            eager_output = model(*x).detach().cpu()
            eager_first_ms = (time.perf_counter() - start) * 1000

            start = time.perf_counter()
            compiled_output = compiled(*x).detach().cpu()
            compile_first_ms = (time.perf_counter() - start) * 1000

            eager_output = eager_output.float()
            compiled_output = compiled_output.float()

        print(
            f"Batch {batch} first call: Eager={eager_first_ms:.3f} ms, "
            f"Compile={compile_first_ms:.3f} ms",
            flush=True
        )

        max_diff = (eager_output - compiled_output).abs().max().item()
        is_close = torch.allclose(
            eager_output, compiled_output,
            rtol=1e-2, atol=1e-2
        )

        print(
            f"Batch {batch}: max_diff={max_diff:.6f}, "
            f"outputs_close={is_close}",
            flush=True
        )

        eager_matches_cpu = torch.allclose(
            eager_output, reference_output, rtol=1e-2, atol=1e-2
        )
        compile_matches_cpu = torch.allclose(
            compiled_output, reference_output, rtol=1e-2, atol=1e-2
        )
        print(
            f"CPU checks: Eager={eager_matches_cpu}, "
            f"Compile={compile_matches_cpu}",
            flush=True,
        )

        if not (is_close and eager_matches_cpu and compile_matches_cpu):
            results.append({
                "batch_size": batch,
                **configuration,
                "status": "correctness_failed",
                "outputs_close": is_close,
                "eager_matches_cpu": eager_matches_cpu,
                "compile_matches_cpu": compile_matches_cpu,
                "max_abs_diff": max_diff,
            })
            continue

        eager_samples = benchmark(model, x)
        compile_samples = benchmark(compiled, x)
        eager_ms = sum(eager_samples) / len(eager_samples)
        compile_ms = sum(compile_samples) / len(compile_samples)

        print(
            f"{batch},{eager_ms:.3f},{compile_ms:.3f},"
            f"{eager_ms / compile_ms:.2f}",
            flush=True
        )
        results.append({
            "batch_size": batch,
            **configuration,
            "status": "passed",
            "eager_matches_cpu": eager_matches_cpu,
            "compile_matches_cpu": compile_matches_cpu,
            "eager_first_call_ms": eager_first_ms,
            "compile_first_call_ms": compile_first_ms,
            "eager_warm_samples_ms": eager_samples,
            "compile_warm_samples_ms": compile_samples,
            "eager_warm_mean_ms": eager_ms,
            "compile_warm_mean_ms": compile_ms,
            "compile_warm_speedup": eager_ms / compile_ms,
            "max_abs_diff": max_diff,
            "outputs_close": is_close,
        })

import platform
from importlib.metadata import version

report = {
    "environment": {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torchvision": version("torchvision"),
        "furiosa_torch": version("furiosa-torch"),
    },
    "model": "SmallCNN" if args.model == "smallcnn" else args.model,
    "device": "furiosa:0",
    "dtype": "bfloat16",
    "seed": 42,
    "input_seed": 43,
    "cpu_fallback": False,
    "warmup_iterations": 10,
    "timed_iterations": 50,
    "rtol": 1e-2,
    "atol": 1e-2,
    "timing_scope": "Input already on NPU; output copy to CPU included",
    "first_call_protocol": "Per batch shape, same process, Eager first, cache state uncontrolled",
    "results": results,
}
with args.output.open("x") as output:
    json.dump(report, output, indent=2)
    output.write("\n")
print(f"Saved {len(results)} results to {args.output}")
if any(row["status"] != "passed" for row in results):
    raise SystemExit("One or more batches failed the correctness check.")
