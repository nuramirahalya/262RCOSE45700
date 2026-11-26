# Attention benchmark — November 26, 2026

Reports, logs, manifest, and summary are in attention-repeated-20261126T114845Z/.

The benchmark uses the shared AttentionCore implementation: unmasked scaled dot-product attention without dropout or learned projections.

Configuration:
- Batch size 1, four heads, head dimension 64.
- Sequence lengths 64 and 128.
- Reproducible synthetic BF16 Q, K, and V inputs with input seed 43.
- CPU fallback disabled.
- CPU-reference and Eager/Compile comparisons at rtol=0.01 and atol=0.01.
- Five sequential fresh processes, each testing both sequence lengths.
- Ten warm-up calls and 50 timed calls per mode and configuration.

All five runs passed correctness checks. The reports contain 1000 individual warm timings.

| Sequence | Mode | Mean (ms) | p50 (ms) | p95 (ms) | Sequences/s | Run-mean CV |
|---:|---|---:|---:|---:|---:|---:|
| 64 | Eager | 0.462 | 0.453 | 0.540 | 2163.4 | 0.68% |
| 64 | Compile | 0.180 | 0.178 | 0.202 | 5545.6 | 1.28% |
| 128 | Eager | 0.456 | 0.450 | 0.525 | 2192.7 | 1.10% |
| 128 | Compile | 0.182 | 0.180 | 0.195 | 5495.1 | 3.08% |

Mean, p50, and p95 use 250 pooled warm samples per row. The p95 uses inclusive linear interpolation. CV is the sample standard deviation of the five run means divided by their mean. Throughput is batch size times 1000 divided by mean latency in milliseconds.

Inputs are already on the NPU. Timing includes model execution and output transfer to CPU. Persistent compiler cache state is uncontrolled. First-call values remain exploratory. Each configuration uses one input seed.

## Reproduce

From the repository root in the Furiosa environment:

    python experiments/ai-baseline/repeat_benchmarks.py --models attention_core --output-dir attention-results
    python experiments/ai-baseline/summarize_benchmarks.py attention-results

Use a new output directory for each experiment. Software versions are recorded in each JSON report.

The default SmallCNN and ResNet benchmark commands remain available. Attention summary rows use throughput_sequences_per_second and include sequence length, head count, and head dimension.
