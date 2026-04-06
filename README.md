# LLM Kernel Optimization on AMD MI355X

This repository is a compact case study in low-level LLM inference optimization on AMD Instinct MI355X GPUs. The work came from three kernel-focused qualifier tasks from the AMD x GPU MODE competition:

- MXFP4 GEMM
- MoE MXFP4
- Mixed MLA decode

The focus here is not raw submission history. It is the engineering process behind quantized inference optimization: forming narrow hypotheses, validating correctness, benchmarking on remote judge hardware, and keeping only the variants that were both safe and measurably better.

## Results At A Glance

| Workload | Final file | Best local safe result | Main optimization theme |
| --- | --- | --- | --- |
| MXFP4 GEMM | `submissions/amd-mxfp4-mm/submission_v6.py` | `~21.67 µs` geomean | Kernel-dispatch and asm-path steering |
| MoE MXFP4 | `submissions/amd-moe-mxfp4/submission_v2.py` | `183.31 µs -> 173.75 µs` | Runtime-path tuning |
| Mixed MLA | `submissions/amd-mixed-mla/submission_v5.py` | `179.29 µs -> 132.45 µs` | Metadata reuse and split tuning |

## What This Project Demonstrates

- Quantization-aware inference work using FP8 and MXFP4 paths
- GPU runtime and kernel-dispatch reasoning on AMD hardware
- Latency-focused optimization for inference workloads
- Benchmark-driven engineering instead of intuition-driven tuning
- Debugging under real platform constraints such as stream-safety failures and hidden-evaluation mismatch

## Technical Themes

Across the three workloads, the work included:

- kernel dispatch tuning
- split strategy experiments
- metadata and workspace reuse
- correctness checks for quantized execution
- graph-path and native-kernel exploration
- rollback of variants that benchmarked worse or failed judge constraints

One consistent lesson was that wrapper-level tuning helps only until the main kernel becomes the bottleneck. After that point, large gains typically require changing the dominant kernel itself.

## Why This Matters For AI/LLM Engineering

This maps directly to modern inference and ML systems work:

- quantized inference execution
- latency optimization
- memory-layout sensitivity
- runtime debugging
- performance measurement under production-like constraints

For AI/LLM engineering roles, this kind of project is useful because it shows systems-level reasoning beyond model APIs or notebook experimentation.

## Start Here

If you are reading this as a collaborator or reviewer, the fastest path is:

1. [Benchmark summary](docs/benchmark-summary.md) for the kept variants and main performance outcomes.
2. [Lessons learned](docs/lessons-learned.md) for the engineering takeaways.
3. `submissions/` for the final selected code artifacts.

## Repository Layout

```text
llm-kernel-optimization-amd-mi355x/
├── README.md
├── .gitignore
├── docs/
│   ├── benchmark-summary.md
│   └── lessons-learned.md
└── submissions/
    ├── amd-mixed-mla/
    │   └── submission_v5.py
    ├── amd-moe-mxfp4/
    │   └── submission_v2.py
    └── amd-mxfp4-mm/
        └── submission_v6.py
```

## Notes

- This repository intentionally keeps only the final selected variants, not every experimental file.
- The main value is the technical process: correctness, measurement, iteration, and systems-level analysis.
- If this repository is made public, competition sharing rules should be rechecked first.
