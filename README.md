# LLM Kernel Optimization on AMD MI355X

This repository documents my work on low-level inference optimization for quantized LLM workloads on AMD Instinct MI355X GPUs.

The project was based on three kernel-focused workloads from the AMD x GPU MODE qualifier track:
- MXFP4 GEMM
- MoE MXFP4
- Mixed MLA decode

Rather than treating the work as a pure hackathon submission dump, this repository is organized as a performance-engineering case study: hypothesis-driven optimization, correctness validation, benchmark comparison, and analysis of why some approaches helped while others failed.

## Scope

I focused on:
- quantized inference paths using FP8 and MXFP4
- GPU kernel/runtime behavior on AMD MI355X
- latency-oriented optimization for inference kernels
- benchmark-driven iteration and rollback of weak variants
- debugging judge/runtime issues such as stream-safety violations and hidden-eval mismatch

## Final Selected Variants

The repository keeps the best safe variant I found for each workload:

| Workload | Final file | Notes |
| --- | --- | --- |
| MXFP4 GEMM | `submissions/amd-mxfp4-mm/submission_v6.py` | Best safe kernel-dispatch variant for the GEMM task |
| MoE MXFP4 | `submissions/amd-moe-mxfp4/submission_v2.py` | Best safe runtime-path tuning result |
| Mixed MLA | `submissions/amd-mixed-mla/submission_v5.py` | Best safe split-tuning result after extensive follow-up experiments |

## What This Work Involved

Across the three workloads, the optimization work included:
- kernel dispatch tuning
- split strategy experiments
- metadata/workspace reuse
- correctness validation for quantized execution
- benchmark comparison across many candidate variants
- investigation of native-kernel and graph-based paths
- debugging runtime constraints from the remote evaluation environment

## Why This Is Relevant to AI/LLM Engineering

This project maps directly to modern LLM inference engineering work:
- quantization-aware execution
- inference latency optimization
- GPU memory and layout sensitivity
- runtime/kernel debugging
- performance measurement under real evaluation constraints

The main lesson was that small wrapper-level tuning can help only up to a point. After that, meaningful gains usually require changing the dominant kernel itself.

## Repository Layout

```text
llm-kernel-optimization-amd-mi355x/
├── README.md
├── .gitignore
├── docs/
│   ├── benchmark-summary.md
│   ├── interview-notes.md
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

- This repository intentionally includes only the final selected variants, not every experimental file.
- Public leaderboard ranking is not the main point of this repository. The technical process and systems-level learning are the main value.
- If publishing publicly, it is worth double-checking the competition sharing rules before posting additional raw submission history or logs.
