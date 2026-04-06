# Benchmark Summary

This document summarizes the final kept variants and the main performance outcomes from the project. The numbers below are local safe benchmark observations used for engineering decisions, not a claim about final hidden-evaluation ranking.

## Final Kept Variants

| Workload | Final file | Best local safe result | Strongest improvement observed | Main takeaway |
| --- | --- | --- | --- | --- |
| MXFP4 GEMM | `submissions/amd-mxfp4-mm/submission_v6.py` | `~21.67 µs` geomean | Small-shape asm steering beat later split-K follow-ups | Dispatch tuning helped, but gains plateaued quickly |
| MoE MXFP4 | `submissions/amd-moe-mxfp4/submission_v2.py` | `~173.75 µs` geomean | `183.31 µs -> 173.75 µs` | Runtime-path selection mattered more than later env-var toggles |
| Mixed MLA | `submissions/amd-mixed-mla/submission_v5.py` | `~132.45 µs` geomean | `179.29 µs -> 132.45 µs` | Metadata reuse delivered the biggest gain; later split tuning was incremental |

## Workload Notes

### MXFP4 GEMM

- Final file: `submissions/amd-mxfp4-mm/submission_v6.py`
- Best local safe geomean: `~21.67 µs`
- What helped: steering small-`m` shapes onto a better asm path
- What did not help enough: later exact-shape split-K experiments and graph follow-ups
- Engineering takeaway: once the safe kernel-dispatch space was explored, further gains likely required a fundamentally different kernel path

### MoE MXFP4

- Final file: `submissions/amd-moe-mxfp4/submission_v2.py`
- Best local safe geomean: `~173.75 µs`
- Best measured improvement: `183.31 µs -> 173.75 µs`
- What helped: a runtime-path change outperformed later `AITER_USE_NT` follow-ups
- Engineering takeaway: the first major win came from choosing a better execution path, not layering on more toggles

### Mixed MLA Decode

- Final file: `submissions/amd-mixed-mla/submission_v5.py`
- Best local safe geomean: `~132.45 µs`
- Best measured improvement: `179.29 µs -> 132.45 µs`
- What helped most: caching persistent metadata and work buffers
- What helped slightly: changing `num_kv_splits` from the stronger cached baseline to `8`
- What failed or was unsafe: several graph, non-persistent, and native-kernel experiments either benchmarked worse or hit judge stream-safety constraints

## How To Read These Numbers

- These were benchmark results used to decide which variants to keep.
- Some locally stronger variants did not produce a stronger visible public result.
- That mismatch is part of the engineering story: hidden evaluation and platform constraints matter.

## Recruiter-Friendly Takeaway

The strongest signal from this project is not a single leaderboard rank. It is the ability to run a disciplined optimization loop on real GPU inference workloads: make a narrow change, validate correctness, benchmark it, and keep only what survives measurement.
