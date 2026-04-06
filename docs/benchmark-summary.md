# Benchmark Summary

This file summarizes the final safe variants kept for each workload and the main benchmark takeaways from the optimization process.

## Final Safe Variants

| Workload | Final file | Main result |
| --- | --- | --- |
| MXFP4 GEMM | `submission_v6.py` | Best safe local benchmark result among tested GEMM variants |
| MoE MXFP4 | `submission_v2.py` | Improved over the starter path through runtime-path tuning |
| Mixed MLA | `submission_v5.py` | Best safe split-tuning result after metadata and split-count experiments |

## MXFP4 GEMM

- Final file: `submissions/amd-mxfp4-mm/submission_v6.py`
- Best local safe benchmark geomean observed: about `21.67 µs`
- Key finding: small-shape asm-path steering helped more than later split-K tweaks
- Important negative result: exact-shape split-K follow-up and graph-path follow-up did not improve the best safe result

## MoE MXFP4

- Final file: `submissions/amd-moe-mxfp4/submission_v2.py`
- Best local benchmark geomean observed: about `173.75 µs`
- Baseline comparison: about `183.31 µs -> 173.75 µs`
- Key finding: a runtime-path change improved performance more than later follow-up toggles

## Mixed MLA Decode

- Final file: `submissions/amd-mixed-mla/submission_v5.py`
- Best local safe benchmark geomean observed: about `132.45 µs`
- Earlier strong step: metadata caching reduced the benchmark substantially versus the starter path
- Final split tuning: `num_kv_splits = 8` was slightly better than the stronger earlier safe baseline

## Important Caveat

Local or public benchmark geomeans and public leaderboard-visible scores are not always identical. During the project, some variants looked better locally but did not translate cleanly to the visible public result, which is a useful reminder that hidden evaluation behavior matters.
