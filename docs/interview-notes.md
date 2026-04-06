# Interview Notes

## 30-Second Version

I worked on low-level optimization for quantized LLM inference workloads on AMD Instinct MI355X GPUs. The project covered MXFP4 GEMM, MoE MXFP4, and MLA decode. I built multiple variants, benchmarked them on remote judge hardware, debugged correctness and stream-safety issues, and learned where runtime tuning helps and where a custom kernel becomes necessary.

## If Asked “Did You Win?”

Use this answer:

> I did not finish at the top, but the project was still valuable because it involved real GPU inference engineering. I worked on quantized execution paths, kernel dispatch decisions, split-strategy tuning, benchmark validation, and runtime debugging on AMD MI355X. The main value was the engineering depth and systems understanding I gained.

## If Asked “Why Is This Relevant to AI/LLM Roles?”

Use this answer:

> Modern LLM engineering is not only about model code. It also includes inference latency, quantization, memory movement, runtime behavior, and serving efficiency. This project gave me hands-on experience in exactly those areas.

## Technical Talking Points

- optimized quantized inference workloads using FP8 and MXFP4
- compared multiple kernel/runtime strategies under correctness constraints
- tuned split strategies and dispatch behavior for shape-sensitive GPU workloads
- debugged stream-safety violations and hidden-evaluation mismatch
- learned when wrapper-level changes stop helping and the bottleneck shifts to the kernel itself

## Resume-Ready Bullets

- Optimized quantized LLM inference workloads on AMD Instinct MI355X across MXFP4 GEMM, MoE MXFP4, and MLA decode tasks.
- Built and benchmarked multiple GPU-inference variants using split tuning, dispatch tuning, metadata reuse, and quantization-aware runtime changes.
- Debugged low-level inference issues including stream-safety violations, correctness/performance tradeoffs, and mismatch between local and remote evaluation behavior.
- Applied benchmark-driven performance engineering to GPU inference codepaths relevant to LLM serving and ML systems roles.

## What To Emphasize

- real engineering difficulty
- measurement discipline
- low-level systems understanding
- quantized inference knowledge
- ability to explain failures and tradeoffs clearly

## What To Avoid

- saying it was “just a hackathon”
- focusing on not winning
- presenting the work as random trial-and-error
- talking only about rank without explaining the technical work
