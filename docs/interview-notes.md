# Interview Notes

## 30-Second Version

I worked on low-level optimization for quantized LLM inference workloads on AMD Instinct MI355X GPUs. The project covered MXFP4 GEMM, MoE MXFP4, and MLA decode. I built multiple variants, benchmarked them on remote judge hardware, debugged correctness and stream-safety issues, and learned where runtime tuning helps and where a custom kernel becomes necessary.

## 60-Second Version

I used a GPU optimization competition as a practical ML systems project. I worked on three quantized inference workloads on AMD MI355X: MXFP4 GEMM, MoE MXFP4, and MLA decode. For each task, I formed narrow performance hypotheses, implemented targeted variants, validated correctness, and benchmarked them on the remote evaluation environment. The work involved kernel dispatch tuning, split-strategy changes, metadata reuse, and debugging issues like stream-safety failures and hidden-evaluation mismatch. The main lesson was that small wrapper-level changes can help for a while, but once the dominant kernel becomes the bottleneck, real gains usually require a different kernel implementation.

## If Asked “Did You Win?”

Use this answer:

> I did not finish at the top, but the project was still valuable because it involved real GPU inference engineering. I worked on quantized execution paths, kernel dispatch decisions, split-strategy tuning, benchmark validation, and runtime debugging on AMD MI355X. The main value was the engineering depth and systems understanding I gained.

## If Asked “Why Is This Relevant to AI/LLM Roles?”

Use this answer:

> Modern LLM engineering is not only about model code. It also includes inference latency, quantization, memory movement, runtime behavior, and serving efficiency. This project gave me hands-on experience in exactly those areas.

## If Asked “What Was Actually Hard?”

Use this answer:

> The hard part was that many changes that looked reasonable either did not help on benchmark or failed because of judge constraints such as stream-safety rules. So the project became an exercise in disciplined performance engineering: make one narrow change, validate correctness, benchmark it, and keep only what is supported by data.

## Technical Talking Points

- optimized quantized inference workloads using FP8 and MXFP4
- compared multiple kernel and runtime strategies under correctness constraints
- tuned split strategies and dispatch behavior for shape-sensitive GPU workloads
- debugged stream-safety violations and mismatch between local signal and visible public outcomes
- learned when wrapper-level changes stop helping and the bottleneck shifts to the kernel itself

## Strong Resume Bullets

- Optimized quantized LLM inference workloads on AMD Instinct MI355X across MXFP4 GEMM, MoE MXFP4, and MLA decode tasks.
- Built and benchmarked multiple GPU-inference variants using dispatch tuning, split tuning, metadata reuse, and quantization-aware runtime changes.
- Debugged low-level inference issues including stream-safety violations, correctness and performance tradeoffs, and mismatch between local and remote evaluation behavior.
- Applied benchmark-driven performance engineering to GPU inference codepaths relevant to LLM serving and ML systems roles.

## Role-Specific Framing

### For AI/ML Engineer roles

- Emphasize quantized inference, latency reduction, and production-style benchmarking.
- Position the project as practical experience beyond notebooks and model fine-tuning.

### For LLM Engineer roles

- Emphasize inference bottlenecks, serving efficiency, quantization, and runtime behavior.
- Talk about how model performance in practice depends on memory movement and kernel selection, not only architecture quality.

### For ML Systems or Performance roles

- Emphasize kernel dispatch, low-level debugging, measurement discipline, and failure analysis.
- Position the project as systems work under hardware and platform constraints.

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
