# Job Application Kit

This document turns the project into ready-to-use application material for AI/ML, LLM, and ML systems roles.

## Resume Project Title

Use one of these:

- LLM Inference Kernel Optimization on AMD MI355X
- Quantized LLM Inference Optimization on AMD GPUs
- GPU Inference Performance Engineering for LLM Workloads

## Short Resume Summary

Use this if you want a 1-2 line project description under a project heading:

> Optimized quantized LLM inference workloads on AMD Instinct MI355X across MXFP4 GEMM, MoE MXFP4, and MLA decode tasks. Built and benchmarked multiple low-level variants using dispatch tuning, split tuning, metadata reuse, and correctness-checked quantized execution.

## Resume Bullets

Pick 3-4 bullets, not all of them.

- Optimized quantized LLM inference workloads on AMD Instinct MI355X across MXFP4 GEMM, MoE MXFP4, and MLA decode tasks.
- Built and benchmarked multiple GPU-inference variants using dispatch tuning, split tuning, metadata reuse, and quantization-aware runtime changes.
- Improved measured benchmark performance on MoE MXFP4 and Mixed MLA workloads through runtime-path selection and cache-aware tuning.
- Debugged low-level inference issues including stream-safety violations, correctness and performance tradeoffs, and mismatch between local and remote evaluation behavior.
- Applied benchmark-driven performance engineering to GPU inference codepaths relevant to LLM serving and ML systems roles.

## Role-Tailored Bullets

### AI/ML Engineer

- Worked on practical inference optimization for quantized LLM workloads rather than only model training or notebook experimentation.
- Improved inference-path efficiency through measurement-driven tuning of runtime dispatch, split strategy, and metadata reuse.
- Validated correctness and performance behavior on remote evaluation hardware under production-like constraints.

### LLM Engineer

- Optimized quantized inference paths for LLM workloads involving MXFP4 GEMM, MoE, and MLA decode on AMD MI355X GPUs.
- Investigated latency bottlenecks in serving-style workloads, including kernel dispatch behavior, memory-layout sensitivity, and execution-path selection.
- Used benchmark-driven iteration to identify where wrapper-level tuning helps and where custom kernel work would be required.

### ML Systems / Performance Engineer

- Performed low-level GPU performance engineering on quantized inference kernels under correctness and platform constraints.
- Compared multiple execution strategies across shape-sensitive workloads using targeted hypotheses, remote benchmarking, and rollback of weak variants.
- Debugged runtime constraints such as stream-safety failures and hidden-evaluation mismatch in a GPU judge environment.

## LinkedIn Project Description

Use this as a project entry or post summary:

> Worked on low-level optimization of quantized LLM inference workloads on AMD Instinct MI355X GPUs as part of the AMD x GPU MODE qualifier track. Focused on MXFP4 GEMM, MoE MXFP4, and MLA decode workloads, with experiments around kernel dispatch tuning, split-strategy selection, metadata reuse, correctness validation, and remote benchmark analysis. The project was a practical exercise in GPU inference engineering, quantization-aware execution, and performance measurement under real platform constraints.

## GitHub Repo Description

Use this for the repository description field:

> Low-level LLM inference optimization on AMD MI355X: MXFP4 GEMM, MoE MXFP4, and MLA decode.

## “Tell Me About This Project” Answer

Use this 60-90 second answer:

> I treated this as a practical ML systems project around quantized LLM inference on AMD MI355X GPUs. The work covered three workloads: MXFP4 GEMM, MoE MXFP4, and MLA decode. For each one, I formed a narrow optimization hypothesis, changed one part of the runtime path, validated correctness, and benchmarked the result on remote judge hardware. The main changes involved dispatch tuning, split-strategy experiments, metadata reuse, and investigation of graph or native-kernel paths. A big part of the learning was seeing that some ideas looked good locally but failed under judge constraints such as stream-safety rules or hidden evaluation behavior. The main takeaway was that strong inference engineering is mostly about disciplined measurement and understanding when wrapper-level tuning stops helping and the kernel itself becomes the bottleneck.

## “What Did You Learn?” Answer

> I learned that inference performance depends on much more than model architecture or dtype choice. Memory layout, dispatch policy, synchronization behavior, and platform constraints can dominate the result. I also learned the value of disciplined benchmarking: make one narrow change, validate correctness, measure it, and keep only what survives real evaluation.

## “Why Does This Matter For AI/LLM Jobs?” Answer

> Modern AI engineering is not just training models. A lot of real work is in inference efficiency, quantization, runtime behavior, serving latency, and debugging performance bottlenecks. This project gave me direct experience in those areas.

## Recruiter-Friendly Summary

If someone is non-technical, use this:

> This project was about making LLM inference kernels run faster on AMD GPUs. I worked on low-level performance tuning, measured different implementations, and learned how quantization and GPU runtime behavior affect real inference speed.

## What To Emphasize

- this was hands-on GPU inference engineering
- the work was measurement-driven
- you dealt with real constraints, not toy examples
- you can explain both wins and failures clearly

## What To Avoid

- saying the project only matters if you won
- underselling it as just a hackathon
- focusing only on rank without explaining the engineering work
