# Lessons Learned

## 1. Benchmark Discipline Matters More Than Intuition

Many ideas that looked reasonable were neutral or worse once benchmarked. The useful workflow was:
- make a narrow hypothesis
- change one thing
- validate correctness
- benchmark
- keep or discard based on numbers

## 2. Wrapper-Level Tuning Has a Ceiling

For all three workloads, there was a point where dispatch, split, and caching tweaks stopped giving meaningful gains. That usually means the main kernel is now the bottleneck, and real progress needs a different kernel implementation.

## 3. Judge Constraints Are Part of the Problem

Some seemingly good paths failed because of the remote judge environment:
- stream-safety violations
- mismatch between local-looking wins and public visible results
- runtime behavior that differs from a simple local mental model

In practice, performance engineering under platform constraints is closer to production work than to isolated benchmarking.

## 4. Quantized Inference Is About More Than Dtype Labels

FP8 and MXFP4 optimization involved:
- memory layout sensitivity
- kernel dispatch selection
- scale handling
- quantization/dequantization overhead
- shape-dependent behavior

The dtype name alone does not determine performance. The surrounding runtime path matters just as much.

## 5. Hidden Evaluation Changes What “Best” Means

The best local benchmark variant is not always the best visible leaderboard outcome. That reinforces an important engineering habit: separate local signal, public signal, and hidden-eval risk.

## 6. Not Winning Can Still Produce Strong Engineering Evidence

Even without a top leaderboard finish, the project demonstrated:
- low-level reasoning about GPU inference workloads
- systematic experiment design
- correctness/performance tradeoff handling
- ability to work through failed ideas without losing rigor
