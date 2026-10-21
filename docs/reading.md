# Reading: LoRA and quantization for the interview

About 25 minutes. Read one section, then cover it and explain it out loud in 3 sentences, as if to an interviewer. Section 4 is a self-test: answer before you peek. Numbers use this project's model (Qwen2.5-0.5B: hidden size 896, 24 layers, k/v projections 896 -> 128).

## 1. LoRA: what it really does

Idea. Fine-tuning changes a weight `W` (d_out x d_in) by some `ΔW`. The LoRA bet: `ΔW` has low rank, so write it as `B @ A` with a small inner size `r`. Analogy: a full update is repainting the whole wall; LoRA is a few stencils, and most changes you need fit through a few stencils.

Mechanism, in `train.py`:
- `y = W x + (alpha / r) * B (A x)`. `W` is frozen; only `A` (r x d_in) and `B` (d_out x r) train.
- `B` starts at zero, `A` random. So at step 0 the model is exactly the base model: no damage before learning starts.
- `alpha / r` keeps the update's size roughly stable when you change `r`, so you need not retune the learning rate each time.

Count it (r=8, `q_proj` + `v_proj`):
- `q_proj` 896 -> 896: 8 x (896 + 896) = 14,336
- `v_proj` 896 -> 128: 8 x (896 + 128) = 8,192
- x 24 layers = 540,672, plus the score head 896 x 3 = 2,688
- About 0.54M of ~0.5B: ~0.1% trains. Check it against `params/trainable_pct` in W&B.

What LoRA saves, and what it does not (a favourite trap):
- Saves: gradients and AdamW state (2 extra numbers per param) only for the 0.1%. Checkpoints are a few MB.
- Does NOT save: activations. Backprop still flows through the frozen `W` to reach earlier layers, so activation memory is about the same as full fine-tuning. Long sequences and big batches still eat RAM (that is your swap).
- Merge: `W <- W + (alpha/r) B A` gives zero extra cost at inference. Price: you lose the cheap adapter swap (one base, many adapters).

Where it stops working: tasks far from the base model's knowledge (new language, new domain) may need a higher rank or full fine-tuning.

QLoRA in one line: base weights stored in 4-bit (NF4), LoRA adapters in 16-bit, gradients flow through the dequantized base. Lets a 65B model fine-tune on one 48 GB GPU. Its finding: putting LoRA on all linear layers matters more than a big `r`.

## 2. Quantization: a ruler with 256 marks

Mechanism. int8 keeps 256 values. Map a float to an int with a scale: `q = round(x / scale)`, back with `x ≈ q * scale` (plus a zero point for asymmetric ranges).

Tiny example. Weights in [-0.5, 0.5], symmetric int8: `scale = 0.5 / 127 ≈ 0.0039`. Rounding error is at most `scale / 2 ≈ 0.002`. Fine.

Now add one outlier, 5.0: `scale = 5 / 127 ≈ 0.039`. Every weight smaller than 0.02 rounds to 0. One outlier stretched the ruler and erased the fine detail.

- **per-tensor**: one ruler for the whole matrix. One outlier hurts every row.
- **per-channel**: one ruler per output row. The outlier only hurts its own row. Costs one scale per row, almost nothing. That is the experiment in this project.

Kinds of PTQ:
- **weight-only** (GPTQ, AWQ, GGUF): weights int4/int8, math in float. Most common for LLMs.
- **dynamic** (this project): weights int8 ahead of time, activation scale computed on each call. No calibration data needed.
- **static**: activation scales fixed from calibration data. Faster, but a bad calibration set hurts.
- **QAT** (quantization-aware training): simulate rounding during training, so the model learns to live with it. Best quality, costs training.

Why LLMs are hard to quantize: a few activation channels carry huge values (outliers ~100x the rest). Activation quantization then breaks. Fixes: LLM.int8() keeps the outlier channels in fp16; SmoothQuant moves the outlier scale from activations into weights. That is also why weight-only is popular.

Why int8 is faster, when it is: LLM inference is mostly memory-bound (moving weights costs more than multiplying them), so 4x fewer bytes means less waiting on memory. Plus int8 SIMD instructions. Where it stops: on some CPUs (possibly your M1, estimate) ONNX Runtime's int8 kernels are not much faster than its fp32 ones, and the quantize/dequantize steps can eat the gain. Measure, do not assume.

Size check for this project: Qwen's embedding table is 151,936 x 896 ≈ 136M params, about a quarter of the model. If ORT quantizes the `Gather` op, it shrinks too; with `quant.op_types=[MatMul]` it stays fp32 and the file is less than 4x smaller.

## 3. Measuring honestly (your statistics edge)

- **Warm-up**: the first calls allocate memory and pick kernels. `train.py` skips 5 before timing.
- **Latency vs throughput**: batch 1 latency is what a user feels; big batches give throughput. Do not mix them in one claim.
- **Noise in accuracy**: 400 eval examples, accuracy ~0.85: standard error ≈ sqrt(0.85 x 0.15 / 400) ≈ 0.018. So fp32 vs int8 differences under ~2-3 points can be noise. Same examples for every variant, so a paired test (McNemar on the disagreements) is the right one, and much more sensitive than comparing two accuracies.
- Look at macro-F1 too: quantization can hurt a rare class (here `negative`) while accuracy barely moves.

## 4. Self-test (answer, then check above)

1. Why does `B` start at zero? What breaks if both `A` and `B` start random?
2. LoRA trains 0.1% of params. Why does training still need lots of RAM?
3. What does `alpha / r` buy you?
4. Merged vs unmerged adapter: one advantage each.
5. Per-tensor vs per-channel: explain with one outlier.
6. Dynamic vs static quantization: what is computed when?
7. Why is weight-only quantization so common for LLMs?
8. int8 model is 4x smaller but not faster. Name two reasons.
9. int8 accuracy drops by 1 point on 400 examples. Do you report a regression? Why?

Short answers: (1) so the model starts exactly as the base; random B adds noise to every layer at step 0. (2) activations. (3) stable update size across ranks, less LR retuning. (4) merged: zero inference cost; unmerged: swap adapters on one base. (5) section 2. (6) dynamic: activation scales at each call; static: from calibration, once. (7) activation outliers. (8) kernels not faster on this CPU; quantize/dequantize overhead; small batch so compute is not the bottleneck. (9) no, inside noise (SE ~1.8 points); run a paired test or more data.

## 5. If you have more time (primary sources)

Read only the named parts.
- LoRA, Hu et al. 2021, arXiv:2106.09685. Section 4 (method) and 7 (why low rank works).
- QLoRA, Dettmers et al. 2023, arXiv:2305.14314. Section 3 (NF4, double quantization).
- LLM.int8(), Dettmers et al. 2022, arXiv:2208.07339. The figure on outlier features.
- SmoothQuant, Xiao et al. 2022, arXiv:2211.10438. Figure 1 is the whole idea.
- GPTQ (arXiv:2210.17323) and AWQ (arXiv:2306.00978): know the one-line idea of each: GPTQ fixes rounding errors column by column with second-order info; AWQ protects the ~1% of weights tied to large activations.
