# Five experiments

Baseline (defaults: r 8, alpha 16, q_proj + v_proj, lr 1e-3, 2 epochs, 1500 train / 400 eval):
fp32 accuracy 0.825, macro-F1 0.809; int8 accuracy 0.800, macro-F1 about 0.76. One run takes about 8 minutes.

Measured sweep `lora.r=4,8,16` (alpha fixed at 16):

| r | trainable | fp32 acc / F1 | int8 per-tensor acc / F1 | int8 per-channel acc / F1 |
|---|---|---|---|---|
| 4 | 273k | 0.787 / 0.790 | 0.760 / 0.746 | 0.790 / 0.726 |
| 8 | 543k | 0.825 / 0.809 | 0.800 / 0.760 | 0.800 / 0.753 |
| 16 | 1.08M | 0.815 / 0.799 | 0.807 / 0.759 | 0.800 / 0.742 |

The eval set has 400 sentences, so one sentence is 0.25 points, and the noise between seeds is probably 1-2 points. Expectations below are estimates, not results. Write what you see in `NOTES.md`.

## 1. Rank with the scale held fixed

```bash
.venv/bin/python train.py -m lora.r=4,16,64 'lora.alpha=${lora.r}'
```

`${lora.r}` is a Hydra interpolation, so alpha follows r and the scale `alpha / r` stays 1.

**Expect:** accuracy within about 2 points across r; trainable parameters grow 16x. Compared with the measured sweep, r=16 should not fall below r=8 here.

**Mechanism:** the LoRA output is `(alpha / r) * B @ A @ x`. In the sweep above alpha was fixed, so going from r=8 to r=16 halved the scale, which acts like a smaller learning rate for the adapter. Holding `alpha / r` fixed removes that confound. What is left is capacity, and a 3-class task needs very little: the needed weight change is low-rank. **Where it stops:** with a much bigger r and a small scale, Adam's step size no longer matches the scale. That is why rsLoRA uses `alpha / sqrt(r)`.

## 2. alpha at fixed r

```bash
.venv/bin/python train.py -m lora.r=8 lora.alpha=4,16,64
```

**Expect:** alpha=4 learns slowly (lower accuracy after 2 epochs, loss falls late). alpha=64 makes the loss fall fastest in the first steps, then gives a noisier curve, and maybe lower accuracy than 16.

**Mechanism:** Adam makes each step of `A` and `B` about `lr` in size, whatever the gradient scale. The change in output is then `alpha / r` times that step, so alpha works like a multiplier on the adapter's learning rate. The new `score` head is not scaled by alpha, so it learns at the same speed in all three runs.

## 3. Which layers get LoRA

```bash
.venv/bin/python train.py 'lora.target_modules=[q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj]'
```

**Expect:** about 4.4M trainable parameters (8x the baseline), +1 to +4 accuracy points, and each step about 1.3-1.5x slower.

**Mechanism:** a LoRA layer adds `r * (d_in + d_out)` parameters. `q_proj` and `v_proj` are small: `v_proj` maps 896 -> 128, because Qwen2.5 uses grouped-query attention with 2 KV heads. The MLP layers map 896 <-> 4864 and hold most of the model's weights, so adapting them adds the most capacity per layer. Step time grows less than the parameter count, because the frozen forward pass still dominates the cost.

## 4. Quantization granularity vs model quality

```bash
.venv/bin/python train.py train.lr=2e-4 train.epochs=1    # weak model
.venv/bin/python train.py                                  # strong model (baseline)
```

Compare `int8_per_tensor` and `int8_per_channel` in the `quant/comparison` table of each run.

**Expect** (I saw this once, while building): the weak model loses 14-19 accuracy points to int8 (0.66 -> 0.47 / 0.52), and per-channel loses less. The strong model loses about 2.5 points, and the two granularities are within noise. Sizes differ by about 2 MB: per-channel stores one fp32 scale per output column, which is a few KB per matrix.

**Mechanism:** int8 adds roughly constant noise to the logits. A sentence flips only when that noise is bigger than its margin (top logit minus the runner-up). A weak model has small margins, so many flip. Per-channel shrinks the weight error, because one large column no longer inflates the scale of the whole matrix. But in dynamic quantization the activations are still quantized per-tensor at run time, and LLM activations have a few huge outlier channels. Once the weight error is small, that activation error dominates, which is why per-channel stops helping.

## 5. Keep the embeddings in fp32

```bash
.venv/bin/python train.py quant.op_types=[MatMul]
```

**Expect** (measured once while building: agreement with fp32 went from 0.873 to 0.883): size about 907 MB instead of 499, accuracy the same within 1 point, latency the same.

**Mechanism:** the embedding table has 151936 x 896 = 136M weights, 27% of the model, but it is a lookup (`Gather`), not a matrix multiply. Quantizing it saves memory and costs almost no accuracy: each token reads one row, and that row's int8 error is small next to what the 24 layers do after it. It also does not change speed, because a lookup has no FLOPs to speed up. The speedup comes from the int8 `MatMul`s.
