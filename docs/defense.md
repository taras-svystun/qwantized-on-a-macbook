# Defense pack

How to defend this project: a pitch, the evidence, conclusions at three levels, the code lines people point at, likely questions with answers, and the weak spots to own before anyone else names them. Theory lives in `docs/reading.md`; this file is about *this* project's evidence.

Labels used below: **[measured]** this repo produced the number; **[literature]** from a paper, check before quoting; **[estimate]** my reasoning, not measured.

## 0. The 90-minute plan

| Minutes | Do | Done when |
|---|---|---|
| 0-10 | Section 1: say the 2-minute pitch out loud twice, timed. | Under 2:15 without reading. |
| 10-30 | `docs/reading.md` sections 1-3, then its self-test. | 8 of 9 right. |
| 30-50 | Sections 2-3 here. For each senior conclusion, close the file and say the causal chain in 3 sentences. | You can draw the ablation table from memory. |
| 50-60 | Section 4 with `train.py` open: explain each line. | No line you would skip. |
| 60-80 | Section 5: read the question, answer aloud, then open the answer. Mark misses. | Misses re-done once. |
| 80-90 | Section 6 (weak spots), then the pitch once more. | Each weak spot has your one-line answer. |

## 1. Pitch

**30 seconds.** I fine-tuned Qwen2.5-0.5B with LoRA for financial sentiment on a MacBook, training 0.1% of the weights, and got 0.825 accuracy against 0.72 for a TF-IDF baseline. Then I quantized it to int8 with ONNX Runtime: 4x smaller, 3x faster on CPU. The interesting part is what I found when I asked *why* int8 changed 12% of the predictions: almost all the damage comes from activation outliers in one layer type, `down_proj`. I proved that with an ablation, not by assumption.

**2 minutes.** Add, in this order:
1. Stack: Hydra for config (every field overridable, sweeps with `-m`), W&B for tracking (resolved config, curves, confusion matrix, a comparison table per run, sweeps grouped).
2. Pipeline: LoRA on `q_proj`/`v_proj`, merge the adapter into the weights, export to ONNX, dynamic int8 per-tensor and per-channel.
3. Finding 1: quantizability depends on training. An undertrained model lost 14-19 points to int8; a well-trained one lost 2.5, because confident predictions have larger margins.
4. Finding 2: the loss is not spread evenly. The inputs of `down_proj` have channels 1500x larger than the median, so one int8 scale per tensor zeroes half of the values. Keeping just those 24 matrices in fp32 cuts the disagreements with fp32 by 60%.
5. Honesty: with 400 eval sentences the accuracy drop is not statistically significant (McNemar p = 0.18), so I report agreement with fp32 as well, and I name that the eval set was also used to pick the learning rate.

## 2. The evidence

All on the same 400 eval sentences (61% neutral). Single seed (42); the split is deterministic and the baseline run reproduced exactly twice. **[measured]**

**Main results**

| Model | Size | CPU latency, batch 1 | Accuracy | Macro-F1 |
|---|---|---|---|---|
| Majority class | - | - | 0.613 | - |
| TF-IDF + logistic regression | 73k params | 0.4 ms | 0.723 | 0.637 |
| Qwen + LoRA, fp32 ONNX | 1979 MB | 83-91 ms | 0.825 | 0.809 |
| int8 per-tensor | 497 MB | 28-31 ms | 0.800 | 0.760 |
| int8 per-channel | 499 MB | 28-30 ms | 0.800 | 0.753 |

**Training choice matters more than rank**

| Run | fp32 acc / F1 | int8 acc |
|---|---|---|
| lr 2e-4, 1 epoch | 0.660 / 0.549 | 0.470 (per-tensor), 0.522 (per-channel) |
| lr 1e-3, 2 epochs, r=4 / 8 / 16 | 0.787 / 0.825 / 0.815 | 0.76-0.81 |

**Where int8 hurts: ablation** (`analysis/quant_ablation.py`; all dynamic rows per-channel; "lost/won" = sentences only fp32 got right / only the variant got right)

| Variant | Size | Latency | Acc | F1 | Agree with fp32 | Lost / won | McNemar p |
|---|---|---|---|---|---|---|---|
| dynamic int8, all MatMuls | 499 MB | 28 ms | 0.800 | 0.753 | 0.882 | 28 / 18 | 0.18 |
| ... keep `down_proj` fp32 (24 of 169) | 812 MB | 45 ms | 0.815 | 0.790 | **0.953** | 11 / 7 | 0.48 |
| ... keep `gate`/`up` fp32 (48) | 1125 MB | 63 ms | 0.807 | 0.762 | 0.877 | 27 / 20 | 0.38 |
| ... keep attention fp32 (96) | 631 MB | 35 ms | 0.790 | 0.739 | 0.880 | 31 / 17 | 0.06 |
| weight-only int8 per-tensor (fake quant) | - | - | 0.805 | 0.782 | 0.973 | 9 / 1 | 0.02 |
| weight-only int8 per-channel (fake quant) | - | - | 0.807 | 0.794 | 0.978 | 8 / 1 | 0.04 |

**Why: activation statistics of the base model** (`analysis/activation_outliers.py`, 64 sentences)

| Input of | max \|x\| | Top channel / median channel | Values a per-tensor int8 scale rounds to 0 |
|---|---|---|---|
| `q_proj` (= `k`, `v`) | 47 | 21x | 17% |
| `o_proj` | 12 | 7x | 13% |
| `gate_proj` (= `up`) | 336 | 98x | 19% |
| `down_proj` | 1940 | 1552x | **48%** |

Prediction counts (neg / neu / pos): truth 45 / 245 / 110, fp32 51 / 233 / 116, int8 46 / 257 / 97. int8 drifts toward `neutral`.

## 3. Conclusions at three levels

### Level 1: literal (what happened)

- LoRA works: training 0.11% of the parameters (543k of 495M) lifted a random head to 0.825 accuracy in 5 minutes on an M1.
- Hyperparameters mattered: 5x the learning rate and 2x the steps took accuracy from 0.66 to 0.825.
- Quantization works: int8 is 4x smaller and 3x faster, and loses about 2.5 accuracy points.
- Per-channel and per-tensor ended up within noise of each other.
- Hydra and W&B together made the sweep a one-liner and the runs comparable.

### Level 2: interview (what you can claim and explain)

- "I have built the full path from fine-tuning to a deployable int8 artifact: LoRA with PEFT, merge, ONNX export, ONNX Runtime quantization, measured on CPU."
- "I can explain LoRA's parameter count by hand: `r * (d_in + d_out)` per layer, and why it saves optimizer memory but not activation memory."
- "I know the PTQ family: weight-only, dynamic, static, QAT, and I can say what each one computes when."
- "I measured honestly: warm-up, batch-1 latency, same eval set for every variant, macro-F1 next to accuracy because the classes are imbalanced."
- "I set up a reproducible experiment loop: config composition and sweeps in Hydra, all runs and their configs in W&B, pinned dependencies."

### Level 3: senior (causes, analogies, consequences)

Each one: claim, then the causal chain, then what it implies. The pattern behind all of them: **find the dominant term before you optimise, and know what your evidence can and cannot support.**

**S1. Quantizability is a property of the training run, not only of the architecture.** [measured]
Chain: int8 adds roughly fixed noise to the logits → a prediction flips only when the noise exceeds its margin (top logit minus the runner-up) → an undertrained model has small margins → it lost 14-19 points, the trained one 2.5.
Analogy: bit errors in a radio link depend on the signal-to-noise ratio. The quantizer fixes the noise, and training sets the signal. It is also the margin idea from SVMs: a large margin means robustness to perturbations.
Consequence: if the model ships in int8, pick hyperparameters by **int8** accuracy, not fp32. The best fp32 run need not be the best int8 run. Cheap idea: add a margin-promoting loss (label smoothing changes margins too, in either direction; that is worth testing).

**S2. The error is concentrated, so fix it locally.** [measured]
Chain: `down_proj` takes `SiLU(gate) * up`, a product of two activations, which makes the tails heavy → a few channels reach 1940 → dynamic quantization uses one scale per tensor (`max|x| / 127` ≈ 15) → every value under about 7.6 rounds to zero, which is 48% of them → keeping only those 24 matrices in fp32 cut disagreements from 47 to 19, while keeping 96 attention matrices in fp32 changed nothing.
Analogy: a max-based scale is like using the range instead of the IQR in statistics: one outlier sets the ruler for everyone. Photographers know it as dynamic range: one bright window and the room goes black. HDR is their per-region scale.
Consequence: mixed precision guided by measurement is the cheap fix. The principled fixes attack the outliers themselves. SmoothQuant moves the scale from activations into weights [literature]; LLM.int8() keeps outlier channels in fp16 [literature]; rotation methods (QuaRot, SpinQuant) multiply by an orthogonal matrix so the outlier energy spreads over all channels [literature]. That last one is the same trick as whitening before PCA, or dithering in audio.

**S3. Per-channel did not help because it optimised the smaller error term.** [measured]
Chain: weight-only quantization changes about 2-3% of predictions; dynamic quantization, which adds activation quantization, changes 12% → most of the error comes from activations → per-channel only refines the weights, so it barely moves the total.
Analogy: Amdahl's law for error. Speeding up 10% of the runtime cannot give you more than 10%.
Consequence: before choosing a technique, decompose the error by source. That is one ablation, here 3 minutes of compute.

**S4. Accuracy hid the change; agreement showed it.** [measured]
Chain: int8 flipped 47 of 400 decisions, 28 for the worse and 18 for the better → accuracy moved only 2.5 points, and McNemar says that is not significant (p = 0.18) → yet 12% of users would get a different answer.
Analogy: "prediction churn" when a recommender model is updated. Downstream systems and humans react to changed decisions, not to the aggregate.
Consequence: report agreement (or churn) with the reference model as a release metric, next to accuracy. Use paired tests (McNemar) for paired data. Two separate accuracies with error bars throw the pairing away.

**S5. The winner's curse: my fp32 baseline is optimistically biased.** [measured + reasoning]
Chain: I chose the learning rate by looking at this same eval set → fp32 is partly fitted to these 400 sentences, including the borderline ones it got right by luck → any perturbation pulls those back toward chance → weight-only quantization changed only 10 predictions, yet 9 of the 10 went wrong (p = 0.02). An unbiased reference would give a split closer to 50/50.
Analogy: regression to the mean. The pilots praised for a great landing do worse the next time. It is also why Kaggle has a private leaderboard.
Consequence: keep a test set you never look at while tuning. Here: split off 400 more sentences (the dataset has 4846) and report on those once.

**S6. Memory and compute are different budgets, with different levers.** [measured]
Chain: the embedding table is 27% of the parameters but a lookup with no FLOPs → quantizing it saved about 400 MB (907 → 499 MB) at no accuracy cost (from the earlier probe) and no speed change → the 3x speedup came from int8 `MatMul`s, because at batch 1 inference is memory-bound: you wait for weights to arrive, so fewer bytes mean less waiting [estimate for this M1].
Consequence: "4x smaller" and "3x faster" are separate claims with separate causes. On a server at large batch size the speedup would come from int8 compute instead, and could be smaller or larger [estimate].

**S7. The constraint was optimisation, not capacity.** [measured]
Chain: 4x more rank (r=4 → 16) moved accuracy by 3 points at most, inside noise, while 5x the learning rate moved it by 16 → 1500 sentences and 188 steps limit what the adapter learns, not its rank → the update this task needs is low-rank (the LoRA paper's "intrinsic dimension" claim [literature]).
Consequence: tune learning rate and steps before rank. Also, alpha and the learning rate are coupled through `alpha / r`, so a sweep over r with fixed alpha also changes the effective learning rate (experiment 1 in `docs/experiments.md` separates the two).

**S8. Quantization noise pushes predictions toward the prior.** [measured + reasoning]
Chain: int8 predicted `neutral` 257 times against 233 for fp32 (truth 245), at the cost of `positive` → noisier features carry less evidence → the decision falls back toward the majority class the head learned → macro-F1 falls twice as much as accuracy (-5.6 against -2.5 points).
Analogy: Bayesian updating: weaker evidence leaves the posterior closer to the prior.
Consequence: watch per-class recall after compression. A cheap fix to test: re-fit only the 3 logit biases on a small labelled set after quantization. It is like recalibration, and it costs seconds.

**S9. Is the LLM worth it? Ask the cost question first.** [measured + estimate]
Chain: TF-IDF gets 0.72 at 0.4 ms; the LLM gets 0.825 at 28 ms (int8), 70x slower, for +10 accuracy and +17 macro-F1 points → whether that is worth it depends on the cost of an error, which is a business number, not a model number.
Consequence: in production I would also try a small encoder (a FinBERT-class model of about 110M parameters, bidirectional, built for classification) [estimate: likely similar accuracy at a fraction of the latency], or distil the LLM into it. A decoder like Qwen classifies from the last token only, because with causal attention that is the only token that has seen the whole sentence.

## 4. Lines people will point at (`train.py`)

| Line | What and why |
|---|---|
| 38 | Under `-m` the W&B group is the sweep folder, so all runs of one sweep sit together. |
| 47 | `resolve=True`: W&B gets the values after interpolation, not strings like `${lora.r}`. |
| 53 | `latin-1`: the dataset's text file is not UTF-8; UTF-8 decoding would fail on some characters. |
| 59 | `rsplit("@", 1)`: split on the *last* `@`, because a sentence could contain `@`. |
| 74 | `eager` attention: the simplest attention code, which exports to ONNX cleanly. `sdpa` may be faster for training. |
| 76 | The head reads the hidden state at the last non-pad token. Without `pad_token_id` it cannot find it in a padded batch. |
| 77 | No KV cache: we do one forward pass, not generation. It also keeps cache tensors out of the ONNX graph. |
| 82 | `SEQ_CLS` makes PEFT keep the new `score` head trainable (`modules_to_save`). Otherwise the head would stay random and frozen. |
| 98 | The optimizer gets only the trainable 0.1%, so AdamW state is tiny. |
| 114 | Passing `labels` makes the model compute cross-entropy itself. |
| 117 | `zero_grad` after `step`: PyTorch adds gradients up by default. |
| 135 | Macro-F1 weighs the 3 classes equally; accuracy alone would reward predicting `neutral` (61%). |
| 149 | `merge_and_unload`: `W += (alpha / r) B A`, then drop PEFT, so ONNX sees a plain model. |
| 163 | Dynamic batch and sequence axes, so one file serves any input length. |
| 165 | External data: the fp32 weights are 1.98 GB, close to protobuf's 2 GB limit. |
| 188 | Warm-up: the first calls allocate memory and choose kernels, so they are not timed. |
| 191 | `perf_counter`: a monotonic clock with high resolution, suited to timing. |

## 5. Questions and answers

Answer aloud first, then open.

### Design

<details><summary>1. Why a 0.5B decoder for 3-class sentiment, and not BERT?</summary>

The goal was to practise LLM fine-tuning and quantization, which are the skills that transfer to bigger models. For this task alone I would benchmark a small encoder first (S9). I did measure the cheap baseline: TF-IDF gets 0.72 against 0.825.
</details>

<details><summary>2. How does a decoder-only model do classification?</summary>

`AutoModelForSequenceClassification` puts a linear layer (896 → 3) on the hidden state of the last non-pad token. With causal attention that token is the only one that has attended to the whole sentence. That is why `pad_token_id` must be set.
</details>

<details><summary>3. Why a hand-written loop and not HF Trainer?</summary>

To see every step: forward, loss, backward, step, zero_grad. Trainer would add LR schedules, gradient accumulation, mixed precision and checkpointing. I would use it, or Accelerate, for real multi-GPU work.
</details>

<details><summary>4. Isn't lr 1e-3 high?</summary>

For full fine-tuning, yes. LoRA trains few parameters and `B` starts at zero, so higher rates (1e-4 to 1e-3) are common. Here 2e-4 underfitted in 94 steps (0.66). I picked 1e-3 on the eval set, which biases it (S5).
</details>

<details><summary>5. Why fp32, not bf16, on the Mac?</summary>

[estimate] The M1 GPU has no native bf16, and fp16 training needs loss scaling to stay stable. At 0.5B, fp32 fits in 16 GB, so I chose correctness over speed. On a CUDA GPU I would use bf16.
</details>

### LoRA

<details><summary>6. Count the trainable parameters.</summary>

Per layer: `q_proj` 8 × (896 + 896) = 14,336, `v_proj` 8 × (896 + 128) = 8,192. Times 24 layers = 540,672, plus the head 896 × 3 = 2,688, total 543,360. That matches the log exactly.
</details>

<details><summary>7. Why did r=16 not beat r=8?</summary>

Two reasons. The task needs a low-rank update, and the limit was steps and data (S7). Also, with alpha fixed at 16, r=16 halves the scale `alpha / r`, which is like a lower learning rate. The difference is inside noise either way.
</details>

<details><summary>8. Merge first, then quantize: why that order?</summary>

So the quantizer sees the final weights. If you quantize the base and keep the adapter in float (QLoRA-style), you keep cheap adapter swapping but pay an extra matmul per layer, and the adapter never learned to compensate for the base's rounding. A better route: train LoRA *on* a quantized base, so the adapter learns to correct the quantization error. That is QAT-lite.
</details>

<details><summary>9. LoRA trains 0.1%; why does it still need much memory?</summary>

Activations. Backprop passes through the frozen layers, so their activations must be stored. LoRA saves gradients and optimizer state for frozen weights, not activations. Gradient checkpointing is the lever for activations.
</details>

### Quantization

<details><summary>10. What does dynamic quantization do at run time?</summary>

Weights were stored as int8 ahead of time with fixed scales. At each call, ORT finds the activation tensor's range, quantizes it to uint8 with one scale, runs an integer matmul, and dequantizes the result. No calibration data is needed, but there is one scale per tensor, which is the weak point (S2).
</details>

<details><summary>11. Why was per-channel no better than per-tensor?</summary>

Per-channel only improves the weights, and weights are the smaller error source: weight-only changes 2-3% of predictions, dynamic changes 12% (S3).
</details>

<details><summary>12. Why `down_proj`?</summary>

Its input is `SiLU(gate) * up`, a product of activations, which gives heavy tails. In layer 2 we measured channels up to 1940, 1552x the median. One per-tensor scale then zeroes 48% of the values. The literature calls these "massive activations" and finds that they act like fixed biases or attention sinks [literature: Sun et al. 2024].
</details>

<details><summary>13. How would you fix it without keeping fp32 layers?</summary>

Per-token activation scales; static quantization with percentile clipping on calibration data; SmoothQuant (divide activations by a per-channel factor, multiply the weights by it); rotations (QuaRot/SpinQuant) to spread the outliers. Or go weight-only (int4/int8 weights, float math), which sidesteps activation quantization altogether.
</details>

<details><summary>14. Why is int8 3x faster here, and when would it not be?</summary>

At batch 1, inference is memory-bound: 4x fewer weight bytes means less waiting, and the M1 has int8 dot-product instructions [estimate]. It would gain less if the quantize/dequantize overhead dominates (small matrices), if the CPU lacks fast int8 kernels, or on a GPU path without int8 kernels.
</details>

<details><summary>15. Is the int8 accuracy drop real?</summary>

Not proven at n = 400: McNemar on the 46 discordant sentences gives p = 0.18. The change in decisions is real, though: 12% of them. I would report both, and get a larger test set before claiming a regression.
</details>

### Measurement and MLOps

<details><summary>16. How exactly did you measure latency? What would you add?</summary>

ORT on CPU, batch 1, no padding, 5 warm-up calls, `perf_counter` around 400 calls, mean per sentence. I would add p50/p95 instead of the mean, a fixed thread count, sequence-length buckets, and repeats on an idle machine.
</details>

<details><summary>17. What does Hydra give you that argparse does not?</summary>

Composition: configs in groups, swapped with `lora=...`. Typed overrides of any nested field. Interpolation (`alpha=${lora.r}`). Multirun sweeps. A fresh output folder per run, with the exact config saved in `.hydra/`.
</details>

<details><summary>18. What exactly is in W&B, and how do you compare sweep runs?</summary>

The resolved config, train loss per step, eval accuracy and macro-F1, the confusion matrix, parameter counts, and a table with size, latency, accuracy and F1 per variant. Summary keys like `int8_per_channel/accuracy` make runs comparable in the runs table. Sweep runs share a `group`.
</details>

<details><summary>19. Is it reproducible?</summary>

Pinned top-level dependencies, a seed for Python and torch, and a deterministic split. The baseline reproduced to the same three decimals in two runs. Caveats: transitive dependencies are not pinned, and MPS kernels are not guaranteed to be bit-deterministic.
</details>

### Hostile

<details><summary>20. "You tuned on your eval set."</summary>

Yes, and I can show its effect (S5: weight-only lost 9, won 1). The fix is cheap: split off an untouched test set from the 2900 unused sentences and report on it once.
</details>

<details><summary>21. "One seed. How do you know r=8 beats r=4?"</summary>

I don't. The standard error is about 2 points, and r=4 vs r=8 differs by 3.8 (0.787 vs 0.825), which is borderline. I would run 3-5 seeds and a paired test before claiming it.
</details>

<details><summary>22. "Papers get higher accuracy on this dataset."</summary>

Different setups: they often use the all-agree subset (easier) or full fine-tuning with more data and epochs. Mine is 50-agree, 1500 sentences, 2 epochs, and was never a state-of-the-art attempt. The project is about the mechanics, and the error analysis.
</details>

<details><summary>23. "Why should a 0.5B toy tell us anything about real LLMs?"</summary>

The mechanisms carry over, and they get stronger with size: LLM.int8() reports outlier features becoming systematic around 6.7B parameters [literature]. The method carries over unchanged: measure the error by source, then fix the dominant one.
</details>

## 6. Weak spots: name them first

| Weak spot | Your line |
|---|---|
| Eval set used for tuning | "Biased, and I can show the bias; next step is a held-out test set." |
| One seed, n = 400 | "Differences under about 4 points are noise here; I use paired tests and agreement." |
| No encoder baseline | "TF-IDF is measured; an encoder is the first thing I would add." |
| Latency on one laptop | "Relative numbers only; on a server the ratio could change." |
| Fake-quant weight-only, not a real int8 kernel | "It isolates the weight error; it says nothing about speed." |
| Activation statistics from the base model, not the fine-tuned one | "The outliers come from the pretrained MLP weights, which LoRA leaves frozen; and the ablation on the fine-tuned model points to the same layer." |

## 7. What I would do next (value per hour, best first)

1. A held-out test set and 3 seeds: makes every number above defensible. About 30 minutes of compute.
2. Pick hyperparameters by int8 accuracy (S1), not fp32.
3. SmoothQuant or per-token activation scales, against the `down_proj`-in-fp32 mix (S2).
4. LoRA on all linear layers (experiment 3), then check whether it changes int8 robustness.
5. Distil into a small encoder and compare cost per correct prediction (S9).

## 8. Answers to the questions in NOTES.md

Plain words first; the one formula at the end of each answer is optional.

### Read these 3 first (the 20% that gives 80%)

**Q3. "LoRA does NOT save activations".**
Activations = the numbers that flow *between* layers during one forward pass (the `x` in `y = Wx`). Training is: forward pass, then backward pass. The backward pass goes through the layers in reverse and, at each layer, needs that layer's input from the forward pass. So all of them are kept in memory until backward ends.
Analogy: checking a long calculation backwards. You need every intermediate line written down, even for lines you will not change.
LoRA freezes 99.9% of the weights, so it saves the memory for *their* gradients and optimizer state. But the backward pass still walks through every frozen layer, so all the intermediate lines stay written down. Memory for them grows with batch size x sentence length, which is why our runs went to swap.
Your self-test answer 2 ("0.1% is still a lot") was wrong; this is the answer.

**Q6. Why TF-IDF and not the model without fine-tuning?** You caught a real gap.
The model without fine-tuning has no way to answer. `AutoModelForSequenceClassification` adds a brand new layer (896 numbers -> 3 class scores) with random weights. Before training it guesses at random: about 33%, worse than always saying `neutral` (61.3%). So "base model" as a baseline is just "random", which is why it was skipped.
What the fine-tune bought, with the numbers we have: random head ~0.33 -> majority class 0.613 -> TF-IDF 0.723 -> LoRA 0.825.
The fair question is "how much did LoRA add on top of just training the new head?". The comparison that answers it is a **linear probe**: freeze the whole model, train only the 3-class head. Not run yet (~5 min). Another fair baseline is zero-shot prompting of an instruct model ("Is this positive, negative or neutral?"), which is the agent-world way you already know.

**Q9. Why train/test and not train/val/test?** You caught a real gap again.
I picked the learning rate by looking at the eval set. Once you choose anything with a set, it is a validation set, and its score is too optimistic. We have no clean test set. Section 3, S5 shows the effect. Fix: put 400 of the 2900 unused sentences aside, never look at them while tuning, and report on them once at the end.
Interview line: "my eval set was used for tuning, so it is really a validation set; the honest number needs a held-out test set."

### The rest

**Q1. `q_proj` and `v_proj`.**
Attention lets each word gather information from other words. Each word's vector is turned into three vectors by three weight matrices:
- query (`q_proj`): "what am I looking for?"
- key (`k_proj`): "what do I contain?"
- value (`v_proj`): "what do I give if someone picks me?"
A word compares its query with every key, gets scores, and takes a weighted mix of the values.
Analogy: a library. Query = your request, keys = index cards, values = the books.
LoRA on `q` changes *what each word looks for*; LoRA on `v` changes *what it brings back*. The LoRA paper found this pair best for a small budget [literature].

**Q2. Low rank.**
High level: a weight matrix maps 896 input numbers to 896 output numbers. Its rank is how many independent "directions" it really uses. A full update could change all 896. The bet: a new task needs only a few. Sentiment needs maybe "good news vs bad news", "certain vs vague".
Analogy: a sound engineer. Re-recording the song = full fine-tuning. Moving 8 sliders on the equaliser = LoRA with r=8.
Low level: `ΔW = B @ A`. `A` (8 x 896) reads 8 numbers from the input: each row is one "detector", a dot product. `B` (896 x 8) writes those 8 numbers back into the 896 outputs. Everything passes through an 8-number bottleneck, so the update can never use more than 8 directions.
Cost for `q_proj`: 896 x 896 = 802,816 numbers for a full update; 8 x (896 + 896) = 14,336 with LoRA, 56x fewer.
Evidence here: r = 4, 8 and 16 gave the same accuracy within noise, so 4 directions were enough.

**Q4. Is it like PCA / SVD?** Yes, closely.
SVD writes any matrix as a sum of rank-1 pieces, sorted by size. Keeping the top `r` pieces gives the best rank-`r` copy of the matrix (Eckart-Young theorem). That is exactly how PCA keeps the top components.
LoRA's `B @ A` is a rank-`r` matrix too. The difference: SVD *computes* the best rank-`r` matrix from one you already have. LoRA does not have the full `ΔW` (that is what full fine-tuning would give you), so it *learns* a rank-`r` matrix directly with gradient descent.
The LoRA paper did this check: it took full updates, ran SVD, and found a few top directions carry most of it [literature]. A variant, PiSSA, starts `A` and `B` from the SVD of `W` itself [literature].

**Q5. Why `alpha / r` keeps the update size stable.**
Mechanism with Adam (our optimizer): each trainable number moves by about `lr` per step, whatever its gradient. One entry of the update is `ΔW_ij = B_i1 A_1j + ... + B_ir A_rj`, a sum of `r` terms, each growing by about `lr`. So with r=16 there are twice as many terms as with r=8, and the update grows roughly twice as fast. Dividing by `r` cancels that.
Analogy: a jury. Add up the votes and the result grows with the size of the jury; take the average and it does not.
Where it stops: terms are not perfectly aligned, so the growth is between `√r` and `r`. rsLoRA argues for `alpha / √r` [literature]. In our sweep alpha was fixed at 16, so r=16 had half the scale of r=8: like a smaller learning rate.

**Q7. fp32 vs int8. Why integers?**
- fp32: a 32-bit float, 4 bytes. It is scientific notation, `1.234 x 10^-3`: a huge range of sizes with ~7 digits of precision.
- int8: 1 byte, only 256 values, -128 to 127. Plus one float `scale` per matrix (or row): `real ≈ int x scale`. A ruler with 256 marks.
Why integers: 4x less memory, and chips multiply small integers faster and with less energy than floats.
Can we use small floats instead? Yes, and it is common: fp16 and bf16 (2 bytes) for training, fp8 on new GPUs, NF4 (4-bit) in QLoRA. Floats suit numbers that span many sizes; int + scale suits numbers in a known range.

**Q8. Per-channel: what are rows and columns?**
In PyTorch a linear layer computes `y = W x`, with `W` of shape (outputs, inputs), for `q_proj` (896, 896).
- Row `i`: the weights that make output number `i`. `y_i` = dot product of row `i` with the input. Think of a row as one detector.
- Column `j`: how input number `j` affects all outputs.
"Channel" = output = row. Per-channel gives each detector its own ruler.
Why rows and not columns: with one scale per row, `y_i = scale_i x (integer row · integer x)`. The scale comes out of the sum, so the whole dot product stays in fast integer math, with one multiply at the end. A scale per column would sit inside the sum and break that.

### Your self-test answers, corrected

1. B = 0: your intuition is right. Random `B` adds random noise to every layer, so the model starts damaged. Extra point: why not `A = 0` too? The gradient of `B` is proportional to `A`, and vice versa. Both zero means both gradients are zero forever. One zero, one random is the minimum that can learn.
2. Wrong: see Q3 (activations).
3. See Q5.
4. Merged: no extra cost at inference, and a plain model to deploy. Unmerged: one base model in memory plus many small adapters. 100 customers = 1 base (1 GB) + 100 adapters (2 MB each), not 100 models of 1 GB. You swap the adapter per request.
5. Correct.
6. Weights are always quantized once, ahead of time. Static vs dynamic is only about **activations**. To turn them into int8 you need their range. Dynamic: measure the range for each input, live. Static: measure ranges once on ~100 sample inputs (calibration) and freeze them. Analogy: camera auto-exposure for each shot vs exposure fixed from test shots.
7. Yes, activations can be quantized: they are the `x` in `y = Wx`. Fast integer math needs **both** sides as int8: int8 weights x int8 activations. Weight-only keeps `x` in float and turns the int8 weights back to float just before multiplying: memory saved, math the same. LLMs like weight-only because (a) at batch 1 the slow part is moving weights from memory, which weight-only already fixes, and (b) activations have outliers (our `down_proj`: up to 1940) that break int8.
8. Your answer is half right: the number of multiplications does not change. The other half: time = fetching weights from memory + multiplying. int8 makes fetching 4x cheaper. Analogy: moving house with the same number of boxes, each 4x lighter. Carrying is faster; unpacking is not. Here it *was* 3x faster [measured]. It would not be when converting int8 <-> float costs more than it saves, or when the chip has no fast int8 instructions.
9. Good, and senior: a small drop can be worth it for a 4x smaller model that fits a phone. Add the statistics: here p = 0.18, so the drop is not proven. But 12% of predictions changed, and users feel changed answers. "Not significant" does not mean "no difference".

### Small fixes to your explanations

- It is `(alpha / r) * B @ A`, not `r / alpha`. And 0.1% of weights, not 0.5%.
- LoRA does not take "the low rank of q and v". It adds a low-rank *update* next to the frozen q and v weights.
- "fp32 vs int8 showed no significant difference, which is nice": not significant on 400 sentences, yet 47 of 400 answers changed. Say both.
