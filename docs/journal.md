# Journal

Newest at the bottom.

## 2026-10-21: interview reading guide

Added `docs/reading.md`: LoRA and quantization by mechanism, with numbers from this model, a self-test and primary sources. Why: Taras reads it while runs train, to close his interview gaps. New terms went to `GLOSSARY.md`.

## 2026-10-21: setup, spike, pipeline

- Set up the learning-lab docs, a public GitHub repo, and noreply commit author.
- Data: financial_phrasebank has no parquet version and its loading script no longer runs with `datasets` 5, so `train.py` reads `Sentences_50Agree.txt` from the zip in the HF repo. `datasets` is not needed.
- Spike: `torch.onnx.export(dynamo=True, external_data=True)` exports Qwen2 with `eager` attention on the first try; ONNX logits match PyTorch to 6e-5. The fp32 file is 1.98 GB, so external data is required (the protobuf limit is 2 GB).
- Pipeline in one script, plain PyTorch loop instead of HF `Trainer` so each step is visible.
- First defaults (lr 2e-4, 1 epoch) gave a weak model: accuracy 0.66, and int8 fell to 0.47-0.52. With lr 1e-3 and 2 epochs: fp32 0.825 / F1 0.809, int8 0.800. A model with larger logit margins survives quantization noise better. A run takes about 8 minutes.
- Probe of the int8 drop (agreement with fp32 on 300 sentences, per-channel): all ops 0.873, MatMul only (embeddings fp32) 0.883, head excluded 0.873. So neither the embedding nor the head causes the drop; the transformer MatMuls do, most likely through per-tensor dynamic activation scales.

## 2026-10-21: sweep and experiments

- Sweep `-m lora.r=4,8,16`, run offline: fp32 accuracy 0.787 / 0.825 / 0.815. All three runs carry the W&B group `sweep-2026-10-21_11-39-44`; ONNX files were deleted after each run.
- Hydra sets the root logger to INFO, which printed thousands of ONNX-exporter lines; `hydra.job_logging.root.level` is now WARNING.
- Added `quant.op_types` so experiment 5 can keep embeddings in fp32.
- `docs/experiments.md`: five experiments with the expected result and the mechanism.
