# Plan

## Purpose

Get hands-on with Hydra, W&B, LoRA and post-training quantization before an interview, by fine-tuning Qwen/Qwen2.5-0.5B for 3-class sentiment on a MacBook M1 and comparing fp32 against int8 ONNX variants. Budget: 2-3 hours, no paid GPU.

## What Taras asked for

1. [x] `.venv` with pip, pinned `requirements.txt`, `.gitignore` for `outputs/`, `multirun/`, `wandb/`.
2. [x] Hydra: `conf/config.yaml` with groups `model/`, `data/`, `lora/`, `train/`. Everything overridable from the CLI. Override examples and a sweep `lora.r=4,8,16`.
3. [x] Train with Transformers + PEFT (`AutoModelForSequenceClassification` + `LoraConfig`) on a subset, under about 10 minutes per run.
4. [x] W&B: resolved Hydra config, train loss, eval accuracy and macro-F1, confusion matrix, trainable vs total parameters. Sweep runs grouped.
5. [x] Merge the adapter, export to ONNX, dynamic int8 quantization per-tensor and per-channel. Log a table to the same run: variant, size MB, CPU latency per sample, accuracy, macro-F1.
6. [x] When it works: 5 experiments for Taras to run, each with the expected result and its mechanism. `NOTES.md` for his surprises.

## Agreed (2026-10-21)

- The AI writes the code; Taras reads it and must be able to explain every file.
- Data: `financial_phrasebank`, `Sentences_50Agree.txt` read straight from the zip in its HF repo (the repo has no parquet version, and its loading script no longer runs). Fallback if that breaks: `zeroshot/twitter-financial-news-sentiment` (also 3 classes), not `ag_news` (4 classes).
- Training loop: plain PyTorch, not HF `Trainer`, so every step (forward, loss, backward, optimizer) is visible.
- ONNX files are deleted after their numbers are logged; only the last run keeps them.
- W&B: Taras's free account.
- GitHub: public repo, as an interview portfolio.

## Steps

Each step has its check. Steps 1-7 are done (2026-10-21); see `docs/journal.md`.

1. Environment: `.venv`, pinned `requirements.txt`. Check: imports work, MPS is available.
2. ONNX spike: export the plain Qwen classifier to ONNX and run it in ONNX Runtime. Check: ONNX logits match PyTorch to about 1e-3.
3. Hydra config: `conf/` with groups `model/`, `data/`, `lora/`, `train/`. Check: `python train.py --cfg job` prints the resolved config, and an override changes it.
4. Training + W&B: `train.py` trains LoRA and logs config, loss, accuracy, macro-F1, confusion matrix and parameter counts. Check: one run under 10 minutes, all of these visible in W&B.
5. Merge, export, quantize: fp32, int8 per-tensor and int8 per-channel ONNX, plus a comparison table in the same W&B run. Check: the table has 3 rows; the int8 files are about 4x smaller.
6. Sweep: `-m lora.r=4,8,16`, grouped in W&B. Check: 3 runs in one group.
7. [x] Five experiments for Taras, with the expected result and its mechanism, in `docs/experiments.md`.

Last step, always: clean up the Mac (venv, outputs, ONNX, Hugging Face caches).
