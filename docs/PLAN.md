# Plan

## Purpose

Get hands-on with Hydra, W&B, LoRA and post-training quantization before an interview, by fine-tuning Qwen/Qwen2.5-0.5B for 3-class sentiment on a MacBook M1 and comparing fp32 against int8 ONNX variants. Budget: 2-3 hours, no paid GPU.

## What Taras asked for

1. `.venv` with pip, pinned `requirements.txt`, `.gitignore` for `outputs/`, `multirun/`, `wandb/`.
2. Hydra: `conf/config.yaml` with groups `model/`, `data/`, `lora/`, `train/`. Everything overridable from the CLI. Override examples and a sweep `lora.r=4,8,16`.
3. Train with Transformers + PEFT (`AutoModelForSequenceClassification` + `LoraConfig`) on a subset, under about 10 minutes per run.
4. W&B: resolved Hydra config, train loss, eval accuracy and macro-F1, confusion matrix, trainable vs total parameters. Sweep runs grouped.
5. Merge the adapter, export to ONNX, dynamic int8 quantization per-tensor and per-channel. Log a table to the same run: variant, size MB, CPU latency per sample, accuracy, macro-F1.
6. When it works: 5 experiments for Taras to run, each with the expected result and its mechanism. `NOTES.md` for his surprises.

## Agreed (2026-10-21)

- The AI writes the code; Taras reads it and must be able to explain every file.
- Data: `financial_phrasebank` from its parquet version. Fallback: `zeroshot/twitter-financial-news-sentiment` (also 3 classes), not `ag_news` (4 classes).
- ONNX files are deleted after their numbers are logged; only the last run keeps them.
- W&B: Taras's free account.
- GitHub: public repo, as an interview portfolio.

## Steps

(To be written next.)

Last step, always: clean up the Mac (venv, outputs, ONNX, Hugging Face caches).
