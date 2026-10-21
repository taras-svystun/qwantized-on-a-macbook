# Lora-avec-Quantization

A small learning lab: LoRA-fine-tune Qwen2.5-0.5B for 3-class sentiment, then quantize it with ONNX Runtime and compare, with Hydra for config and W&B for tracking. It exists to prepare Taras for an ML interview.

- `GLOSSARY.md`: the project's words. Use them in code, docs and commits.
- `docs/PLAN.md`: the plan and what we agreed. Read it when starting a step.
- `docs/journal.md`: what each finished step did and why.
- `NOTES.md`: Taras's own notes on what surprised him. Only he writes there; never edit his entries.
- `docs/adr/`: decisions that are hard to undo (created when the first one is needed).

## Learning lab rules

- Taras is learning Hydra, W&B, LoRA and post-training quantization. The AI writes the code; Taras owns the ideas and must be able to explain every line.
- Keep code small and explicit: plain top-to-bottom scripts, no helper classes or wrapper abstractions.
- Before writing each code file, explain in 2-3 sentences what it does and why.
- Define each term the first time it appears, and add it to `GLOSSARY.md`.
- Explain each new piece by its mechanism: the idea, how the code works step by step, and why it behaves as it does.

## Constraints

- Runs on a MacBook Pro M1 (2020), 16 GB RAM, no paid GPU. One training run must take under about 10 minutes.
- Disk is tight. Large exports (ONNX files) must not pile up across runs: delete them once their numbers are logged to W&B, keep only the last run's.
- The repo is a public portfolio on GitHub. No secrets or personal data in files or commits.
- When the project is done, clean up the Mac: delete `.venv/`, `outputs/`, `multirun/`, `wandb/`, ONNX files and this project's Hugging Face caches. List them with sizes first.

## Finishing a step

- `README.md` has a copy-paste command for every way to run the project.
