# qwantized-on-a-macbook

LoRA-fine-tune Qwen/Qwen2.5-0.5B for 3-class financial sentiment (financial_phrasebank) on a MacBook M1, then export it to ONNX, quantize it to int8 with ONNX Runtime and compare size, CPU latency and quality. Hydra holds the config; Weights & Biases tracks every run.

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/wandb login            # once; paste the key from https://wandb.ai/authorize
```

## Run

One run: train, merge, export, quantize, compare (about 5 minutes on an M1):

```bash
.venv/bin/python train.py
```

Print the resolved config without running:

```bash
.venv/bin/python train.py --cfg job
```

Override any field from the CLI:

```bash
.venv/bin/python train.py lora.r=16 lora.alpha=32
.venv/bin/python train.py 'lora.target_modules=[q_proj,k_proj,v_proj,o_proj]'
.venv/bin/python train.py train.lr=1e-4 train.epochs=2 data.train_size=3000
.venv/bin/python train.py quant.enabled=false          # train only
.venv/bin/python train.py quant.keep_onnx=true         # keep the ONNX files in outputs/
.venv/bin/python train.py wandb.mode=offline           # log locally; upload later with `wandb sync`
.venv/bin/python train.py data.train_size=64 data.eval_size=32   # smoke test, about 2 minutes
```

Sweep: one run per value, all in one W&B group:

```bash
.venv/bin/python train.py -m lora.r=4,8,16
.venv/bin/python train.py -m lora.r=4,16 lora.alpha=16,32   # grid: 4 runs
```

Single runs write to `outputs/<date>/<time>/`, sweeps to `multirun/<date>/<time>/<job>/`.

## Layout

- `conf/config.yaml`: picks one option per group, plus W&B and quantization settings.
- `conf/{model,data,lora,train}/`: config groups; add a file to add an option.
- `train.py`: the whole pipeline, top to bottom.
