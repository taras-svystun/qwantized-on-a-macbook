"""LoRA-fine-tune Qwen2.5-0.5B for 3-class sentiment, then export to ONNX,
quantize to int8 two ways, and compare everything in one W&B run.

Run: python train.py            (see README.md for overrides and sweeps)
"""

import os
import random
import shutil
import time
import zipfile

import hydra
import numpy as np
import onnxruntime as ort
import torch
import wandb
from huggingface_hub import hf_hub_download
from hydra.core.hydra_config import HydraConfig
from hydra.types import RunMode
from omegaconf import DictConfig, OmegaConf
from onnxruntime.quantization import QuantType, quantize_dynamic
from peft import LoraConfig, get_peft_model
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    hydra_cfg = HydraConfig.get()
    out_dir = hydra_cfg.runtime.output_dir  # outputs/<date>/<time> or multirun/<date>/<time>/<job>
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    # ---- W&B run: one run per Hydra job; all jobs of one sweep share a group.
    group = cfg.wandb.group
    if group is None:
        if hydra_cfg.mode == RunMode.MULTIRUN:
            group = "sweep-" + "_".join(hydra_cfg.sweep.dir.split("/")[-2:])
        else:
            group = "single"
    run = wandb.init(
        project=cfg.wandb.project,
        group=group,
        name=f"r{cfg.lora.r}_a{cfg.lora.alpha}_{'-'.join(cfg.lora.target_modules)}",
        mode=cfg.wandb.mode,
        config=OmegaConf.to_container(cfg, resolve=True),  # the full resolved Hydra config
        dir=out_dir,
    )

    # ---- Data: lines look like "Profit rose 20 percent .@positive".
    zip_path = hf_hub_download(cfg.data.repo, cfg.data.zip_file, repo_type="dataset")
    lines = zipfile.ZipFile(zip_path).read(cfg.data.txt_file).decode("latin-1").splitlines()
    labels = list(cfg.data.labels)
    examples = []
    for line in lines:
        if not line.strip():
            continue
        text, label = line.rsplit("@", 1)
        examples.append((text.strip(), labels.index(label.strip())))
    random.shuffle(examples)
    train_examples = examples[: cfg.data.train_size]
    eval_examples = examples[cfg.data.train_size : cfg.data.train_size + cfg.data.eval_size]
    eval_texts = [text for text, _ in eval_examples]
    eval_labels = [label for _, label in eval_examples]
    print(f"train {len(train_examples)}, eval {len(eval_examples)}")

    # ---- Model: base Qwen + a fresh 3-way linear head ("score") on the last token.
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.name)
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model.name,
        num_labels=len(labels),
        dtype=torch.float32,
        attn_implementation=cfg.model.attn_implementation,
    )
    model.config.pad_token_id = tokenizer.pad_token_id  # tells the head where each sentence ends
    model.config.use_cache = False  # no KV cache: we classify, we do not generate

    # ---- LoRA: freeze everything, add B @ A next to each target layer.
    # task_type SEQ_CLS also keeps the new "score" head fully trainable.
    lora_config = LoraConfig(
        task_type="SEQ_CLS",
        r=cfg.lora.r,
        lora_alpha=cfg.lora.alpha,
        lora_dropout=cfg.lora.dropout,
        target_modules=list(cfg.lora.target_modules),
    )
    model = get_peft_model(model, lora_config)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    run.summary["params/trainable"] = trainable_params
    run.summary["params/total"] = total_params
    run.summary["params/trainable_pct"] = 100 * trainable_params / total_params
    print(f"trainable {trainable_params:,} of {total_params:,} ({100 * trainable_params / total_params:.2f}%)")

    # ---- Train: a plain PyTorch loop.
    model.to(cfg.device)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=cfg.train.lr)
    step = 0
    start_time = time.time()
    for epoch in range(cfg.train.epochs):
        model.train()
        random.shuffle(train_examples)
        for start in range(0, len(train_examples), cfg.train.batch_size):
            batch = train_examples[start : start + cfg.train.batch_size]
            inputs = tokenizer(
                [text for text, _ in batch],
                padding=True,
                truncation=True,
                max_length=cfg.model.max_length,
                return_tensors="pt",
            ).to(cfg.device)
            targets = torch.tensor([label for _, label in batch], device=cfg.device)
            loss = model(**inputs, labels=targets).loss  # cross-entropy on the 3 logits
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            step += 1
            wandb.log({"train/loss": loss.item(), "train/epoch": epoch}, step=step)

        # Eval after each epoch, in PyTorch.
        model.eval()
        preds = []
        with torch.no_grad():
            for start in range(0, len(eval_texts), cfg.train.batch_size):
                inputs = tokenizer(
                    eval_texts[start : start + cfg.train.batch_size],
                    padding=True,
                    truncation=True,
                    max_length=cfg.model.max_length,
                    return_tensors="pt",
                ).to(cfg.device)
                preds += model(**inputs).logits.argmax(-1).tolist()
        accuracy = accuracy_score(eval_labels, preds)
        macro_f1 = f1_score(eval_labels, preds, average="macro")
        wandb.log({"eval/accuracy": accuracy, "eval/macro_f1": macro_f1}, step=step)
        print(f"epoch {epoch}: accuracy {accuracy:.3f}, macro-F1 {macro_f1:.3f}, {time.time() - start_time:.0f}s")

    run.summary["train/seconds"] = time.time() - start_time
    wandb.log(
        {"eval/confusion_matrix": wandb.plot.confusion_matrix(y_true=eval_labels, preds=preds, class_names=labels)},
        step=step,
    )
    if not cfg.quant.enabled:
        run.finish()
        return

    # ---- Merge: W <- W + (alpha / r) * B @ A. The model is a plain Qwen again, no PEFT.
    model = model.merge_and_unload().to("cpu").eval()

    # ---- Export fp32 ONNX. Batch and sequence length stay dynamic.
    onnx_dir = os.path.join(out_dir, "onnx")
    os.makedirs(onnx_dir, exist_ok=True)
    fp32_path = os.path.join(onnx_dir, "fp32.onnx")
    example = tokenizer(["Profit rose 20 percent ."], return_tensors="pt")
    torch.onnx.export(
        model,
        (),
        fp32_path,
        kwargs={"input_ids": example["input_ids"], "attention_mask": example["attention_mask"]},
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_shapes={"input_ids": {0: "batch", 1: "seq"}, "attention_mask": {0: "batch", 1: "seq"}},
        dynamo=True,
        external_data=True,  # weights go to fp32.onnx.data: a single protobuf file is capped at 2 GB
    )

    # ---- Dynamic int8: weights become int8 now; activation scales are computed per call.
    variants = {"fp32": fp32_path}
    for name, per_channel in [("int8_per_tensor", False), ("int8_per_channel", True)]:
        path = os.path.join(onnx_dir, f"{name}.onnx")
        quantize_dynamic(
            fp32_path,
            path,
            per_channel=per_channel,
            weight_type=QuantType.QInt8,
            op_types_to_quantize=list(cfg.quant.op_types) if cfg.quant.op_types else None,  # None: all it can
            use_external_data_format=True,
        )
        variants[name] = path

    # ---- Compare on CPU, one sentence at a time (batch 1, no padding).
    table = wandb.Table(columns=["variant", "size_mb", "latency_ms", "accuracy", "macro_f1"])
    for name, path in variants.items():
        size_mb = (os.path.getsize(path) + os.path.getsize(path + ".data")) / 1e6
        session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        encoded = [tokenizer(text, truncation=True, max_length=cfg.model.max_length, return_tensors="np") for text in eval_texts]
        for inputs in encoded[:5]:  # warm-up: first calls allocate memory
            session.run(None, {"input_ids": inputs["input_ids"], "attention_mask": inputs["attention_mask"]})
        preds = []
        start = time.perf_counter()
        for inputs in encoded:
            logits = session.run(None, {"input_ids": inputs["input_ids"], "attention_mask": inputs["attention_mask"]})[0]
            preds.append(int(np.argmax(logits[0])))
        latency_ms = 1000 * (time.perf_counter() - start) / len(encoded)
        accuracy = accuracy_score(eval_labels, preds)
        macro_f1 = f1_score(eval_labels, preds, average="macro")
        table.add_data(name, size_mb, latency_ms, accuracy, macro_f1)
        run.summary[f"{name}/size_mb"] = size_mb  # summary keys let the UI compare runs side by side
        run.summary[f"{name}/latency_ms"] = latency_ms
        run.summary[f"{name}/accuracy"] = accuracy
        run.summary[f"{name}/macro_f1"] = macro_f1
        print(f"{name:17s} {size_mb:7.0f} MB {latency_ms:6.1f} ms  acc {accuracy:.3f}  F1 {macro_f1:.3f}")
    wandb.log({"quant/comparison": table}, step=step)

    if not cfg.quant.keep_onnx:
        shutil.rmtree(onnx_dir)
    run.finish()


if __name__ == "__main__":
    main()
