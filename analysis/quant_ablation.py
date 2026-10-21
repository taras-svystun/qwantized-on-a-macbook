"""Where does the int8 accuracy loss come from? A causal test, one change at a time.

Needs the ONNX files of a trained run:
    .venv/bin/python train.py quant.keep_onnx=true
    .venv/bin/python analysis/quant_ablation.py outputs/<date>/<time>/onnx

Variants:
- dynamic int8 everywhere (what train.py does), then the same but keeping one layer kind in fp32;
- weight-only "fake quant": weights rounded to int8 and back to float, activations untouched.
  If weight-only is near fp32, the loss comes from quantizing activations.
Each variant is compared with fp32 on the same 400 sentences: accuracy, macro-F1,
agreement, and an exact McNemar test on the sentences where the two disagree.
"""

import glob
import os
import random
import sys
import time
import zipfile

import numpy as np
import onnx
import onnxruntime as ort
from huggingface_hub import hf_hub_download
from onnx import numpy_helper
from onnxruntime.quantization import QuantType, quantize_dynamic
from scipy.stats import binomtest
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoTokenizer

onnx_dir = sys.argv[1]
fp32_path = os.path.join(onnx_dir, "fp32.onnx")

# ---- Same eval split as train.py (wandb.init does not touch Python's random state).
random.seed(42)
zip_path = hf_hub_download("takala/financial_phrasebank", "data/FinancialPhraseBank-v1.0.zip", repo_type="dataset")
lines = zipfile.ZipFile(zip_path).read("FinancialPhraseBank-v1.0/Sentences_50Agree.txt").decode("latin-1").splitlines()
labels = ["negative", "neutral", "positive"]
examples = []
for line in lines:
    if not line.strip():
        continue
    text, label = line.rsplit("@", 1)
    examples.append((text.strip(), labels.index(label.strip())))
random.shuffle(examples)
eval_examples = examples[1500:1900]
y = np.array([label for _, label in eval_examples])
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
encoded = [tokenizer(text, truncation=True, max_length=64, return_tensors="np") for text, _ in eval_examples]

# ---- Sort the MatMul nodes by the shape of their weight [d_in, d_out].
graph = onnx.load(fp32_path, load_external_data=False).graph
weight_shape = {init.name: tuple(init.dims) for init in graph.initializer}
kind_by_shape = {(896, 896): "attention", (896, 128): "attention", (896, 4864): "gate_up", (4864, 896): "down", (896, 3): "head"}
nodes_by_kind = {}
weight_names = []
for node in graph.node:
    if node.op_type == "MatMul" and node.input[1] in weight_shape:
        kind = kind_by_shape[weight_shape[node.input[1]]]
        nodes_by_kind.setdefault(kind, []).append(node.name)
        weight_names.append(node.input[1])
print({kind: len(nodes) for kind, nodes in nodes_by_kind.items()})

results = []
fp32_preds = None
for name in ["fp32", "dynamic_all", "dynamic_keep_down_fp32", "dynamic_keep_gate_up_fp32",
             "dynamic_keep_attention_fp32", "weight_only_per_tensor", "weight_only_per_channel"]:
    path = os.path.join(onnx_dir, f"ablation_{name}.onnx")
    if name == "fp32":
        path = fp32_path
    elif name.startswith("dynamic"):
        exclude = []
        if name != "dynamic_all":
            exclude = nodes_by_kind[name.removeprefix("dynamic_keep_").removesuffix("_fp32")]
        quantize_dynamic(fp32_path, path, per_channel=True, weight_type=QuantType.QInt8,
                         nodes_to_exclude=exclude, use_external_data_format=True)
    else:
        # Round every MatMul weight to int8 and back; the graph stays fp32.
        model = onnx.load(fp32_path)
        for init in model.graph.initializer:
            if init.name in weight_names:
                w = numpy_helper.to_array(init)
                if name.endswith("per_tensor"):
                    scale = np.abs(w).max() / 127
                else:
                    scale = np.abs(w).max(axis=0, keepdims=True) / 127  # one scale per output column
                init.CopyFrom(numpy_helper.from_array((np.round(w / scale) * scale).astype(np.float32), init.name))
        onnx.save(model, path, save_as_external_data=True, location=os.path.basename(path) + ".data")
        del model

    session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    for inputs in encoded[:5]:
        session.run(None, {"input_ids": inputs["input_ids"], "attention_mask": inputs["attention_mask"]})
    preds = []
    start = time.perf_counter()
    for inputs in encoded:
        logits = session.run(None, {"input_ids": inputs["input_ids"], "attention_mask": inputs["attention_mask"]})[0]
        preds.append(int(np.argmax(logits[0])))
    latency_ms = 1000 * (time.perf_counter() - start) / len(encoded)
    preds = np.array(preds)
    size_mb = sum(os.path.getsize(f) for f in glob.glob(path + "*")) / 1e6
    if fp32_preds is None:
        fp32_preds = preds
    # McNemar: among sentences where exactly one model is right, is the split 50/50?
    fp32_only = int(np.sum((fp32_preds == y) & (preds != y)))
    variant_only = int(np.sum((fp32_preds != y) & (preds == y)))
    p_value = binomtest(fp32_only, fp32_only + variant_only).pvalue if fp32_only + variant_only else 1.0
    row = (name, size_mb, latency_ms, accuracy_score(y, preds), f1_score(y, preds, average="macro"),
           np.mean(preds == fp32_preds), fp32_only, variant_only, p_value, np.bincount(preds, minlength=3))
    results.append(row)
    print(f"{name:28s} {size_mb:5.0f} MB {latency_ms:5.1f} ms acc {row[3]:.3f} F1 {row[4]:.3f} "
          f"agree {row[5]:.3f} lost {fp32_only:2d} won {variant_only:2d} p {p_value:.3f} counts {row[9]}", flush=True)
    if path != fp32_path:
        for f in glob.glob(path + "*"):
            os.remove(f)

print("true counts", np.bincount(y, minlength=3))
