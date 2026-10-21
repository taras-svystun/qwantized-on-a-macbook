"""How heavy-tailed are the inputs of each Linear layer in Qwen2.5-0.5B?

Dynamic int8 quantizes each activation tensor with ONE scale (max |x| / 127).
If a few channels are huge, that scale is huge and small values round to 0.
This script measures that, per layer type, on 64 real sentences.

Run: .venv/bin/python analysis/activation_outliers.py
"""

import zipfile

import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer

name = "Qwen/Qwen2.5-0.5B"
tokenizer = AutoTokenizer.from_pretrained(name)
model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32).eval()

zip_path = hf_hub_download("takala/financial_phrasebank", "data/FinancialPhraseBank-v1.0.zip", repo_type="dataset")
lines = zipfile.ZipFile(zip_path).read("FinancialPhraseBank-v1.0/Sentences_50Agree.txt").decode("latin-1").splitlines()
texts = [line.rsplit("@", 1)[0] for line in lines if line.strip()][:64]

# For each Linear layer, collect the absolute max of its input per channel (over all tokens),
# and the share of input values that a per-tensor int8 scale rounds to zero.
stats = {}  # layer name -> list of (channel_absmax, zero_fraction) per sentence


def make_hook(layer_name):
    def hook(module, inputs, output):
        x = inputs[0][0]  # (tokens, channels), batch 1
        channel_absmax = x.abs().amax(dim=0)
        scale = x.abs().max() / 127
        zero_fraction = ((x / scale).round() == 0).float().mean().item()
        stats.setdefault(layer_name, []).append((channel_absmax, zero_fraction))

    return hook


for layer_name, module in model.named_modules():
    if isinstance(module, torch.nn.Linear) and ".layers." in layer_name:
        module.register_forward_hook(make_hook(layer_name))

with torch.no_grad():
    for text in texts:
        model(**tokenizer(text, return_tensors="pt"))

# Group by layer type. The input of q/k/v is the same tensor, so show q_proj only.
print(f"{'input of':10s} {'max|x|':>8s} {'top/median channel':>19s} {'rounded to 0':>13s}")
for kind in ["q_proj", "o_proj", "gate_proj", "down_proj"]:
    names = [n for n in stats if n.endswith(kind)]
    max_abs, ratios, zeros = [], [], []
    for n in names:
        absmax = torch.stack([s[0] for s in stats[n]]).amax(dim=0)  # per channel, over sentences
        max_abs.append(absmax.max().item())
        ratios.append((absmax.max() / absmax.median()).item())
        zeros += [s[1] for s in stats[n]]
    print(f"{kind:10s} {max(max_abs):8.0f} {max(ratios):19.0f}x {sum(zeros) / len(zeros):12.0%}")

# Worst single layer for down_proj, the usual home of "massive activations".
worst = max((n for n in stats if n.endswith("down_proj")), key=lambda n: max(s[0].max().item() for s in stats[n]))
print("largest down_proj input:", worst)
