# Glossary

Lora-avec-Quantization: a learning lab that LoRA-fine-tunes a small LLM for sentiment classification, then quantizes it and compares speed, size and quality.

- **LoRA** (Low-Rank Adaptation): freeze the base weight `W` and learn a small update `B @ A`, where `A` is `r x d_in` and `B` is `d_out x r`. Only `A` and `B` train.
- **rank `r`**: the inner size of the LoRA update. Bigger `r` means more trainable parameters and more capacity.
- **alpha**: LoRA scale. The update is multiplied by `alpha / r`.
- **target modules**: the layers (for example `q_proj`, `v_proj`) that get a LoRA update.
- **merge**: fold `W + (alpha/r) * B @ A` into one plain weight, so the model runs without PEFT.
- **PTQ** (post-training quantization): turn trained float weights into low-bit integers without retraining.
- **dynamic int8 quantization**: weights are stored as int8 ahead of time; activation scales are computed on the fly at each inference.
- **per-tensor**: one scale (and zero point) for a whole weight matrix.
- **per-channel**: one scale per output channel (row) of a weight matrix.
- **Hydra config group**: a folder under `conf/` (for example `lora/`) whose files are swappable options.
- **multirun**: Hydra's `-m` mode, which runs the script once per value in a sweep, for example `lora.r=4,8,16`.
- **W&B group**: a W&B run field that puts related runs (for example one sweep) together in the UI.
- **ONNX**: a framework-neutral file format for a model's compute graph and weights. ONNX Runtime runs it.
- **external data**: ONNX weights stored in a side file (`model.onnx.data`), because one protobuf file is capped at 2 GB.
- **macro-F1**: the F1 score per class, averaged with equal weight, so a rare class counts as much as a common one.
- **score head**: the new `896 -> 3` linear layer that `AutoModelForSequenceClassification` puts on the last non-pad token.
- **static quantization**: activation scales are fixed ahead of time from a small calibration set.
- **QAT** (quantization-aware training): rounding is simulated during training, so the model adapts to it.
- **weight-only quantization**: weights are stored in low bits; activations and math stay in float. GPTQ and AWQ are examples.
- **QLoRA**: LoRA on top of a base model stored in 4-bit NF4.
- **activation outliers**: a few hidden channels in LLMs with values ~100x the rest; they make activation quantization hard.
- **margin**: top logit minus the runner-up. A prediction flips under noise only when the noise exceeds the margin.
- **agreement**: share of inputs where a variant predicts the same class as fp32. One minus agreement is the **churn**.
- **McNemar test**: a paired test on the inputs where exactly one of two models is right; asks if those split 50/50.
- **fake quantization**: round weights to int8 and back to float, so the graph stays fp32. Isolates the rounding error from kernel speed.
- **activation outliers**: a few channels of a layer's input with values hundreds of times the median; they ruin a per-tensor activation scale.
- **mixed precision (quantization)**: quantize most layers, keep the sensitive ones in fp32.
- **activations**: the numbers that flow between layers in a forward pass (the `x` in `y = Wx`). Training keeps them for the backward pass.
- **linear probe**: freeze the whole model and train only a new head on top. It shows how much the frozen features already know.
- **calibration**: running ~100 sample inputs once to measure activation ranges for static quantization.
