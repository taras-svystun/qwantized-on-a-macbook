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
