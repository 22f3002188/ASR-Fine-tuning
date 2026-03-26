"""
LoRA adapter setup via PEFT.

Reads lora.yaml config and applies get_peft_model() to the loaded Whisper model.
Supports applying LoRA to decoder-only, encoder-only, or both.
"""

from transformers import WhisperForConditionalGeneration
from peft import LoraConfig, get_peft_model, TaskType


def apply_lora(
    model: WhisperForConditionalGeneration,
    cfg,
) -> WhisperForConditionalGeneration:
    """
    Wrap the model with LoRA adapters according to lora.yaml.

    Args:
        model : base WhisperForConditionalGeneration (post load_model())
        cfg   : merged OmegaConf config

    Returns:
        PeftModel wrapping the original model. Only adapter weights are
        trainable — base weights are frozen by PEFT automatically.
    """
    lora_cfg = cfg.lora
    apply_to = lora_cfg.get("apply_to", "decoder_only")

    # Build the list of target module patterns based on apply_to scope.
    # Whisper's encoder and decoder share the same projection names,
    # so we qualify them with "model.encoder" / "model.decoder" prefixes
    # to restrict which side gets adapters.
    base_modules = list(lora_cfg.target_modules)   # ["q_proj", "v_proj"]

    if apply_to == "decoder_only":
        target_modules = [f"model.decoder.{m}" for m in base_modules]
    elif apply_to == "encoder_only":
        target_modules = [f"model.encoder.{m}" for m in base_modules]
    else:  # "both"
        target_modules = base_modules  # PEFT matches by suffix across all layers

    peft_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=lora_cfg.r,
        lora_alpha=lora_cfg.lora_alpha,
        lora_dropout=lora_cfg.lora_dropout,
        bias=lora_cfg.bias,
        target_modules=target_modules,
    )

    model = get_peft_model(model, peft_config)
    return model


def print_trainable_parameters(model) -> None:
    """Log the ratio of trainable vs total parameters."""
    trainable, total = 0, 0
    for _, p in model.named_parameters():
        total += p.numel()
        if p.requires_grad:
            trainable += p.numel()
    pct = 100 * trainable / total if total > 0 else 0
    print(f"Trainable params : {trainable:,}")
    print(f"Total params     : {total:,}")
    print(f"Trainable %      : {pct:.4f}%")