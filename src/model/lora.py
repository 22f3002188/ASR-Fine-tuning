"""
LoRA adapter setup via PEFT.

Confirmed architecture (whisper-large-v3, 32 encoder + 32 decoder layers):
  Encoder: model.encoder.layers.N.self_attn.{q,v}_proj
  Decoder: model.decoder.layers.N.self_attn.{q,v}_proj
           model.decoder.layers.N.encoder_attn.{q,v}_proj

PEFT target_modules behaviour (confirmed via inspection):
  - Matches by checking if the key ends with any string in target_modules
  - Does NOT support full dotted paths or regex
  - layers_to_transform is unreliable across PEFT versions

Safest approach for decoder-only: explicitly enumerate the full dotted
parameter names for decoder layers and pass them as target_modules.
This is version-agnostic and unambiguous.
"""

from transformers import WhisperForConditionalGeneration
from peft import LoraConfig, get_peft_model, TaskType


# Whisper large-v3 has 32 decoder layers
N_DECODER_LAYERS = 32


def _build_decoder_target_modules(
    base_modules: list[str],
    n_layers: int = N_DECODER_LAYERS,
) -> list[str]:
    """
    Enumerate the full dotted names for every decoder projection we want to adapt.

    For each layer i and each attention type (self_attn, encoder_attn),
    generates:
        model.decoder.layers.{i}.self_attn.q_proj
        model.decoder.layers.{i}.self_attn.v_proj
        model.decoder.layers.{i}.encoder_attn.q_proj
        model.decoder.layers.{i}.encoder_attn.v_proj
    """
    modules = []
    attn_types = ["self_attn", "encoder_attn"]
    for i in range(n_layers):
        for attn in attn_types:
            for proj in base_modules:
                modules.append(f"model.decoder.layers.{i}.{attn}.{proj}")
    return modules


def _build_encoder_target_modules(
    base_modules: list[str],
    n_layers: int = N_DECODER_LAYERS,   # encoder also has 32 layers on large-v3
) -> list[str]:
    modules = []
    for i in range(n_layers):
        for proj in base_modules:
            modules.append(f"model.encoder.layers.{i}.self_attn.{proj}")
    return modules


def apply_lora(
    model: WhisperForConditionalGeneration,
    cfg,
) -> WhisperForConditionalGeneration:
    """
    Wrap the model with LoRA adapters according to lora.yaml.

    Uses explicit full module paths to guarantee correct targeting
    regardless of PEFT version.
    """
    lora_cfg  = cfg.lora
    apply_to  = lora_cfg.get("apply_to", "decoder_only")
    base_mods = list(lora_cfg.target_modules)   # ["q_proj", "v_proj"]

    # Dynamically read the actual layer counts from the loaded model
    n_dec = len(model.model.decoder.layers)
    n_enc = len(model.model.encoder.layers)

    if apply_to == "decoder_only":
        target_modules = _build_decoder_target_modules(base_mods, n_layers=n_dec)
    elif apply_to == "encoder_only":
        target_modules = _build_encoder_target_modules(base_mods, n_layers=n_enc)
    else:  # "both"
        target_modules = (
            _build_decoder_target_modules(base_mods, n_layers=n_dec)
            + _build_encoder_target_modules(base_mods, n_layers=n_enc)
        )

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
        n = p.numel()
        total += n
        if p.requires_grad:
            trainable += n
    pct = 100 * trainable / total if total > 0 else 0
    print(f"Trainable params : {trainable:,}")
    print(f"Total params     : {total:,}")
    print(f"Trainable %      : {pct:.4f}%")