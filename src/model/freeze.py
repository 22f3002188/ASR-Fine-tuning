"""
Layer freezing strategies for Whisper finetuning.

Called after apply_lora() — PEFT has already frozen base weights,
but we still need to explicitly freeze the encoder (PEFT only freezes
non-adapter params, it doesn't know about our encoder/decoder split intent).

Freeze order matters:
    1. load_model()         — loads weights, sets generation config
    2. apply_lora()         — wraps with adapters, freezes non-adapter params
    3. apply_freeze()       — enforces encoder freeze + embed freeze on top
    4. print_trainable_parameters()  — verify the result
"""

from transformers import WhisperForConditionalGeneration


def apply_freeze(model, cfg) -> None:
    """
    Apply layer freezing according to lora.yaml freeze section.

    Modifies model in-place (sets requires_grad=False on targeted params).

    Args:
        model : PeftModel (output of apply_lora())
        cfg   : merged OmegaConf config
    """
    freeze_cfg = cfg.get("freeze", {})

    if freeze_cfg.get("encoder", True):
        _freeze_encoder(model)

    n_frozen_decoder = freeze_cfg.get("freeze_decoder_layers", 0)
    if n_frozen_decoder > 0:
        _freeze_decoder_layers(model, n_frozen_decoder)

    if freeze_cfg.get("freeze_embed_positions", True):
        _freeze_embed_positions(model)


# ── Individual freeze helpers ──────────────────────────────────────────────────

def _freeze_encoder(model) -> None:
    """
    Freeze the entire Whisper encoder.

    Strongly recommended for low-resource finetuning: the encoder's acoustic
    representations are already strong from multilingual pretraining. Training
    it on <100hrs risks catastrophic forgetting of other languages and degrades
    the model's robustness to noise and accents.
    """
    # PEFT wraps the original model — access base via model.model or model.base_model
    base = _get_base_model(model)
    for param in base.model.encoder.parameters():
        param.requires_grad = False
    print("Frozen: encoder")


def _freeze_decoder_layers(model, n: int) -> None:
    """
    Freeze the first n decoder transformer layers.

    Early decoder layers encode syntactic / language-model priors from
    pretraining. Freezing them when data is scarce prevents overfitting
    to the training distribution and preserves the LM prior.
    """
    base = _get_base_model(model)
    layers = base.model.decoder.layers
    for i, layer in enumerate(layers[:n]):
        for param in layer.parameters():
            param.requires_grad = False
    print(f"Frozen: first {n} decoder layers")


def _freeze_embed_positions(model) -> None:
    """
    Freeze encoder + decoder positional embeddings.

    Whisper uses fixed sinusoidal embed_positions — these are never updated
    during pretraining either, so freezing them is a no-op in practice,
    but making it explicit prevents accidents if the architecture changes.
    """
    base = _get_base_model(model)
    for name, param in base.model.named_parameters():
        if "embed_positions" in name:
            param.requires_grad = False
    print("Frozen: embed_positions")


def _get_base_model(model):
    """
    Unwrap PeftModel to get the underlying WhisperForConditionalGeneration.
    Works whether or not LoRA has been applied.
    """
    if hasattr(model, "base_model"):
        return model.base_model   # PeftModel.base_model is the wrapped module
    return model