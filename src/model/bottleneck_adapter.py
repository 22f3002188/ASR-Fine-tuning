"""
Bottleneck Adapter (BA) module for Whisper finetuning.

Architecture (from Liu et al. 2024, Figure 4):
    input
      │
    LayerNorm
      │
    Linear (d_model → d)      ← down-projection
      │
    GELU
      │
    Linear (d → d_model)      ← up-projection
      │
    + input                   ← residual connection
      │
    output

Inserted after every encoder and decoder layer as the last sub-module.
During BAFT only BA parameters are trained — all original Whisper weights
are frozen.

Key hyperparameter: d (bottleneck dimension)
    d=32  → ~0.4% trainable params
    d=64  → ~0.8% trainable params  ← best per Liu et al.
    d=128 → ~1.6% trainable params
    d=256 → ~3.2% trainable params
"""

import torch
import torch.nn as nn
from transformers import WhisperForConditionalGeneration


# ── Bottleneck Adapter module ─────────────────────────────────────────────────

class BottleneckAdapter(nn.Module):
    """
    Single BA module inserted after a Whisper transformer layer.

    Args:
        d_model : hidden dimension of the Whisper model (e.g. 1280 for large-v3)
        d       : bottleneck dimension (default 64 per paper)
        dropout : dropout on the up-projection output
    """

    def __init__(self, d_model: int, d: int = 64, dropout: float = 0.0):
        super().__init__()
        self.layer_norm   = nn.LayerNorm(d_model)
        self.down_proj    = nn.Linear(d_model, d, bias=True)
        self.activation   = nn.GELU()
        self.up_proj      = nn.Linear(d, d_model, bias=True)
        self.dropout      = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Initialise up_proj to zero so the adapter is a no-op at init.
        # This ensures training starts from the pretrained model's behaviour.
        nn.init.zeros_(self.up_proj.weight)
        nn.init.zeros_(self.up_proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.layer_norm(x)
        x = self.down_proj(x)
        x = self.activation(x)
        x = self.up_proj(x)
        x = self.dropout(x)
        return x + residual


# ── Whisper layer wrapper ─────────────────────────────────────────────────────

class WhisperLayerWithAdapter(nn.Module):
    """
    Wraps a WhisperEncoderLayer or WhisperDecoderLayer with a BA appended.

    The original layer runs first, then the BA is applied to its output.
    This matches Figure 4 — BA is inserted as the LAST module of each layer.
    """

    def __init__(self, original_layer: nn.Module, adapter: BottleneckAdapter):
        super().__init__()
        self.layer   = original_layer
        self.adapter = adapter

    def forward(self, *args, **kwargs):
        # Run the original Whisper layer
        outputs = self.layer(*args, **kwargs)

        # outputs is a tuple — first element is the hidden state tensor
        hidden = outputs[0]
        hidden = self.adapter(hidden)

        # Return with the adapted hidden state, rest of outputs unchanged
        return (hidden,) + outputs[1:]


# ── Adapter injection ─────────────────────────────────────────────────────────

def inject_bottleneck_adapters(
    model: WhisperForConditionalGeneration,
    d: int = 64,
    dropout: float = 0.0,
) -> WhisperForConditionalGeneration:
    """
    Insert a BottleneckAdapter after every encoder and decoder layer.

    Args:
        model   : base WhisperForConditionalGeneration
        d       : bottleneck dimension (64 recommended per Liu et al.)
        dropout : dropout rate in the adapter

    Returns:
        Model with adapters injected and all original weights frozen.
        Only BA parameters have requires_grad=True.
    """
    d_model = model.config.d_model

    # ── Freeze all original parameters ────────────────────────────────────────
    for param in model.parameters():
        param.requires_grad = False

    # ── Inject into encoder ───────────────────────────────────────────────────
    encoder_layers = model.model.encoder.layers
    for i in range(len(encoder_layers)):
        adapter = BottleneckAdapter(d_model=d_model, d=d, dropout=dropout)
        encoder_layers[i] = WhisperLayerWithAdapter(encoder_layers[i], adapter)

    # ── Inject into decoder ───────────────────────────────────────────────────
    decoder_layers = model.model.decoder.layers
    for i in range(len(decoder_layers)):
        adapter = BottleneckAdapter(d_model=d_model, d=d, dropout=dropout)
        decoder_layers[i] = WhisperLayerWithAdapter(decoder_layers[i], adapter)

    print(f"Injected BA (d={d}) into "
          f"{len(encoder_layers)} encoder + {len(decoder_layers)} decoder layers")

    # Cast all adapter parameters to match the base model's dtype.
    # nn.Linear and nn.LayerNorm default to float32 — if the model is
    # in bf16 or fp16 the adapter forward pass will raise a dtype mismatch.
    model_dtype = next(
        p for name, p in model.named_parameters() if "adapter" not in name
    ).dtype
    for module in model.modules():
        if isinstance(module, BottleneckAdapter):
            module.to(model_dtype)

    print(f"  Adapter dtype cast to: {model_dtype}")
    return model


def freeze_non_adapter_params(model: WhisperForConditionalGeneration) -> None:
    """
    Ensure only BottleneckAdapter parameters are trainable.
    Called after inject_bottleneck_adapters() as a safety check.
    """
    for name, param in model.named_parameters():
        if "adapter" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False


def count_adapter_params(model: WhisperForConditionalGeneration) -> dict:
    """Return trainable vs total parameter counts."""
    trainable, total = 0, 0
    for _, p in model.named_parameters():
        n = p.numel()
        total += n
        if p.requires_grad:
            trainable += n
    return {
        "trainable": trainable,
        "total":     total,
        "pct":       round(100 * trainable / total, 4) if total > 0 else 0,
    }


def save_adapter_weights(
    model: WhisperForConditionalGeneration,
    save_dir: str,
) -> None:
    """
    Save only the adapter weights to a small checkpoint file (~50MB for d=64).
    The base Whisper weights are not saved — reload from HuggingFace at merge time.
    """
    import os, torch
    os.makedirs(save_dir, exist_ok=True)

    adapter_state = {
        name: param
        for name, param in model.state_dict().items()
        if "adapter" in name
    }
    torch.save(adapter_state, os.path.join(save_dir, "adapter_weights.pt"))
    print(f"Adapter weights saved → {save_dir}/adapter_weights.pt  "
          f"({len(adapter_state)} tensors)")


def load_adapter_weights(
    model: WhisperForConditionalGeneration,
    adapter_dir: str,
    strict: bool = True,
) -> WhisperForConditionalGeneration:
    """
    Load adapter weights into an already-injected model.

    Usage:
        model = load_model(cfg)
        model = inject_bottleneck_adapters(model, d=64)
        model = load_adapter_weights(model, "checkpoints/final_adapter")
    """
    import torch
    path = os.path.join(adapter_dir, "adapter_weights.pt")
    state = torch.load(path, map_location="cpu")
    missing, unexpected = model.load_state_dict(state, strict=False)

    if strict and (missing or unexpected):
        raise RuntimeError(
            f"Adapter weight mismatch.\n"
            f"  Missing   : {missing[:5]}\n"
            f"  Unexpected: {unexpected[:5]}"
        )
    print(f"Adapter weights loaded from {path}")
    return model