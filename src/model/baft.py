"""
Bottleneck Adapter Fine-Tuning (BAFT) module for Whisper.

Injects custom down-up projection bottleneck adapters into the Whisper model's 
Transformer layers (specifically after linear projections) via PyTorch module wrapping,
and strictly freezes all other original model parameters.

This allows parameter-efficient fine-tuning (PEFT) without breaking Whisper's 
native generate() loops or requiring complex wrapper abstractions.
"""

import torch
import torch.nn as nn
from transformers import WhisperForConditionalGeneration


class BAFTAdapter(nn.Module):
    """
    Standard Houlsby-style Bottleneck Adapter.
    Projects down to a bottleneck dimension `r`, applies a non-linearity (GELU), 
    and projects back up to the original hidden dimension.
    """
    def __init__(self, d_model: int, r: int, act_fn: nn.Module = None):
        super().__init__()
        self.down = nn.Linear(d_model, r)
        self.act = act_fn if act_fn is not None else nn.GELU()
        self.up = nn.Linear(r, d_model)
        
        # Zero-init the up-projection so the adapter starts out mathematically 
        # as a perfect identity function (does not disrupt base Whisper logic on step 0)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)
        
    def forward(self, x):
        return self.up(self.act(self.down(x)))


class BAFTWrappedLinear(nn.Module):
    """
    Wraps an existing Whisper nn.Linear layer. 
    Computes: original_layer(x) + adapter(original_layer(x))
    Original layer parameters are strictly frozen internally.
    """
    def __init__(self, original_linear: nn.Linear, r: int):
        super().__init__()
        
        # 1. Store the original layer and freeze its gradients
        self.original = original_linear
        for p in self.original.parameters():
            p.requires_grad = False
            
        # 2. Initialize the adapter on top of the original layer's output dimension
        out_features = original_linear.out_features
        self.adapter = BAFTAdapter(d_model=out_features, r=r)

    def forward(self, x):
        # Forward pass through the (frozen) original weights
        orig_out = self.original(x)
        # Add the adapter's residual update
        return orig_out + self.adapter(orig_out)


def apply_baft(
    model: WhisperForConditionalGeneration, 
    cfg
) -> WhisperForConditionalGeneration:
    """
    Recursively iterate through the model and replace target linear layers 
    (fc2, out_proj) with BAFT wrapped modules.
    """
    
    # Try to extract BAFT config, default to r=32 and targeting out_proj / fc2 if block missing
    if hasattr(cfg, "baft") and cfg.baft is not None:
        r = getattr(cfg.baft, "r", 32)
        target_modules = getattr(cfg.baft, "target_modules", ["out_proj", "fc2"])
    else:
        print("[BAFT] 'baft' config section not found. Defaulting to r=32, targets=['out_proj', 'fc2']")
        r = 32
        target_modules = ["out_proj", "fc2"]

    # 1. Freeze the ENTIRE backbone to be safe
    for p in model.parameters():
        p.requires_grad = False
        
    try:
        total_encoder_layers = len(model.model.encoder.layers)
    except AttributeError:
        total_encoder_layers = 32
        
    p_encoder_top = getattr(cfg.baft, "p_encoder_top", 9) if hasattr(cfg, "baft") and cfg.baft else 9
    min_encoder_layer_idx = total_encoder_layers - p_encoder_top

    def _replace_modules_recursive(module, name=""):
        # Modifying children dynamically during iteration
        for child_name, child_module in list(module.named_children()):
            full_name = f"{name}.{child_name}" if name else child_name
            
            # Check if this linear layer is within an encoder layer we want to skip
            skip_layer = False
            if "encoder.layers." in full_name:
                try:
                    # Extract the index from e.g. "model.encoder.layers.5.self_attn.out_proj"
                    idx_str = full_name.split("encoder.layers.")[1].split(".")[0]
                    if int(idx_str) < min_encoder_layer_idx:
                        skip_layer = True
                except ValueError:
                    pass
            
            # If the child is an nn.Linear and its name matches our targets
            is_target = any(target in child_name for target in target_modules)
            
            if isinstance(child_module, nn.Linear) and is_target and not skip_layer:
                # Replace the original nn.Linear with our BAFT adapter wrapper
                wrapped_layer = BAFTWrappedLinear(child_module, r=r)
                setattr(module, child_name, wrapped_layer)
            else:
                # Recursively search deeper
                _replace_modules_recursive(child_module, full_name)
                
    # 2. Inject Adapters
    print(f"[BAFT] Injecting adapters (r={r}) into targets '{target_modules}', targeting Top {p_encoder_top} encoder layers.")
    _replace_modules_recursive(model, name="model")
    
    # At this point, only modules inside BAFTWrappedLinear.adapter have requires_grad=True
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
