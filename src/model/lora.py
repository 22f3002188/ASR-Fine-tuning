"""
LoRA utilities for Whisper.

Why custom wrapper?
PEFT's default seq2seq wrapper passes `input_ids` to the base model.
Whisper expects `input_features`, so we use a custom PeftModel wrapper
that calls Whisper correctly.
"""

from __future__ import annotations

from typing import List

import torch
from peft import LoraConfig, PeftModel, TaskType
from peft.utils import PeftType


def _resolve_target_modules(model, base_targets: List[str], apply_to: str) -> List[str]:
    """
    Expand shorthand target names like q_proj/v_proj into actual Whisper module names.
    """
    full_targets = []

    for module_name, _ in model.named_modules():
        if apply_to == "decoder_only" and not module_name.startswith("model.decoder"):
            continue
        if apply_to == "encoder_only" and not module_name.startswith("model.encoder"):
            continue
        if apply_to == "both":
            pass

        for target in base_targets:
            if module_name.endswith(target):
                full_targets.append(module_name)

    return sorted(set(full_targets))


class WhisperTuner(PeftModel):
    """
    Custom PEFT wrapper for Whisper.

    Main fix:
    - accept input_features
    - ignore/strip accidental input_ids and inputs_embeds
    - call base Whisper model with input_features=...
    """

    def __init__(self, model: torch.nn.Module, peft_config, adapter_name: str = "default") -> None:
        super().__init__(model, peft_config, adapter_name)
        self.base_model_prepare_inputs_for_generation = self.base_model.prepare_inputs_for_generation
        self.base_model_prepare_encoder_decoder_kwargs_for_generation = (
            self.base_model._prepare_encoder_decoder_kwargs_for_generation
        )

    def forward(
        self,
        input_features=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        decoder_inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        task_ids=None,
        **kwargs,
    ):
        peft_config = self.active_peft_config

        # PEFT sometimes forwards keys Whisper doesn't accept
        if input_features is None and "input_ids" in kwargs and kwargs["input_ids"] is not None:
            input_features = kwargs.pop("input_ids")

        kwargs.pop("input_ids", None)
        kwargs.pop("inputs_embeds", None)

        if not peft_config.is_prompt_learning:
            if peft_config.peft_type == PeftType.POLY:
                kwargs["task_ids"] = task_ids

            with self._enable_peft_forward_hooks(**kwargs):
                kwargs = {
                    k: v for k, v in kwargs.items()
                    if k not in self.special_peft_forward_args
                }

                return self.base_model(
                    input_features=input_features,
                    attention_mask=attention_mask,
                    decoder_input_ids=decoder_input_ids,
                    decoder_attention_mask=decoder_attention_mask,
                    decoder_inputs_embeds=decoder_inputs_embeds,
                    labels=labels,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                    **kwargs,
                )

        raise NotImplementedError("Prompt-learning PEFT is not implemented for WhisperTuner.")

    def generate(self, **kwargs):
        peft_config = self.active_peft_config

        self.base_model.prepare_inputs_for_generation = self.prepare_inputs_for_generation
        self.base_model._prepare_encoder_decoder_kwargs_for_generation = (
            self._prepare_encoder_decoder_kwargs_for_generation
        )

        # Safety: remove keys Whisper.generate / Whisper.forward don't want
        if "input_features" not in kwargs and "input_ids" in kwargs:
            kwargs["input_features"] = kwargs.pop("input_ids")
        else:
            kwargs.pop("input_ids", None)

        kwargs.pop("inputs_embeds", None)

        try:
            if not peft_config.is_prompt_learning:
                with self._enable_peft_forward_hooks(**kwargs):
                    kwargs = {
                        k: v for k, v in kwargs.items()
                        if k not in self.special_peft_forward_args
                    }
                    outputs = self.base_model.generate(**kwargs)
            else:
                raise NotImplementedError("Prompt-learning PEFT is not implemented for WhisperTuner.")
        finally:
            self.base_model.prepare_inputs_for_generation = self.base_model_prepare_inputs_for_generation
            self.base_model._prepare_encoder_decoder_kwargs_for_generation = (
                self.base_model_prepare_encoder_decoder_kwargs_for_generation
            )

        return outputs

    def prepare_inputs_for_generation(self, *args, task_ids: torch.Tensor = None, **kwargs):
        peft_config = self.active_peft_config
        model_kwargs = self.base_model_prepare_inputs_for_generation(*args, **kwargs)

        if peft_config.peft_type == PeftType.POLY:
            model_kwargs["task_ids"] = task_ids

        if (
            model_kwargs.get("past_key_values", None) is None
            and peft_config.peft_type == PeftType.PREFIX_TUNING
        ):
            batch_size = model_kwargs["decoder_input_ids"].shape[0]
            past_key_values = self.get_prompt(batch_size)
            model_kwargs["past_key_values"] = past_key_values

        return model_kwargs


def apply_lora(model, cfg):
    """
    Apply LoRA adapters to Whisper using the custom WhisperTuner wrapper.
    """
    if not hasattr(cfg, "lora") or not cfg.lora.get("enabled", False):
        return model

    lora_cfg = cfg.lora
    base_targets = list(lora_cfg.get("target_modules", ["q_proj", "v_proj"]))
    apply_to = lora_cfg.get("apply_to", "decoder_only")

    full_targets = _resolve_target_modules(model, base_targets, apply_to)

    if not full_targets:
        raise ValueError(
            f"No LoRA target modules found for target_modules={base_targets} "
            f"with apply_to={apply_to}"
        )

    print("Resolved LoRA target modules:")
    for name in full_targets[:20]:
        print(f"  {name}")
    if len(full_targets) > 20:
        print(f"  ... ({len(full_targets)} total)")

    peft_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        inference_mode=False,
        r=int(lora_cfg.r),
        lora_alpha=int(lora_cfg.lora_alpha),
        lora_dropout=float(lora_cfg.get("lora_dropout", 0.05)),
        bias=str(lora_cfg.get("bias", "none")),
        target_modules=full_targets,
    )

    model = WhisperTuner(model, peft_config)
    return model


def print_trainable_parameters(model) -> None:
    trainable_params = 0
    total_params = 0

    for _, param in model.named_parameters():
        total_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()

    pct = 100 * trainable_params / total_params if total_params > 0 else 0.0

    print(f"Trainable params : {trainable_params:,}")
    print(f"Total params     : {total_params:,}")
    print(f"Trainable %      : {pct:.6f}%")