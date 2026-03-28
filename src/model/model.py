"""
Whisper model loader for vanilla (full) finetuning.

No LoRA, no adapter wrapping, no layer freezing.
The full model is trained end-to-end with the optimizer
updating all parameters.
"""

from transformers import WhisperForConditionalGeneration, WhisperProcessor


def load_model(cfg) -> WhisperForConditionalGeneration:
    """
    Load Whisper from HuggingFace Hub and configure generation settings.

    Args:
        cfg : merged OmegaConf config (output of load_config())

    Returns:
        WhisperForConditionalGeneration with generation config set.
    """
    model_name = cfg.model.name
    model = WhisperForConditionalGeneration.from_pretrained(model_name)

    # Disable forced_decoder_ids on model.config — the Trainer uses
    # generation_config instead, and having both causes a shape mismatch
    # during predict_with_generate=True evaluation.
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens    = []

    # Set language + task on generation_config so every generate() call
    # is constrained to Punjabi transcription automatically.
    processor = WhisperProcessor.from_pretrained(
        model_name,
        language=cfg.model.language,
        task=cfg.model.task,
    )
    forced_ids = processor.get_decoder_prompt_ids(
        language=cfg.model.language,
        task=cfg.model.task,
    )
    model.generation_config.forced_decoder_ids = forced_ids
    model.generation_config.suppress_tokens    = []

    return model