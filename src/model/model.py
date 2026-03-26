"""
Whisper model loader.

Loads the base WhisperForConditionalGeneration checkpoint and configures
the forced decoder ids and language/task tokens so generation is
constrained to Punjabi transcription without beam search guessing.
"""

from transformers import WhisperForConditionalGeneration, WhisperProcessor


def load_model(cfg) -> WhisperForConditionalGeneration:
    """
    Load Whisper from HuggingFace Hub and apply generation config.

    Args:
        cfg : merged OmegaConf config (output of load_config())

    Returns:
        WhisperForConditionalGeneration ready for LoRA / freeze application.
    """
    model_name = cfg.model.name

    model = WhisperForConditionalGeneration.from_pretrained(model_name)

    # Disable default forced_decoder_ids — the Trainer sets these via
    # generation_config during training. Leaving them set causes a shape
    # mismatch when predict_with_generate=True.
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens    = []

    # Tell the model which language + task to use at generation time.
    # This writes into model.generation_config which the Trainer reads.
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