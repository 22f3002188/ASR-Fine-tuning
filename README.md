# whisper-finetune

Whisper finetuning pipeline for low-resource ASR.

## Quickstart

```bash
cp .env.example .env          # add your tokens
make data                     # prepare dataset
make train                    # finetune
make eval                     # evaluate WER / CER
```

## Structure

| Path | Purpose |
|---|---|
| `data/` | Raw audio, processed features, splits |
| `src/data/` | Dataset class, augmentation, splitting |
| `src/model/` | Model loader, LoRA, layer freezing |
| `src/training/` | Trainer, callbacks, scheduler |
| `src/evaluation/` | WER/CER metrics, error analysis |
| `src/inference/` | faster-whisper serving, postprocessing |
| `configs/` | YAML configs per stage |
| `scripts/` | CLI entrypoints |
| `tests/` | Unit tests |
