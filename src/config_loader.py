"""
Central config + environment loader.

Every script (train.py, evaluate.py, etc.) calls load_config() at the top.
This ensures .env is sourced, tokens are registered with HuggingFace and W&B,
and all three YAML configs are merged into a single OmegaConf DictConfig.

Usage:
    from src.config_loader import load_config
    cfg = load_config()

    # Access any value with dot notation:
    cfg.model.name          # "openai/whisper-medium"
    cfg.lora.r              # 32
    cfg.data.buffer_size    # 500
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from omegaconf import OmegaConf, DictConfig


CONFIG_DIR = Path(__file__).parent.parent / "configs"


def load_config(config_dir: Path = CONFIG_DIR) -> DictConfig:
    """
    1. Load .env from project root (silently skip if missing — CI/CD injects vars directly)
    2. Register HF_TOKEN with huggingface_hub
    3. Set WANDB env vars
    4. Merge train.yaml + lora.yaml + data.yaml into one DictConfig
    5. Apply SMOKE_TEST override if env var is set
    """
    _load_env()
    _register_hf_token()
    _register_wandb()
    cfg = _merge_yaml_configs(config_dir)
    cfg = _apply_smoke_test(cfg)
    return cfg


# ── Private helpers ───────────────────────────────────────────────────────────

def _load_env() -> None:
    """Load .env from the project root. No-op if the file doesn't exist."""
    project_root = Path(__file__).parent.parent
    env_path = project_root / ".env"
    load_dotenv(dotenv_path=env_path, override=False)  # don't override existing env vars


def _register_hf_token() -> None:
    """Push HF_TOKEN into huggingface_hub so all dataset/model calls are authenticated."""
    token = os.environ.get("HF_TOKEN")
    if token:
        try:
            from huggingface_hub import login
            login(token=token, add_to_git_credential=False)
        except Exception:
            pass  # non-fatal — public datasets don't need a token


def _register_wandb() -> None:
    """
    W&B reads WANDB_API_KEY, WANDB_PROJECT, WANDB_ENTITY directly from the environment.
    Nothing to do here — just validate they're present and warn if not.
    """
    if not os.environ.get("WANDB_API_KEY"):
        import warnings
        warnings.warn(
            "WANDB_API_KEY is not set. Training will log locally only. "
            "Set it in .env or the environment to enable W&B tracking.",
            stacklevel=2,
        )


def _merge_yaml_configs(config_dir: Path) -> DictConfig:
    """Load and merge the three YAML configs into one DictConfig."""
    train = OmegaConf.load(config_dir / "train.yaml")
    lora  = OmegaConf.load(config_dir / "lora.yaml")
    data  = OmegaConf.load(config_dir / "data.yaml")
    return OmegaConf.merge(train, lora, data)


def _apply_smoke_test(cfg: DictConfig) -> DictConfig:
    """
    If SMOKE_TEST=true in the environment, cap max_steps at smoke_test_steps.
    Lets you validate the full pipeline end-to-end in <2 minutes.
    """
    if os.environ.get("SMOKE_TEST", "false").lower() == "true":
        smoke_steps = cfg.training.get("smoke_test_steps", 10)
        OmegaConf.update(cfg, "training.max_steps", smoke_steps)
        OmegaConf.update(cfg, "training.eval_steps", smoke_steps)
        OmegaConf.update(cfg, "training.save_steps", smoke_steps)
        OmegaConf.update(cfg, "training.logging_steps", 1)
        print(f"[SMOKE TEST] max_steps capped at {smoke_steps}")
    return cfg