"""
Simple config loader for ASR pipeline.

Loads:
- .env (optional)
- train.yaml
- data.yaml
- lora.yaml (optional but recommended)

Applies:
- HF token login
- optional smoke test override
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf


CONFIG_DIR = Path(__file__).parent.parent / "configs"


def load_config(config_dir: Path = CONFIG_DIR) -> DictConfig:
    _load_env()
    _register_hf_token()

    cfg = _load_yaml_configs(config_dir)
    cfg = _apply_smoke_test(cfg)

    return cfg


# ────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────

def _load_env() -> None:
    """Load .env if present."""
    env_path = Path(__file__).parent.parent / ".env"
    load_dotenv(dotenv_path=env_path, override=False)


def _register_hf_token() -> None:
    """Login to Hugging Face if token exists."""
    token = os.environ.get("HF_TOKEN")
    if token:
        try:
            from huggingface_hub import login
            login(token=token, add_to_git_credential=False)
        except Exception:
            pass


def _load_yaml_configs(config_dir: Path) -> DictConfig:
    """
    Load and merge:
    - train.yaml
    - data.yaml
    - lora.yaml (if present)

    Merge order matters:
    train <- data <- lora
    """
    train_cfg = OmegaConf.load(config_dir / "train.yaml")
    data_cfg = OmegaConf.load(config_dir / "data.yaml")

    cfgs = [train_cfg, data_cfg]

    lora_path = config_dir / "lora.yaml"
    if lora_path.exists():
        lora_cfg = OmegaConf.load(lora_path)
        cfgs.append(lora_cfg)

    return OmegaConf.merge(*cfgs)


def _apply_smoke_test(cfg: DictConfig) -> DictConfig:
    """Reduce training steps for quick testing."""
    if os.environ.get("SMOKE_TEST", "false").lower() == "true":
        steps = cfg.training.get("smoke_test_steps", 10)

        cfg.training.max_steps = steps
        cfg.training.eval_steps = steps
        cfg.training.save_steps = steps
        cfg.training.logging_steps = 1

        print(f"[SMOKE TEST] max_steps = {steps}")

    return cfg