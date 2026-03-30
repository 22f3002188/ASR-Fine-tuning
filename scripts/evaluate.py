"""
Step 5 entrypoint: evaluate a finetuned Whisper model.

Streams the val split, generates transcriptions, computes WER/CER,
and prints a full error analysis report.

Run:
    python scripts/evaluate.py
    python scripts/evaluate.py --model-dir checkpoints/checkpoint-2000
    python scripts/evaluate.py --n-samples 500   # quick partial eval
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config_loader import load_config
from src.evaluation.evaluate import run_evaluation
from src.evaluation.error_analysis import ErrorAnalyser


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir",  type=str, default=None,
                   help="Path to model dir (default: checkpoints/final_model)")
    p.add_argument("--n-samples",  type=int, default=None,
                   help="Cap evaluation at N samples")
    p.add_argument("--save-predictions", action="store_true",
                   help="Save predictions to predictions.json")
    return p.parse_args()


def main():
    args = parse_args()
    cfg  = load_config()

    results = run_evaluation(
        cfg,
        model_dir=args.model_dir,
        n_samples=args.n_samples,
    )

    # ── Print report ──────────────────────────────────────────────────────────
    analyser = ErrorAnalyser(
        references=[p["reference"]  for p in results["predictions"]],
        hypotheses=[p["hypothesis"] for p in results["predictions"]],
        domains   =[p["domain"]     for p in results["predictions"]],
    )
    analyser.print_report()

    print(f"  Overall WER : {results['wer']:.4f}")
    print(f"  Overall CER : {results['cer']:.4f}")

    # ── Optionally save predictions ────────────────────────────────────────────
    if args.save_predictions:
        out_path = Path(cfg.training.output_dir) / "predictions.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results["predictions"], f, ensure_ascii=False, indent=2)
        print(f"\n  Predictions saved → {out_path}")


if __name__ == "__main__":
    main()