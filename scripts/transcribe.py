"""
Step 6 entrypoint: transcribe audio files with the finetuned model.

Run:
    python scripts/transcribe.py --audio path/to/file.wav
    python scripts/transcribe.py --audio path/to/file.wav --faster-whisper
    python scripts/transcribe.py --audio dir/of/wavs/ --output results.json
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config_loader import load_config
from src.inference.pipeline import ASRPipeline


AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".opus"}


def parse_args():
    p = argparse.ArgumentParser(description="Transcribe audio with finetuned Whisper.")
    p.add_argument("--audio",         required=True,
                   help="Path to audio file or directory of audio files")
    p.add_argument("--model-dir",     default=None,
                   help="Model directory (default: checkpoints/final_model)")
    p.add_argument("--faster-whisper", action="store_true",
                   help="Use faster-whisper backend (requires ct2 conversion)")
    p.add_argument("--output",        default=None,
                   help="Save results to JSON file")
    p.add_argument("--remove-punctuation", action="store_true")
    p.add_argument("--strip-fillers",      action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    cfg  = load_config()

    model_dir = args.model_dir or str(Path(cfg.training.output_dir) / "final_model")

    pipeline = ASRPipeline(
        model_dir=model_dir,
        language=cfg.model.language,
        use_faster=args.faster_whisper,
    )

    # Collect audio files
    audio_path = Path(args.audio)
    if audio_path.is_dir():
        files = sorted(
            f for f in audio_path.rglob("*")
            if f.suffix.lower() in AUDIO_EXTENSIONS
        )
    else:
        files = [audio_path]

    print(f"\nTranscribing {len(files)} file(s)...\n")

    results = []
    for f in files:
        text = pipeline(
            f,
            remove_punctuation=args.remove_punctuation,
            strip_filler_words=args.strip_fillers,
        )
        results.append({"file": str(f), "transcript": text})
        print(f"[{f.name}]\n  {text}\n")

    if args.output:
        with open(args.output, "w", encoding="utf-8") as fh:
            json.dump(results, fh, ensure_ascii=False, indent=2)
        print(f"Results saved → {args.output}")


if __name__ == "__main__":
    main()