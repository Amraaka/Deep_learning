#!/usr/bin/env python3
"""
LoRA training CLI for Whisper medium on Common Voice Mongolian.

Run from this directory so ``import src`` resolves:
 python run_train.py

Reference examples (not used by this script): train_whisper_mn.py, training.ipynb
Interactive checks: sanity-check.ipynb
Demo: python gradio_demo.py --adapter-path results/final-lora

Push: ``--push`` uploads to ``https://huggingface.co/<you>/whisper-medium-mn-lora``
unless you set ``--hub-model-id USER/other-name``.
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

from huggingface_hub import login, whoami
from transformers import WhisperProcessor, WhisperTokenizer

from src.collate import DataCollatorSpeechSeq2SeqWithPadding
from src.config import LANGUAGE, MODEL_NAME, TASK
from src.data import filter_dataset, load_common_voice_validated, split_dataset
from src.eval import run_evaluation
from src.train import run_training

# Default Hub repo name segment when --push is used without --hub-model-id:
#   https://huggingface.co/{username}/whisper-medium-mn-lora
DEFAULT_HUB_REPO_NAME = "whisperMedium-MN"


def _load_hf_token_from_dotenv() -> Optional[str]:
    """Check cwd, then lab3/.env (parent of this repo folder)."""
    for env_path in (
        Path(".env"),
        Path(__file__).resolve().parent / ".env",
        Path(__file__).resolve().parent.parent / ".env",
    ):
        if not env_path.is_file():
            continue
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line.startswith("HF_TOKEN="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    return None


def authenticate_hf() -> None:
    token = os.environ.get("HF_TOKEN")
    if not token:
        token = _load_hf_token_from_dotenv()
        if token:
            os.environ["HF_TOKEN"] = token
    if token:
        login(token=token)
        print("[OK] Authenticated with Hugging Face.")
    else:
        print("[WARN] HF_TOKEN not set. --push will fail.")


def parse_args():
    p = argparse.ArgumentParser(description="Whisper medium Mongolian LoRA training")
    p.add_argument("--data-dir", type=str, default="common_voice_mn")
    p.add_argument("--output-dir", type=str, default="results")
    p.add_argument("--logging-dir", type=str, default="runs/whisper-mn-lora")
    p.add_argument(
        "--hub-model-id",
        type=str,
        default=None,
        help=f"HF repo for --push (default: <your_user>/{DEFAULT_HUB_REPO_NAME} when logged in).",
    )
    p.add_argument("--num-epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument(
        "--eval",
        dest="do_eval",
        action="store_true",
        help="After training, run test WER vs base model.",
    )
    p.add_argument(
        "--push",
        action="store_true",
        help=f"Push adapter to Hub (default repo: USER/{DEFAULT_HUB_REPO_NAME}).",
    )
    return p.parse_args()


def resolve_data_dir(data_dir: str) -> str:
    p = Path(data_dir)
    if p.is_dir() and (p / "validated.tsv").is_file():
        return str(p.resolve())
    for candidate in (
        Path(__file__).resolve().parent.parent / "common_voice_mn",
        Path(__file__).resolve().parent / "common_voice_mn",
    ):
        if candidate.is_dir() and (candidate / "validated.tsv").is_file():
            print(f"[INFO] Using data dir: {candidate}")
            return str(candidate)
    return str(p.resolve())


def resolve_hub_model_id(explicit_id: Optional[str]) -> str:
    """Use explicit --hub-model-id or default ``{whoami}/{DEFAULT_HUB_REPO_NAME}``."""
    if explicit_id:
        return explicit_id
    try:
        user = whoami()["name"]
    except Exception as exc:
        print(
            "[ERROR] --push needs --hub-model-id USER/repo or a valid HF_TOKEN login."
        )
        print(f"        ({exc})")
        sys.exit(1)
    hub_id = f"{user}/{DEFAULT_HUB_REPO_NAME}"
    print(f"[INFO] Default Hub repo: {hub_id}")
    return hub_id


def main() -> None:
    args = parse_args()
    args.data_dir = resolve_data_dir(args.data_dir)

    print("=" * 60)
    print("  Whisper Medium - Mongolian LoRA (run_train.py)")
    print("=" * 60)
    print(f"  Data dir:   {args.data_dir}")
    print(f"  Output dir: {args.output_dir}\n")

    authenticate_hf()

    tokenizer = WhisperTokenizer.from_pretrained(
        MODEL_NAME, language=LANGUAGE, task=TASK
    )
    processor = WhisperProcessor.from_pretrained(
        MODEL_NAME, language=LANGUAGE, task=TASK
    )

    ds = load_common_voice_validated(args.data_dir)
    ds = filter_dataset(ds, tokenizer)
    dsd = split_dataset(ds)

    trainer = run_training(args, dsd, tokenizer, processor)
    model = trainer.model
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

    if args.do_eval:
        run_evaluation(
            model, dsd, data_collator, processor, tokenizer, args.batch_size
        )
    else:
        print("\n[SKIP] Evaluation (pass --eval to run test WER)")

    if args.push:
        hub_id = resolve_hub_model_id(args.hub_model_id)
        model.push_to_hub(hub_id)
        tokenizer.push_to_hub(hub_id)
        print(f"[OK] Pushed to https://huggingface.co/{hub_id}")
    else:
        print(
            f"\n[SKIP] Hub push (pass --push; default repo <you>/{DEFAULT_HUB_REPO_NAME})"
        )

    print("\n[DONE] For Gradio: python gradio_demo.py --adapter-path results/final-lora")


if __name__ == "__main__":
    main()
