#!/usr/bin/env python3
"""
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

import torch
from huggingface_hub import login, whoami
from transformers import WhisperProcessor

from src.config import (
    EVAL_STEPS,
    FEATURE_CACHE_SUBDIR,
    GRADIENT_ACCUMULATION_STEPS,
    LEARNING_RATE,
    MAX_STEPS,
    MODEL_ID,
    PER_DEVICE_EVAL_BATCH_SIZE,
    PER_DEVICE_TRAIN_BATCH_SIZE,
    SAVE_STEPS,
    WARMUP_STEPS,
    WEIGHT_DECAY,
)
from src.data import load_validated_frame, log_missing_clips, train_val_test_split_df
from src.train import prepare_datasets, run_training

DEFAULT_HUB_REPO_NAME = "whisper-md-mn"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("run_train")


def _load_hf_token_from_dotenv() -> Optional[str]:
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
    import os

    token = os.environ.get("HF_TOKEN")
    if not token:
        token = _load_hf_token_from_dotenv()
        if token:
            os.environ["HF_TOKEN"] = token
    if token:
        login(token=token)
        log.info("Authenticated with Hugging Face.")
    else:
        log.warning("HF_TOKEN not set. Model download may fail for gated repos; --push will fail.")


def parse_args():
    p = argparse.ArgumentParser(description="Whisper Medium full fine-tune (Mongolian CV)")
    p.add_argument("--data-dir", type=str, default="common_voice_mn")
    p.add_argument("--output-dir", type=str, default="results")
    p.add_argument("--logging-dir", type=str, default=None, help="Default: OUTPUT_DIR/runs")
    p.add_argument(
        "--hub-model-id",
        type=str,
        default=None,
        help=f"HF repo for --push (default: <you>/{DEFAULT_HUB_REPO_NAME}).",
    )
    p.add_argument("--max-steps", type=int, default=MAX_STEPS)
    p.add_argument("--eval-steps", type=int, default=EVAL_STEPS)
    p.add_argument("--save-steps", type=int, default=SAVE_STEPS)
    p.add_argument("--per-device-train-batch-size", type=int, default=PER_DEVICE_TRAIN_BATCH_SIZE)
    p.add_argument("--per-device-eval-batch-size", type=int, default=PER_DEVICE_EVAL_BATCH_SIZE)
    p.add_argument("--gradient-accumulation-steps", type=int, default=GRADIENT_ACCUMULATION_STEPS)
    p.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    p.add_argument("--warmup-steps", type=int, default=WARMUP_STEPS)
    p.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    p.add_argument("--push", action="store_true", help=f"Push checkpoints to Hub (default USER/{DEFAULT_HUB_REPO_NAME}).")
    p.add_argument("--skip-test", action="store_true", help="Skip full test-set WER after training.")
    return p.parse_args()


def resolve_data_dir(data_dir: str) -> Path:
    p = Path(data_dir)
    if p.is_dir() and (p / "validated.tsv").is_file():
        return p.resolve()
    for candidate in (
        Path(__file__).resolve().parent.parent / "common_voice_mn",
        Path(__file__).resolve().parent / "common_voice_mn",
    ):
        if candidate.is_dir() and (candidate / "validated.tsv").is_file():
            log.info("Using data dir: %s", candidate)
            return candidate.resolve()
    return p.resolve()


def resolve_hub_model_id(explicit_id: Optional[str]) -> str:
    if explicit_id:
        return explicit_id
    try:
        user = whoami()["name"]
    except Exception as exc:
        log.error("--push needs --hub-model-id or HF login: %s", exc)
        sys.exit(1)
    hub_id = f"{user}/{DEFAULT_HUB_REPO_NAME}"
    log.info("Default Hub repo: %s", hub_id)
    return hub_id


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        log.error("CUDA is required for this training script (bf16 + bitsandbytes 8-bit Adam).")
        sys.exit(1)

    data_root = resolve_data_dir(args.data_dir)
    clips_dir = data_root / "clips"
    if not clips_dir.is_dir():
        log.error("clips/ not found under %s", data_root)
        sys.exit(1)

    out = Path(args.output_dir).resolve()
    log_dir = Path(args.logging_dir).resolve() if args.logging_dir else out / "runs"
    cache_dir = out / FEATURE_CACHE_SUBDIR

    args.output_dir = str(out)
    args.logging_dir = str(log_dir)
    args.push = bool(args.push)
    args.skip_test = bool(args.skip_test)

    if args.push:
        args.hub_model_id = resolve_hub_model_id(args.hub_model_id)
    else:
        args.hub_model_id = None

    log.info("Data root: %s", data_root)
    log.info("Output: %s", out)
    log.info("Cache:     %s", cache_dir)

    authenticate_hf()

    df = load_validated_frame(data_root)
    log_missing_clips(df, clips_dir)
    df_train, df_val, df_test = train_val_test_split_df(df)

    processor = WhisperProcessor.from_pretrained(MODEL_ID, language="mongolian", task="transcribe")
    log.info("Processor loaded  sr=%s", processor.feature_extractor.sampling_rate)

    train_ds, val_ds, test_ds = prepare_datasets(
        df_train, df_val, df_test, clips_dir, cache_dir, processor
    )

    trainer = run_training(args, train_ds, val_ds, test_ds, processor)

    if args.push:
        hub_id = args.hub_model_id
        assert hub_id
        log.info("Final push metadata → %s", hub_id)
        trainer.push_to_hub(
            dataset_tags="mozilla-foundation/common_voice_17_0",
            dataset="Common Voice Mongolian 25.0",
            dataset_args="config: mn, split: validated",
            language="mn",
            model_name="Whisper Medium - Mongolian",
            finetuned_from=MODEL_ID,
            tasks="automatic-speech-recognition",
        )
        processor.push_to_hub(hub_id)
        log.info("Done  https://huggingface.co/%s", hub_id)
    else:
        log.info("Skipped Hub push (pass --push). Default repo would be <you>/%s", DEFAULT_HUB_REPO_NAME)

    log.info("TensorBoard: tensorboard --logdir %s", log_dir)
    log.info("Gradio: python gradio_demo.py --model-path %s", out)


if __name__ == "__main__":
    main()
