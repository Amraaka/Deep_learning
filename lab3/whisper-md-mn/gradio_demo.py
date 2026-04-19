#!/usr/bin/env python3
"""
  python gradio_demo.py --model-path results
"""

import argparse
from pathlib import Path

import gradio as gr
import librosa
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from src.config import LANGUAGE, TASK


def _resolve_model_dir(model_path: str) -> str:
    """Return a path that actually contains model weight files."""
    base = Path(model_path)
    weight_files = ("model.safetensors", "pytorch_model.bin")

    if any((base / w).exists() for w in weight_files):
        return str(base)

    checkpoints = sorted(
        (p for p in base.glob("checkpoint-*") if p.is_dir()),
        key=lambda p: int(p.name.split("-")[-1]) if p.name.split("-")[-1].isdigit() else -1,
    )
    for ckpt in reversed(checkpoints):
        if any((ckpt / w).exists() for w in weight_files):
            return str(ckpt)

    return str(base)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--model-path",
        type=str,
        default="results",
        help="Directory with config.json, model weights, and tokenizer (from training)",
    )
    p.add_argument("--share", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_dir = _resolve_model_dir(args.model_path)

    processor = WhisperProcessor.from_pretrained(args.model_path)
    model = WhisperForConditionalGeneration.from_pretrained(model_dir)
    model.to(device)
    model.eval()

    def transcribe(audio_path):
        if audio_path is None:
            return "No audio provided."
        audio_arr, _ = librosa.load(audio_path, sr=16_000)
        inputs = processor(audio_arr, sampling_rate=16_000, return_tensors="pt")
        input_features = inputs.input_features.to(device)
        with torch.no_grad():
            ids = model.generate(input_features, language=LANGUAGE, task=TASK)
        return processor.batch_decode(ids, skip_special_tokens=True)[0]

    gr.Interface(
        fn=transcribe,
        inputs=gr.Audio(sources=["microphone", "upload"], type="filepath"),
        outputs="text",
        title="Whisper Medium — Mongolian (full fine-tune)",
        description=f"Model: {model_dir} | Processor: {args.model_path}",
    ).launch(share=True)


if __name__ == "__main__":
    main()
