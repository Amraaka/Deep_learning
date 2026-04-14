#!/usr/bin/env python3
"""
Gradio ASR for a **full** fine-tuned Whisper checkpoint (not LoRA).

  python gradio_demo.py --model-path results

Use the training output directory (or a checkpoint-* folder with config + weights).
"""

import argparse

import gradio as gr
import librosa
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from src.config import LANGUAGE, TASK


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

    processor = WhisperProcessor.from_pretrained(args.model_path)
    model = WhisperForConditionalGeneration.from_pretrained(args.model_path)
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
        description=f"Model: {args.model_path}",
    ).launch(share=args.share)


if __name__ == "__main__":
    main()
