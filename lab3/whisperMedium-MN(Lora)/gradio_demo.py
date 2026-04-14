#!/usr/bin/env python3
"""
Local Gradio UI for a saved LoRA adapter (directory with adapter_config + weights).

    python gradio_demo.py --adapter-path results/final-lora
"""

import argparse

import gradio as gr
import librosa
import torch
from peft import PeftModel
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from src.config import LANGUAGE, MODEL_NAME, TASK


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--adapter-path",
        type=str,
        default="results/final-lora",
        help="Directory with adapter + tokenizer (e.g. final-lora from training)",
    )
    p.add_argument("--share", action="store_true", help="gradio share=True")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    processor = WhisperProcessor.from_pretrained(args.adapter_path)

    base = WhisperForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(base, args.adapter_path)
    model.eval()

    def transcribe(audio_path):
        if audio_path is None:
            return "No audio provided."
        audio_arr, _ = librosa.load(audio_path, sr=16_000)
        inputs = processor.feature_extractor(
            audio_arr, sampling_rate=16_000, return_tensors="pt"
        )
        input_features = inputs.input_features.to(model.device, dtype=torch.bfloat16)
        model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(
            language=LANGUAGE, task=TASK
        )
        with torch.no_grad():
            ids = model.generate(input_features=input_features)
        return processor.tokenizer.batch_decode(ids, skip_special_tokens=True)[0]

    gr.Interface(
        fn=transcribe,
        inputs=gr.Audio(sources=["microphone", "upload"], type="filepath"),
        outputs="text",
        title="Whisper Medium — Mongolian (LoRA)",
        description=f"Adapter: {args.adapter_path}",
    ).launch(share=args.share)


if __name__ == "__main__":
    main()
