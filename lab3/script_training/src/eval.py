"""Evaluate a fine-tuned Whisper checkpoint on the local Common Voice (mn) test split.

Supports:
  * Full fine-tuned HF models saved under --model_dir
  * PEFT / LoRA adapters saved under --model_dir (auto-detects adapter_config.json
    and loads the base model from it)
"""
import argparse
import logging
import os
from pathlib import Path

import evaluate
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from transformers.models.whisper.english_normalizer import BasicTextNormalizer

from data_collate import DataCollatorSpeechSeq2SeqWithPadding
from local_dataset import load_common_voice_mn
from train import normalize_mn

logging.basicConfig(format="%(message)s", level=logging.INFO)


def _is_peft_checkpoint(model_dir: Path) -> bool:
    return (model_dir / "adapter_config.json").exists()


def _load_model_and_processor(model_dir: str):
    path = Path(model_dir)
    if _is_peft_checkpoint(path):
        from peft import PeftConfig, PeftModel

        peft_cfg = PeftConfig.from_pretrained(str(path))
        base_id = peft_cfg.base_model_name_or_path
        logging.info(f"PEFT adapter detected. base={base_id}, adapter={path}")
        base = WhisperForConditionalGeneration.from_pretrained(base_id)
        model = PeftModel.from_pretrained(base, str(path))
        # Processor is usually saved alongside adapters; fall back to base otherwise.
        proc_src = str(path) if (path / "preprocessor_config.json").exists() else base_id
    else:
        logging.info(f"Loading full HF model from {path}")
        model = WhisperForConditionalGeneration.from_pretrained(str(path))
        proc_src = str(path)

    processor = WhisperProcessor.from_pretrained(proc_src, language="mn", task="transcribe")
    return model, processor


def _preprocess_test_split(dataset, processor, max_input_length: float, num_workers: int):
    def _prep(batch):
        audio = batch["audio"]
        text = normalize_mn(batch["transcription"])
        inputs = processor(
            audio=audio["array"],
            sampling_rate=audio["sampling_rate"],
            text=text,
        )
        inputs["input_length"] = len(audio["array"]) / audio["sampling_rate"]
        inputs["labels_length"] = len(inputs["labels"])
        return inputs

    dataset = dataset.map(
        _prep,
        num_proc=num_workers,
        remove_columns=dataset.column_names,
        desc="preprocess/test",
    )

    def _keep(batch):
        return [
            (il < max_input_length) and (0 < ll < 448)
            for il, ll in zip(batch["input_length"], batch["labels_length"])
        ]

    return dataset.filter(_keep, batched=True, batch_size=1000, num_proc=num_workers)


@torch.no_grad()
def evaluate_checkpoint(
    model_dir: str,
    data_root: str,
    batch_size: int = 8,
    num_workers: int = 4,
    max_input_length: float = 20.0,
    num_beams: int = 5,
    val_ratio: float = 0.05,
    test_ratio: float = 0.05,
    seed: int = 42,
):
    device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    logging.info(f"Device: {device}")

    model, processor = _load_model_and_processor(model_dir)
    model.to(device)
    model.eval()

    # Match fp16 on GPU where available to match training-time throughput.
    if device == "cuda":
        model = model.to(dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16)

    sr = processor.feature_extractor.sampling_rate
    ds = load_common_voice_mn(
        data_root=data_root,
        sampling_rate=sr,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )
    test_ds = _preprocess_test_split(ds["test"], processor, max_input_length, num_workers)
    logging.info(f"test samples after filter: {len(test_ds)}")

    collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
    )
    loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        collate_fn=collator,
        num_workers=num_workers,
        pin_memory=(device == "cuda"),
        shuffle=False,
    )

    wer_metric = evaluate.load("wer")
    normalizer = BasicTextNormalizer()
    preds, refs = [], []

    for batch in tqdm(loader, desc="eval"):
        input_features = batch["input_features"].to(device)
        if device == "cuda":
            input_features = input_features.to(dtype=model.dtype)

        generated = model.generate(
            input_features=input_features,
            max_new_tokens=225,
            num_beams=num_beams,
            language="mn",
            task="transcribe",
        )
        pred_str = processor.batch_decode(generated, skip_special_tokens=True)

        labels = batch["labels"].clone()
        labels[labels == -100] = processor.tokenizer.pad_token_id
        label_str = processor.batch_decode(labels, skip_special_tokens=True)

        preds.extend(normalizer(p) for p in pred_str)
        refs.extend(normalizer(r) for r in label_str)

    pairs = [(p, r) for p, r in zip(preds, refs) if r.strip()]
    preds, refs = zip(*pairs) if pairs else ([], [])
    wer = wer_metric.compute(predictions=list(preds), references=list(refs))
    logging.info(f"WER on CV-mn test split: {wer:.4f}  ({len(preds)} samples)")
    return wer


def main():
    parser = argparse.ArgumentParser(description="Evaluate Whisper checkpoint on CV-mn test split.")
    parser.add_argument("--model_dir", type=str, required=True,
                        help="Path to a full HF checkpoint or PEFT adapter directory.")
    parser.add_argument("--data_root", type=str, default="lab3/common_voice_mn")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_input_length", type=float, default=20.0)
    parser.add_argument("--num_beams", type=int, default=5)
    parser.add_argument("--val_ratio", type=float, default=0.05)
    parser.add_argument("--test_ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    evaluate_checkpoint(
        model_dir=args.model_dir,
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_input_length=args.max_input_length,
        num_beams=args.num_beams,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
