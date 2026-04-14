#!/usr/bin/env python3
"""
Fine-tune Whisper medium on Common Voice Mongolian using LoRA.

Usage:
    # Sanity check (no training, just dataset stats + base model preview):
    python train_whisper_mn.py --sanity-check

    # Full training + evaluation + Gradio demo:
    python train_whisper_mn.py

    # Training only (skip optional stages):
    python train_whisper_mn.py --skip-eval --skip-push --skip-gradio

    # View TensorBoard logs (in another terminal):
    tensorboard --logdir runs/
"""

import argparse
import gc
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Union

import evaluate
import numpy as np
import pandas as pd
import soundfile as sf
import torch
from datasets import Dataset, DatasetDict
from huggingface_hub import login
from peft import LoraConfig, get_peft_model
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperFeatureExtractor,
    WhisperForConditionalGeneration,
    WhisperProcessor,
    WhisperTokenizer,
)

# ---------------------------------------------------------------------------
# CLI arguments
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune Whisper medium on Common Voice Mongolian with LoRA"
    )
    parser.add_argument(
        "--sanity-check",
        action="store_true",
        help="Print dataset stats, sample predictions from the base model, then exit.",
    )
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="Skip post-training WER evaluation.",
    )
    parser.add_argument(
        "--skip-push",
        action="store_true",
        help="Skip pushing the LoRA adapter to Hugging Face Hub.",
    )
    parser.add_argument(
        "--skip-gradio",
        action="store_true",
        help="Skip launching the Gradio demo after training.",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="common_voice_mn",
        help="Path to the Common Voice Mongolian dataset directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Directory for checkpoints and final model.",
    )
    parser.add_argument(
        "--logging-dir",
        type=str,
        default="runs/whisper-mn-lora",
        help="TensorBoard logging directory.",
    )
    parser.add_argument(
        "--hub-model-id",
        type=str,
        default=None,
        help="HF Hub repo id for pushing. Auto-generated if not set.",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=5,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Per-device train/eval batch size.",
    )
    parser.add_argument(
        "--grad-accum",
        type=int,
        default=8,
        help="Gradient accumulation steps (effective batch = batch_size * grad_accum).",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
        help="Peak learning rate for LoRA training.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_NAME = "openai/whisper-medium"
LANGUAGE = "Mongolian"
LANGUAGE_ABBR = "mn"
TASK = "transcribe"
SEED = 42
MAX_AUDIO_SEC = 30.0
MIN_AUDIO_SEC = 1.0


# ---------------------------------------------------------------------------
# Hugging Face authentication
# ---------------------------------------------------------------------------

def authenticate_hf():
    token = os.environ.get("HF_TOKEN")

    # Fall back to .env file in the working directory
    if not token:
        env_path = Path(".env")
        if env_path.exists():
            for line in env_path.read_text().splitlines():
                line = line.strip()
                if line.startswith("HF_TOKEN="):
                    token = line.split("=", 1)[1].strip()
                    os.environ["HF_TOKEN"] = token
                    break

    if token:
        login(token=token)
        print("[OK] Authenticated with Hugging Face.")
    else:
        print("[WARN] HF_TOKEN not found in environment or .env file. Push-to-hub will fail.")
        print("       Set it with: export HF_TOKEN=hf_...")


# ---------------------------------------------------------------------------
# Dataset loading and filtering
# ---------------------------------------------------------------------------

def load_common_voice_validated(data_dir: str) -> Dataset:
    """Load validated.tsv and return a HuggingFace Dataset with audio file paths."""
    data_path = Path(data_dir)
    tsv_path = data_path / "validated.tsv"
    clips_dir = data_path / "clips"

    if not tsv_path.exists():
        sys.exit(f"[ERROR] validated.tsv not found at {tsv_path}")
    if not clips_dir.exists():
        sys.exit(f"[ERROR] clips directory not found at {clips_dir}")

    df = pd.read_csv(tsv_path, sep="\t")
    print(f"[INFO] Loaded validated.tsv: {len(df)} rows")

    # Build absolute audio paths
    df["audio_path"] = df["path"].apply(lambda p: str(clips_dir / p))

    # Keep only rows where the audio file actually exists
    exists_mask = df["audio_path"].apply(os.path.isfile)
    missing = (~exists_mask).sum()
    if missing > 0:
        print(f"[WARN] {missing} audio files not found on disk — skipping them.")
    df = df[exists_mask].reset_index(drop=True)

    # Store paths as strings — audio is decoded on-the-fly with soundfile
    ds = Dataset.from_dict({
        "audio_path": df["audio_path"].tolist(),
        "sentence": df["sentence"].tolist(),
    })
    return ds


def filter_dataset(ds: Dataset, tokenizer: WhisperTokenizer) -> Dataset:
    """Filter by audio duration (via soundfile header) and label token length."""
    max_target_tokens = 448  # Whisper default max_target_positions

    def is_valid(example):
        try:
            info = sf.info(example["audio_path"])
            duration = info.duration
        except Exception:
            return False
        if duration < MIN_AUDIO_SEC or duration > MAX_AUDIO_SEC:
            return False
        token_ids = tokenizer(example["sentence"], verbose=False).input_ids
        if len(token_ids) > max_target_tokens:
            return False
        return True

    before = len(ds)
    ds = ds.filter(is_valid)
    after = len(ds)
    print(f"[INFO] Filtered: {before} -> {after} samples "
          f"(removed {before - after} by duration/label length)")
    return ds


# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------

def split_dataset(ds: Dataset) -> DatasetDict:
    """Split into 80% train / 10% validation / 10% test."""
    # First split: 80% train, 20% rest
    split1 = ds.train_test_split(test_size=0.2, seed=SEED)
    # Second split: 50/50 of the 20% -> 10% val, 10% test
    split2 = split1["test"].train_test_split(test_size=0.5, seed=SEED)

    dsd = DatasetDict({
        "train": split1["train"],
        "validation": split2["train"],
        "test": split2["test"],
    })
    for name, part in dsd.items():
        print(f"  {name:>12s}: {len(part)} samples")
    return dsd


# ---------------------------------------------------------------------------
# Audio loading
# ---------------------------------------------------------------------------

def load_audio_array(audio_path: str) -> np.ndarray:
    """Load an audio file, convert to mono, and resample to 16kHz if needed."""
    audio_arr, sr = sf.read(audio_path, dtype="float32", always_2d=False)

    if getattr(audio_arr, "ndim", 1) == 2:
        audio_arr = audio_arr.mean(axis=1)

    if sr != 16_000:
        import librosa

        audio_arr = librosa.resample(audio_arr, orig_sr=sr, target_sr=16_000)

    return np.asarray(audio_arr, dtype=np.float32)


# ---------------------------------------------------------------------------
# Data collator
# ---------------------------------------------------------------------------

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        input_feats = []
        label_feats = []

        for feature in features:
            audio_arr = load_audio_array(feature["audio_path"])
            input_feats.append(
                {
                    "input_features": self.processor.feature_extractor(
                        audio_arr, sampling_rate=16_000
                    ).input_features[0]
                }
            )
            label_feats.append(
                {
                    "input_ids": self.processor.tokenizer(
                        feature["sentence"], verbose=False
                    ).input_ids
                }
            )

        batch = self.processor.feature_extractor.pad(input_feats, return_tensors="pt")
        batch["input_features"] = batch["input_features"].to(dtype=torch.bfloat16)
        labels_padded = self.processor.tokenizer.pad(label_feats, return_tensors="pt")
        labels = labels_padded["input_ids"].masked_fill(
            labels_padded.attention_mask.ne(1), -100
        )
        batch["labels"] = labels
        return batch


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def make_compute_metrics(tokenizer):
    metric = evaluate.load("wer")

    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids
        label_ids[label_ids == -100] = tokenizer.pad_token_id
        pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        wer = 100 * metric.compute(predictions=pred_str, references=label_str)
        return {"wer": wer}

    return compute_metrics


# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------

def run_sanity_check(dsd: DatasetDict, tokenizer, feature_extractor, processor):
    """Print dataset stats and run base Whisper on a few test samples."""
    print("\n" + "=" * 60)
    print("  SANITY CHECK")
    print("=" * 60)

    # Dataset stats
    for split_name, split_ds in dsd.items():
        print(f"\n--- {split_name} split ---")
        print(f"  Samples: {len(split_ds)}")

    # Duration stats from train split (sample up to 500 for speed, uses soundfile header only)
    print("\n--- Duration statistics (train split, sampled) ---")
    sample_ds = dsd["train"].select(range(min(500, len(dsd["train"]))))
    durations = []
    for s in sample_ds:
        try:
            durations.append(sf.info(s["audio_path"]).duration)
        except Exception:
            pass
    durations = np.array(durations)
    print(f"  Min:    {durations.min():.2f}s")
    print(f"  Max:    {durations.max():.2f}s")
    print(f"  Mean:   {durations.mean():.2f}s")
    print(f"  Median: {np.median(durations):.2f}s")

    # Show 3 sample transcripts
    print("\n--- Sample transcripts (test split) ---")
    for i in range(min(3, len(dsd["test"]))):
        sample = dsd["test"][i]
        print(f"  [{i}] {sample['sentence']}")
        print(f"       {sample['audio_path']}")

    # VRAM estimate
    print("\n--- VRAM estimate ---")
    if torch.cuda.is_available():
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        reserved = torch.cuda.memory_reserved(0) / 1e9
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  Total VRAM: {total:.1f} GB")
        print(f"  Currently reserved: {reserved:.2f} GB")
        model_est = 1.5  # Whisper medium in FP16/BF16
        lora_est = 0.02  # LoRA params + optimizer states
        batch_est = 2.0  # activations for batch_size=4
        overhead = 1.0
        total_est = model_est + lora_est + batch_est + overhead
        print(f"  Estimated training VRAM: ~{total_est:.1f} GB")
        if total_est < total:
            print(f"  --> Should fit in {total:.0f} GB VRAM")
        else:
            print(f"  [WARN] Might be tight! Consider reducing batch size.")
    else:
        print("  [WARN] No CUDA device detected.")

    # Base model predictions on 3 test samples
    print("\n--- Base model predictions (no fine-tuning) ---")
    model = WhisperForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()

    forced_decoder_ids = processor.get_decoder_prompt_ids(
        language=LANGUAGE, task=TASK
    )
    model.config.forced_decoder_ids = forced_decoder_ids

    metric = evaluate.load("wer")
    preds_all, refs_all = [], []

    for i in range(min(3, len(dsd["test"]))):
        sample = dsd["test"][i]
        audio_arr = load_audio_array(sample["audio_path"])
        inputs = feature_extractor(audio_arr, sampling_rate=16_000, return_tensors="pt")
        input_features = inputs.input_features.to(model.device, dtype=torch.bfloat16)

        with torch.no_grad():
            generated_ids = model.generate(input_features=input_features)

        transcription = tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0]
        reference = sample["sentence"]
        preds_all.append(transcription)
        refs_all.append(reference)
        print(f"\n  Sample {i}:")
        print(f"    Reference:  {reference}")
        print(f"    Prediction: {transcription}")

    wer_score = 100 * metric.compute(predictions=preds_all, references=refs_all)
    print(f"\n  Base model WER on {len(preds_all)} samples: {wer_score:.2f}%")

    del model
    gc.collect()
    torch.cuda.empty_cache()

    print("\n" + "=" * 60)
    print("  Sanity check complete. Exiting without training.")
    print("=" * 60 + "\n")


# ---------------------------------------------------------------------------
# Evaluation (post-training)
# ---------------------------------------------------------------------------

def evaluate_model(model, dataloader, tokenizer, model_name="Model"):
    """Run inference on a DataLoader and return WER."""
    from tqdm import tqdm

    model.eval()
    metric = evaluate.load("wer")
    progress = tqdm(dataloader, desc=f"Evaluating {model_name}")

    for batch in progress:
        input_features = batch["input_features"].to(
            model.device, dtype=torch.bfloat16
        )
        labels = batch["labels"].to(model.device)

        with torch.no_grad():
            generated_tokens = (
                model.generate(
                    input_features=input_features,
                )
                .cpu()
                .numpy()
            )
            labels = labels.cpu().numpy()
            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

            decoded_preds = tokenizer.batch_decode(
                generated_tokens, skip_special_tokens=True
            )
            decoded_labels = tokenizer.batch_decode(
                labels, skip_special_tokens=True
            )
            metric.add_batch(predictions=decoded_preds, references=decoded_labels)

        gc.collect()
        torch.cuda.empty_cache()

    return 100 * metric.compute()


def run_evaluation(model, dsd, data_collator, processor, tokenizer, args):
    """Compare base Whisper vs fine-tuned LoRA model WER."""
    from torch.utils.data import DataLoader

    print("\n" + "=" * 60)
    print("  EVALUATION")
    print("=" * 60)

    eval_dataloader = DataLoader(
        dsd["test"], batch_size=args.batch_size, collate_fn=data_collator
    )

    # Fine-tuned model WER
    lora_wer = evaluate_model(
        model, eval_dataloader, tokenizer,
        model_name="Fine-tuned Whisper medium + LoRA"
    )

    # Base model WER
    print("\nLoading base model for comparison...")
    base_model = WhisperForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    base_model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(
        language=LANGUAGE, task=TASK
    )
    base_model.config.suppress_tokens = []
    base_wer = evaluate_model(
        base_model, eval_dataloader, tokenizer,
        model_name="Whisper medium (base)"
    )

    del base_model
    gc.collect()
    torch.cuda.empty_cache()

    print("\n--- Evaluation Results ---")
    results = pd.DataFrame({
        "Model": ["Whisper medium (base)", "Fine-tuned Whisper medium + LoRA"],
        "WER (%)": [base_wer, lora_wer],
    })
    print(results.to_string(index=False))
    improvement = base_wer - lora_wer
    print(f"\nImprovement: {improvement:.2f} WER points")

    if lora_wer < 48.0:
        print(f"[OK] Target WER < 48% achieved! ({lora_wer:.2f}%)")
    else:
        print(f"[INFO] WER is {lora_wer:.2f}%. Consider more epochs or larger LoRA rank.")

    return lora_wer, base_wer


# ---------------------------------------------------------------------------
# Gradio demo
# ---------------------------------------------------------------------------

def launch_gradio(model, processor, tokenizer):
    """Launch a Gradio microphone transcription demo."""
    import gradio as gr

    model.eval()

    def transcribe(audio_path):
        if audio_path is None:
            return "No audio provided."
        import librosa
        audio_arr, sr = librosa.load(audio_path, sr=16_000)
        inputs = processor.feature_extractor(
            audio_arr, sampling_rate=16_000, return_tensors="pt"
        )
        input_features = inputs.input_features.to(model.device, dtype=torch.bfloat16)

        forced_decoder_ids = processor.get_decoder_prompt_ids(
            language=LANGUAGE, task=TASK
        )
        model.config.forced_decoder_ids = forced_decoder_ids

        with torch.no_grad():
            generated_ids = model.generate(input_features=input_features)
        transcription = tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0]
        return transcription

    demo = gr.Interface(
        fn=transcribe,
        inputs=gr.Audio(sources=["microphone", "upload"], type="filepath"),
        outputs="text",
        title="Whisper Medium - Mongolian (LoRA Fine-tuned)",
        description="Speak in Mongolian or upload an audio file to transcribe.",
    )
    demo.launch(share=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    print("=" * 60)
    print("  Whisper Medium - Mongolian LoRA Fine-tuning")
    print("=" * 60)
    print(f"  Model:      {MODEL_NAME}")
    print(f"  Language:   {LANGUAGE} ({LANGUAGE_ABBR})")
    print(f"  Data dir:   {args.data_dir}")
    print(f"  Output dir: {args.output_dir}")
    print(f"  Epochs:     {args.num_epochs}")
    print(f"  Batch size: {args.batch_size} x {args.grad_accum} grad accum "
          f"= {args.batch_size * args.grad_accum} effective")
    print(f"  LR:         {args.learning_rate}")
    print()

    # ---- HF auth ----
    authenticate_hf()

    # ---- Tokenizer & feature extractor (needed for filtering) ----
    tokenizer = WhisperTokenizer.from_pretrained(
        MODEL_NAME, language=LANGUAGE, task=TASK
    )
    feature_extractor = WhisperFeatureExtractor.from_pretrained(MODEL_NAME)
    processor = WhisperProcessor.from_pretrained(
        MODEL_NAME, language=LANGUAGE, task=TASK
    )

    # ---- Load dataset ----
    print("\n[STAGE] Loading dataset...")
    ds = load_common_voice_validated(args.data_dir)

    # ---- Filter ----
    print("\n[STAGE] Filtering by duration and label length...")
    ds = filter_dataset(ds, tokenizer)

    # ---- Split ----
    print("\n[STAGE] Splitting 80/10/10...")
    dsd = split_dataset(ds)

    # ---- Sanity check (before expensive feature extraction) ----
    if args.sanity_check:
        run_sanity_check(dsd, tokenizer, feature_extractor, processor)
        sys.exit(0)

    # ---- Data collator ----
    print("\n[STAGE] Preparing on-the-fly batch preprocessing...")
    print("  Raw rows stay in the dataset; audio is loaded per batch to reduce RAM usage.")
    print(f"  Train columns: {dsd['train'].column_names}")
    print(f"  Train samples: {len(dsd['train'])}")
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

    # ---- Load model ----
    print("\n[STAGE] Loading Whisper medium (BF16)...")
    model = WhisperForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    # Configure decoder for Mongolian
    forced_decoder_ids = processor.get_decoder_prompt_ids(
        language=LANGUAGE, task=TASK
    )
    model.config.forced_decoder_ids = forced_decoder_ids
    model.config.suppress_tokens = []

    # ---- LoRA ----
    print("\n[STAGE] Applying LoRA adapters...")
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj"],
    )
    model.enable_input_require_grads()
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ---- Training arguments ----
    print("\n[STAGE] Setting up training...")
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        warmup_steps=50,
        num_train_epochs=args.num_epochs,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        bf16=True,
        generation_max_length=225,
        logging_steps=25,
        logging_dir=args.logging_dir,
        report_to=["tensorboard"],
        remove_unused_columns=False,
        label_names=["labels"],
        predict_with_generate=True,
        seed=SEED,
        dataloader_num_workers=0,
        save_total_limit=2,
    )

    compute_metrics = make_compute_metrics(tokenizer)

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=dsd["train"],
        eval_dataset=dsd["validation"],
        data_collator=data_collator,
        processing_class=processor,
        compute_metrics=compute_metrics,
    )

    # ---- Train ----
    print("\n[STAGE] Training...")
    print(f"  TensorBoard logs: {args.logging_dir}/")
    print(f"  Run 'tensorboard --logdir runs/' in another terminal to monitor.\n")
    trainer.train()

    # Save final LoRA adapter
    final_dir = os.path.join(args.output_dir, "final-lora")
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"\n[OK] LoRA adapter saved to {final_dir}")

    # ---- Evaluation ----
    if not args.skip_eval:
        run_evaluation(model, dsd, data_collator, processor, tokenizer, args)
    else:
        print("\n[SKIP] Evaluation (--skip-eval)")

    # ---- Push to Hub ----
    if not args.skip_push:
        print("\n[STAGE] Pushing LoRA adapter to Hugging Face Hub...")
        hub_id = args.hub_model_id
        if hub_id is None:
            hub_id = f"whisper-medium-mn-lora"
        model.push_to_hub(hub_id)
        tokenizer.push_to_hub(hub_id)
        print(f"[OK] Pushed to https://huggingface.co/{hub_id}")
    else:
        print("\n[SKIP] Push to Hub (--skip-push)")

    # ---- Gradio demo ----
    if not args.skip_gradio:
        print("\n[STAGE] Launching Gradio demo...")
        launch_gradio(model, processor, tokenizer)
    else:
        print("\n[SKIP] Gradio demo (--skip-gradio)")

    print("\n[DONE] All stages complete.")


if __name__ == "__main__":
    main()
