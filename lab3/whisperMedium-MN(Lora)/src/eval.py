"""Post-training test-set WER (used by run_train.py)."""

import gc
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from datasets import DatasetDict
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import WhisperForConditionalGeneration

import evaluate

from src.collate import DataCollatorSpeechSeq2SeqWithPadding
from src.config import LANGUAGE, MODEL_NAME, TASK


def evaluate_model(model, dataloader, tokenizer, model_name: str = "Model") -> float:
    """Run inference on a DataLoader and return WER (%)."""
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
            labels_np = labels.cpu().numpy()
            labels_np = np.where(labels_np != -100, labels_np, tokenizer.pad_token_id)

            decoded_preds = tokenizer.batch_decode(
                generated_tokens, skip_special_tokens=True
            )
            decoded_labels = tokenizer.batch_decode(
                labels_np, skip_special_tokens=True
            )
            metric.add_batch(predictions=decoded_preds, references=decoded_labels)

        gc.collect()
        torch.cuda.empty_cache()

    return 100 * metric.compute()


def run_evaluation(
    model,
    dsd: DatasetDict,
    data_collator: DataCollatorSpeechSeq2SeqWithPadding,
    processor,
    tokenizer,
    batch_size: int,
) -> Tuple[float, float]:
    """Compare base Whisper vs fine-tuned LoRA WER on the test split."""
    print("\n" + "=" * 60)
    print("  EVALUATION")
    print("=" * 60)

    eval_dataloader = DataLoader(
        dsd["test"], batch_size=batch_size, collate_fn=data_collator
    )

    lora_wer = evaluate_model(
        model, eval_dataloader, tokenizer,
        model_name="Fine-tuned Whisper medium + LoRA",
    )

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
        model_name="Whisper medium (base)",
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
