"""LoRA training: checkpoints + final-lora adapter save."""

import os
from typing import Any

import torch
from datasets import DatasetDict
from peft import LoraConfig, get_peft_model
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperForConditionalGeneration,
    WhisperProcessor,
    WhisperTokenizer,
)

from src.collate import DataCollatorSpeechSeq2SeqWithPadding
from src.config import (
    GENERATION_MAX_LENGTH,
    LANGUAGE,
    LORA_ALPHA,
    LORA_DROPOUT,
    LORA_R,
    LORA_TARGET_MODULES,
    MODEL_NAME,
    SEED,
    TASK,
    WARMUP_STEPS,
)
from src.metrics import make_compute_metrics


def run_training(
    args: Any,
    dsd: DatasetDict,
    tokenizer: WhisperTokenizer,
    processor: WhisperProcessor,
) -> Seq2SeqTrainer:
    """Train LoRA; write checkpoints to args.output_dir and adapter to output_dir/final-lora."""
    print("\n[STAGE] Preparing on-the-fly batch preprocessing...")
    print("  Raw rows stay in the dataset; audio is loaded per batch to reduce RAM usage.")
    print(f"  Train columns: {dsd['train'].column_names}")
    print(f"  Train samples: {len(dsd['train'])}")
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

    print("\n[STAGE] Loading Whisper medium (BF16)...")
    model = WhisperForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    forced_decoder_ids = processor.get_decoder_prompt_ids(
        language=LANGUAGE, task=TASK
    )
    model.config.forced_decoder_ids = forced_decoder_ids
    model.config.suppress_tokens = []

    print("\n[STAGE] Applying LoRA adapters...")
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES,
    )
    model.enable_input_require_grads()
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    print("\n[STAGE] Setting up training...")
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        warmup_steps=WARMUP_STEPS,
        num_train_epochs=args.num_epochs,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        bf16=True,
        generation_max_length=GENERATION_MAX_LENGTH,
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

    print("\n[STAGE] Training...")
    print(f"  TensorBoard logs: {args.logging_dir}/")
    print("  Run 'tensorboard --logdir runs/' in another terminal to monitor.\n")
    trainer.train()

    final_dir = os.path.join(args.output_dir, "final-lora")
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"\n[OK] LoRA adapter saved to {final_dir}")

    return trainer
