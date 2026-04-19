"""Full Whisper fine-tuning with memmap datasets (from finetune.py)."""

import logging
import shutil
from pathlib import Path
from typing import Any, Optional, Tuple

import numpy as np
import torch
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)

from src.collate import DataCollatorSpeechSeq2SeqWithPadding
from src.config import (
    DEFAULT_EVAL_TASK,
    GENERATION_MAX_LENGTH,
    LANGUAGE,
    LOGGING_STEPS,
    MIN_DISK_GB_FOR_CACHE,
    MODEL_ID,
    SAVE_TOTAL_LIMIT,
    SEED,
)
from src.memmap import WhisperMemmapDataset, preprocess_to_memmap
from src.metrics import make_compute_metrics

log = logging.getLogger(__name__)


def check_disk_space(cache_dir: Path, min_gb: float = MIN_DISK_GB_FOR_CACHE) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    free_gb = shutil.disk_usage(str(cache_dir)).free / 1e9
    log.info("Free disk: %.1f GB  (need ~%.0f GB for feature cache)", free_gb, min_gb)
    if free_gb < min_gb:
        raise RuntimeError(f"Not enough disk space: {free_gb:.1f} GB (need >= {min_gb})")


def build_model(eval_task: str = DEFAULT_EVAL_TASK) -> WhisperForConditionalGeneration:
    model = WhisperForConditionalGeneration.from_pretrained(MODEL_ID)
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.generation_config.language = LANGUAGE
    model.generation_config.task = eval_task
    model.generation_config.forced_decoder_ids = None
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    log.info("Model loaded  params=%.0fM  grad_ckpt=on", n_params)
    return model


def prepare_datasets(
    df_train,
    df_val,
    df_test,
    clips_dir: Path,
    cache_dir: Path,
    processor: WhisperProcessor,
    task_mode: str,
    eval_task: str,
) -> Tuple[WhisperMemmapDataset, WhisperMemmapDataset, WhisperMemmapDataset]:
    check_disk_space(cache_dir)
    target_sr = processor.feature_extractor.sampling_rate
    include_translate_labels = task_mode in {"translate", "multitask"}
    eval_mode = eval_task if task_mode == "multitask" else task_mode
    log.info("Preprocessing (first run long; instant on cache hit)...")
    train_ds = preprocess_to_memmap(
        df_train,
        "train",
        clips_dir,
        cache_dir,
        processor,
        target_sr,
        language=LANGUAGE,
        mode=task_mode,
        include_translate_labels=include_translate_labels,
    )
    val_ds = preprocess_to_memmap(
        df_val,
        "val",
        clips_dir,
        cache_dir,
        processor,
        target_sr,
        language=LANGUAGE,
        mode=eval_mode,
        include_translate_labels=include_translate_labels,
    )
    test_ds = preprocess_to_memmap(
        df_test,
        "test",
        clips_dir,
        cache_dir,
        processor,
        target_sr,
        language=LANGUAGE,
        mode=eval_mode,
        include_translate_labels=include_translate_labels,
    )
    log.info(
        "Ready  train=%s  val=%s  test=%s",
        f"{len(train_ds):,}",
        f"{len(val_ds):,}",
        f"{len(test_ds):,}",
    )
    return train_ds, val_ds, test_ds


def run_training(
    args: Any,
    train_dataset: WhisperMemmapDataset,
    val_dataset: WhisperMemmapDataset,
    test_dataset: WhisperMemmapDataset,
    processor: WhisperProcessor,
    task_mode: str,
    eval_task: str,
) -> Seq2SeqTrainer:
    output_dir = Path(args.output_dir)
    log_dir = Path(args.logging_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=processor.tokenizer.bos_token_id,
    )

    model = build_model(eval_task=eval_task)
    compute_metrics = make_compute_metrics(processor, task=eval_task)

    use_cuda = torch.cuda.is_available()
    workers = 4 if use_cuda else 0
    pin_memory = bool(use_cuda)

    hub_id: Optional[str] = getattr(args, "hub_model_id", None) if args.push else None

    ta_kwargs = dict(
        output_dir=str(output_dir),
        bf16=use_cuda and torch.cuda.is_bf16_supported(),
        fp16=False,
        gradient_checkpointing=True,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        optim="adamw_bnb_8bit",
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        max_steps=args.max_steps,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=SAVE_TOTAL_LIMIT,
        load_best_model_at_end=True,
        metric_for_best_model="wer" if eval_task == "transcribe" else "bleu",
        greater_is_better=False if eval_task == "transcribe" else True,
        predict_with_generate=True,
        generation_max_length=GENERATION_MAX_LENGTH,
        logging_dir=str(log_dir),
        logging_steps=LOGGING_STEPS,
        report_to=["tensorboard"],
        seed=SEED,
        dataloader_num_workers=workers,
        dataloader_pin_memory=pin_memory,
        remove_unused_columns=False,
    )
    if args.push:
        ta_kwargs["push_to_hub"] = True
        ta_kwargs["hub_model_id"] = hub_id
        ta_kwargs["hub_strategy"] = "checkpoint"
    else:
        ta_kwargs["push_to_hub"] = False

    training_args = Seq2SeqTrainingArguments(**ta_kwargs)

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        processing_class=processor.feature_extractor,
    )
    processor.save_pretrained(str(output_dir))
    log.info("Trainer ready. Processor saved to %s", output_dir)

    log.info("Starting training  steps=%s  eval_every=%s", args.max_steps, args.eval_steps)
    train_result = trainer.train()
    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)
    trainer.save_state()

    log.info("Training complete")
    log.info("  loss        : %.4f", train_result.metrics.get("train_loss", float("nan")))
    log.info(
        "  samples/sec : %.2f",
        train_result.metrics.get("train_samples_per_second", 0.0),
    )
    log.info(
        "  runtime     : %.2f h",
        train_result.metrics.get("train_runtime", 0) / 3600,
    )

    if not getattr(args, "skip_test", False):
        log.info("Evaluating on test set...")
        test_results = trainer.evaluate(
            eval_dataset=test_dataset, metric_key_prefix="test"
        )
        trainer.log_metrics("test", test_results)
        trainer.save_metrics("test", test_results)
        test_wer = test_results.get("test_wer")
        test_bleu = test_results.get("test_bleu")
        log.info("=" * 50)
        if test_wer is not None:
            log.info("  Test WER : %.2f%%", test_wer)
        elif test_bleu is not None:
            log.info("  Test BLEU: %.2f", test_bleu)
        else:
            log.info("  Expected metric missing from test results")
        log.info("  Eval task: %s", eval_task)
        if test_wer is not None:
            status = "PASSED" if test_wer < 48.0 else "NOT YET — try more steps or lower LR"
            log.info("  Status   : %s", status)
        elif test_bleu is not None:
            log.info("  Status   : BLEU reported (higher is better)")
        log.info("=" * 50)
    else:
        log.info("[SKIP] Full test WER (--skip-test)")

    _log_sample_predictions(trainer.model, test_dataset, processor, eval_task=eval_task)

    return trainer


def _log_sample_predictions(
    model: WhisperForConditionalGeneration,
    test_dataset: WhisperMemmapDataset,
    processor: WhisperProcessor,
    eval_task: str,
    n: int = 5,
) -> None:
    if not torch.cuda.is_available():
        log.info("Skipping sample predictions (no CUDA)")
        return
    device = torch.device("cuda")
    model.eval()
    model.to(device)
    log.info("Sample predictions on %s test examples:", n)
    for i in range(min(n, len(test_dataset))):
        sample = test_dataset[i]
        mel = np.array(sample["input_features"], dtype=np.float32)
        input_features = torch.tensor(mel).unsqueeze(0).to(device)
        with torch.no_grad():
            predicted_ids = model.generate(
                input_features, language=LANGUAGE, task=eval_task
            )
        label_ids = [
            t if t != -100 else processor.tokenizer.pad_token_id
            for t in sample["labels"]
        ]
        ref = processor.tokenizer.decode(label_ids, skip_special_tokens=True)
        hyp = processor.tokenizer.decode(predicted_ids[0], skip_special_tokens=True)
        log.info("  [%s] REF : %s", i + 1, ref)
        log.info("       HYP : %s", hyp)
