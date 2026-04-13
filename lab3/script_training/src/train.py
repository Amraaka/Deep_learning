"""Fine-tune Whisper-medium on Common Voice Mongolian (local) with LoRA + Accelerate."""
import logging
import math
import os
import re

import evaluate
import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    get_cosine_schedule_with_warmup,
)
from transformers.models.whisper.english_normalizer import BasicTextNormalizer

from data_collate import DataCollatorSpeechSeq2SeqWithPadding
from local_dataset import load_common_voice_mn
from utils import compute_module_sizes, count_parameters

logging.basicConfig(format="%(message)s", level=logging.INFO)


# ------------------------------ setup ------------------------------

def initialize_accelerator(args):
    return Accelerator(
        mixed_precision=args.mixed_precision,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        log_with="all",
    )


def set_environment(accelerator, seed):
    set_seed(seed)
    if accelerator.device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        # TF32 matmul on Ampere+ (incl. RTX 5080 Blackwell) speeds up fp32 ops.
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    logging.info(
        f"Device: {accelerator.device} | precision: {accelerator.mixed_precision} "
        f"| grad_accum: {accelerator.gradient_accumulation_steps}"
    )


# ------------------------------ data ------------------------------

_MN_PUNCT_RE = re.compile(r"[^\w\sа-яА-ЯөӨүҮёЁ'\-]", re.UNICODE)


def normalize_mn(text: str) -> str:
    text = text.lower().strip()
    text = _MN_PUNCT_RE.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_and_prepare_datasets(args, processor):
    logging.info("Loading local Common Voice Mongolian...")
    sr = processor.feature_extractor.sampling_rate
    datasets = load_common_voice_mn(
        data_root=args.data_root,
        sampling_rate=sr,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )

    if args.debug:
        logging.info(f"Debug mode: {args.debug_subset_size} samples per split.")
        for k in list(datasets.keys()):
            datasets[k] = datasets[k].select(
                range(min(args.debug_subset_size, len(datasets[k])))
            )

    def preprocess(batch):
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

    logging.info("Preprocessing...")
    for k in list(datasets.keys()):
        datasets[k] = datasets[k].map(
            preprocess,
            num_proc=args.num_workers,
            remove_columns=datasets[k].column_names,
            desc=f"preprocess/{k}",
        )

    def keep(batch):
        return [
            (il < args.max_input_length) and (ll > 0) and (ll < 448)
            for il, ll in zip(batch["input_length"], batch["labels_length"])
        ]

    logging.info("Filtering by length...")
    for k in list(datasets.keys()):
        datasets[k] = datasets[k].filter(
            keep, batched=True, batch_size=1000, num_proc=args.num_workers
        )
        logging.info(f"  {k}: {len(datasets[k])} after filter")

    return datasets


# ------------------------------ model ------------------------------

def setup_model(args, processor):
    logging.info(f"Loading {args.model_name_or_path}...")
    model = WhisperForConditionalGeneration.from_pretrained(args.model_name_or_path)

    # SpecAugment on encoder feature frames — big WER win for low-resource ASR.
    model.config.apply_spec_augment = True
    model.config.mask_time_prob = 0.05
    model.config.mask_time_length = 10
    model.config.mask_feature_prob = 0.05
    model.config.mask_feature_length = 10

    # Force language=mn, task=transcribe; drop suppress_tokens to let model predict freely.
    model.generation_config.language = "mn"
    model.generation_config.task = "transcribe"
    model.generation_config.forced_decoder_ids = None
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    model.config.use_cache = False  # required for grad checkpointing

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()

    if args.use_lora:
        lora_config = LoraConfig(
            inference_mode=False,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=0.05,
            # Targeting q/k/v/o + MLP projections improves capacity vs q,v only.
            target_modules=["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"],
            bias="none",
        )
        model = get_peft_model(model, lora_config)

    count_parameters(model)
    sizes = compute_module_sizes(model)
    logging.info(f"Model size: {sizes[''] * 1e-9:.2f} GB")
    return model


# ------------------------------ loaders / optim ------------------------------

def prepare_dataloaders(args, datasets, data_collator):
    train_loader = DataLoader(
        datasets["train"],
        batch_size=args.train_batch_size,
        collate_fn=data_collator,
        num_workers=args.num_workers,
        pin_memory=True,
        shuffle=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        datasets["validation"],
        batch_size=args.eval_batch_size,
        collate_fn=data_collator,
        num_workers=args.num_workers,
        pin_memory=True,
        shuffle=False,
    )
    return train_loader, val_loader


def setup_optimizer_scheduler(args, model, num_training_steps):
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=args.learning_rate,
        betas=(0.9, 0.98),
        weight_decay=args.weight_decay,
    )
    num_warmup = int(num_training_steps * args.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup,
        num_training_steps=num_training_steps,
    )
    return optimizer, scheduler


# ------------------------------ loops ------------------------------

def train_epoch(model, loader, optimizer, scheduler, accelerator):
    model.train()
    total_loss = 0.0
    n = 0
    bar = tqdm(loader, desc="train", disable=not accelerator.is_local_main_process, leave=False)
    for batch in bar:
        with accelerator.accumulate(model):
            outputs = model(**batch)
            loss = outputs.loss
            accelerator.backward(loss)
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        total_loss += loss.detach().float().item()
        n += 1
        bar.set_postfix(loss=f"{loss.item():.3f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")
    return total_loss / max(n, 1)


@torch.no_grad()
def validate(args, model, loader, processor, wer_metric, normalizer, accelerator):
    model.eval()
    unwrapped = accelerator.unwrap_model(model)
    # PEFT models expose the underlying HF model via .generate directly.
    gen_model = unwrapped

    total_loss = 0.0
    n_batches = 0
    preds_all, refs_all = [], []
    bar = tqdm(loader, desc="val", disable=not accelerator.is_local_main_process, leave=False)
    for batch in bar:
        outputs = model(**batch)
        total_loss += outputs.loss.detach().float().item()
        n_batches += 1

        # WER must be computed from generate(), NOT argmax over teacher-forced logits.
        generated = gen_model.generate(
            input_features=batch["input_features"],
            max_new_tokens=225,
            num_beams=args.eval_num_beams,
            language="mn",
            task="transcribe",
        )
        labels = batch["labels"].clone()
        labels[labels == -100] = processor.tokenizer.pad_token_id

        pred_str = processor.batch_decode(generated, skip_special_tokens=True)
        label_str = processor.batch_decode(labels, skip_special_tokens=True)

        pred_str = [normalizer(p) for p in pred_str]
        label_str = [normalizer(l) for l in label_str]
        preds_all.extend(pred_str)
        refs_all.extend(label_str)

    # Filter empty references (normalizer may drop a few) to avoid divide-by-zero.
    pairs = [(p, r) for p, r in zip(preds_all, refs_all) if r.strip()]
    preds_all, refs_all = zip(*pairs) if pairs else ([], [])
    wer = wer_metric.compute(predictions=list(preds_all), references=list(refs_all))
    return total_loss / max(n_batches, 1), wer


def save_model(model, processor, output_dir, accelerator):
    if accelerator.is_main_process:
        os.makedirs(output_dir, exist_ok=True)
        accelerator.unwrap_model(model).save_pretrained(output_dir)
        processor.save_pretrained(output_dir)
        logging.info(f"Saved to {output_dir}")


# ------------------------------ entry ------------------------------

def train(args):
    accelerator = initialize_accelerator(args)
    set_environment(accelerator, args.seed)

    logging.info(f"Loading processor from {args.model_name_or_path}...")
    processor = WhisperProcessor.from_pretrained(
        args.model_name_or_path, language="mn", task="transcribe"
    )

    datasets = load_and_prepare_datasets(args, processor)

    model = setup_model(args, processor)

    # Use the model's real decoder_start_token_id (<|startoftranscript|>),
    # not tokenizer.bos_token_id, which is <|endoftext|> for Whisper. PeftModel
    # proxies .config to the underlying HF model, so this works in both cases.
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
    )

    train_loader, val_loader = prepare_dataloaders(args, datasets, data_collator)

    steps_per_epoch = math.ceil(len(train_loader) / args.gradient_accumulation_steps)
    num_training_steps = steps_per_epoch * args.num_train_epochs
    optimizer, scheduler = setup_optimizer_scheduler(args, model, num_training_steps)

    model, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, val_loader, scheduler
    )

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)

    wer_metric = evaluate.load("wer")
    normalizer = BasicTextNormalizer()

    best_wer = float("inf")
    epochs_no_improve = 0

    logging.info(f"Training {args.num_train_epochs} epochs, {num_training_steps} optimizer steps")
    for epoch in range(args.num_train_epochs):
        logging.info(f"\nEpoch {epoch + 1}/{args.num_train_epochs}")
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, accelerator)
        logging.info(f"train_loss={train_loss:.4f}")

        val_loss, val_wer = validate(
            args, model, val_loader, processor, wer_metric, normalizer, accelerator
        )
        logging.info(f"val_loss={val_loss:.4f}  val_WER={val_wer:.4f}")

        if val_wer < best_wer - args.early_stopping_min_delta:
            best_wer = val_wer
            epochs_no_improve = 0
            save_model(model, processor, args.output_dir, accelerator)
            logging.info(f"new best WER={best_wer:.4f}")
        else:
            epochs_no_improve += 1
            logging.info(f"no improvement for {epochs_no_improve} epoch(s)")
            if epochs_no_improve >= args.early_stopping_patience:
                logging.info("early stopping")
                break

    logging.info(f"Done. Best WER={best_wer:.4f}  artifacts at {args.output_dir}")
