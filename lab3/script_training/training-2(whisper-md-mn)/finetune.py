#!/usr/bin/env python3
"""
Fine-Tune Whisper Medium on Mongolian Common Voice 25.0
Goal: WER < 48%

Usage:
    cd lab3/script_training/training-2
    source ../../venv/bin/activate      # adjust to your venv
    python finetune.py

Monitor training (separate terminal):
    tensorboard --logdir ./whisper-medium-mn/runs

Requirements (install once):
    pip install transformers datasets accelerate evaluate jiwer tensorboard \
                bitsandbytes soundfile librosa huggingface_hub scikit-learn \
                matplotlib python-dotenv tqdm
"""

# ── Stdlib ─────────────────────────────────────────────────────────────────
import json
import logging
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Union

# ── Third-party ────────────────────────────────────────────────────────────
import librosa
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm

from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
from transformers.models.whisper.english_normalizer import BasicTextNormalizer
import evaluate
from huggingface_hub import login, whoami

# ── Logging ────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════
# 1. GPU CHECK
# ═══════════════════════════════════════════════════════════════════════════
log.info(subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout)
assert torch.cuda.is_available(), "No CUDA GPU detected!"
device = torch.device("cuda")
log.info(f"GPU  : {torch.cuda.get_device_name(0)}")
log.info(f"VRAM : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
log.info(f"BF16 : {torch.cuda.is_bf16_supported()}")

# ═══════════════════════════════════════════════════════════════════════════
# 2. CONFIGURATION  — edit here to tune
# ═══════════════════════════════════════════════════════════════════════════
SCRIPT_DIR = Path(__file__).parent.resolve()

load_dotenv(SCRIPT_DIR.parents[1] / ".env")   # lab3/.env → sets HF_TOKEN

MODEL_ID  = "openai/whisper-medium"
LANGUAGE  = "mongolian"
TASK      = "transcribe"

DATASET_ROOT  = SCRIPT_DIR.parents[2] / "lab3" / "common_voice_mn"
CLIPS_DIR     = DATASET_ROOT / "clips"
VALIDATED_TSV = DATASET_ROOT / "validated.tsv"
OUTPUT_DIR    = SCRIPT_DIR / "whisper-medium-mn"
LOG_DIR       = OUTPUT_DIR / "runs"
CACHE_DIR     = OUTPUT_DIR / "feature_cache"

SEED                        = 42
TRAIN_RATIO                 = 0.80

PER_DEVICE_TRAIN_BATCH_SIZE = 8
PER_DEVICE_EVAL_BATCH_SIZE  = 8
GRADIENT_ACCUMULATION_STEPS = 2       # effective batch = 16
LEARNING_RATE               = 1e-5
WARMUP_STEPS                = 500
MAX_STEPS                   = 6000
EVAL_STEPS                  = 500
SAVE_STEPS                  = 500
WEIGHT_DECAY                = 0.01

MAX_SENTENCE_CHARS          = 200     # filters 17 corrupted rows (p99 = 111 chars)

HF_TOKEN = os.environ.get("HF_TOKEN", "")

for d in (OUTPUT_DIR, LOG_DIR, CACHE_DIR):
    d.mkdir(parents=True, exist_ok=True)

log.info(f"Dataset  : {DATASET_ROOT}")
log.info(f"Output   : {OUTPUT_DIR}")
log.info(f"Cache    : {CACHE_DIR}")
log.info(f"TSV      : {VALIDATED_TSV}  exists={VALIDATED_TSV.exists()}")

# ═══════════════════════════════════════════════════════════════════════════
# 3. HUGGING FACE LOGIN
# ═══════════════════════════════════════════════════════════════════════════
assert HF_TOKEN, "HF_TOKEN is not set (check lab3/.env)"
login(token=HF_TOKEN)
HF_USERNAME = whoami()["name"]
HUB_REPO    = f"{HF_USERNAME}/whisper-medium-mn"
log.info(f"HF user  : {HF_USERNAME}  →  hub repo: {HUB_REPO}")

# ═══════════════════════════════════════════════════════════════════════════
# 4. LOAD & SPLIT DATASET
# ═══════════════════════════════════════════════════════════════════════════
df = pd.read_csv(VALIDATED_TSV, sep="\t")
df = df.dropna(subset=["path", "sentence"]).reset_index(drop=True)
log.info(f"Validated rows after dropna: {len(df):,}")

df_train, df_temp = train_test_split(df, test_size=(1 - TRAIN_RATIO), random_state=SEED, shuffle=True)
df_val,   df_test = train_test_split(df_temp, test_size=0.5, random_state=SEED)
df_train = df_train.reset_index(drop=True)
df_val   = df_val.reset_index(drop=True)
df_test  = df_test.reset_index(drop=True)
log.info(f"Split  train={len(df_train):,}  val={len(df_val):,}  test={len(df_test):,}")

# ═══════════════════════════════════════════════════════════════════════════
# 5. SANITY CHECK  (saves plots to disk)
# ═══════════════════════════════════════════════════════════════════════════
missing = [f for f in df["path"] if not (CLIPS_DIR / f).exists()]
if missing:
    log.warning(f"{len(missing)} missing audio files — first 5: {missing[:5]}")
else:
    log.info(f"All {len(df):,} audio files present.")

log.info(f"Sample path      : {df_train.iloc[0]['path']}")
log.info(f"Sample transcript: {df_train.iloc[0]['sentence']}")

fig, axes = plt.subplots(1, 2, figsize=(14, 4))
sent_lengths = df_train["sentence"].str.len()
axes[0].hist(sent_lengths, bins=50, color="steelblue", edgecolor="white")
axes[0].set_title("Sentence length (chars)")
axes[0].set_xlabel("Characters")
axes[0].set_ylabel("Count")
dur_tsv = DATASET_ROOT / "clip_durations.tsv"
if dur_tsv.exists():
    df_dur = pd.read_csv(dur_tsv, sep="\t")
    df_dur = df_dur.rename(columns={c: "duration_ms" for c in df_dur.columns if "duration" in c.lower()})
    df_dur["duration_ms"] = pd.to_numeric(df_dur["duration_ms"], errors="coerce")
    durations_sec = df_dur[df_dur["clip"].isin(set(df["path"]))]["duration_ms"] / 1000
    axes[1].hist(durations_sec, bins=50, color="coral", edgecolor="white")
    axes[1].set_title("Audio duration (s)")
    total_h = durations_sec.sum() / 3600
    log.info(f"Total validated audio: {total_h:.1f} h  |  train est: {total_h * TRAIN_RATIO:.1f} h")
plt.tight_layout()
plot_path = OUTPUT_DIR / "sanity_check.png"
plt.savefig(plot_path, dpi=100)
plt.close()
log.info(f"Sanity plot → {plot_path}")

# ═══════════════════════════════════════════════════════════════════════════
# 6. PROCESSOR
# ═══════════════════════════════════════════════════════════════════════════
processor = WhisperProcessor.from_pretrained(MODEL_ID, language=LANGUAGE, task=TASK)
feature_extractor = processor.feature_extractor
tokenizer         = processor.tokenizer
TARGET_SR         = feature_extractor.sampling_rate
log.info(f"Processor loaded  language={LANGUAGE}  sr={TARGET_SR}")

# ═══════════════════════════════════════════════════════════════════════════
# 7. PREPROCESSING — numpy memmap (peak RAM: ~2 MB; skipped on re-runs)
#
# Why not datasets.map(): datasets 4.8.4 buffers the entire output in
# memory before writing Arrow files — Python hit 50–61 GB RSS before the
# OOM killer fired. np.memmap writes each row immediately to disk; RAM
# usage stays constant at ~2 MB for the entire preprocessing loop.
# ═══════════════════════════════════════════════════════════════════════════
free_gb = shutil.disk_usage(str(CACHE_DIR)).free / 1e9
log.info(f"Free disk: {free_gb:.1f} GB  (need ~16 GB for feature cache)")
assert free_gb >= 16, f"Not enough disk space: {free_gb:.1f} GB"


class WhisperMemmapDataset(torch.utils.data.Dataset):
    """Read-only view of precomputed float16 mel spectrograms via memmap.

    __getitem__ copies one (80, 3000) slice from the mmap file (~480 KB read).
    Safe with dataloader_num_workers > 0: read-only mmap files are
    copy-on-write shareable across forked workers on Linux.
    """

    def __init__(self, feat_path: Path, label_path: Path, n_samples: int):
        self.features = np.memmap(
            str(feat_path), dtype=np.float16, mode="r", shape=(n_samples, 80, 3000)
        )
        with open(label_path) as f:
            self.labels = json.load(f)
        assert len(self.labels) == n_samples, (
            f"Feature/label count mismatch: {n_samples} vs {len(self.labels)}"
        )

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Dict:
        return {
            "input_features": np.array(self.features[idx]),  # float16 [80, 3000]
            "labels": self.labels[idx],
        }


def preprocess_to_memmap(df_split: pd.DataFrame, split_name: str) -> WhisperMemmapDataset:
    """Decode audio → extract log-Mel (float16) → write memmap + labels JSON.

    Skips preprocessing if cache files already exist with matching size.
    Peak RAM during preprocessing: ~2–5 MB (one audio + one mel at a time).
    """
    feat_path  = CACHE_DIR / f"{split_name}_features.bin"
    label_path = CACHE_DIR / f"{split_name}_labels.json"

    # Filter corrupted rows (17 samples with sentences up to 16,984 chars)
    df_clean = df_split[
        df_split["sentence"].str.strip().str.len().between(1, MAX_SENTENCE_CHARS)
    ].reset_index(drop=True)
    n = len(df_clean)

    # Cache hit: file exists and has the expected byte size
    expected_bytes = n * 80 * 3000 * 2   # float16 = 2 bytes per element
    if feat_path.exists() and label_path.exists() and feat_path.stat().st_size == expected_bytes:
        log.info(f"{split_name}: cache hit ({n:,} samples) — skipping")
        return WhisperMemmapDataset(feat_path, label_path, n)

    log.info(f"{split_name}: preprocessing {n:,} samples → {expected_bytes/1e9:.2f} GB")

    # Pre-allocate the full .bin file on disk (does not load into RAM)
    mmap = np.memmap(str(feat_path), dtype=np.float16, mode="w+", shape=(n, 80, 3000))

    all_labels: List[List[int]] = []
    for i, (_, row) in enumerate(tqdm(df_clean.iterrows(), total=n, desc=split_name)):
        audio, _ = librosa.load(str(CLIPS_DIR / row["path"]), sr=TARGET_SR, mono=True)
        mel = feature_extractor(audio, sampling_rate=TARGET_SR).input_features[0]
        mmap[i] = mel.astype(np.float16)   # written to disk immediately
        all_labels.append(tokenizer(row["sentence"]).input_ids)
        del audio, mel                      # free ~2 MB per iteration

    del mmap   # flush remaining dirty pages and release file handle

    with open(label_path, "w") as f:
        json.dump(all_labels, f)

    log.info(f"{split_name}: done  {feat_path.stat().st_size/1e9:.2f} GB → {feat_path}")
    return WhisperMemmapDataset(feat_path, label_path, n)


log.info("Preprocessing (first run ~15–30 min for train, instant on re-runs)...")
train_dataset = preprocess_to_memmap(df_train, "train")
val_dataset   = preprocess_to_memmap(df_val,   "val")
test_dataset  = preprocess_to_memmap(df_test,  "test")
log.info(f"Ready  train={len(train_dataset):,}  val={len(val_dataset):,}  test={len(test_dataset):,}")

# ═══════════════════════════════════════════════════════════════════════════
# 8. DATA COLLATOR
# ═══════════════════════════════════════════════════════════════════════════
@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    """Pad float16 precomputed features and tokenised labels into a batch."""
    processor: Any
    decoder_start_token_id: int

    def __call__(
        self, features: List[Dict[str, Union[List[int], np.ndarray]]]
    ) -> Dict[str, torch.Tensor]:
        # Cast float16 → float32 for padding (WhisperFeatureExtractor.pad requirement)
        input_features = [
            {"input_features": np.array(f["input_features"], dtype=np.float32)}
            for f in features
        ]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch


data_collator = DataCollatorSpeechSeq2SeqWithPadding(
    processor=processor,
    decoder_start_token_id=processor.tokenizer.bos_token_id,
)

# ═══════════════════════════════════════════════════════════════════════════
# 9. MODEL
# ═══════════════════════════════════════════════════════════════════════════
model = WhisperForConditionalGeneration.from_pretrained(MODEL_ID)
model.config.forced_decoder_ids  = None
model.config.suppress_tokens     = []
model.config.use_cache           = False   # required with gradient_checkpointing
model.gradient_checkpointing_enable()
model.generation_config.language          = LANGUAGE
model.generation_config.task              = TASK
model.generation_config.forced_decoder_ids = None
log.info(
    f"Model loaded  params={sum(p.numel() for p in model.parameters())/1e6:.0f}M"
    "  grad_ckpt=on"
)

# ═══════════════════════════════════════════════════════════════════════════
# 10. WER METRIC
# ═══════════════════════════════════════════════════════════════════════════
wer_metric = evaluate.load("wer")
normalizer = BasicTextNormalizer()


def compute_metrics(pred):
    pred_ids  = pred.predictions
    label_ids = pred.label_ids
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str  = processor.tokenizer.batch_decode(pred_ids,  skip_special_tokens=True)
    label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    pred_norm  = [normalizer(p) for p in pred_str]
    label_norm = [normalizer(l) for l in label_str]

    pairs = [(p, l) for p, l in zip(pred_norm, label_norm) if len(l) > 0]
    if not pairs:
        return {"wer": 100.0}
    pred_norm, label_norm = zip(*pairs)
    wer = 100 * wer_metric.compute(
        predictions=list(pred_norm), references=list(label_norm)
    )
    return {"wer": wer}

# ═══════════════════════════════════════════════════════════════════════════
# 11. TRAINING ARGUMENTS
# ═══════════════════════════════════════════════════════════════════════════
training_args = Seq2SeqTrainingArguments(
    output_dir=str(OUTPUT_DIR),

    # Precision & memory
    bf16=True,
    fp16=False,
    gradient_checkpointing=True,

    # Batch sizes
    per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
    per_device_eval_batch_size=PER_DEVICE_EVAL_BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,

    # Optimiser
    optim="adamw_bnb_8bit",
    learning_rate=LEARNING_RATE,
    warmup_steps=WARMUP_STEPS,
    weight_decay=WEIGHT_DECAY,

    # Steps
    max_steps=MAX_STEPS,

    # Evaluation & checkpointing
    eval_strategy="steps",
    eval_steps=EVAL_STEPS,
    save_strategy="steps",
    save_steps=SAVE_STEPS,
    save_total_limit=3,
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,

    # Generation during eval
    predict_with_generate=True,
    generation_max_length=225,

    # Logging
    logging_dir=str(LOG_DIR),
    logging_steps=25,
    report_to=["tensorboard"],

    # Hub
    push_to_hub=True,
    hub_model_id=HUB_REPO,
    hub_strategy="checkpoint",

    # DataLoader — workers read memmap files: no CUDA, fork-safe
    seed=SEED,
    dataloader_num_workers=4,
    dataloader_pin_memory=True,
    remove_unused_columns=False,
)
log.info(
    f"TrainingArgs  steps={MAX_STEPS}  lr={LEARNING_RATE}"
    f"  eff_batch={PER_DEVICE_TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}"
)
log.info(f"TensorBoard → tensorboard --logdir {LOG_DIR}")

# ═══════════════════════════════════════════════════════════════════════════
# 12. TRAINER
# ═══════════════════════════════════════════════════════════════════════════
trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    processing_class=processor.feature_extractor,
)
processor.save_pretrained(str(OUTPUT_DIR))
log.info("Trainer ready. Processor saved to output dir.")

# ═══════════════════════════════════════════════════════════════════════════
# 13. TRAIN
# ═══════════════════════════════════════════════════════════════════════════
log.info(f"Starting training  steps={MAX_STEPS}  eval_every={EVAL_STEPS}")
train_result = trainer.train()
trainer.log_metrics("train", train_result.metrics)
trainer.save_metrics("train", train_result.metrics)
trainer.save_state()

log.info("Training complete")
log.info(f"  loss        : {train_result.metrics.get('train_loss', 'N/A'):.4f}")
log.info(f"  samples/sec : {train_result.metrics.get('train_samples_per_second', 'N/A'):.2f}")
log.info(f"  runtime     : {train_result.metrics.get('train_runtime', 0)/3600:.2f} h")

# ═══════════════════════════════════════════════════════════════════════════
# 14. EVALUATE ON TEST SET
# ═══════════════════════════════════════════════════════════════════════════
log.info("Evaluating on test set...")
test_results = trainer.evaluate(eval_dataset=test_dataset, metric_key_prefix="test")
trainer.log_metrics("test", test_results)
trainer.save_metrics("test", test_results)

test_wer = test_results.get("test_wer")
log.info("=" * 50)
log.info(f"  Test WER : {test_wer:.2f}%" if test_wer is not None else "  WER not in results")
log.info(f"  Target   : < 48.00%")
if test_wer is not None:
    status = "PASSED" if test_wer < 48.0 else "NOT YET — try more steps or lower LR"
    log.info(f"  Status   : {status}")
log.info("=" * 50)

# ═══════════════════════════════════════════════════════════════════════════
# 15. SAMPLE PREDICTIONS
# ═══════════════════════════════════════════════════════════════════════════
model.eval()
model.to(device)
log.info("Sample predictions on 5 test examples:")
for i in range(5):
    sample = test_dataset[i]
    mel = np.array(sample["input_features"], dtype=np.float32)
    # float32 inputs: model weights/biases are fp32; manual bf16 here breaks Conv1d (bias dtype mismatch).
    input_features = torch.tensor(mel).unsqueeze(0).to(device)

    with torch.no_grad():
        predicted_ids = model.generate(input_features, language=LANGUAGE, task=TASK)

    label_ids = [
        l if l != -100 else processor.tokenizer.pad_token_id
        for l in sample["labels"]
    ]
    ref = processor.tokenizer.decode(label_ids,        skip_special_tokens=True)
    hyp = processor.tokenizer.decode(predicted_ids[0], skip_special_tokens=True)
    log.info(f"  [{i+1}] REF : {ref}")
    log.info(f"       HYP : {hyp}")

# ═══════════════════════════════════════════════════════════════════════════
# 16. PUSH TO HUB
# ═══════════════════════════════════════════════════════════════════════════
log.info(f"Pushing best model → {HUB_REPO}")
trainer.push_to_hub(
    dataset_tags="mozilla-foundation/common_voice_17_0",
    dataset="Common Voice Mongolian 25.0",
    dataset_args="config: mn, split: validated",
    language="mn",
    model_name="Whisper Medium - Mongolian",
    finetuned_from=MODEL_ID,
    tasks="automatic-speech-recognition",
)
processor.push_to_hub(HUB_REPO)
log.info(f"Done  https://huggingface.co/{HUB_REPO}")
