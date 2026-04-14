"""Dataset load / filter / split for training."""

import os
import sys
from pathlib import Path

import pandas as pd
import soundfile as sf
from datasets import Dataset, DatasetDict
from transformers import WhisperTokenizer

from src.config import MAX_AUDIO_SEC, MAX_TARGET_TOKENS, MIN_AUDIO_SEC, SEED


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

    df["audio_path"] = df["path"].apply(lambda p: str(clips_dir / p))

    exists_mask = df["audio_path"].apply(os.path.isfile)
    missing = (~exists_mask).sum()
    if missing > 0:
        print(f"[WARN] {missing} audio files not found on disk — skipping them.")
    df = df[exists_mask].reset_index(drop=True)

    ds = Dataset.from_dict({
        "audio_path": df["audio_path"].tolist(),
        "sentence": df["sentence"].tolist(),
    })
    return ds


def filter_dataset(ds: Dataset, tokenizer: WhisperTokenizer) -> Dataset:
    """Filter by audio duration (via soundfile header) and label token length."""

    def is_valid(example):
        try:
            info = sf.info(example["audio_path"])
            duration = info.duration
        except Exception:
            return False
        if duration < MIN_AUDIO_SEC or duration > MAX_AUDIO_SEC:
            return False
        token_ids = tokenizer(example["sentence"], verbose=False).input_ids
        if len(token_ids) > MAX_TARGET_TOKENS:
            return False
        return True

    before = len(ds)
    ds = ds.filter(is_valid)
    after = len(ds)
    print(f"[INFO] Filtered: {before} -> {after} samples "
          f"(removed {before - after} by duration/label length)")
    return ds


def split_dataset(ds: Dataset) -> DatasetDict:
    """Split into 80% train / 10% validation / 10% test."""
    split1 = ds.train_test_split(test_size=0.2, seed=SEED)
    split2 = split1["test"].train_test_split(test_size=0.5, seed=SEED)

    dsd = DatasetDict({
        "train": split1["train"],
        "validation": split2["train"],
        "test": split2["test"],
    })
    for name, part in dsd.items():
        print(f"  {name:>12s}: {len(part)} samples")
    return dsd
