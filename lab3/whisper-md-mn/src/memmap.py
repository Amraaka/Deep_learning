"""Precomputed float16 log-mel memmap datasets (from finetune.py)."""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Union

import librosa
import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
from transformers import WhisperProcessor

from src.config import MAX_SENTENCE_CHARS, N_FRAMES, N_MEL

log = logging.getLogger(__name__)


class WhisperMemmapDataset(torch.utils.data.Dataset):
    """Read-only memmap of shape (N, 80, 3000) float16 + labels JSON."""

    def __init__(self, feat_path: Path, label_path: Path, n_samples: int):
        self.features = np.memmap(
            str(feat_path),
            dtype=np.float16,
            mode="r",
            shape=(n_samples, N_MEL, N_FRAMES),
        )
        with open(label_path) as f:
            self.labels = json.load(f)
        assert len(self.labels) == n_samples, (
            f"Feature/label count mismatch: {n_samples} vs {len(self.labels)}"
        )

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Dict[str, Union[np.ndarray, List[int]]]:
        return {
            "input_features": np.array(self.features[idx]),
            "labels": self.labels[idx],
        }


def preprocess_to_memmap(
    df_split: pd.DataFrame,
    split_name: str,
    clips_dir: Path,
    cache_dir: Path,
    processor: WhisperProcessor,
    target_sr: int,
) -> WhisperMemmapDataset:
    """Decode audio to log-mel float16 on disk; skip if cache size matches."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    feat_path = cache_dir / f"{split_name}_features.bin"
    label_path = cache_dir / f"{split_name}_labels.json"

    tokenizer = processor.tokenizer
    feature_extractor = processor.feature_extractor

    df_clean = df_split[
        df_split["sentence"].str.strip().str.len().between(1, MAX_SENTENCE_CHARS)
    ].reset_index(drop=True)
    n = len(df_clean)

    expected_bytes = n * N_MEL * N_FRAMES * 2
    if (
        feat_path.exists()
        and label_path.exists()
        and feat_path.stat().st_size == expected_bytes
    ):
        log.info("%s: cache hit (%s samples) — skipping", split_name, f"{n:,}")
        return WhisperMemmapDataset(feat_path, label_path, n)

    log.info(
        "%s: preprocessing %s samples → %.2f GB",
        split_name,
        f"{n:,}",
        expected_bytes / 1e9,
    )

    mmap = np.memmap(str(feat_path), dtype=np.float16, mode="w+", shape=(n, N_MEL, N_FRAMES))

    all_labels: List[List[int]] = []
    for _, row in tqdm(df_clean.iterrows(), total=n, desc=split_name):
        audio, _ = librosa.load(str(clips_dir / row["path"]), sr=target_sr, mono=True)
        mel = feature_extractor(audio, sampling_rate=target_sr).input_features[0]
        mmap[len(all_labels)] = mel.astype(np.float16)
        all_labels.append(tokenizer(row["sentence"]).input_ids)
        del audio, mel

    del mmap

    with open(label_path, "w") as f:
        json.dump(all_labels, f)

    log.info(
        "%s: done  %.2f GB → %s",
        split_name,
        feat_path.stat().st_size / 1e9,
        feat_path,
    )
    return WhisperMemmapDataset(feat_path, label_path, n)
