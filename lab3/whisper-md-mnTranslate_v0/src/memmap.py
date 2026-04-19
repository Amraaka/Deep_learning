"""Precomputed float16 log-mel memmap datasets (from finetune.py)."""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

import librosa
import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
from transformers import WhisperProcessor, WhisperTokenizer

from src.config import MAX_SENTENCE_CHARS, N_FRAMES, N_MEL

log = logging.getLogger(__name__)


class WhisperMemmapDataset(torch.utils.data.Dataset):
    """shape (N, 80, 3000) """

    def __init__(
        self,
        feat_path: Path,
        label_transcribe_path: Path,
        n_samples: int,
        mode: Literal["transcribe", "translate", "multitask"] = "transcribe",
        label_translate_path: Optional[Path] = None,
    ):
        self.features = np.memmap(
            str(feat_path),
            dtype=np.float16,
            mode="r",
            shape=(n_samples, N_MEL, N_FRAMES),
        )
        self.mode = mode
        with open(label_transcribe_path) as f:
            self.labels_transcribe = json.load(f)

        self.labels_translate = None
        if label_translate_path and label_translate_path.is_file():
            with open(label_translate_path) as f:
                self.labels_translate = json.load(f)

        assert len(self.labels_transcribe) == n_samples, (
            f"Feature/label count mismatch: {n_samples} vs {len(self.labels_transcribe)}"
        )
        if self.labels_translate is not None:
            assert len(self.labels_translate) == n_samples, (
                f"Feature/translate-label mismatch: {n_samples} vs {len(self.labels_translate)}"
            )
        if self.mode in ("translate", "multitask") and self.labels_translate is None:
            raise ValueError(
                "Translate labels are missing in cache. Rebuild cache with translate labels enabled."
            )

    def __len__(self) -> int:
        base = len(self.labels_transcribe)
        if self.mode == "multitask":
            return base * 2
        return base

    def __getitem__(self, idx: int) -> Dict[str, Union[np.ndarray, List[int]]]:
        task = self.mode
        base_idx = idx
        if self.mode == "multitask":
            base_idx = idx // 2
            task = "transcribe" if (idx % 2) == 0 else "translate"

        labels = self.labels_transcribe[base_idx]
        if task == "translate":
            assert self.labels_translate is not None
            labels = self.labels_translate[base_idx]

        return {
            "input_features": np.array(self.features[base_idx]),
            "labels": labels,
            "task": task,
        }


def preprocess_to_memmap(
    df_split: pd.DataFrame,
    split_name: str,
    clips_dir: Path,
    cache_dir: Path,
    processor: WhisperProcessor,
    target_sr: int,
    language: str,
    mode: Literal["transcribe", "translate", "multitask"] = "transcribe",
    include_translate_labels: bool = False,
) -> WhisperMemmapDataset:
    """Decode audio to log-mel float16 on disk; skip if cache size matches."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    feat_path = cache_dir / f"{split_name}_features.bin"
    label_transcribe_path = cache_dir / f"{split_name}_labels_transcribe.json"
    label_translate_path = cache_dir / f"{split_name}_labels_translate.json"

    feature_extractor = processor.feature_extractor

    tokenizer_name = processor.tokenizer.name_or_path
    tokenizer_transcribe = WhisperTokenizer.from_pretrained(
        tokenizer_name,
        language=language,
        task="transcribe",
    )
    tokenizer_translate = WhisperTokenizer.from_pretrained(
        tokenizer_name,
        language=language,
        task="translate",
    )

    df_clean = df_split.copy()
    df_clean = df_clean[
        df_clean["sentence"].astype(str).str.strip().str.len().between(1, MAX_SENTENCE_CHARS)
    ]
    if include_translate_labels:
        if "sentence_en" not in df_clean.columns:
            raise ValueError(
                f"split={split_name} requires sentence_en for mode={mode}, but column is missing"
            )
        df_clean = df_clean[
            df_clean["sentence_en"].astype(str).str.strip().str.len().between(1, MAX_SENTENCE_CHARS)
        ]
    df_clean = df_clean.reset_index(drop=True)

    n = len(df_clean)

    expected_bytes = n * N_MEL * N_FRAMES * 2
    if (
        feat_path.exists()
        and label_transcribe_path.exists()
        and feat_path.stat().st_size == expected_bytes
        and (not include_translate_labels or label_translate_path.exists())
    ):
        log.info("%s: cache hit (%s samples) — skipping", split_name, f"{n:,}")
        return WhisperMemmapDataset(
            feat_path,
            label_transcribe_path,
            n,
            mode=mode,
            label_translate_path=label_translate_path,
        )

    log.info(
        "%s: preprocessing %s samples → %.2f GB",
        split_name,
        f"{n:,}",
        expected_bytes / 1e9,
    )

    mmap = np.memmap(str(feat_path), dtype=np.float16, mode="w+", shape=(n, N_MEL, N_FRAMES))

    all_labels_transcribe: List[List[int]] = []
    all_labels_translate: List[List[int]] = []
    for _, row in tqdm(df_clean.iterrows(), total=n, desc=split_name):
        audio, _ = librosa.load(str(clips_dir / row["path"]), sr=target_sr, mono=True)
        mel = feature_extractor(audio, sampling_rate=target_sr).input_features[0]
        mmap[len(all_labels_transcribe)] = mel.astype(np.float16)
        all_labels_transcribe.append(tokenizer_transcribe(str(row["sentence"])).input_ids)
        if include_translate_labels:
            all_labels_translate.append(tokenizer_translate(str(row["sentence_en"])).input_ids)
        del audio, mel

    del mmap

    with open(label_transcribe_path, "w") as f:
        json.dump(all_labels_transcribe, f)
    if include_translate_labels:
        with open(label_translate_path, "w") as f:
            json.dump(all_labels_translate, f)

    log.info(
        "%s: done  %.2f GB → %s",
        split_name,
        feat_path.stat().st_size / 1e9,
        feat_path,
    )
    return WhisperMemmapDataset(
        feat_path,
        label_transcribe_path,
        n,
        mode=mode,
        label_translate_path=label_translate_path,
    )
