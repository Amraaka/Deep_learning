"""Load validated.tsv and 80/10/10 split"""

import logging
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split

from src.config import SEED, TRAIN_RATIO

log = logging.getLogger(__name__)


def load_validated_frame(data_root: Path) -> pd.DataFrame:
    tsv = data_root / "validated.tsv"
    if not tsv.is_file():
        raise FileNotFoundError(f"validated.tsv not found: {tsv}")
    df = pd.read_csv(tsv, sep="\t")
    df = df.dropna(subset=["path", "sentence"]).reset_index(drop=True)
    log.info("Validated rows after dropna: %s", f"{len(df):,}")
    return df


def load_training_frame(
    data_root: Path,
    dataset_id: Optional[str],
    require_sentence_en: bool,
) -> pd.DataFrame:
    if dataset_id:
        log.info("Loading translated dataset from HF: %s", dataset_id)
        ds = load_dataset(dataset_id, split="train")
        df = ds.to_pandas()

        if "path" not in df.columns and "audio" in df.columns:
            def _audio_to_name(audio_obj):
                if isinstance(audio_obj, dict):
                    p = audio_obj.get("path")
                    if p:
                        return Path(str(p)).name
                return None

            df["path"] = df["audio"].apply(_audio_to_name)

        if "path" not in df.columns:
            raise ValueError("HF dataset must contain a 'path' column or an 'audio.path' field.")
        if "sentence" not in df.columns:
            raise ValueError("HF dataset must contain a 'sentence' column.")
        if require_sentence_en and "sentence_en" not in df.columns:
            raise ValueError("HF dataset must contain 'sentence_en' for translate/multitask.")

        keep_cols = ["path", "sentence"]
        if "sentence_en" in df.columns:
            keep_cols.append("sentence_en")
        df = df[keep_cols].copy()
    else:
        df = load_validated_frame(data_root)

    required = ["path", "sentence"]
    if require_sentence_en:
        if "sentence_en" not in df.columns:
            raise ValueError("sentence_en is required for translate/multitask, but column is missing.")
        required.append("sentence_en")

    df = df.dropna(subset=required).reset_index(drop=True)
    df["path"] = df["path"].astype(str).map(lambda p: Path(p).name)
    if "sentence_en" in df.columns:
        df["sentence_en"] = df["sentence_en"].astype(str)

    log.info(
        "Training rows after schema/dropna checks: %s (sentence_en=%s)",
        f"{len(df):,}",
        "yes" if "sentence_en" in df.columns else "no",
    )
    return df


def train_val_test_split_df(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df_train, df_temp = train_test_split(
        df,
        test_size=(1 - TRAIN_RATIO),
        random_state=SEED,
        shuffle=True,
    )
    df_val, df_test = train_test_split(
        df_temp,
        test_size=0.5,
        random_state=SEED,
    )
    df_train = df_train.reset_index(drop=True)
    df_val = df_val.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)
    log.info(
        "Split  train=%s  val=%s  test=%s",
        f"{len(df_train):,}",
        f"{len(df_val):,}",
        f"{len(df_test):,}",
    )
    return df_train, df_val, df_test


def log_missing_clips(df: pd.DataFrame, clips_dir: Path) -> None:
    missing = [f for f in df["path"] if not (clips_dir / f).exists()]
    if missing:
        log.warning("%s missing audio files — first 5: %s", len(missing), missing[:5])
    else:
        log.info("All %s audio files present.", f"{len(df):,}")
