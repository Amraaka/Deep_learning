"""Load validated.tsv and 80/10/10 split (sklearn, same as finetune.py)."""

import logging
from pathlib import Path
from typing import Tuple

import pandas as pd
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
