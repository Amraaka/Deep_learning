"""Load Mozilla Common Voice (Mongolian) from local TSV and split validated.tsv."""
import logging
import os
from pathlib import Path

import pandas as pd
from datasets import Audio, Dataset, DatasetDict


def _read_validated(data_root: Path) -> pd.DataFrame:
    tsv_path = data_root / "validated.tsv"
    if not tsv_path.exists():
        raise FileNotFoundError(f"validated.tsv not found at {tsv_path}")

    df = pd.read_csv(tsv_path, sep="\t", low_memory=False)
    if "path" not in df.columns or "sentence" not in df.columns:
        raise ValueError(f"validated.tsv missing required columns, got: {list(df.columns)}")

    df = df[["path", "sentence"]].dropna()
    df["sentence"] = df["sentence"].astype(str).str.strip()
    df = df[df["sentence"].str.len() > 0]

    clips_dir = data_root / "clips"
    df["path"] = df["path"].apply(lambda p: str(clips_dir / p))
    df = df[df["path"].apply(os.path.exists)]
    return df.reset_index(drop=True)


def load_common_voice_mn(
    data_root: str,
    sampling_rate: int = 16000,
    val_ratio: float = 0.05,
    test_ratio: float = 0.05,
    seed: int = 42,
) -> DatasetDict:
    """Load validated.tsv and deterministically split into train/validation/test.

    Splitting is done on client_id-agnostic shuffled rows; for strict speaker-
    disjoint splits, swap in a GroupShuffleSplit on `client_id`.
    """
    root = Path(data_root)
    logging.info(f"Loading Common Voice (mn) from {root}")
    df = _read_validated(root)
    logging.info(f"validated rows after existence filter: {len(df)}")

    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    n = len(df)
    n_test = int(n * test_ratio)
    n_val = int(n * val_ratio)
    test_df = df.iloc[:n_test]
    val_df = df.iloc[n_test : n_test + n_val]
    train_df = df.iloc[n_test + n_val :]

    def to_hf(d: pd.DataFrame) -> Dataset:
        ds = Dataset.from_pandas(
            d.rename(columns={"path": "audio", "sentence": "transcription"}),
            preserve_index=False,
        )
        return ds.cast_column("audio", Audio(sampling_rate=sampling_rate))

    ds = DatasetDict(
        {
            "train": to_hf(train_df),
            "validation": to_hf(val_df),
            "test": to_hf(test_df),
        }
    )
    logging.info(
        f"Splits — train: {len(ds['train'])}, val: {len(ds['validation'])}, test: {len(ds['test'])}"
    )
    return ds
