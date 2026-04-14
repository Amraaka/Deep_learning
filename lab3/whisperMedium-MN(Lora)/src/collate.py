"""On-the-fly audio featurization and batching for Seq2SeqTrainer."""

from dataclasses import dataclass
from typing import Any, Dict, List, Union

import librosa
import numpy as np
import soundfile as sf
import torch


def load_audio_array(audio_path: str) -> np.ndarray:
    """Load an audio file, convert to mono, and resample to 16kHz if needed."""
    audio_arr, sr = sf.read(audio_path, dtype="float32", always_2d=False)

    if getattr(audio_arr, "ndim", 1) == 2:
        audio_arr = audio_arr.mean(axis=1)

    if sr != 16_000:
        audio_arr = librosa.resample(audio_arr, orig_sr=sr, target_sr=16_000)

    return np.asarray(audio_arr, dtype=np.float32)


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        input_feats = []
        label_feats = []

        for feature in features:
            audio_arr = load_audio_array(feature["audio_path"])
            input_feats.append(
                {
                    "input_features": self.processor.feature_extractor(
                        audio_arr, sampling_rate=16_000
                    ).input_features[0]
                }
            )
            label_feats.append(
                {
                    "input_ids": self.processor.tokenizer(
                        feature["sentence"], verbose=False
                    ).input_ids
                }
            )

        batch = self.processor.feature_extractor.pad(input_feats, return_tensors="pt")
        batch["input_features"] = batch["input_features"].to(dtype=torch.bfloat16)
        labels_padded = self.processor.tokenizer.pad(label_feats, return_tensors="pt")
        labels = labels_padded["input_ids"].masked_fill(
            labels_padded.attention_mask.ne(1), -100
        )
        batch["labels"] = labels
        return batch
