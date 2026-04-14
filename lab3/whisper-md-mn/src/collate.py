"""Pad precomputed features + labels (from finetune.py)."""

from dataclasses import dataclass
from typing import Any, Dict, List, Union

import numpy as np
import torch

from transformers import WhisperProcessor


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: WhisperProcessor
    decoder_start_token_id: int

    def __call__(
        self, features: List[Dict[str, Union[List[int], np.ndarray]]]
    ) -> Dict[str, torch.Tensor]:
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
