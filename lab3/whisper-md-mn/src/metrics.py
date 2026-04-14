"""WER with BasicTextNormalizer (from finetune.py)."""

from typing import Any, Dict

import evaluate
from transformers import WhisperProcessor
from transformers.models.whisper.english_normalizer import BasicTextNormalizer


def make_compute_metrics(processor: WhisperProcessor):
    wer_metric = evaluate.load("wer")
    normalizer = BasicTextNormalizer()

    def compute_metrics(pred) -> Dict[str, Any]:
        pred_ids = pred.predictions
        label_ids = pred.label_ids
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

        pred_str = processor.tokenizer.batch_decode(
            pred_ids, skip_special_tokens=True
        )
        label_str = processor.tokenizer.batch_decode(
            label_ids, skip_special_tokens=True
        )

        pred_norm = [normalizer(p) for p in pred_str]
        label_norm = [normalizer(l) for l in label_str]

        pairs = [(p, l) for p, l in zip(pred_norm, label_norm) if len(l) > 0]
        if not pairs:
            return {"wer": 100.0}
        pred_norm, label_norm = zip(*pairs)
        wer = 100 * wer_metric.compute(
            predictions=list(pred_norm),
            references=list(label_norm),
        )
        return {"wer": float(wer)}

    return compute_metrics
