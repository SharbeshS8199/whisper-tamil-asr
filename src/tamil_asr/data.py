"""Load and prepare Tamil speech datasets for Whisper (16 kHz log-Mel features + token labels)."""

from __future__ import annotations

TEXT_CANDIDATES = ["transcription", "sentence", "text", "transcript", "normalized", "verbatim", "raw_transcription"]


def pick_text_column(columns) -> str:
    cols = list(columns)
    for c in TEXT_CANDIDATES:
        if c in cols:
            return c
    raise ValueError(f"no transcript column among {cols}")


def load_one(spec: dict, streaming: bool = False):
    from datasets import Audio, load_dataset

    kw = {"trust_remote_code": True} if spec["name"] == "google/fleurs" else {}
    ds = load_dataset(spec["name"], spec.get("config"), split=spec["split"], streaming=streaming, **kw)
    text = spec["text"] if spec.get("text", "auto") != "auto" else pick_text_column(ds.column_names)
    audio = "audio" if "audio" in ds.column_names else next(c for c in ds.column_names if "audio" in c)
    ds = ds.rename_columns({k: v for k, v in {audio: "audio", text: "sentence"}.items() if k != v})
    ds = ds.select_columns(["audio", "sentence"]).cast_column("audio", Audio(sampling_rate=16_000))
    if spec.get("max_samples") and not streaming:
        ds = ds.shuffle(seed=0).select(range(min(spec["max_samples"], len(ds))))
    return ds


def load_train(cfg: dict):
    from datasets import concatenate_datasets

    parts = [load_one(s) for s in cfg["train_sets"] if s.get("enabled", True)]
    return concatenate_datasets(parts) if len(parts) > 1 else parts[0]


def prepare(ds, processor, max_label_tokens: int = 448, max_audio_seconds: float = 30.0, num_proc: int = 2):
    def fn(batch):
        a = batch["audio"]
        batch["input_features"] = processor.feature_extractor(a["array"], sampling_rate=a["sampling_rate"]).input_features[0]
        batch["labels"] = processor.tokenizer(batch["sentence"]).input_ids
        batch["seconds"] = len(a["array"]) / a["sampling_rate"]
        return batch

    ds = ds.map(fn, remove_columns=ds.column_names, num_proc=num_proc)
    # Whisper's encoder sees at most 30 s and its decoder at most 448 tokens: drop anything longer
    ds = ds.filter(lambda x: x["seconds"] <= max_audio_seconds and len(x["labels"]) <= max_label_tokens)
    return ds.remove_columns(["seconds"])


class SpeechCollator:
    """Pads features and labels separately; label padding becomes -100; drops a leading
    decoder-start token if the tokenizer already added it (the model adds its own)."""

    def __init__(self, processor, decoder_start_token_id: int):
        self.processor, self.start = processor, decoder_start_token_id

    def __call__(self, features):
        inputs = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(inputs, return_tensors="pt")
        labels = self.processor.tokenizer.pad([{"input_ids": f["labels"]} for f in features], return_tensors="pt")
        y = labels["input_ids"].masked_fill(labels.attention_mask.ne(1), -100)
        if (y[:, 0] == self.start).all().cpu().item():
            y = y[:, 1:]
        batch["labels"] = y
        return batch
