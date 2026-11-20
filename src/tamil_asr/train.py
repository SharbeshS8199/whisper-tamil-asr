"""Fine-tune Whisper on Tamil.

    python -m tamil_asr.train --config configs/whisper-small-ta.yaml [--max-steps 1500] [--push]
"""

from __future__ import annotations

import argparse

import yaml

from tamil_asr.data import SpeechCollator, load_one, load_train, prepare
from tamil_asr.metrics import cer, wer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/whisper-small-ta.yaml")
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--push", action="store_true", help="push the final model to the Hugging Face Hub")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    t = cfg["train"]

    import torch
    from transformers import (Seq2SeqTrainer, Seq2SeqTrainingArguments, WhisperForConditionalGeneration,
                              WhisperProcessor)

    processor = WhisperProcessor.from_pretrained(cfg["base_model"], language=cfg["language"], task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained(cfg["base_model"])
    model.generation_config.language = cfg["language"]
    model.generation_config.task = "transcribe"
    model.generation_config.forced_decoder_ids = None

    train_ds = prepare(load_train(cfg), processor, t["max_label_tokens"], t["max_audio_seconds"])
    eval_ds = prepare(load_one(cfg["eval_set"]), processor, t["max_label_tokens"], t["max_audio_seconds"])
    print(f"train {len(train_ds)} utterances, eval {len(eval_ds)}")

    def compute_metrics(pred):
        ids = pred.label_ids
        ids[ids == -100] = processor.tokenizer.pad_token_id
        hyp = processor.batch_decode(pred.predictions, skip_special_tokens=True)
        ref = processor.batch_decode(ids, skip_special_tokens=True)
        return {"wer": wer(ref, hyp), "cer": cer(ref, hyp)}

    targs = Seq2SeqTrainingArguments(
        output_dir=cfg["output_dir"], per_device_train_batch_size=t["batch_size"],
        per_device_eval_batch_size=t["batch_size"] // 2, gradient_accumulation_steps=t["grad_accum"],
        learning_rate=t["learning_rate"], warmup_steps=t["warmup_steps"], max_steps=args.max_steps or t["max_steps"],
        gradient_checkpointing=True, fp16=torch.cuda.is_available(), eval_strategy="steps",
        eval_steps=t["eval_steps"], save_steps=t["eval_steps"], save_total_limit=2, logging_steps=50,
        predict_with_generate=True, generation_max_length=225, load_best_model_at_end=True,
        metric_for_best_model="wer", greater_is_better=False, report_to="none",
        push_to_hub=args.push, hub_model_id=cfg.get("hub_model_id"),
    )
    trainer = Seq2SeqTrainer(model=model, args=targs, train_dataset=train_ds, eval_dataset=eval_ds,
                             data_collator=SpeechCollator(processor, model.config.decoder_start_token_id),
                             compute_metrics=compute_metrics, processing_class=processor.feature_extractor)
    trainer.train()
    trainer.save_model(f"{cfg['output_dir']}/final")
    processor.save_pretrained(f"{cfg['output_dir']}/final")
    if args.push:
        trainer.push_to_hub(language="ta", finetuned_from=cfg["base_model"], tasks="automatic-speech-recognition",
                            dataset="google/fleurs", model_name="Whisper Small Tamil (Sharbesh S)")


if __name__ == "__main__":
    main()
