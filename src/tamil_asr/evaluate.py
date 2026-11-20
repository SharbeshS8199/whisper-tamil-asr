"""Zero-shot base Whisper vs the fine-tuned model on the FLEURS Tamil test set.

    python -m tamil_asr.evaluate --finetuned outputs/whisper-small-ta/final [--limit 300]

Writes reports/{metrics.json, predictions.jsonl, report.md}. The report also shows how much
Whisper's default normaliser (which strips Tamil vowel signs) would understate the WER.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from tamil_asr.metrics import cer, wer
from tamil_asr.normalize import strips_marks_like_whisper


def transcribe_all(model_id: str, ds, language: str, batch_size: int = 16) -> list[str]:
    import torch
    from transformers import pipeline

    asr = pipeline("automatic-speech-recognition", model=model_id, device=0 if torch.cuda.is_available() else -1,
                   torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32)
    out = asr([s["audio"] for s in ds], batch_size=batch_size,
              generate_kwargs={"language": language, "task": "transcribe"})
    return [o["text"] for o in out]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/whisper-small-ta.yaml")
    ap.add_argument("--finetuned", default="outputs/whisper-small-ta/final")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default="reports")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))

    from tamil_asr.data import load_one

    ds = load_one(cfg["eval_set"])
    if args.limit:
        ds = ds.select(range(min(args.limit, len(ds))))
    refs = list(ds["sentence"])
    out = Path(args.out)
    out.mkdir(exist_ok=True)
    results, preds = {"n": len(refs)}, {}
    for name, model_id in (("zero_shot", cfg["base_model"]), ("fine_tuned", args.finetuned)):
        hyps = transcribe_all(model_id, ds, cfg["language"])
        preds[name] = hyps
        results[name] = {"wer": round(wer(refs, hyps), 2), "cer": round(cer(refs, hyps), 2),
                         "wer_if_marks_stripped": round(wer([strips_marks_like_whisper(r) for r in refs],
                                                            [strips_marks_like_whisper(h) for h in hyps], normalize=False), 2)}
        print(name, results[name])
    z, f = results["zero_shot"]["wer"], results["fine_tuned"]["wer"]
    results["relative_wer_reduction_pct"] = round(100 * (z - f) / z, 1) if z else None
    (out / "metrics.json").write_text(json.dumps(results, indent=2))
    with open(out / "predictions.jsonl", "w", encoding="utf-8") as fh:
        for i, r in enumerate(refs):
            fh.write(json.dumps({"ref": r, "zero_shot": preds["zero_shot"][i], "fine_tuned": preds["fine_tuned"][i]},
                                ensure_ascii=False) + "\n")
    lines = ["# Whisper Tamil evaluation (FLEURS ta_in test)", "", f"n = {len(refs)} utterances", "",
             "| model | WER % | CER % | WER % with Whisper's default normaliser |", "|---|---|---|---|"]
    for name in ("zero_shot", "fine_tuned"):
        r = results[name]
        lines.append(f"| {name.replace('_', ' ')} | {r['wer']} | {r['cer']} | {r['wer_if_marks_stripped']} |")
    lines += ["", f"**Relative WER reduction: {results['relative_wer_reduction_pct']}%**", "", "## Examples", "",
              "| reference | zero-shot | fine-tuned |", "|---|---|---|"]
    lines += [f"| {refs[i]} | {preds['zero_shot'][i]} | {preds['fine_tuned'][i]} |" for i in range(min(8, len(refs)))]
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
