# Whisper for Tamil: fine-tuning, with WER measured properly

**Author:** [Sharbesh S](https://github.com/SharbeshS8199) · MIT License

Fine-tunes OpenAI **Whisper-small** for Tamil speech recognition on **Google FLEURS** and, optionally,
**AI4Bharat IndicVoices**. It reports word and character error rate before and after fine-tuning,
publishes the model on the Hugging Face Hub, and includes a microphone demo.

## The finding most Tamil Whisper results miss

Whisper's built-in `BasicTextNormalizer` replaces every Unicode *mark* with a space. In Tamil, the
vowel signs (ா ி ு …) and the pulli (்) are marks, so one word is cut into fragments before WER is
computed:

| | Proper Tamil normalisation (this repo) | Whisper's default normaliser |
|---|---|---|
| "கல்வி" (education) becomes | `கல்வி` (1 word) | `கல வ` (3 fragments) |
| WER for ref "நான் பள்ளிக்கு சென்றேன்" vs hyp "நான் பள்ளி சென்றேன்" | **33.3%** (1 wrong word of 3) | **20.0%** |

So the default normaliser **understates the error**. This repo scores everything with a
Tamil-safe normaliser (`normalize.py`, unit-tested). The evaluation report also shows the
default-normaliser number side by side, so you can see the gap on real outputs.

## Results

> **To fill in after training.** Run `python -m tamil_asr.evaluate`, then copy `reports/report.md` here.

| Model | WER % | CER % | WER % (Whisper's default normaliser) |
|---|---|---|---|
| whisper-small, zero-shot | _run me_ | _run me_ | _run me_ |
| whisper-small + FLEURS fine-tune (this repo) | _run me_ | _run me_ | _run me_ |
| + AI4Bharat IndicVoices (optional) | _run me_ | _run me_ | _run me_ |

**What to expect:** public Whisper-small Tamil fine-tunes on FLEURS report a zero-shot WER of about
35% and 15–21% after fine-tuning (see `sgangireddy/whisper-small-ta` and `steja/whisper-small-tamil`
on the Hub). Those numbers use Whisper's default normaliser, though. Compare your results against
them using the third column, and lead with the honest first column.

## Quickstart

Open `notebooks/train_colab.ipynb` on a free **T4** GPU. It runs a quick baseline, fine-tunes,
evaluates and pushes to `SharbeshS8199/whisper-small-tamil`. From a terminal:

```bash
pip install -r requirements.txt && pip install -e .
python -m tamil_asr.evaluate --finetuned openai/whisper-small --limit 200 --out reports/baseline   # before
python -m tamil_asr.train --config configs/whisper-small-ta.yaml --push                            # ~4-5 h on T4
python -m tamil_asr.evaluate --finetuned outputs/whisper-small-ta/final                            # after
python -m tamil_asr.app --model SharbeshS8199/whisper-small-tamil                                  # mic demo
pytest     # normaliser + metric tests, no GPU needed
```

**Adding AI4Bharat data:** accept the terms at `huggingface.co/datasets/ai4bharat/IndicVoices`, run
`huggingface-cli login`, then set `enabled: true` under it in `configs/whisper-small-ta.yaml`. The
transcript column is detected automatically. IndicVoices is natural speech from across Tamil Nadu,
which is the best way to make the model robust beyond FLEURS's read speech.

## How it works

```
FLEURS ta_in (+ IndicVoices) ─► resample 16 kHz ─► log-Mel features │ tokenised text (language=tamil, task=transcribe)
                              ─► drop >30 s audio / >448 tokens (Whisper's limits)
whisper-small ─► Seq2SeqTrainer: lr 1e-5, warmup 500, fp16, gradient checkpointing, eval WER every 1000 steps,
                  keep best checkpoint by WER ─► push to Hub
evaluate ─► same FLEURS test set for zero-shot and fine-tuned ─► WER, CER, default-normaliser WER, examples
```

```
src/tamil_asr/  normalize.py  metrics.py  data.py  train.py  evaluate.py  app.py
configs/        whisper-small-ta.yaml      notebooks/train_colab.ipynb      tests/
```

**Data credit:** FLEURS (Conneau et al., 2022, CC-BY-4.0); IndicVoices (Javed et al., AI4Bharat, 2024, CC-BY-4.0).
