"""Whisper Tamil dashboard.   streamlit run dashboard/app.py"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "dashboard")]
import ui  # noqa: E402
from tamil_asr.metrics import cer, wer  # noqa: E402
from tamil_asr.normalize import normalize_tamil, strips_marks_like_whisper  # noqa: E402

ui.page("Whisper · Tamil Speech Recognition", "Zero-shot vs fine-tuned WER/CER · error analysis · honest Tamil normalisation", "🎙️")
REPORTS = ROOT / "reports"
with st.sidebar:
    folder = st.selectbox("Evaluation folder", sorted({str(p.parent.relative_to(ROOT)) for p in ROOT.rglob("reports*/**/metrics.json")}
                                                      | {"reports"}))
REPORTS = ROOT / folder
m = json.loads((REPORTS / "metrics.json").read_text()) if (REPORTS / "metrics.json").exists() else None
zs, ft = (m or {}).get("zero_shot"), (m or {}).get("fine_tuned")
ui.kpis([("Model", "whisper-small", "244M params · language=tamil"),
         ("WER zero-shot", f"{zs['wer']:.1f}%" if zs else "–", "before fine-tuning"),
         ("WER fine-tuned", f"{ft['wer']:.1f}%" if ft else "–",
          f"{m['relative_wer_reduction_pct']:.0f}% relative reduction" if m and m.get("relative_wer_reduction_pct") else "train to see"),
         ("CER fine-tuned", f"{ft['cer']:.1f}%" if ft else "–", "character error rate"),
         ("Test utterances", f"{m['n']:,}" if m else "–", "FLEURS ta_in test")])
st.write("")
t_res, t_err, t_norm, t_try = st.tabs(["📊 Results", "🔎 Error analysis", "🔤 Normaliser explorer", "🎧 Transcribe"])

with t_res:
    if not m:
        ui.empty_state("No evaluation yet", "Run the Colab notebook (<code>notebooks/train_colab.ipynb</code>, free T4) "
                       "and copy <code>reports/</code> here. The 5-minute baseline cell already fills the zero-shot numbers.",
                       "python -m tamil_asr.evaluate --finetuned outputs/whisper-small-ta/final")
    else:
        rows = []
        for name, r in (("zero-shot", zs), ("fine-tuned", ft)):
            rows += [{"model": name, "metric": "WER (Tamil-safe)", "value": r["wer"]},
                     {"model": name, "metric": "CER", "value": r["cer"]},
                     {"model": name, "metric": "WER (Whisper default normaliser)", "value": r["wer_if_marks_stripped"]}]
        st.plotly_chart(ui.style_fig(px.bar(pd.DataFrame(rows), x="metric", y="value", color="model", barmode="group",
                                            text="value", title="Error rates (%), lower is better"), 380), use_container_width=True)
        st.caption("The third pair shows what you'd report with Whisper's default normaliser, which splits Tamil words at "
                   "vowel signs. Lead with the Tamil-safe WER.")

with t_err:
    p = REPORTS / "predictions.jsonl"
    if not p.exists():
        st.caption("Per-utterance analysis appears after evaluation (predictions.jsonl).")
    else:
        df = pd.DataFrame([json.loads(l) for l in p.read_text(encoding="utf-8").splitlines()])
        df["words"] = df["ref"].map(lambda s: len(normalize_tamil(s).split()))
        df["WER zero-shot"] = [wer([r], [h]) for r, h in zip(df["ref"], df["zero_shot"])]
        df["WER fine-tuned"] = [wer([r], [h]) for r, h in zip(df["ref"], df["fine_tuned"])]
        df["improvement"] = df["WER zero-shot"] - df["WER fine-tuned"]
        a, b = st.columns(2)
        a.plotly_chart(ui.style_fig(px.scatter(df, x="WER zero-shot", y="WER fine-tuned", size="words", hover_data=["ref"],
                                               title="Each utterance: below the diagonal = fine-tuning helped"), 360),
                       use_container_width=True)
        b.plotly_chart(ui.style_fig(px.histogram(df, x="improvement", nbins=30, title="WER improvement per utterance"), 360),
                       use_container_width=True)
        st.markdown("**Hardest utterances after fine-tuning**")
        st.dataframe(df.sort_values("WER fine-tuned", ascending=False).head(25)[["ref", "fine_tuned", "WER fine-tuned", "zero_shot"]],
                     hide_index=True, use_container_width=True)

with t_norm:
    st.markdown("Type a reference and a transcript to see how Whisper's default normaliser **fragments Tamil words** "
                "and understates the error.")
    ref = st.text_input("Reference", "நான் பள்ளிக்கு சென்றேன்")
    hyp = st.text_input("Transcript", "நான் பள்ளி சென்றேன்")
    a, b = st.columns(2)
    with a.container(border=True):
        st.markdown("**Tamil-safe normaliser (this repo)**")
        st.code(f"ref: {normalize_tamil(ref)}\nhyp: {normalize_tamil(hyp)}", language="text")
        st.metric("WER", f"{wer([ref], [hyp]):.1f}%")
        st.metric("CER", f"{cer([ref], [hyp]):.1f}%")
    with b.container(border=True):
        st.markdown("**Whisper's default normaliser**")
        r2, h2 = strips_marks_like_whisper(ref), strips_marks_like_whisper(hyp)
        st.code(f"ref: {r2}\nhyp: {h2}", language="text")
        st.metric("WER", f"{wer([r2], [h2], normalize=False):.1f}%")
        st.metric("'words' in reference", f"{len(r2.split())} (really {len(normalize_tamil(ref).split())})")

with t_try:
    model_dirs = [p for p in [ROOT / "outputs" / "whisper-small-ta" / "final"] if p.exists()]
    model_id = st.text_input("Model (local folder or Hub id)", str(model_dirs[0]) if model_dirs else "SharbeshS8199/whisper-small-tamil")
    audio = st.file_uploader("Tamil audio (wav/mp3/m4a)", type=["wav", "mp3", "m4a", "flac", "ogg"])
    if audio is not None and st.button("Transcribe", type="primary"):
        try:
            import tempfile

            from transformers import pipeline
        except ImportError:
            st.error("Install the training requirements first: pip install -r requirements.txt")
            st.stop()
        tmp = Path(tempfile.mkdtemp()) / audio.name
        tmp.write_bytes(audio.getbuffer())
        with st.spinner("Transcribing…"):
            asr = pipeline("automatic-speech-recognition", model=model_id, chunk_length_s=30)
            st.success(asr(str(tmp), generate_kwargs={"language": "tamil", "task": "transcribe"})["text"])
    st.caption("Microphone demo: python -m tamil_asr.app")

ui.footer("whisper-small · FLEURS ta_in (+ AI4Bharat IndicVoices)")
