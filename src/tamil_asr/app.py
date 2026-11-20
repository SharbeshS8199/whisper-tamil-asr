"""Gradio demo: speak or upload Tamil audio, get text.

    python -m tamil_asr.app --model SharbeshS8199/whisper-small-tamil
"""

from __future__ import annotations

import argparse


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="outputs/whisper-small-ta/final")
    ap.add_argument("--share", action="store_true")
    args = ap.parse_args()

    import gradio as gr
    import torch
    from transformers import pipeline

    asr = pipeline("automatic-speech-recognition", model=args.model, chunk_length_s=30,
                   device=0 if torch.cuda.is_available() else -1)

    def run(path):
        if path is None:
            return ""
        return asr(path, generate_kwargs={"language": "tamil", "task": "transcribe"})["text"]

    gr.Interface(run, gr.Audio(sources=["microphone", "upload"], type="filepath", label="Tamil audio"),
                 gr.Textbox(label="Transcript", lines=4), title="தமிழ் பேச்சு → உரை (Tamil speech to text)",
                 description="Whisper-small fine-tuned on Tamil by Sharbesh S. Long audio is chunked into 30 s windows.",
                 ).launch(share=args.share)


if __name__ == "__main__":
    main()
