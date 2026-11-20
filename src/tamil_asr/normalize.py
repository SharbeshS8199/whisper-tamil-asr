"""Tamil-safe text normalisation for WER/CER.

Whisper's BasicTextNormalizer replaces every Unicode mark, symbol and punctuation character with a
space. In Tamil the marks are the vowel signs and the pulli (்), so one word such as "கல்வி" becomes
three fragments "கல வ". WER computed that way counts fragments, not words. This normaliser keeps
every mark and only removes what shouldn't count: punctuation, symbols, zero-width joiners and
extra whitespace.
"""

from __future__ import annotations

import re
import unicodedata

_ZW = re.compile("[​‌‍﻿]")
_SPACE = re.compile(r"\s+")


def normalize_tamil(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = _ZW.sub("", text)
    out = []
    for ch in text:
        cat = unicodedata.category(ch)
        if cat[0] in "LMN":            # letters, combining marks (vowel signs, pulli), numbers
            out.append(ch.lower())
        elif ch.isspace():
            out.append(" ")
        else:                          # punctuation (P*), symbols (S*), control chars
            out.append(" ")
    return _SPACE.sub(" ", "".join(out)).strip()


def strips_marks_like_whisper(text: str) -> str:
    """Re-implementation of whisper.normalizers.basic.remove_symbols (used by BasicTextNormalizer):
    NFKC, then marks/symbols/punctuation -> space. For the comparison column in the eval report."""
    text = unicodedata.normalize("NFKC", text).lower()
    return _SPACE.sub(" ", "".join(" " if unicodedata.category(c)[0] in "MSP" else c for c in text)).strip()
