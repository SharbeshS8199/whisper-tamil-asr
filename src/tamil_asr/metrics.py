"""WER / CER via Levenshtein distance (same definitions as jiwer, no dependency)."""

from __future__ import annotations

from tamil_asr.normalize import normalize_tamil


def edit_distance(a: list, b: list) -> int:
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def wer(refs: list[str], hyps: list[str], normalize: bool = True) -> float:
    errs = words = 0
    for r, h in zip(refs, hyps):
        if normalize:
            r, h = normalize_tamil(r), normalize_tamil(h)
        rw, hw = r.split(), h.split()
        errs += edit_distance(rw, hw)
        words += len(rw)
    return 100 * errs / max(words, 1)


def cer(refs: list[str], hyps: list[str], normalize: bool = True) -> float:
    errs = chars = 0
    for r, h in zip(refs, hyps):
        if normalize:
            r, h = normalize_tamil(r), normalize_tamil(h)
        errs += edit_distance(list(r), list(h))
        chars += len(r)
    return 100 * errs / max(chars, 1)
