from tamil_asr.data import pick_text_column
from tamil_asr.metrics import cer, edit_distance, wer
from tamil_asr.normalize import normalize_tamil, strips_marks_like_whisper


def test_normalizer_keeps_tamil_vowel_signs_and_pulli():
    s = "கல்வி, அழகு!  வணக்கம்‍."
    assert normalize_tamil(s) == "கல்வி அழகு வணக்கம்"
    assert "்" in normalize_tamil("கல்வி")                      # pulli survives
    assert normalize_tamil("Chennai-ல் 2024") == "chennai ல் 2024"


def test_whisper_style_normalizer_fragments_tamil_words():
    assert strips_marks_like_whisper("கல்வி") == "கல வ"            # one word -> fragments
    ref, hyp = "நான் பள்ளிக்கு சென்றேன்", "நான் பள்ளி சென்றேன்"
    assert len(strips_marks_like_whisper(ref).split()) > len(ref.split())
    proper = wer([ref], [hyp])                                    # 1 wrong word out of 3
    whisper_style = wer([strips_marks_like_whisper(ref)], [strips_marks_like_whisper(hyp)], normalize=False)
    assert round(proper, 2) == 33.33 and whisper_style != proper


def test_wer_cer_values():
    assert edit_distance(list("kitten"), list("sitting")) == 3
    ref, hyp = ["நான் பள்ளிக்கு சென்றேன்"], ["நான் பள்ளி சென்றேன்"]
    assert round(wer(ref, hyp), 2) == 33.33
    assert 0 < cer(ref, hyp) < 30
    assert wer(["வணக்கம்"], ["வணக்கம்."]) == 0.0                  # punctuation ignored


def test_text_column_detection():
    assert pick_text_column(["audio", "transcription", "raw_transcription"]) == "transcription"
    assert pick_text_column(["audio_filepath", "text"]) == "text"
