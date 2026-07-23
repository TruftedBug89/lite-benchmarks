"""Programmatic verifiers for IFEval instructions.

Implements the official verification logic from google/IFEval so that
instruction-following can be scored deterministically without an LLM judge.
"""

from __future__ import annotations

import json
import re
import string


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _count_words(text: str) -> int:
    return len(text.split())


def _count_sentences(text: str) -> int:
    sentences = re.split(r"[.!?]+", text)
    return len([s for s in sentences if s.strip()])


def _count_paragraphs(text: str) -> int:
    paragraphs = re.split(r"\n\s*\n", text)
    return len([p for p in paragraphs if p.strip()])


def _get_paragraphs(text: str) -> list[str]:
    paragraphs = re.split(r"\n\s*\n", text)
    return [p.strip() for p in paragraphs if p.strip()]


# ---------------------------------------------------------------------------
# Verifiers — each returns True if the response satisfies the constraint
# ---------------------------------------------------------------------------

def verify_number_words(response: str, **kw) -> bool:
    count = _count_words(response)
    if "min_words" in kw and kw["min_words"] is not None:
        if count < kw["min_words"]:
            return False
    if "max_words" in kw and kw["max_words"] is not None:
        if count > kw["max_words"]:
            return False
    return True


def verify_number_sentences(response: str, **kw) -> bool:
    count = _count_sentences(response)
    if "min_sentences" in kw and kw["min_sentences"] is not None:
        if count < kw["min_sentences"]:
            return False
    if "max_sentences" in kw and kw["max_sentences"] is not None:
        if count > kw["max_sentences"]:
            return False
    return True


def verify_number_paragraphs(response: str, **kw) -> bool:
    count = _count_paragraphs(response)
    if "num_paragraphs" in kw and kw["num_paragraphs"] is not None:
        return count == kw["num_paragraphs"]
    if "min_paragraphs" in kw and kw["min_paragraphs"] is not None:
        if count < kw["min_paragraphs"]:
            return False
    if "max_paragraphs" in kw and kw["max_paragraphs"] is not None:
        if count > kw["max_paragraphs"]:
            return False
    return True


def verify_nth_paragraph_first_word(response: str, **kw) -> bool:
    nth = kw.get("nth_paragraph", 1)
    first_word = kw.get("first_word", "")
    paragraphs = _get_paragraphs(response)
    if nth > len(paragraphs):
        return False
    para = paragraphs[nth - 1]
    words = para.split()
    if not words:
        return False
    return words[0].strip(string.punctuation).lower() == first_word.lower()


def verify_number_placeholders(response: str, **kw) -> bool:
    num = kw.get("num_placeholders", 0)
    placeholders = re.findall(r"\[.*?\]", response)
    return len(placeholders) >= num


def verify_postscript(response: str, **kw) -> bool:
    marker = kw.get("postscript_marker", "P.S.")
    return marker.lower() in response.lower()


def verify_number_bullet_lists(response: str, **kw) -> bool:
    num = kw.get("num_bullets", 0)
    bullets = re.findall(r"^\s*[\*\-\+]\s+", response, re.MULTILINE)
    return len(bullets) == num


def verify_constrained_response(response: str, **kw) -> bool:
    options = kw.get("options", [])
    stripped = response.strip()
    return any(stripped.lower() == opt.lower() for opt in options)


def verify_number_highlighted_sections(response: str, **kw) -> bool:
    num = kw.get("num_highlights", 0)
    highlights = re.findall(r"\*[^*]+\*", response)
    return len(highlights) >= num


def verify_multiple_sections(response: str, **kw) -> bool:
    num = kw.get("num_sections", 0)
    sections = re.findall(r"^#{1,6}\s+", response, re.MULTILINE)
    return len(sections) >= num


def verify_json_format(response: str, **kw) -> bool:
    stripped = response.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```\w*\n?", "", stripped)
        stripped = re.sub(r"\n?```$", "", stripped)
        stripped = stripped.strip()
    try:
        json.loads(stripped)
        return True
    except (json.JSONDecodeError, ValueError):
        return False


def verify_title(response: str, **kw) -> bool:
    return bool(re.search(r"<<.+?>>", response))


def verify_keywords_existence(response: str, **kw) -> bool:
    keywords = kw.get("keywords", [])
    response_lower = response.lower()
    return all(kw.lower() in response_lower for kw in keywords)


def verify_keyword_frequency(response: str, **kw) -> bool:
    keyword = kw.get("keyword", "")
    frequency = kw.get("frequency", 1)
    if not keyword:
        return True
    count = response.lower().count(keyword.lower())
    return count == frequency


def verify_forbidden_words(response: str, **kw) -> bool:
    forbidden = kw.get("forbidden_words", [])
    response_lower = response.lower()
    return all(w.lower() not in response_lower for w in forbidden)


def verify_letter_frequency(response: str, **kw) -> bool:
    letter = kw.get("letter", "")
    frequency = kw.get("frequency", 0)
    if not letter:
        return True
    count = response.lower().count(letter.lower())
    return count == frequency


def verify_response_language(response: str, **kw) -> bool:
    language = kw.get("language", "English")
    try:
        from langdetect import detect
        detected = detect(response)
        lang_map = {
            "english": "en", "french": "fr", "spanish": "es", "german": "de",
            "italian": "it", "portuguese": "pt", "dutch": "nl", "russian": "ru",
            "chinese": "zh-cn", "japanese": "ja", "korean": "ko", "arabic": "ar",
            "hindi": "hi", "turkish": "tr", "polish": "pl", "swedish": "sv",
        }
        expected = lang_map.get(language.lower(), language.lower())
        return detected == expected
    except Exception:
        return False


def verify_two_responses(response: str, **kw) -> bool:
    return "******" in response


def verify_repeat_prompt(response: str, **kw) -> bool:
    prompt = kw.get("prompt", "")
    if not prompt:
        return True
    return response.strip().startswith(prompt.strip())


def verify_end_checker(response: str, **kw) -> bool:
    end_phrase = kw.get("end_phrase", "")
    return response.strip().endswith(end_phrase)


def verify_quotation(response: str, **kw) -> bool:
    stripped = response.strip()
    return (
        (stripped.startswith('"') and stripped.endswith('"'))
        or (stripped.startswith("'") and stripped.endswith("'"))
    )


def verify_english_lowercase(response: str, **kw) -> bool:
    letters = [c for c in response if c.isalpha()]
    return all(c.islower() for c in letters) if letters else True


def verify_english_capital(response: str, **kw) -> bool:
    letters = [c for c in response if c.isalpha()]
    return all(c.isupper() for c in letters) if letters else True


def verify_no_comma(response: str, **kw) -> bool:
    return "," not in response


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

VERIFIERS: dict[str, callable] = {
    "length_constraints:number_words": verify_number_words,
    "length_constraints:number_sentences": verify_number_sentences,
    "length_constraints:number_paragraphs": verify_number_paragraphs,
    "length_constraints:nth_paragraph_first_word": verify_nth_paragraph_first_word,
    "detectable_content:number_placeholders": verify_number_placeholders,
    "detectable_content:postscript": verify_postscript,
    "detectable_format:number_bullet_lists": verify_number_bullet_lists,
    "detectable_format:constrained_response": verify_constrained_response,
    "detectable_format:number_highlighted_sections": verify_number_highlighted_sections,
    "detectable_format:multiple_sections": verify_multiple_sections,
    "detectable_format:json_format": verify_json_format,
    "detectable_format:title": verify_title,
    "keywords:existence": verify_keywords_existence,
    "keywords:frequency": verify_keyword_frequency,
    "keywords:forbidden_words": verify_forbidden_words,
    "keywords:letter_frequency": verify_letter_frequency,
    "language:response_language": verify_response_language,
    "combination:two_responses": verify_two_responses,
    "combination:repeat_prompt": verify_repeat_prompt,
    "startend:end_checker": verify_end_checker,
    "startend:quotation": verify_quotation,
    "change_case:english_lowercase": verify_english_lowercase,
    "change_case:english_capital": verify_english_capital,
    "punctuation:no_comma": verify_no_comma,
}


def verify_instruction(instruction_id: str, response: str, kwargs: dict) -> bool:
    """Verify a single IFEval instruction. Returns True if satisfied."""
    verifier = VERIFIERS.get(instruction_id)
    if verifier is None:
        return False  # Unknown instruction → fail safe
    try:
        return verifier(response, **kwargs)
    except Exception:
        return False


def verify_all(instruction_ids: list[str], response: str, kwargs_list: list[dict]) -> bool:
    """Verify all instructions for an IFEval question. All must pass."""
    for iid, kw in zip(instruction_ids, kwargs_list):
        if not verify_instruction(iid, response, kw or {}):
            return False
    return True
