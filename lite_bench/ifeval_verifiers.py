"""Strict IFEval verifiers compatible with the official dataset schema.

The dataset stores the instruction arguments alongside every prompt.  These
functions deliberately use those argument names and strict semantics so that a
score can be compared with IFEval's strict prompt-level metric.
"""

from __future__ import annotations

import json
import re
import string
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from langdetect import LangDetectException, detect


def _count_sentences(value: str) -> int:
    return len([sentence for sentence in re.split(r"[.!?]+", value) if sentence.strip()])


def _count_words(value: str) -> int:
    return len(value.split())


def _matches_relation(count: int, relation: object, target: object) -> bool:
    if not isinstance(target, int):
        return False
    if relation == "less than":
        return count < target
    if relation == "at least":
        return count >= target
    return False


def _paragraphs(value: str) -> list[str] | None:
    paragraphs = re.split(r"\s?\*\*\*\s?", value)
    for index, paragraph in enumerate(paragraphs):
        if not paragraph.strip():
            if index in (0, len(paragraphs) - 1):
                continue
            return None
    return [paragraph for paragraph in paragraphs if paragraph.strip()]


def verify_number_words(response: str, **kwargs: Any) -> bool:
    return _matches_relation(
        _count_words(response), kwargs.get("relation"), kwargs.get("num_words")
    )


def verify_number_sentences(response: str, **kwargs: Any) -> bool:
    return _matches_relation(
        _count_sentences(response), kwargs.get("relation"), kwargs.get("num_sentences")
    )


def verify_number_paragraphs(response: str, **kwargs: Any) -> bool:
    paragraphs = _paragraphs(response)
    return paragraphs is not None and len(paragraphs) == kwargs.get("num_paragraphs")


def verify_nth_paragraph_first_word(response: str, **kwargs: Any) -> bool:
    paragraphs = _paragraphs(response)
    nth = kwargs.get("nth_paragraph")
    first_word = kwargs.get("first_word")
    if paragraphs is None or not isinstance(nth, int) or not isinstance(first_word, str):
        return False
    if nth < 1 or nth > len(paragraphs):
        return False
    words = paragraphs[nth - 1].split()
    return bool(words) and words[0].strip(string.punctuation).lower() == first_word.lower()


def verify_number_placeholders(response: str, **kwargs: Any) -> bool:
    required = kwargs.get("num_placeholders")
    return isinstance(required, int) and len(re.findall(r"\[.*?\]", response)) >= required


def verify_postscript(response: str, **kwargs: Any) -> bool:
    marker = kwargs.get("postscript_marker")
    if not isinstance(marker, str):
        return False
    if marker == "P.P.S":
        pattern = r"\s*p\.\s?p\.\s?s.*$"
    elif marker == "P.S.":
        pattern = r"\s*p\.\s?s\..*$"
    else:
        pattern = rf"\s*{re.escape(marker.lower())}.*$"
    return bool(re.findall(pattern, response.lower(), flags=re.MULTILINE))


def verify_number_bullet_lists(response: str, **kwargs: Any) -> bool:
    expected = kwargs.get("num_bullets")
    if not isinstance(expected, int):
        return False
    stars = re.findall(r"^\s*\*[^\*].*$", response, flags=re.MULTILINE)
    dashes = re.findall(r"^\s*-.*$", response, flags=re.MULTILINE)
    return len(stars) + len(dashes) == expected


def verify_constrained_response(response: str, **kwargs: Any) -> bool:
    options = kwargs.get(
        "options", ("My answer is yes.", "My answer is no.", "My answer is maybe.")
    )
    return isinstance(options, Sequence) and any(
        isinstance(option, str) and option in response.strip() for option in options
    )


def verify_number_highlighted_sections(response: str, **kwargs: Any) -> bool:
    expected = kwargs.get("num_highlights")
    if not isinstance(expected, int):
        return False
    highlights = re.findall(r"\*[^\n\*]*\*", response)
    double_highlights = re.findall(r"\*\*[^\n\*]*\*\*", response)
    count = sum(bool(highlight.strip("*").strip()) for highlight in highlights)
    count += sum(
        bool(highlight.removeprefix("**").removesuffix("**").strip())
        for highlight in double_highlights
    )
    return count >= expected


def verify_multiple_sections(response: str, **kwargs: Any) -> bool:
    splitter = kwargs.get("section_spliter")
    expected = kwargs.get("num_sections")
    if not isinstance(splitter, str) or not isinstance(expected, int):
        return False
    pattern = rf"\s?{re.escape(splitter)}\s?\d+\s?"
    return len(re.split(pattern, response)) - 1 >= expected


def verify_json_format(response: str, **kwargs: Any) -> bool:
    try:
        json.loads(response)
    except (json.JSONDecodeError, TypeError):
        return False
    return True


def verify_title(response: str, **kwargs: Any) -> bool:
    return bool(re.search(r"<<.+?>>", response))


def verify_keywords_existence(response: str, **kwargs: Any) -> bool:
    keywords = kwargs.get("keywords")
    return isinstance(keywords, Sequence) and all(
        isinstance(keyword, str) and re.search(keyword, response, flags=re.IGNORECASE)
        for keyword in keywords
    )


def verify_keyword_frequency(response: str, **kwargs: Any) -> bool:
    keyword = kwargs.get("keyword")
    frequency = kwargs.get("frequency")
    if not isinstance(keyword, str) or not isinstance(frequency, int):
        return False
    return len(re.findall(keyword, response, flags=re.IGNORECASE)) == frequency


def verify_forbidden_words(response: str, **kwargs: Any) -> bool:
    forbidden_words = kwargs.get("forbidden_words")
    return isinstance(forbidden_words, Sequence) and all(
        isinstance(word, str) and not re.search(word, response, flags=re.IGNORECASE)
        for word in forbidden_words
    )


def verify_letter_frequency(response: str, **kwargs: Any) -> bool:
    letter = kwargs.get("letter")
    if not isinstance(letter, str) or len(letter) != 1:
        return False
    return _matches_relation(
        response.lower().count(letter.lower()),
        kwargs.get("let_relation"),
        kwargs.get("let_frequency"),
    )


def verify_response_language(response: str, **kwargs: Any) -> bool:
    language = kwargs.get("language")
    if not isinstance(language, str):
        return False
    expected = {"english": "en", "chinese": "zh-cn"}.get(language.lower(), language.lower())
    try:
        return detect(response) == expected
    except LangDetectException:
        # The reference evaluator treats indeterminate short text as compliant.
        return True


def verify_two_responses(response: str, **kwargs: Any) -> bool:
    return "******" in response


def verify_repeat_prompt(response: str, **kwargs: Any) -> bool:
    prompt = kwargs.get("prompt_to_repeat")
    return isinstance(prompt, str) and response.strip().startswith(prompt.strip())


def verify_end_checker(response: str, **kwargs: Any) -> bool:
    end_phrase = kwargs.get("end_phrase")
    return isinstance(end_phrase, str) and response.strip().endswith(end_phrase)


def verify_quotation(response: str, **kwargs: Any) -> bool:
    value = response.strip()
    return (value.startswith('"') and value.endswith('"')) or (
        value.startswith("'") and value.endswith("'")
    )


def verify_capital_word_frequency(response: str, **kwargs: Any) -> bool:
    words = re.findall(r"\b[A-Z]+\b", response)
    return _matches_relation(
        len(words), kwargs.get("capital_relation"), kwargs.get("capital_frequency")
    )


def verify_english_lowercase(response: str, **kwargs: Any) -> bool:
    return all(letter.islower() for letter in re.findall(r"[a-zA-Z]", response))


def verify_english_capital(response: str, **kwargs: Any) -> bool:
    return all(letter.isupper() for letter in re.findall(r"[a-zA-Z]", response))


def verify_no_comma(response: str, **kwargs: Any) -> bool:
    return "," not in response


Verifier = Callable[..., bool]

VERIFIERS: dict[str, Verifier] = {
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
    "change_case:capital_word_frequency": verify_capital_word_frequency,
    "change_case:english_lowercase": verify_english_lowercase,
    "change_case:english_capital": verify_english_capital,
    "punctuation:no_comma": verify_no_comma,
}


def verify_instruction(instruction_id: str, response: str, kwargs: Mapping[str, Any]) -> bool:
    """Check one IFEval instruction and fail closed on malformed data."""
    verifier = VERIFIERS.get(instruction_id)
    if verifier is None:
        return False
    try:
        return verifier(response, **kwargs)
    except (TypeError, ValueError, re.error):
        return False


def verify_all(
    instruction_ids: Sequence[str], response: str, kwargs_list: Sequence[Mapping[str, Any]]
) -> bool:
    """Return true only when every instruction in a prompt is satisfied."""
    if len(instruction_ids) != len(kwargs_list):
        return False
    return all(
        verify_instruction(instruction_id, response, kwargs)
        for instruction_id, kwargs in zip(instruction_ids, kwargs_list, strict=True)
    )
