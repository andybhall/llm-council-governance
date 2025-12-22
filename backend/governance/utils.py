"""Utility functions for governance structures."""

import re
from collections import Counter
from typing import List, Optional


def extract_final_answer(response: str) -> Optional[str]:
    """
    Extract final answer from a response that includes 'FINAL ANSWER: X'.

    Args:
        response: The full response text from an LLM

    Returns:
        The extracted answer string, or None if not found
    """
    pattern = r"FINAL ANSWER:\s*(.+?)(?:\n|$)"
    match = re.search(pattern, response, re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


def extract_final_answer_with_fallback(response: str) -> str:
    """
    Extract final answer with fallback to last sentence.

    Args:
        response: The full response text from an LLM

    Returns:
        The extracted answer, or fallback if pattern not found
    """
    answer = extract_final_answer(response)
    if answer:
        return answer

    # Fallback: return last non-empty sentence
    text = response.strip()
    if not text:
        return ""

    # Split by sentence-ending punctuation
    sentences = re.split(r"[.!?]+", text)
    sentences = [s.strip() for s in sentences if s.strip()]

    if sentences:
        return sentences[-1]

    # Last resort: return last 100 chars
    return text[-100:]


def majority_vote(answers: List[str], tiebreaker: Optional[str] = None) -> str:
    """
    Return the most common answer from a list.

    Args:
        answers: List of answer strings
        tiebreaker: Optional answer to prefer in case of tie

    Returns:
        The most common answer, or tiebreaker if tied
    """
    if not answers:
        return ""

    counts = Counter(answers)
    max_count = max(counts.values())
    winners = [ans for ans, count in counts.items() if count == max_count]

    if len(winners) == 1:
        return winners[0]
    elif tiebreaker and tiebreaker in winners:
        return tiebreaker
    else:
        return winners[0]  # Arbitrary if no tiebreaker


def normalize_answer(answer: str) -> str:
    """
    Normalize an answer for comparison.

    Strips whitespace, lowercases, and removes common punctuation.

    Args:
        answer: The answer string to normalize

    Returns:
        Normalized answer string
    """
    if not answer:
        return ""

    # Strip and lowercase
    normalized = answer.strip().lower()

    # Remove trailing punctuation
    normalized = re.sub(r"[.,;:!?]+$", "", normalized)

    return normalized


def majority_vote_normalized(
    answers: List[str], tiebreaker: Optional[str] = None
) -> str:
    """
    Return the most common answer, normalizing for comparison but returning original.

    Args:
        answers: List of answer strings
        tiebreaker: Optional answer to prefer in case of tie

    Returns:
        The most common answer (original form, not normalized)
    """
    if not answers:
        return ""

    # Map normalized -> list of original answers
    normalized_to_originals: dict[str, list[str]] = {}
    for ans in answers:
        norm = normalize_answer(ans)
        if norm not in normalized_to_originals:
            normalized_to_originals[norm] = []
        normalized_to_originals[norm].append(ans)

    # Count by normalized form
    counts = {norm: len(originals) for norm, originals in normalized_to_originals.items()}
    max_count = max(counts.values())
    winners = [norm for norm, count in counts.items() if count == max_count]

    # Handle tiebreaker
    if len(winners) > 1 and tiebreaker:
        norm_tiebreaker = normalize_answer(tiebreaker)
        if norm_tiebreaker in winners:
            winners = [norm_tiebreaker]

    # Return first original form of winning normalized answer
    winning_norm = winners[0]
    return normalized_to_originals[winning_norm][0]
