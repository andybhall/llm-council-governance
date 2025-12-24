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


def build_stage1_prompt(query: str) -> str:
    """
    Build the standard Stage-1 prompt with FINAL ANSWER instruction.

    All governance structures should use this for Stage-1 to ensure
    consistent prompting across structures (eliminates prompting as a confound).

    Args:
        query: The raw query/question text

    Returns:
        Formatted prompt with FINAL ANSWER instruction
    """
    return f"""{query}

After your reasoning, state your final answer in this exact format:
FINAL ANSWER: [your answer]"""


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


def normalize_numeric_answer(answer: str) -> Optional[float]:
    """
    Normalize a numeric answer for voting comparison.

    Converts string to float for numeric comparison.
    Handles commas, dollar signs, and whitespace.

    Args:
        answer: String representation of number

    Returns:
        Float value or None if not parseable
    """
    if not answer:
        return None

    # Remove common prefixes/suffixes
    cleaned = answer.strip()
    cleaned = cleaned.lstrip("$")
    cleaned = cleaned.replace(",", "")
    cleaned = cleaned.rstrip(".")

    try:
        return float(cleaned)
    except ValueError:
        return None


def normalize_letter_answer(answer: str) -> Optional[str]:
    """
    Normalize a letter answer for voting comparison.

    Extracts and uppercases single letter (A-J).

    Args:
        answer: String possibly containing a letter answer

    Returns:
        Uppercase letter or None if not found
    """
    if not answer:
        return None

    # Look for single letter
    match = re.search(r"([A-Ja-j])", answer.strip())
    if match:
        return match.group(1).upper()
    return None


def majority_vote_numeric(
    answers: List[str], tiebreaker: Optional[str] = None
) -> Optional[str]:
    """
    Majority vote for numeric answers, comparing by value not string.

    This ensures that "42", "42.0", and "$42" are all treated as the same vote.

    Args:
        answers: List of answer strings (numbers)
        tiebreaker: Optional tiebreaker answer

    Returns:
        Winning answer in original string form, or None if no valid answers
    """
    if not answers:
        return None

    # Group by numeric value
    value_to_originals: dict[float, list[str]] = {}
    for ans in answers:
        val = normalize_numeric_answer(ans)
        if val is not None:
            if val not in value_to_originals:
                value_to_originals[val] = []
            value_to_originals[val].append(ans)

    if not value_to_originals:
        return None

    # Count votes
    counts = {val: len(originals) for val, originals in value_to_originals.items()}
    max_count = max(counts.values())
    winners = [val for val, count in counts.items() if count == max_count]

    # Handle tiebreaker
    if len(winners) > 1 and tiebreaker:
        tb_val = normalize_numeric_answer(tiebreaker)
        if tb_val in winners:
            winners = [tb_val]

    # Return first original form of winner
    winning_val = winners[0]
    return value_to_originals[winning_val][0]


def majority_vote_letter(
    answers: List[str], tiebreaker: Optional[str] = None
) -> Optional[str]:
    """
    Majority vote for letter answers (A, B, C, etc.).

    Normalizes to uppercase before comparing.

    Args:
        answers: List of answer strings (letters)
        tiebreaker: Optional tiebreaker answer

    Returns:
        Winning letter (uppercase), or None if no valid answers
    """
    if not answers:
        return None

    # Normalize to uppercase letters
    normalized = []
    for ans in answers:
        letter = normalize_letter_answer(ans)
        if letter:
            normalized.append(letter)

    if not normalized:
        return None

    # Count votes
    counts = Counter(normalized)
    max_count = max(counts.values())
    winners = [letter for letter, count in counts.items() if count == max_count]

    # Handle tiebreaker
    if len(winners) > 1 and tiebreaker:
        tb_letter = normalize_letter_answer(tiebreaker)
        if tb_letter in winners:
            winners = [tb_letter]

    return winners[0]


def smart_majority_vote(
    answers: List[str], tiebreaker: Optional[str] = None
) -> str:
    """
    Smart majority vote that auto-detects answer type.

    - If all answers parse as numbers → numeric voting
    - If all answers are single letters → letter voting
    - Otherwise → normalized string voting

    Args:
        answers: List of answer strings
        tiebreaker: Optional tiebreaker answer

    Returns:
        Winning answer string
    """
    if not answers:
        return ""

    # Try numeric voting first
    if all(normalize_numeric_answer(a) is not None for a in answers):
        result = majority_vote_numeric(answers, tiebreaker)
        if result:
            return result

    # Try letter voting
    if all(normalize_letter_answer(a) is not None for a in answers):
        result = majority_vote_letter(answers, tiebreaker)
        if result:
            return result

    # Fall back to normalized string voting
    return majority_vote_normalized(answers, tiebreaker)
