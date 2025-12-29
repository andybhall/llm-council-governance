"""Utility functions for governance structures."""

import re
from collections import Counter
from typing import Any, Dict, List, Literal, Optional, Tuple


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


def extract_vote_accept_veto(response: str) -> Optional[Literal["ACCEPT", "VETO"]]:
    """
    Extract ACCEPT or VETO vote from a response.

    Looks for patterns like:
    - "FINAL VOTE: ACCEPT"
    - "FINAL VOTE: VETO"
    - "I vote ACCEPT"
    - "My vote is VETO"

    Args:
        response: The full response text from an LLM

    Returns:
        "ACCEPT" or "VETO" if found, None otherwise
    """
    if not response:
        return None

    text = response.upper()

    # Try structured format first
    pattern = r"FINAL\s*VOTE:\s*(ACCEPT|VETO)"
    match = re.search(pattern, text)
    if match:
        return match.group(1)  # type: ignore

    # Fallback: look for the words anywhere
    if "VETO" in text:
        return "VETO"
    if "ACCEPT" in text:
        return "ACCEPT"

    return None


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
        # Sort for deterministic tie resolution (alphabetical order)
        return sorted(winners)[0]


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
    # Sort for deterministic tie resolution
    winning_norm = sorted(winners)[0]
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
    # Sort for deterministic tie resolution (numeric order)
    winning_val = sorted(winners)[0]
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

    # Sort for deterministic tie resolution (alphabetical order)
    return sorted(winners)[0]


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


def weighted_majority_vote(
    answers: List[str],
    models: List[str],
    weights: dict,
    tiebreaker: Optional[str] = None,
) -> str:
    """
    Weighted majority vote where each model's vote is weighted by its accuracy.

    Args:
        answers: List of answer strings (parallel to models list)
        models: List of model names (parallel to answers list)
        weights: Dictionary mapping model names to weights (typically accuracy rates)
        tiebreaker: Optional tiebreaker answer

    Returns:
        Winning answer string (highest weighted vote total)
    """
    if not answers or not models:
        return ""

    if len(answers) != len(models):
        raise ValueError("answers and models must have the same length")

    # Aggregate weights for each unique answer
    answer_weights: dict = {}
    answer_originals: dict = {}  # Store first occurrence for each answer

    for answer, model in zip(answers, models):
        if answer is None:
            continue

        # Normalize for comparison
        norm_answer = normalize_answer(answer)
        weight = weights.get(model, 1.0)  # Default weight of 1.0 if not found

        if norm_answer not in answer_weights:
            answer_weights[norm_answer] = 0.0
            answer_originals[norm_answer] = answer

        answer_weights[norm_answer] += weight

    if not answer_weights:
        return ""

    # Find highest weighted answer
    max_weight = max(answer_weights.values())
    winners = [ans for ans, w in answer_weights.items() if w == max_weight]

    # Handle ties
    if len(winners) > 1 and tiebreaker:
        norm_tiebreaker = normalize_answer(tiebreaker)
        if norm_tiebreaker in winners:
            return answer_originals[norm_tiebreaker]

    # Return winner (original form), sorted for deterministic tie resolution
    return answer_originals[sorted(winners)[0]]


def weighted_majority_vote_numeric(
    answers: List[str],
    models: List[str],
    weights: dict,
    tiebreaker: Optional[str] = None,
) -> Optional[str]:
    """
    Weighted majority vote for numeric answers.

    Args:
        answers: List of answer strings (numbers)
        models: List of model names (parallel to answers)
        weights: Dictionary mapping model names to weights
        tiebreaker: Optional tiebreaker answer

    Returns:
        Winning answer in original string form, or None if no valid answers
    """
    if not answers or not models:
        return None

    if len(answers) != len(models):
        raise ValueError("answers and models must have the same length")

    # Group by numeric value
    value_weights: dict = {}
    value_originals: dict = {}

    for answer, model in zip(answers, models):
        val = normalize_numeric_answer(answer)
        if val is None:
            continue

        weight = weights.get(model, 1.0)

        if val not in value_weights:
            value_weights[val] = 0.0
            value_originals[val] = answer

        value_weights[val] += weight

    if not value_weights:
        return None

    # Find highest weighted
    max_weight = max(value_weights.values())
    winners = [val for val, w in value_weights.items() if w == max_weight]

    # Handle ties
    if len(winners) > 1 and tiebreaker:
        tb_val = normalize_numeric_answer(tiebreaker)
        if tb_val in winners:
            return value_originals[tb_val]

    # Sort for deterministic tie resolution (numeric order)
    return value_originals[sorted(winners)[0]]


def weighted_majority_vote_letter(
    answers: List[str],
    models: List[str],
    weights: dict,
    tiebreaker: Optional[str] = None,
) -> Optional[str]:
    """
    Weighted majority vote for letter answers (A, B, C, etc.).

    Args:
        answers: List of answer strings (letters)
        models: List of model names (parallel to answers)
        weights: Dictionary mapping model names to weights
        tiebreaker: Optional tiebreaker answer

    Returns:
        Winning letter (uppercase), or None if no valid answers
    """
    if not answers or not models:
        return None

    if len(answers) != len(models):
        raise ValueError("answers and models must have the same length")

    # Group by letter
    letter_weights: dict = {}

    for answer, model in zip(answers, models):
        letter = normalize_letter_answer(answer)
        if letter is None:
            continue

        weight = weights.get(model, 1.0)

        if letter not in letter_weights:
            letter_weights[letter] = 0.0

        letter_weights[letter] += weight

    if not letter_weights:
        return None

    # Find highest weighted
    max_weight = max(letter_weights.values())
    winners = [letter for letter, w in letter_weights.items() if w == max_weight]

    # Handle ties
    if len(winners) > 1 and tiebreaker:
        tb_letter = normalize_letter_answer(tiebreaker)
        if tb_letter in winners:
            return tb_letter

    # Sort for deterministic tie resolution (alphabetical order)
    return sorted(winners)[0]


def smart_weighted_majority_vote(
    answers: List[str],
    models: List[str],
    weights: dict,
    tiebreaker: Optional[str] = None,
) -> str:
    """
    Smart weighted majority vote that auto-detects answer type.

    - If all answers parse as numbers → weighted numeric voting
    - If all answers are single letters → weighted letter voting
    - Otherwise → weighted normalized string voting

    Args:
        answers: List of answer strings
        models: List of model names (parallel to answers)
        weights: Dictionary mapping model names to weights (typically accuracy rates)
        tiebreaker: Optional tiebreaker answer

    Returns:
        Winning answer string
    """
    if not answers:
        return ""

    # Filter out None answers while keeping model alignment
    valid_pairs = [
        (ans, model)
        for ans, model in zip(answers, models)
        if ans is not None
    ]

    if not valid_pairs:
        return ""

    valid_answers = [pair[0] for pair in valid_pairs]
    valid_models = [pair[1] for pair in valid_pairs]

    # Try numeric voting first
    if all(normalize_numeric_answer(a) is not None for a in valid_answers):
        result = weighted_majority_vote_numeric(
            valid_answers, valid_models, weights, tiebreaker
        )
        if result:
            return result

    # Try letter voting
    if all(normalize_letter_answer(a) is not None for a in valid_answers):
        result = weighted_majority_vote_letter(
            valid_answers, valid_models, weights, tiebreaker
        )
        if result:
            return result

    # Fall back to weighted normalized string voting
    return weighted_majority_vote(valid_answers, valid_models, weights, tiebreaker)


def compute_vote_metadata(
    extracted_answers: Dict[str, Optional[str]],
    tiebreaker: Optional[str] = None,
) -> Tuple[str, Dict[str, Any]]:
    """
    Compute vote metadata for regular (unweighted) majority voting.

    Args:
        extracted_answers: Dictionary mapping model names to extracted answers
        tiebreaker: Optional tiebreaker answer (e.g., chairman's answer)

    Returns:
        Tuple of (winning_answer, metadata_dict) where metadata_dict contains:
        - raw_answers: {model: raw_answer}
        - normalized_answers: {model: normalized_answer_or_none}
        - vote_counts: {normalized_answer: count}
        - is_tie: bool
        - winning_answer: normalized winning answer
        - tiebreaker_used: bool
    """
    # Build raw and normalized answer maps
    raw_answers = {}
    normalized_answers = {}

    for model, answer in extracted_answers.items():
        raw_answers[model] = answer
        if answer is not None:
            normalized_answers[model] = normalize_answer(answer)
        else:
            normalized_answers[model] = None

    # Count votes by normalized answer
    valid_normalized = [na for na in normalized_answers.values() if na is not None]

    if not valid_normalized:
        # No valid answers
        return "", {
            "raw_answers": raw_answers,
            "normalized_answers": normalized_answers,
            "vote_counts": {},
            "is_tie": False,
            "winning_answer": "",
            "tiebreaker_used": False,
        }

    vote_counts = dict(Counter(valid_normalized))
    max_count = max(vote_counts.values())
    winners = [ans for ans, count in vote_counts.items() if count == max_count]
    is_tie = len(winners) > 1

    # Determine winner
    tiebreaker_used = False
    if is_tie and tiebreaker:
        norm_tiebreaker = normalize_answer(tiebreaker)
        if norm_tiebreaker in winners:
            winning_norm = norm_tiebreaker
            tiebreaker_used = True
        else:
            winning_norm = sorted(winners)[0]
    else:
        winning_norm = sorted(winners)[0] if winners else ""

    # Get original form of winning answer
    valid_answers = [a for a in extracted_answers.values() if a is not None]
    winning_answer = smart_majority_vote(valid_answers, tiebreaker=tiebreaker)

    return winning_answer, {
        "raw_answers": raw_answers,
        "normalized_answers": normalized_answers,
        "vote_counts": vote_counts,
        "is_tie": is_tie,
        "winning_answer": winning_norm,
        "tiebreaker_used": tiebreaker_used,
    }


def compute_weighted_vote_metadata(
    extracted_answers: Dict[str, Optional[str]],
    weights: Dict[str, float],
    tiebreaker: Optional[str] = None,
) -> Tuple[str, Dict[str, Any]]:
    """
    Compute vote metadata for weighted majority voting.

    Args:
        extracted_answers: Dictionary mapping model names to extracted answers
        weights: Dictionary mapping model names to weights
        tiebreaker: Optional tiebreaker answer (e.g., chairman's answer)

    Returns:
        Tuple of (winning_answer, metadata_dict) where metadata_dict contains:
        - raw_answers: {model: raw_answer}
        - normalized_answers: {model: normalized_answer_or_none}
        - vote_counts: {normalized_answer: total_weight}
        - is_tie: bool
        - winning_answer: normalized winning answer
        - tiebreaker_used: bool
    """
    # Build raw and normalized answer maps
    raw_answers = {}
    normalized_answers = {}

    for model, answer in extracted_answers.items():
        raw_answers[model] = answer
        if answer is not None:
            normalized_answers[model] = normalize_answer(answer)
        else:
            normalized_answers[model] = None

    # Count weighted votes by normalized answer
    vote_weights: Dict[str, float] = {}
    for model, answer in extracted_answers.items():
        if answer is None:
            continue
        norm = normalize_answer(answer)
        weight = weights.get(model, 1.0)
        vote_weights[norm] = vote_weights.get(norm, 0.0) + weight

    if not vote_weights:
        # No valid answers
        return "", {
            "raw_answers": raw_answers,
            "normalized_answers": normalized_answers,
            "vote_counts": {},
            "is_tie": False,
            "winning_answer": "",
            "tiebreaker_used": False,
        }

    max_weight = max(vote_weights.values())
    winners = [ans for ans, w in vote_weights.items() if w == max_weight]
    is_tie = len(winners) > 1

    # Determine winner
    tiebreaker_used = False
    if is_tie and tiebreaker:
        norm_tiebreaker = normalize_answer(tiebreaker)
        if norm_tiebreaker in winners:
            winning_norm = norm_tiebreaker
            tiebreaker_used = True
        else:
            winning_norm = sorted(winners)[0]
    else:
        winning_norm = sorted(winners)[0] if winners else ""

    # Get original form of winning answer using weighted voting
    models = list(extracted_answers.keys())
    answers = [extracted_answers[m] for m in models]
    valid_pairs = [(a, m) for a, m in zip(answers, models) if a is not None]

    if valid_pairs:
        valid_answers = [p[0] for p in valid_pairs]
        valid_models = [p[1] for p in valid_pairs]
        winning_answer = smart_weighted_majority_vote(
            valid_answers, valid_models, weights, tiebreaker=tiebreaker
        )
    else:
        winning_answer = ""

    return winning_answer, {
        "raw_answers": raw_answers,
        "normalized_answers": normalized_answers,
        "vote_counts": vote_weights,
        "is_tie": is_tie,
        "winning_answer": winning_norm,
        "tiebreaker_used": tiebreaker_used,
    }
