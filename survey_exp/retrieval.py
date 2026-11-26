from __future__ import annotations

import math
import random
import re
from collections import Counter
from typing import Any


def tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def cosine_text(a: str, b: str) -> float:
    left, right = Counter(tokens(a)), Counter(tokens(b))
    if not left or not right:
        return 0.0
    dot = sum(value * right.get(key, 0) for key, value in left.items())
    return dot / math.sqrt(sum(v * v for v in left.values()) * sum(v * v for v in right.values()))


def retrieve(query: str, bank: list[dict[str, Any]], k: int = 3) -> list[dict[str, Any]]:
    scored = []
    for item in bank:
        score = cosine_text(query, f"{item.get('goal', '')} {item.get('output', '')}")
        scored.append((score, str(item.get("experience_id", "")), item))
    scored.sort(key=lambda value: (-value[0], value[1]))
    return [{**item, "retrieval_score": score} for score, _, item in scored[:k]]


def build_index(bank: list[dict[str, Any]]) -> list[tuple[Counter[str], float, dict[str, Any]]]:
    """Tokenize the fixed experience bank once before any model generation."""
    index = []
    for item in bank:
        counts = Counter(tokens(f"{item.get('goal', '')} {item.get('output', '')}"))
        norm = math.sqrt(sum(value * value for value in counts.values()))
        index.append((counts, norm, item))
    return index


def retrieve_indexed(
    query: str,
    index: list[tuple[Counter[str], float, dict[str, Any]]],
    k: int = 3,
) -> list[dict[str, Any]]:
    query_counts = Counter(tokens(query))
    query_norm = math.sqrt(sum(value * value for value in query_counts.values()))
    scored = []
    for counts, norm, item in index:
        if not query_norm or not norm:
            score = 0.0
        else:
            score = sum(value * counts.get(key, 0) for key, value in query_counts.items()) / (query_norm * norm)
        scored.append((score, str(item.get("experience_id", "")), item))
    scored.sort(key=lambda value: (-value[0], value[1]))
    return [{**item, "retrieval_score": score} for score, _, item in scored[:k]]


def matched_random(
    relevant: list[dict[str, Any]], bank: list[dict[str, Any]], seed: int
) -> list[dict[str, Any]]:
    """Random control with the same record count and approximately matched text lengths."""
    rng = random.Random(seed)
    excluded = {item.get("experience_id") for item in relevant}
    available = [item for item in bank if item.get("experience_id") not in excluded]
    rng.shuffle(available)
    selected = []
    for target in relevant:
        target_len = len(str(target.get("output", "")))
        pool = available[: min(1000, len(available))]
        if not pool:
            break
        choice = min(pool, key=lambda item: (abs(len(str(item.get("output", ""))) - target_len), str(item.get("experience_id"))))
        selected.append({**choice, "retrieval_score": None, "matched_length_to": target.get("experience_id")})
        available.remove(choice)
    return selected

