"""Listwise partitioning adapted from HAKARI's MIT TypeSafe adapter."""

import hashlib
import math
from collections.abc import Callable
from typing import Any

from .errors import ContextLimitError

SHUFFLE_SEED = "hakari-typesafe-listwise-chunk-v1"


def balanced_chunks(
    indices: list[int], lengths: list[int], count: int
) -> list[list[int]]:
    base, remainder = divmod(len(indices), count)
    capacities = [base + (i >= count - remainder) for i in range(count)]
    chunks: list[list[int]] = [[] for _ in range(count)]
    totals = [0] * count
    for index in sorted(indices, key=lambda i: (-lengths[i], i)):
        target = min(
            (i for i in range(count) if len(chunks[i]) < capacities[i]),
            key=lambda i: (totals[i], len(chunks[i]), i),
        )
        chunks[target].append(index)
        totals[target] += lengths[index]
    return chunks


def score_listwise(
    *,
    query: str,
    lengths: list[int],
    state_budget: int,
    request_budget: int,
    estimate: Callable[[list[int]], dict[str, int]],
    request: Callable[[list[int]], list[float]],
    splits: list[dict[str, Any]],
) -> list[float]:
    scores = [0.0] * len(lengths)
    query_hash = hashlib.sha256(query.encode()).hexdigest()

    def fits(value: dict[str, int], sb: int, rb: int) -> bool:
        return value["state_plus_longest_question"] <= sb and value["request"] <= rb

    def split(indices: list[int], depth: int, sb: int, rb: int, reason: str) -> None:
        if len(indices) == 1:
            raise ContextLimitError(
                f"Query and document index {indices[0]} cannot fit the context budget."
            )
        value = estimate(indices)
        count = min(
            len(indices),
            max(
                2,
                math.ceil(value["state_plus_longest_question"] / sb),
                math.ceil(value["request"] / rb),
            ),
        )
        while True:
            chunks = balanced_chunks(indices, lengths, count)
            if all(fits(estimate(c), sb, rb) for c in chunks) or count == len(indices):
                break
            count += 1
        for chunk in chunks:
            chunk.sort(
                key=lambda i: hashlib.sha256(
                    f"{SHUFFLE_SEED}:{query_hash}:{i}".encode()
                ).digest()
            )
        splits.append(
            {
                "reason": reason,
                "depth": depth,
                "document_indices": indices,
                "state_token_budget": sb,
                "request_token_budget": rb,
                "estimated_tokens": value,
                "chunks": chunks,
            }
        )
        for chunk in chunks:
            score(chunk, depth + 1, sb, rb)

    def score(indices: list[int], depth: int, sb: int, rb: int) -> None:
        if not fits(estimate(indices), sb, rb):
            split(indices, depth, sb, rb, "estimated_token_budget")
            return
        try:
            values = request(indices)
        except ContextLimitError:
            split(
                indices, depth, max(1, sb // 2), max(1, rb // 2), "max_tokens_exceeded"
            )
            return
        for index, value in zip(indices, values, strict=True):
            scores[index] = value

    score(list(range(len(lengths))), 0, state_budget, request_budget)
    return scores
