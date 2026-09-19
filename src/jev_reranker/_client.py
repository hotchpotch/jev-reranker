"""Bounded HTTP retries and strict Noul response validation."""

import json
import math
import random
import time
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from typing import Any

import httpx

from .errors import APIError, ContextLimitError, ResponseValidationError

RETRY_STATUSES = {429, 500, 502, 503, 504, 529}


def json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def retry_delay(attempt: int, retry_after: str | None) -> float:
    if retry_after:
        try:
            seconds = float(retry_after)
        except ValueError:
            try:
                date = parsedate_to_datetime(retry_after)
                if date.tzinfo is None:
                    date = date.replace(tzinfo=UTC)
                seconds = (date - datetime.now(UTC)).total_seconds()
            except (ValueError, TypeError, OverflowError):
                seconds = float("nan")
        if math.isfinite(seconds):
            return min(60.0, max(0.0, seconds))
    return min(60.0, 0.5 * 2 ** min(attempt, 7) + random.uniform(0, 0.2))


def validate(data: Any, keys: list[str]) -> list[float]:
    if not isinstance(data, dict):
        raise ResponseValidationError("Response must be a JSON object.")
    answers = data.get("answers")
    if not isinstance(answers, dict) or set(answers) != set(keys):
        raise ResponseValidationError(
            "Answers must match all requested questions exactly."
        )
    scores = []
    for key in keys:
        answer = answers[key]
        value = answer.get("noul") if isinstance(answer, dict) else None
        if (
            not isinstance(answer, dict)
            or answer.get("type") != "noul"
            or isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not 0 <= value <= 1
        ):
            raise ResponseValidationError(f"Invalid Noul answer for {key}.")
        scores.append(float(value))
    model, usage = data.get("model"), data.get("usage")
    if not isinstance(model, str) or not model.strip() or not isinstance(usage, dict):
        raise ResponseValidationError("Response is missing model or usage metadata.")
    for field in ("input_tokens", "output_tokens"):
        value = usage.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ResponseValidationError(f"Invalid {field} usage.")
    return scores


def request(
    *,
    client: httpx.Client,
    endpoint: str,
    api_key: str,
    timeout: float,
    max_retries: int,
    payload: dict[str, Any],
    trace: dict[str, Any],
) -> tuple[list[float], dict[str, Any]]:
    """Mutate only this request's trace; never record HTTP headers or error bodies."""
    trace.update(attempts=0, retries=0, waits=[], status="running")
    started = time.monotonic()
    try:
        for attempt in range(max_retries + 1):
            trace["attempts"] += 1
            trace["retries"] = attempt
            retry_after = None
            try:
                reply = client.post(
                    endpoint,
                    content=json_text(payload).encode(),
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                    timeout=timeout,
                    follow_redirects=False,
                )
            except httpx.TransportError:
                trace["status_code"] = None
                if attempt == max_retries:
                    raise APIError(
                        "TypeSafe transport failed after bounded retries."
                    ) from None
            else:
                trace["status_code"] = reply.status_code
                if reply.status_code == 200:
                    try:
                        data = reply.json()
                    except (ValueError, UnicodeError):
                        raise ResponseValidationError(
                            "TypeSafe returned invalid JSON."
                        ) from None
                    scores = validate(data, list(payload["questions"]))
                    trace["status"] = "success"
                    return scores, data
                if reply.status_code in {400, 422}:
                    try:
                        error = reply.json()
                    except ValueError:
                        error = None
                    detail = error.get("detail") if isinstance(error, dict) else None
                    if (
                        isinstance(detail, dict)
                        and detail.get("error_type") == "max_tokens_exceeded"
                    ):
                        trace["status"] = "max_tokens_exceeded"
                        raise ContextLimitError(
                            "TypeSafe rejected the request: max_tokens_exceeded."
                        )
                if reply.status_code not in RETRY_STATUSES or attempt == max_retries:
                    raise APIError(
                        f"TypeSafe request failed with HTTP {reply.status_code}.",
                        reply.status_code,
                    )
                retry_after = reply.headers.get("Retry-After")
            delay = retry_delay(attempt, retry_after)
            trace["waits"].append(delay)
            time.sleep(delay)
    finally:
        trace["elapsed_seconds"] = time.monotonic() - started
        if trace["status"] == "running":
            trace["status"] = "failed"
    raise AssertionError("unreachable")
