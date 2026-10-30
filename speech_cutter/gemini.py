from __future__ import annotations

import json
import re
import time
from typing import Callable
from http.client import RemoteDisconnected
from urllib import error, request


DEFAULT_GEMINI_MODEL = "gemini-3.6-flash"
GEMINI_GENERATE_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
GEMINI_REQUEST_TIMEOUT_SECONDS = 75
GEMINI_CHUNK_SIZE = 120
GEMINI_MIN_CHUNK_SIZE = 30
GEMINI_CONTEXT_WORDS = 35
GEMINI_MAX_RATE_LIMIT_RETRIES = 5


class _GeminiSectionTimeout(RuntimeError):
    """Raised internally so an oversized slow section can be divided and retried."""


class _GeminiQuotaExhausted(RuntimeError):
    """Raised internally after the API quota remains unavailable through all retries."""


ChunkFallback = Callable[[list[str], list[str], list[str]], list[str]]


def correct_transcript_words(
    words: list[str],
    *,
    api_key: str,
    model: str = DEFAULT_GEMINI_MODEL,
    cancel_callback: Callable[[], bool] | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
    warning_callback: Callable[[str], None] | None = None,
    quota_fallback: ChunkFallback | None = None,
) -> list[str]:
    """Correct a transcript while preserving its word-level timestamp mapping."""
    if not words:
        return []
    if not api_key.strip():
        raise RuntimeError("Gemini transcript correction is enabled, but no API key was provided.")
    if cancel_callback and cancel_callback():
        raise RuntimeError("Processing was cancelled.")

    corrected_words = list(words)
    chunk_ranges = [
        (start, min(start + GEMINI_CHUNK_SIZE, len(words)))
        for start in range(0, len(words), GEMINI_CHUNK_SIZE)
    ]
    for chunk_number, (start, end) in enumerate(chunk_ranges, start=1):
        if cancel_callback and cancel_callback():
            raise RuntimeError("Processing was cancelled.")
        context_start = max(0, start - GEMINI_CONTEXT_WORDS)
        context_end = min(len(words), end + GEMINI_CONTEXT_WORDS)
        try:
            corrected_chunk = _correct_transcript_chunk_resilient(
                corrected_words[start:end],
                context_before=corrected_words[context_start:start],
                context_after=words[end:context_end],
                api_key=api_key,
                model=model,
                cancel_callback=cancel_callback,
                warning_callback=warning_callback,
                quota_fallback=quota_fallback,
            )
        except _GeminiQuotaExhausted:
            if warning_callback:
                warning_callback(
                    "Warning: Gemini quota remained unavailable after all retries. "
                    f"Keeping corrections from {chunk_number - 1} completed section(s) and leaving "
                    "the remaining transcript unchanged."
                )
            break
        corrected_words[start:end] = corrected_chunk
        if progress_callback:
            progress_callback(chunk_number, len(chunk_ranges))
    return corrected_words


def _correct_transcript_chunk_resilient(
    words: list[str],
    *,
    context_before: list[str],
    context_after: list[str],
    api_key: str,
    model: str,
    cancel_callback: Callable[[], bool] | None,
    warning_callback: Callable[[str], None] | None,
    quota_fallback: ChunkFallback | None,
) -> list[str]:
    try:
        return _correct_transcript_chunk(
            words,
            context_before=context_before,
            context_after=context_after,
            api_key=api_key,
            model=model,
            cancel_callback=cancel_callback,
            warning_callback=warning_callback,
            quota_fallback=quota_fallback,
        )
    except _GeminiSectionTimeout:
        if len(words) <= GEMINI_MIN_CHUNK_SIZE:
            if warning_callback:
                warning_callback(
                    f"Warning: Gemini timed out on a {len(words)}-word transcript fragment; "
                    "only that fragment was left unchanged."
                )
            return list(words)

        midpoint = len(words) // 2
        left_words = _correct_transcript_chunk_resilient(
            words[:midpoint],
            context_before=context_before[-GEMINI_CONTEXT_WORDS:],
            context_after=(words[midpoint:] + context_after)[:GEMINI_CONTEXT_WORDS],
            api_key=api_key,
            model=model,
            cancel_callback=cancel_callback,
            warning_callback=warning_callback,
            quota_fallback=quota_fallback,
        )
        right_words = _correct_transcript_chunk_resilient(
            words[midpoint:],
            context_before=(context_before + left_words)[-GEMINI_CONTEXT_WORDS:],
            context_after=context_after[:GEMINI_CONTEXT_WORDS],
            api_key=api_key,
            model=model,
            cancel_callback=cancel_callback,
            warning_callback=warning_callback,
            quota_fallback=quota_fallback,
        )
        return left_words + right_words


def _correct_transcript_chunk(
    words: list[str],
    *,
    context_before: list[str],
    context_after: list[str],
    api_key: str,
    model: str,
    cancel_callback: Callable[[], bool] | None,
    warning_callback: Callable[[str], None] | None,
    quota_fallback: ChunkFallback | None,
) -> list[str]:
    """Correct one bounded token range while using adjacent words as read-only context."""

    prompt = (
        "You are a meticulous Polish transcript editor. Review the complete transcript for clear "
        "speech-recognition, grammar, punctuation, and logical-context errors. Correct only errors "
        "that can be inferred confidently from the transcript. Do not invent facts or rewrite style. "
        "The TARGET is a JSON array of word tokens in speaking order. Return only TARGET tokens that need "
        "correction in the 'changes' array. Each change must contain the zero-based token 'index' and "
        "its corrected replacement 'word'. Never add, remove, split, merge, or reorder tokens. Do not "
        "return unchanged tokens or changes for the read-only context. Preserve punctuation attached "
        "to a token. If no correction is needed, return an empty changes array.\n\n"
        "READ-ONLY CONTEXT BEFORE:\n"
        + json.dumps(context_before, ensure_ascii=False)
        + "\n\nTARGET TOKENS TO CORRECT:\n"
        + json.dumps(words, ensure_ascii=False)
        + "\n\nREAD-ONLY CONTEXT AFTER:\n"
        + json.dumps(context_after, ensure_ascii=False)
    )
    generation_config: dict[str, object] = {
        "responseMimeType": "application/json",
        "responseJsonSchema": {
            "type": "object",
            "properties": {
                "changes": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "index": {
                                "type": "integer",
                                "minimum": 0,
                                "maximum": len(words) - 1,
                            },
                            "word": {"type": "string"},
                        },
                        "required": ["index", "word"],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["changes"],
            "additionalProperties": False,
        },
    }
    selected_model = model.strip() or DEFAULT_GEMINI_MODEL
    if selected_model.startswith("gemini-3"):
        generation_config["thinkingConfig"] = {"thinkingLevel": "low"}

    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": generation_config,
    }
    url = GEMINI_GENERATE_URL.format(model=selected_model)
    http_request = request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "SpeechCutter/1.0",
            "x-goog-api-key": api_key.strip(),
        },
        method="POST",
    )

    response_data: object | None = None
    retry_delays = (2.0, 4.0, 8.0)
    temporary_attempt = 0
    rate_limit_attempt = 0
    quota_fallback_attempted = False
    while True:
        try:
            with request.urlopen(http_request, timeout=GEMINI_REQUEST_TIMEOUT_SECONDS) as response:
                response_data = json.loads(response.read().decode("utf-8"))
            break
        except error.HTTPError as exc:
            detail = _read_api_error(exc)
            if exc.code == 429 and quota_fallback is not None and not quota_fallback_attempted:
                quota_fallback_attempted = True
                if warning_callback:
                    warning_callback(
                        "Gemini quota is busy; checking this transcript section with the Groq fallback."
                    )
                try:
                    return quota_fallback(words, context_before, context_after)
                except RuntimeError as fallback_exc:
                    if warning_callback:
                        warning_callback(
                            f"Warning: Groq transcript fallback failed ({fallback_exc}); "
                            "waiting for Gemini instead."
                        )
            if exc.code == 429 and rate_limit_attempt < GEMINI_MAX_RATE_LIMIT_RETRIES:
                wait_seconds = _retry_delay(exc, 60.0, detail=detail) + 1.0
                rate_limit_attempt += 1
                if warning_callback:
                    warning_callback(
                        f"Gemini free-tier request limit reached; waiting {wait_seconds:.0f}s and "
                        "then resuming the same transcript section "
                        f"(retry {rate_limit_attempt}/{GEMINI_MAX_RATE_LIMIT_RETRIES})."
                    )
                _sleep_with_cancel(wait_seconds, cancel_callback)
                continue
            if exc.code == 429:
                raise _GeminiQuotaExhausted(detail) from exc
            if exc.code in {500, 502, 503, 504} and temporary_attempt < len(retry_delays):
                wait_seconds = _retry_delay(exc, retry_delays[temporary_attempt], detail=detail)
                temporary_attempt += 1
                _sleep_with_cancel(wait_seconds, cancel_callback)
                continue
            raise RuntimeError(f"Gemini transcript correction failed (HTTP {exc.code}): {detail}") from exc
        except (RemoteDisconnected, ConnectionResetError) as exc:
            if temporary_attempt < len(retry_delays):
                wait_seconds = retry_delays[temporary_attempt]
                temporary_attempt += 1
                if warning_callback:
                    warning_callback(
                        f"Gemini connection closed unexpectedly; retrying in {wait_seconds:.0f}s "
                        f"(retry {temporary_attempt}/{len(retry_delays)})."
                    )
                _sleep_with_cancel(wait_seconds, cancel_callback)
                continue
            raise RuntimeError("Gemini closed the connection repeatedly while checking the transcript.") from exc
        except error.URLError as exc:
            if isinstance(exc.reason, TimeoutError):
                raise _GeminiSectionTimeout(
                    "Gemini transcript correction timed out while checking one transcript section."
                ) from exc
            raise RuntimeError(f"Could not connect to Gemini for transcript correction: {exc.reason}") from exc
        except TimeoutError as exc:
            raise _GeminiSectionTimeout(
                "Gemini transcript correction timed out while checking one transcript section."
            ) from exc
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RuntimeError("Gemini returned an unreadable response.") from exc

    if response_data is None:
        raise RuntimeError("Gemini did not return a transcript correction.")

    if cancel_callback and cancel_callback():
        raise RuntimeError("Processing was cancelled.")

    response_text = _extract_response_text(response_data)
    try:
        corrected_data = json.loads(response_text)
    except json.JSONDecodeError as exc:
        raise RuntimeError("Gemini returned invalid transcript correction JSON.") from exc
    changes = corrected_data.get("changes") if isinstance(corrected_data, dict) else None
    if not isinstance(changes, list):
        raise RuntimeError("Gemini returned an invalid transcript correction list.")

    corrected_words = list(words)
    changed_indexes: set[int] = set()
    for change in changes:
        if not isinstance(change, dict):
            raise RuntimeError("Gemini returned an invalid transcript correction.")
        index = change.get("index")
        replacement = change.get("word")
        if (
            not isinstance(index, int)
            or isinstance(index, bool)
            or index < 0
            or index >= len(words)
            or index in changed_indexes
            or not isinstance(replacement, str)
            or not replacement.strip()
        ):
            raise RuntimeError("Gemini returned an invalid transcript correction.")
        corrected_words[index] = replacement.strip()
        changed_indexes.add(index)
    return corrected_words


def _extract_response_text(response_data: object) -> str:
    try:
        candidates = response_data["candidates"]  # type: ignore[index]
        parts = candidates[0]["content"]["parts"]
        text = "".join(str(part.get("text", "")) for part in parts)
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError("Gemini did not return a transcript correction.") from exc
    if not text.strip():
        raise RuntimeError("Gemini returned an empty transcript correction.")
    return text


def _read_api_error(exc: error.HTTPError) -> str:
    raw_body = exc.read().decode("utf-8", errors="replace")
    try:
        payload = json.loads(raw_body)
        message = payload.get("error", {}).get("message", "")
        if message:
            return str(message)
    except (json.JSONDecodeError, AttributeError):
        pass
    plain_detail = " ".join(raw_body.split())
    if plain_detail and not plain_detail.startswith("<"):
        return plain_detail[:500]
    return "API request was rejected"


def _retry_delay(exc: error.HTTPError, default: float, *, detail: str = "") -> float:
    try:
        retry_after = exc.headers.get("Retry-After")
        if retry_after is not None:
            return min(120.0, max(0.0, float(retry_after)))
    except (AttributeError, TypeError, ValueError):
        pass
    match = re.search(r"retry\s+in\s+([0-9]+(?:\.[0-9]+)?)s", detail, flags=re.IGNORECASE)
    if match:
        return min(120.0, max(0.0, float(match.group(1))))
    return min(120.0, max(0.0, default))


def _sleep_with_cancel(seconds: float, cancel_callback: Callable[[], bool] | None) -> None:
    deadline = time.monotonic() + max(0.0, seconds)
    while True:
        if cancel_callback and cancel_callback():
            raise RuntimeError("Processing was cancelled.")
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return
        time.sleep(min(0.25, remaining))
