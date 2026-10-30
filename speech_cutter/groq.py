from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Callable
from urllib import error, request
import uuid


DEFAULT_GROQ_WHISPER_MODEL = "whisper-large-v3-turbo"
GROQ_WHISPER_MODELS = ("whisper-large-v3-turbo", "whisper-large-v3")
GROQ_TRANSCRIPTION_URL = "https://api.groq.com/openai/v1/audio/transcriptions"
DEFAULT_GROQ_CORRECTION_MODEL = "openai/gpt-oss-20b"
GROQ_CHAT_COMPLETIONS_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_FREE_TIER_MAX_FILE_BYTES = 25_000_000


@dataclass(frozen=True)
class GroqTranscriptionWord:
    start: float
    end: float
    text: str


def correct_transcript_chunk(
    words: list[str],
    *,
    context_before: list[str],
    context_after: list[str],
    api_key: str,
    model: str = DEFAULT_GROQ_CORRECTION_MODEL,
    cancel_callback: Callable[[], bool] | None = None,
) -> list[str]:
    """Correct a timestamp-safe transcript chunk through Groq's text LLM API."""
    if not api_key.strip():
        raise RuntimeError("Groq transcript fallback is enabled, but no API key was provided.")
    if cancel_callback and cancel_callback():
        raise RuntimeError("Processing was cancelled.")

    prompt = (
        "You are a meticulous Polish transcript editor. Correct only clear speech-recognition, grammar, "
        "punctuation, and logical-context errors that can be inferred confidently. Do not invent facts or "
        "rewrite style. TARGET is a JSON array of word tokens. Return a JSON object with only a 'changes' "
        "array. Each change has a zero-based TARGET 'index' and replacement 'word'. Never add, remove, "
        "split, merge, or reorder tokens. Do not return unchanged tokens or read-only context changes. "
        "Preserve punctuation attached to tokens. If uncertain, return no change.\n\n"
        "READ-ONLY CONTEXT BEFORE:\n"
        + json.dumps(context_before, ensure_ascii=False)
        + "\n\nTARGET TOKENS TO CORRECT:\n"
        + json.dumps(words, ensure_ascii=False)
        + "\n\nREAD-ONLY CONTEXT AFTER:\n"
        + json.dumps(context_after, ensure_ascii=False)
    )
    response_schema = {
        "type": "object",
        "properties": {
            "changes": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "index": {"type": "integer", "minimum": 0, "maximum": len(words) - 1},
                        "word": {"type": "string"},
                    },
                    "required": ["index", "word"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["changes"],
        "additionalProperties": False,
    }
    payload = {
        "model": model.strip() or DEFAULT_GROQ_CORRECTION_MODEL,
        "messages": [
            {"role": "user", "content": prompt},
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "transcript_changes",
                "strict": True,
                "schema": response_schema,
            },
        },
        "reasoning_effort": "low",
        "reasoning_format": "hidden",
        "max_completion_tokens": 2048,
    }
    http_request = request.Request(
        GROQ_CHAT_COMPLETIONS_URL,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key.strip()}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "SpeechCutter/1.0",
        },
        method="POST",
    )
    try:
        with request.urlopen(http_request, timeout=75) as response:
            response_data = json.loads(response.read().decode("utf-8"))
    except error.HTTPError as exc:
        raise RuntimeError(f"Groq transcript fallback failed (HTTP {exc.code}): {_read_api_error(exc)}") from exc
    except error.URLError as exc:
        raise RuntimeError(f"Could not connect to Groq transcript fallback: {exc.reason}") from exc
    except TimeoutError as exc:
        raise RuntimeError("Groq transcript fallback timed out.") from exc
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("Groq returned an unreadable transcript correction response.") from exc

    try:
        content = response_data["choices"][0]["message"]["content"]
        corrected_data = json.loads(str(content))
        changes = corrected_data["changes"]
    except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
        raise RuntimeError("Groq returned invalid transcript correction JSON.") from exc
    if not isinstance(changes, list):
        raise RuntimeError("Groq returned an invalid transcript correction list.")

    corrected_words = list(words)
    changed_indexes: set[int] = set()
    for change in changes:
        if not isinstance(change, dict):
            raise RuntimeError("Groq returned an invalid transcript correction.")
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
            raise RuntimeError("Groq returned an invalid transcript correction.")
        corrected_words[index] = replacement.strip()
        changed_indexes.add(index)
    return corrected_words


def transcribe_audio_words(
    audio_path: Path,
    *,
    api_key: str,
    model: str = DEFAULT_GROQ_WHISPER_MODEL,
    language: str | None = None,
    cancel_callback: Callable[[], bool] | None = None,
) -> list[GroqTranscriptionWord]:
    if not api_key.strip():
        raise RuntimeError("Groq transcription is enabled, but no API key was provided.")
    if cancel_callback and cancel_callback():
        raise RuntimeError("Processing was cancelled.")

    try:
        file_size = audio_path.stat().st_size
    except OSError as exc:
        raise RuntimeError("Could not read the prepared audio file for Groq transcription.") from exc
    if file_size > GROQ_FREE_TIER_MAX_FILE_BYTES:
        size_mb = file_size / (1024 * 1024)
        raise RuntimeError(
            f"The prepared audio is {size_mb:.1f} MB, exceeding Groq's 25 MB free-tier upload limit. "
            "Use local Whisper or process a shorter video."
        )

    boundary = f"speech-cutter-{uuid.uuid4().hex}"
    fields = {
        "model": model.strip() or DEFAULT_GROQ_WHISPER_MODEL,
        "response_format": "verbose_json",
        "timestamp_granularities[]": "word",
        "temperature": "0",
    }
    if language:
        fields["language"] = language
    body = _encode_multipart(boundary, fields=fields, file_path=audio_path)
    http_request = request.Request(
        GROQ_TRANSCRIPTION_URL,
        data=body,
        headers={
            "Authorization": f"Bearer {api_key.strip()}",
            "Content-Type": f"multipart/form-data; boundary={boundary}",
            "Accept": "application/json",
            "User-Agent": "SpeechCutter/1.0",
        },
        method="POST",
    )

    try:
        with request.urlopen(http_request, timeout=180) as response:
            response_data = json.loads(response.read().decode("utf-8"))
    except error.HTTPError as exc:
        detail = _read_api_error(exc)
        raise RuntimeError(f"Groq transcription failed (HTTP {exc.code}): {detail}") from exc
    except error.URLError as exc:
        raise RuntimeError(f"Could not connect to Groq for transcription: {exc.reason}") from exc
    except TimeoutError as exc:
        raise RuntimeError("Groq transcription timed out.") from exc
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("Groq returned an unreadable transcription response.") from exc

    if cancel_callback and cancel_callback():
        raise RuntimeError("Processing was cancelled.")

    raw_words = response_data.get("words") if isinstance(response_data, dict) else None
    if not isinstance(raw_words, list):
        raise RuntimeError("Groq did not return word-level timestamps.")

    words: list[GroqTranscriptionWord] = []
    for raw_word in raw_words:
        if not isinstance(raw_word, dict):
            continue
        text = str(raw_word.get("word", "")).strip()
        try:
            start = float(raw_word["start"])
            end = float(raw_word["end"])
        except (KeyError, TypeError, ValueError):
            continue
        if text and end > start:
            words.append(GroqTranscriptionWord(start=max(0.0, start), end=end, text=text))
    return words


def _encode_multipart(boundary: str, *, fields: dict[str, str], file_path: Path) -> bytes:
    delimiter = f"--{boundary}\r\n".encode("ascii")
    chunks: list[bytes] = []
    for name, value in fields.items():
        chunks.extend(
            (
                delimiter,
                f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode("utf-8"),
                value.encode("utf-8"),
                b"\r\n",
            )
        )
    chunks.extend(
        (
            delimiter,
            (
                f'Content-Disposition: form-data; name="file"; filename="{file_path.name}"\r\n'
                "Content-Type: audio/flac\r\n\r\n"
            ).encode("utf-8"),
            file_path.read_bytes(),
            b"\r\n",
            f"--{boundary}--\r\n".encode("ascii"),
        )
    )
    return b"".join(chunks)


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
