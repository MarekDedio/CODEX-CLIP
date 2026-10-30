from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from speech_cutter.groq import (
    DEFAULT_GROQ_CORRECTION_MODEL,
    GROQ_FREE_TIER_MAX_FILE_BYTES,
    correct_transcript_chunk,
    transcribe_audio_words,
)


class _FakeResponse:
    def __init__(self, payload: dict[str, object]) -> None:
        self._payload = json.dumps(payload).encode("utf-8")

    def __enter__(self) -> _FakeResponse:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def read(self) -> bytes:
        return self._payload


class GroqTranscriptionTests(unittest.TestCase):
    @patch("speech_cutter.groq.request.urlopen")
    def test_uploads_audio_and_returns_word_timestamps(self, urlopen_mock) -> None:
        urlopen_mock.return_value = _FakeResponse(
            {
                "words": [
                    {"word": "Hello", "start": 0.0, "end": 0.4},
                    {"word": "world", "start": 0.4, "end": 0.9},
                ]
            }
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            audio_path = Path(temp_dir) / "audio.flac"
            audio_path.write_bytes(b"fake-flac")
            result = transcribe_audio_words(
                audio_path,
                api_key="groq-secret",
                model="whisper-large-v3-turbo",
                language="en",
            )

        self.assertEqual([word.text for word in result], ["Hello", "world"])
        self.assertEqual(result[1].start, 0.4)
        sent_request = urlopen_mock.call_args.args[0]
        self.assertEqual(sent_request.get_header("Authorization"), "Bearer groq-secret")
        self.assertEqual(sent_request.get_header("User-agent"), "SpeechCutter/1.0")
        self.assertNotIn("groq-secret", sent_request.full_url)
        self.assertIn(b'\r\n\r\nwhisper-large-v3-turbo\r\n', sent_request.data)
        self.assertIn(b'\r\n\r\nword\r\n', sent_request.data)
        self.assertIn(b'\r\n\r\nen\r\n', sent_request.data)

    def test_requires_api_key(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "no API key"):
            transcribe_audio_words(Path("audio.flac"), api_key="")

    def test_rejects_file_above_free_tier_limit_before_upload(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            audio_path = Path(temp_dir) / "large.flac"
            with audio_path.open("wb") as handle:
                handle.truncate(GROQ_FREE_TIER_MAX_FILE_BYTES + 1)
            with self.assertRaisesRegex(RuntimeError, "25 MB"):
                transcribe_audio_words(audio_path, api_key="groq-secret")


class GroqCorrectionTests(unittest.TestCase):
    @patch("speech_cutter.groq.request.urlopen")
    def test_corrects_words_with_separate_text_llm_key(self, urlopen_mock) -> None:
        urlopen_mock.return_value = _FakeResponse(
            {
                "choices": [
                    {"message": {"content": '{"changes": [{"index": 1, "word": "ma"}]}'}}
                ]
            }
        )

        result = correct_transcript_chunk(
            ["To", "mo", "sens."],
            context_before=["Witam."],
            context_after=["Dziękuję."],
            api_key="separate-groq-key",
        )

        self.assertEqual(result, ["To", "ma", "sens."])
        sent_request = urlopen_mock.call_args.args[0]
        self.assertEqual(sent_request.get_header("Authorization"), "Bearer separate-groq-key")
        payload = json.loads(sent_request.data.decode("utf-8"))
        self.assertEqual(payload["model"], DEFAULT_GROQ_CORRECTION_MODEL)
        self.assertEqual(payload["response_format"]["type"], "json_schema")
        self.assertTrue(payload["response_format"]["json_schema"]["strict"])
        self.assertEqual(payload["reasoning_effort"], "low")
        self.assertEqual(payload["reasoning_format"], "hidden")
        self.assertNotIn("separate-groq-key", sent_request.full_url)


if __name__ == "__main__":
    unittest.main()
