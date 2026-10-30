from __future__ import annotations

import json
import io
import unittest
from http.client import RemoteDisconnected
from unittest.mock import patch
from urllib import error

from speech_cutter.gemini import (
    GEMINI_CHUNK_SIZE,
    GEMINI_MIN_CHUNK_SIZE,
    GEMINI_REQUEST_TIMEOUT_SECONDS,
    correct_transcript_words,
)


class _FakeResponse:
    def __init__(self, changes: list[dict[str, object]]) -> None:
        result = {"changes": changes}
        self._payload = json.dumps(
            {"candidates": [{"content": {"parts": [{"text": json.dumps(result)}]}}]}
        ).encode("utf-8")

    def __enter__(self) -> _FakeResponse:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def read(self) -> bytes:
        return self._payload


class GeminiCorrectionTests(unittest.TestCase):
    @patch("speech_cutter.gemini.request.urlopen")
    def test_returns_corrected_words_and_sends_key_as_header(self, urlopen_mock) -> None:
        urlopen_mock.return_value = _FakeResponse([{"index": 1, "word": "ma"}])

        result = correct_transcript_words(
            ["To", "mo", "sens."],
            api_key="secret-key",
        )

        self.assertEqual(result, ["To", "ma", "sens."])
        sent_request = urlopen_mock.call_args.args[0]
        self.assertEqual(urlopen_mock.call_args.kwargs["timeout"], GEMINI_REQUEST_TIMEOUT_SECONDS)
        self.assertEqual(GEMINI_REQUEST_TIMEOUT_SECONDS, 75)
        self.assertEqual(sent_request.get_header("X-goog-api-key"), "secret-key")
        sent_payload = json.loads(sent_request.data.decode("utf-8"))
        self.assertNotIn("temperature", sent_payload["generationConfig"])
        self.assertEqual(
            sent_payload["generationConfig"]["thinkingConfig"]["thinkingLevel"],
            "low",
        )
        schema = sent_payload["generationConfig"]["responseJsonSchema"]
        change_schema = schema["properties"]["changes"]["items"]
        self.assertEqual(change_schema["properties"]["index"]["maximum"], 2)
        self.assertNotIn("maxItems", schema["properties"]["changes"])
        self.assertNotIn("secret-key", sent_request.full_url)

    @patch("speech_cutter.gemini.request.urlopen")
    def test_rejects_out_of_range_change(self, urlopen_mock) -> None:
        urlopen_mock.return_value = _FakeResponse([{"index": 3, "word": "mało"}])

        with self.assertRaisesRegex(RuntimeError, "invalid transcript correction"):
            correct_transcript_words(["Jest", "za", "mało"], api_key="secret-key")

    @patch("speech_cutter.gemini.request.urlopen")
    def test_empty_changes_preserve_original_words(self, urlopen_mock) -> None:
        urlopen_mock.return_value = _FakeResponse([])

        words = ["Wszystko", "dobrze."]

        self.assertEqual(correct_transcript_words(words, api_key="secret-key"), words)

    @patch("speech_cutter.gemini.request.urlopen")
    def test_long_transcript_is_checked_in_bounded_contextual_chunks(self, urlopen_mock) -> None:
        words = [f"word-{index}" for index in range(GEMINI_CHUNK_SIZE * 2 + 5)]
        urlopen_mock.side_effect = [_FakeResponse([]), _FakeResponse([]), _FakeResponse([])]
        progress: list[tuple[int, int]] = []

        result = correct_transcript_words(
            words,
            api_key="secret-key",
            progress_callback=lambda done, total: progress.append((done, total)),
        )

        self.assertEqual(result, words)
        self.assertEqual(urlopen_mock.call_count, 3)
        self.assertEqual(progress, [(1, 3), (2, 3), (3, 3)])
        prompts = []
        for call in urlopen_mock.call_args_list:
            sent_payload = json.loads(call.args[0].data.decode("utf-8"))
            prompts.append(sent_payload["contents"][0]["parts"][0]["text"])
        last_first = GEMINI_CHUNK_SIZE - 1
        first_second = GEMINI_CHUNK_SIZE
        self.assertIn('"word-0"', prompts[0])
        self.assertIn(f'"word-{last_first}"', prompts[0])
        self.assertNotIn(
            f'"word-{first_second}"',
            prompts[0].partition("READ-ONLY CONTEXT AFTER:")[0],
        )
        self.assertIn(f'"word-{last_first}"', prompts[1])
        self.assertIn(f'"word-{first_second}"', prompts[1])

    @patch("speech_cutter.gemini.request.urlopen")
    def test_timed_out_section_is_split_and_retried(self, urlopen_mock) -> None:
        words = [f"word-{index}" for index in range(GEMINI_MIN_CHUNK_SIZE * 2)]
        urlopen_mock.side_effect = [TimeoutError(), _FakeResponse([]), _FakeResponse([])]

        result = correct_transcript_words(words, api_key="secret-key")

        self.assertEqual(result, words)
        self.assertEqual(urlopen_mock.call_count, 3)

    @patch("speech_cutter.gemini.request.urlopen")
    def test_single_failed_minimum_fragment_does_not_discard_other_corrections(self, urlopen_mock) -> None:
        words = [f"word-{index}" for index in range(GEMINI_MIN_CHUNK_SIZE * 2)]
        urlopen_mock.side_effect = [
            TimeoutError(),
            _FakeResponse([{"index": 1, "word": "corrected"}]),
            TimeoutError(),
        ]
        warnings: list[str] = []

        result = correct_transcript_words(
            words,
            api_key="secret-key",
            warning_callback=warnings.append,
        )

        self.assertEqual(result[1], "corrected")
        self.assertEqual(result[GEMINI_MIN_CHUNK_SIZE:], words[GEMINI_MIN_CHUNK_SIZE:])
        self.assertEqual(len(warnings), 1)
        self.assertIn("only that fragment was left unchanged", warnings[0])

    def test_requires_api_key(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "no API key"):
            correct_transcript_words(["Tekst"], api_key="")

    def test_empty_transcript_needs_no_request_or_key(self) -> None:
        self.assertEqual(correct_transcript_words([], api_key=""), [])

    @patch("speech_cutter.gemini._sleep_with_cancel")
    @patch("speech_cutter.gemini.request.urlopen")
    def test_retries_temporary_service_error(self, urlopen_mock, sleep_mock) -> None:
        error_body = io.BytesIO(
            json.dumps({"error": {"message": "Temporary overload"}}).encode("utf-8")
        )
        temporary_error = error.HTTPError(
            "https://example.invalid",
            503,
            "Service Unavailable",
            {"Retry-After": "0"},
            error_body,
        )
        urlopen_mock.side_effect = [temporary_error, _FakeResponse([])]

        result = correct_transcript_words(["Gotowe"], api_key="secret-key")

        self.assertEqual(result, ["Gotowe"])
        self.assertEqual(urlopen_mock.call_count, 2)
        sleep_mock.assert_called_once_with(0.0, None)

    @patch("speech_cutter.gemini._sleep_with_cancel")
    @patch("speech_cutter.gemini.request.urlopen")
    def test_retries_unexpected_connection_close(self, urlopen_mock, sleep_mock) -> None:
        urlopen_mock.side_effect = [RemoteDisconnected("peer closed"), _FakeResponse([])]
        warnings: list[str] = []

        result = correct_transcript_words(
            ["Gotowe"],
            api_key="secret-key",
            warning_callback=warnings.append,
        )

        self.assertEqual(result, ["Gotowe"])
        self.assertEqual(urlopen_mock.call_count, 2)
        sleep_mock.assert_called_once_with(2.0, None)
        self.assertTrue(any("connection closed unexpectedly" in line for line in warnings))

    @patch("speech_cutter.gemini._sleep_with_cancel")
    @patch("speech_cutter.gemini.request.urlopen")
    def test_rate_limit_waits_for_api_retry_time_and_resumes_same_section(
        self,
        urlopen_mock,
        sleep_mock,
    ) -> None:
        error_body = io.BytesIO(
            json.dumps(
                {
                    "error": {
                        "message": (
                            "You exceeded your current quota. Please retry in 35.477758742s."
                        )
                    }
                }
            ).encode("utf-8")
        )
        rate_error = error.HTTPError(
            "https://example.invalid",
            429,
            "Too Many Requests",
            {},
            error_body,
        )
        urlopen_mock.side_effect = [rate_error, _FakeResponse([{"index": 0, "word": "Poprawione"}])]
        warnings: list[str] = []

        result = correct_transcript_words(
            ["Oryginał"],
            api_key="secret-key",
            warning_callback=warnings.append,
        )

        self.assertEqual(result, ["Poprawione"])
        self.assertEqual(urlopen_mock.call_count, 2)
        sleep_mock.assert_called_once_with(36.477758742, None)
        self.assertTrue(any("resuming the same transcript section" in line for line in warnings))

    @patch("speech_cutter.gemini.request.urlopen")
    def test_rate_limit_uses_configured_groq_fallback_before_waiting(self, urlopen_mock) -> None:
        rate_error = error.HTTPError(
            "https://example.invalid",
            429,
            "Too Many Requests",
            {},
            io.BytesIO(json.dumps({"error": {"message": "Quota exceeded."}}).encode("utf-8")),
        )
        urlopen_mock.side_effect = rate_error
        fallback_calls: list[tuple[list[str], list[str], list[str]]] = []

        result = correct_transcript_words(
            ["To", "mo", "sens."],
            api_key="gemini-secret",
            quota_fallback=lambda words, before, after: (
                fallback_calls.append((words, before, after)) or ["To", "ma", "sens."]
            ),
        )

        self.assertEqual(result, ["To", "ma", "sens."])
        self.assertEqual(fallback_calls, [(["To", "mo", "sens."], [], [])])
        self.assertEqual(urlopen_mock.call_count, 1)

    @patch("speech_cutter.gemini._sleep_with_cancel")
    @patch("speech_cutter.gemini.request.urlopen")
    def test_exhausted_quota_preserves_already_completed_corrections(
        self,
        urlopen_mock,
        sleep_mock,
    ) -> None:
        def rate_error() -> error.HTTPError:
            return error.HTTPError(
                "https://example.invalid",
                429,
                "Too Many Requests",
                {},
                io.BytesIO(
                    json.dumps(
                        {"error": {"message": "Quota exceeded. Please retry in 35s."}}
                    ).encode("utf-8")
                ),
            )

        urlopen_mock.side_effect = [_FakeResponse([{"index": 0, "word": "Poprawione"}])] + [
            rate_error() for _ in range(6)
        ]
        words = ["Pierwsze"] + [f"word-{index}" for index in range(GEMINI_CHUNK_SIZE)]
        warnings: list[str] = []

        result = correct_transcript_words(
            words,
            api_key="secret-key",
            warning_callback=warnings.append,
        )

        self.assertEqual(result[0], "Poprawione")
        self.assertEqual(result[1:], words[1:])
        self.assertEqual(sleep_mock.call_count, 5)
        self.assertTrue(any("Keeping corrections from 1 completed section" in line for line in warnings))


if __name__ == "__main__":
    unittest.main()
