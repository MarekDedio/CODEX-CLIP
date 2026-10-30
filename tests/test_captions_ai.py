from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest.mock import ANY, patch

from speech_cutter.captions import (
    CaptionSettings,
    CaptionWord,
    _build_groq_correction_fallback,
    _prepare_transcription_audio,
    create_caption_artifacts,
)


class CaptionAiIntegrationTests(unittest.TestCase):
    @patch("speech_cutter.captions.subprocess.run")
    def test_groq_provider_prepares_compressed_flac(self, run_mock) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            result = _prepare_transcription_audio(
                Path("video.mp4"),
                output_dir=Path(temp_dir),
                total_duration=10.0,
                settings=CaptionSettings(transcription_provider="groq"),
                progress_callback=None,
                log_callback=None,
            )

        command = run_mock.call_args.args[0]
        self.assertEqual(result.suffix, ".flac")
        self.assertEqual(command[command.index("-c:a") + 1], "flac")

    @patch("speech_cutter.captions.correct_transcript_words")
    @patch("speech_cutter.captions._transcribe_words")
    @patch("speech_cutter.captions._prepare_transcription_audio")
    def test_ai_correction_happens_before_subtitle_file_is_written(
        self,
        prepare_audio_mock,
        transcribe_mock,
        correct_mock,
    ) -> None:
        prepare_audio_mock.return_value = Path("audio.wav")
        transcribe_mock.return_value = [
            CaptionWord(0.0, 0.4, "To", "To", False),
            CaptionWord(0.4, 0.8, "mo", "mo", False),
            CaptionWord(0.8, 1.2, "sens.", "sens.", False),
        ]
        correct_mock.return_value = ["To", "ma", "sens."]

        with tempfile.TemporaryDirectory() as temp_dir:
            result = create_caption_artifacts(
                Path("video.mp4"),
                Path(temp_dir),
                video_width=1080,
                video_height=1920,
                total_duration=1.2,
                settings=CaptionSettings(
                    enabled=True,
                    ai_correction_enabled=True,
                    gemini_api_key="secret-key",
                ),
            )
            subtitle_text = result.subtitle_path.read_text(encoding="utf-8")  # type: ignore[union-attr]

        self.assertIn("MA", subtitle_text)
        self.assertNotIn(">MO<", subtitle_text)
        correct_mock.assert_called_once_with(
            ["To", "mo", "sens."],
            api_key="secret-key",
            model=CaptionSettings().gemini_model,
            cancel_callback=None,
            progress_callback=ANY,
            warning_callback=None,
            quota_fallback=None,
        )

    @patch("speech_cutter.captions.correct_transcript_words")
    @patch("speech_cutter.captions._transcribe_words")
    @patch("speech_cutter.captions._prepare_transcription_audio")
    def test_ai_failure_keeps_original_transcript_and_allows_export(
        self,
        prepare_audio_mock,
        transcribe_mock,
        correct_mock,
    ) -> None:
        prepare_audio_mock.return_value = Path("audio.wav")
        transcribe_mock.return_value = [CaptionWord(0.0, 0.5, "Oryginał", "Oryginał", False)]
        correct_mock.side_effect = RuntimeError("HTTP 400: invalid argument")
        log_messages: list[str] = []

        with tempfile.TemporaryDirectory() as temp_dir:
            result = create_caption_artifacts(
                Path("video.mp4"),
                Path(temp_dir),
                video_width=1080,
                video_height=1920,
                total_duration=0.5,
                settings=CaptionSettings(
                    enabled=True,
                    ai_correction_enabled=True,
                    gemini_api_key="secret-key",
                ),
                log_callback=log_messages.append,
            )
            subtitle_text = result.subtitle_path.read_text(encoding="utf-8")  # type: ignore[union-attr]

        self.assertIn("ORYGINAŁ", subtitle_text)
        self.assertTrue(any("Continuing with the original transcript" in line for line in log_messages))

    @patch("speech_cutter.captions.correct_transcript_chunk_with_groq")
    def test_groq_fallback_rotates_across_configured_keys(self, groq_mock) -> None:
        groq_mock.side_effect = [RuntimeError("quota"), ["To", "ma", "sens."], ["Jest", "dobrze."]]
        log_messages: list[str] = []
        fallback = _build_groq_correction_fallback(
            ("key-1", "key-2", "key-3"),
            model="openai/gpt-oss-20b",
            cancel_callback=None,
            log_callback=log_messages.append,
        )

        first = fallback(["To", "mo", "sens."], [], [])  # type: ignore[misc]
        second = fallback(["Jest", "dobrze."], [], [])  # type: ignore[misc]

        self.assertEqual(first, ["To", "ma", "sens."])
        self.assertEqual(second, ["Jest", "dobrze."])
        keys = [call.kwargs["api_key"] for call in groq_mock.call_args_list]
        self.assertEqual(keys, ["key-1", "key-2", "key-3"])
        self.assertTrue(any("key 1/3 failed" in line for line in log_messages))


if __name__ == "__main__":
    unittest.main()
