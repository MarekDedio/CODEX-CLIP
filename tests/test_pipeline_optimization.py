from __future__ import annotations

from pathlib import Path
import unittest

from speech_cutter.captions import CaptionArtifactResult, CaptionWord, _remap_words_to_timeline
from speech_cutter.pipeline import (
    AudioBoostSettings,
    BackgroundMusicSettings,
    CropSettings,
    MotionSettings,
    ProcessingOptions,
    _build_filter_script,
    _replace_video_encoder_args,
    _video_encoder_args,
)
from speech_cutter.vad import SpeechSegment


class TimelineMappingTests(unittest.TestCase):
    def test_remaps_source_words_to_concatenated_timeline(self) -> None:
        words = [
            CaptionWord(0.2, 0.6, "outside", "outside", False),
            CaptionWord(1.5, 2.0, "first", "first", False),
            CaptionWord(5.5, 6.0, "second", "second", False),
        ]

        result = _remap_words_to_timeline(words, [(1.0, 3.0), (5.0, 7.0)])

        self.assertEqual([word.text for word in result], ["first", "second"])
        self.assertAlmostEqual(result[0].start, 0.5)
        self.assertAlmostEqual(result[0].end, 1.0)
        self.assertAlmostEqual(result[1].start, 2.5)
        self.assertAlmostEqual(result[1].end, 3.0)

    def test_clips_word_at_kept_segment_boundary(self) -> None:
        word = CaptionWord(0.8, 1.2, "edge", "edge", False)

        result = _remap_words_to_timeline([word], [(1.0, 2.0)])

        self.assertEqual(len(result), 1)
        self.assertAlmostEqual(result[0].start, 0.0)
        self.assertAlmostEqual(result[0].end, 0.2)


class SinglePassFilterTests(unittest.TestCase):
    def test_combines_cuts_captions_profanity_music_and_boost(self) -> None:
        artifacts = CaptionArtifactResult(
            subtitle_path=Path("captions.ass"),
            profanity_intervals=[(0.25, 0.55)],
            caption_event_count=1,
            censored_word_count=1,
            transcribed_word_count=2,
        )

        script = _build_filter_script(
            [SpeechSegment(1.0, 3.0)],
            video_width=1920,
            video_height=1080,
            crop_settings=CropSettings(enabled=True, x=100, y=50, width=1720, height=968),
            motion_settings=MotionSettings(push_in_enabled=True),
            background_music_settings=BackgroundMusicSettings(enabled=True, path=Path("music.mp3")),
            audio_boost_settings=AudioBoostSettings(enabled=True, gain_db=6.0),
            kept_duration=2.0,
            caption_artifacts=artifacts,
        )

        self.assertIn("concat=n=1:v=1:a=1[video_concat][speech_concat]", script)
        self.assertIn("[video_concat]ass=captions.ass[outv]", script)
        self.assertIn("[speech_concat][bga]amix=", script)
        self.assertIn("[audio_mix]volume=6.000dB", script)
        self.assertIn("[audio_processed]asetpts=PTS-STARTPTS,volume=enable=", script)
        self.assertEqual(script.count("concat=n="), 1)


class VideoEncoderTests(unittest.TestCase):
    def test_nvenc_uses_high_quality_gpu_settings(self) -> None:
        args = _video_encoder_args(ProcessingOptions(video_crf=18), use_nvenc=True)

        self.assertEqual(args[args.index("-c:v") + 1], "h264_nvenc")
        self.assertEqual(args[args.index("-preset") + 1], "p6")
        self.assertEqual(args[args.index("-cq") + 1], "16")
        self.assertEqual(args[args.index("-multipass") + 1], "fullres")
        self.assertEqual(args[args.index("-spatial-aq") + 1], "1")
        self.assertEqual(args[args.index("-temporal-aq") + 1], "1")

    def test_cpu_fallback_preserves_original_quality_settings(self) -> None:
        options = ProcessingOptions(video_crf=19, video_preset="fast")
        args = _video_encoder_args(options, use_nvenc=False)

        self.assertEqual(args[:2], ["-c:v", "libx264"])
        self.assertEqual(args[args.index("-preset") + 1], "fast")
        self.assertEqual(args[args.index("-crf") + 1], "19")

    def test_render_command_can_swap_nvenc_for_cpu_without_touching_audio(self) -> None:
        command = [
            "ffmpeg",
            "-map",
            "[outv]",
            "-c:v",
            "h264_nvenc",
            "-preset",
            "p6",
            "-c:a",
            "aac",
            "output.mp4",
        ]

        replaced = _replace_video_encoder_args(
            command,
            ["-c:v", "libx264", "-preset", "veryfast", "-crf", "18"],
        )

        self.assertIn("libx264", replaced)
        self.assertNotIn("h264_nvenc", replaced)
        self.assertEqual(replaced[-3:], ["-c:a", "aac", "output.mp4"])


if __name__ == "__main__":
    unittest.main()
