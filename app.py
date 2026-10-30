from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from speech_cutter.captions import CaptionSettings
from speech_cutter.faces import DEFAULT_FACE_MAX_ZOOM, DEFAULT_FACE_PADDING
from speech_cutter.gui import launch_gui
from speech_cutter.pipeline import (
    CropSettings,
    DEFAULT_PUSH_IN_MAX_ZOOM,
    DEFAULT_PUSH_IN_MIN_SEGMENT_SECONDS,
    FaceZoomSettings,
    MotionSettings,
    build_output_path,
    check_runtime,
    process_video,
)
from speech_cutter.presets import PRESETS, build_options


def _make_cli_progress_callback():
    last_percent = -1
    last_status = ""

    def callback(value: float, status: str) -> None:
        nonlocal last_percent, last_status
        percent = int(value * 100)
        if percent != last_percent or status != last_status:
            print(f"{value * 100:5.1f}%  {status}", flush=True)
            last_percent = percent
            last_status = status

    return callback


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Keep only spoken parts from a video and export the result as MP4."
    )
    parser.add_argument("input", nargs="?", help="Source video file.")
    parser.add_argument(
        "output",
        nargs="?",
        help="Optional legacy argument. Output is now always saved next to the source video.",
    )
    parser.add_argument(
        "--preset",
        choices=list(PRESETS.keys()),
        default="natural",
        help="Speech trimming preset to use in CLI mode.",
    )
    parser.add_argument("--padding-seconds", type=float, help="Extra padding to keep around each speech segment.")
    parser.add_argument("--padding-ms", type=int, help="Legacy option for padding in milliseconds.")
    parser.add_argument("--merge-gap-ms", type=int, help="Merge nearby speech chunks when the pause is shorter than this.")
    parser.add_argument(
        "--captions",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Burn in TikTok-style subtitles on the output video.",
    )
    parser.add_argument(
        "--profanity-filter",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Automatically mute profane words using the subtitle timing pass.",
    )
    parser.add_argument(
        "--caption-model",
        default=CaptionSettings().model_name,
        help="Whisper model name to use for subtitles and profanity timing.",
    )
    parser.add_argument(
        "--transcription-provider",
        choices=("local", "groq"),
        default="local",
        help="Run Whisper locally or use the Groq speech-to-text API.",
    )
    parser.add_argument(
        "--groq-api-key",
        default=os.environ.get("GROQ_API_KEY", ""),
        help="Groq API key. Prefer setting GROQ_API_KEY instead of putting it in shell history.",
    )
    parser.add_argument(
        "--groq-model",
        choices=("whisper-large-v3-turbo", "whisper-large-v3"),
        default=CaptionSettings().groq_model,
        help="Whisper model used by Groq transcription.",
    )
    parser.add_argument(
        "--enhanced-polish-recognition",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use a larger model and Polish-specific decoding for more accurate Polish subtitles.",
    )
    parser.add_argument(
        "--ai-transcript-correction",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Check and correct transcript context with Gemini before creating subtitles.",
    )
    parser.add_argument(
        "--gemini-api-key",
        default=os.environ.get("GEMINI_API_KEY", ""),
        help="Gemini API key. Prefer setting GEMINI_API_KEY instead of putting it in shell history.",
    )
    parser.add_argument(
        "--gemini-model",
        default=CaptionSettings().gemini_model,
        help="Gemini model used for transcript correction.",
    )
    parser.add_argument(
        "--push-in",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Add a slow center push-in to longer speech segments.",
    )
    parser.add_argument(
        "--push-in-min-seconds",
        type=float,
        default=DEFAULT_PUSH_IN_MIN_SEGMENT_SECONDS,
        help="Minimum speech segment length that receives the slow push-in.",
    )
    parser.add_argument(
        "--push-in-zoom",
        type=float,
        default=DEFAULT_PUSH_IN_MAX_ZOOM,
        help="Final push-in zoom amount, such as 1.08 for an 8 percent zoom.",
    )
    parser.add_argument(
        "--face-zoom",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Detect faces and zoom speech segments toward a face instead of the manual crop.",
    )
    parser.add_argument(
        "--face-id",
        type=int,
        action="append",
        default=[],
        metavar="N",
        help="Face id to follow (repeatable). Detected ids are printed during the run. Omit for automatic selection.",
    )
    parser.add_argument(
        "--face-padding",
        type=float,
        default=DEFAULT_FACE_PADDING,
        help="Extra space around a detected face, as a fraction of face size (default 0.35).",
    )
    parser.add_argument(
        "--face-max-zoom",
        type=float,
        default=DEFAULT_FACE_MAX_ZOOM,
        help="Maximum face zoom, such as 1.8 for 80 percent.",
    )
    args = parser.parse_args(argv)

    if args.face_id and not args.face_zoom:
        parser.error("--face-id requires --face-zoom")
    if args.ai_transcript_correction and not args.captions:
        parser.error("--ai-transcript-correction requires --captions")
    if args.ai_transcript_correction and not str(args.gemini_api_key).strip():
        parser.error("--ai-transcript-correction requires GEMINI_API_KEY or --gemini-api-key")
    if (
        args.transcription_provider == "groq"
        and (args.captions or args.profanity_filter)
        and not str(args.groq_api_key).strip()
    ):
        parser.error("Groq transcription requires GROQ_API_KEY or --groq-api-key")

    issues = check_runtime()
    if issues:
        for issue in issues:
            print(f"Error: {issue}", file=sys.stderr)
        return 1

    if args.input:
        output_path = build_output_path(Path(args.input))
        if args.output:
            print(
                f"Output path is automatic in this version. Saving to {output_path}",
                flush=True,
            )
        options = build_options(
            args.preset,
            padding_ms=(
                int(round(args.padding_seconds * 1000))
                if args.padding_seconds is not None
                else args.padding_ms
            ),
            merge_gap_ms=args.merge_gap_ms,
            motion=MotionSettings(
                push_in_enabled=bool(args.push_in),
                push_in_min_segment_seconds=max(0.0, float(args.push_in_min_seconds)),
                push_in_max_zoom=max(1.001, float(args.push_in_zoom)),
            ),
            crop=(
                CropSettings(
                    enabled=True,
                    every_n_segments=1,
                    min_segment_seconds=0.0,
                    zoom_duration_seconds=0.0,
                    mid_zoom_on_long_segments=False,
                )
                if args.face_zoom
                else None
            ),
            faces=FaceZoomSettings(
                enabled=bool(args.face_zoom),
                selected_face_ids=tuple(args.face_id),
                padding=max(0.0, float(args.face_padding)),
                max_zoom=max(1.01, float(args.face_max_zoom)),
            ),
            captions=CaptionSettings(
                enabled=bool(args.captions),
                profanity_filter=bool(args.profanity_filter),
                model_name=str(args.caption_model),
                enhanced_polish_recognition=bool(args.enhanced_polish_recognition),
                transcription_provider=str(args.transcription_provider),
                groq_api_key=str(args.groq_api_key),
                groq_model=str(args.groq_model),
                ai_correction_enabled=bool(args.ai_transcript_correction),
                gemini_api_key=str(args.gemini_api_key),
                gemini_model=str(args.gemini_model),
            ),
        )
        result = process_video(
            Path(args.input),
            output_path,
            options=options,
            progress_callback=_make_cli_progress_callback(),
            log_callback=lambda message: print(message, flush=True),
        )
        print(
            f"Saved {result.output_path} with {len(result.segments)} speech segment(s). "
            f"Kept {result.kept_duration:.1f}s / {result.input_duration:.1f}s."
        )
        return 0

    launch_gui()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
