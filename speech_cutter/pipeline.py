from __future__ import annotations

from dataclasses import dataclass, field
from bisect import bisect_left, bisect_right
from fractions import Fraction
from pathlib import Path
from typing import Callable
import json
import math
import os
import shutil
import subprocess
import tempfile
import time

from .captions import CaptionArtifactResult, CaptionSettings, create_caption_artifacts, needs_transcription
from .vad import (
    DetectionCancelledError,
    SileroOnnxVad,
    SpeechSegment,
    VadSettings,
    detect_speech_in_wav,
)


MODEL_PATH = Path(__file__).resolve().parent / "assets" / "silero_vad_16k_op15.onnx"
DEFAULT_BACKGROUND_MUSIC_PATH = Path(__file__).resolve().parent / "assets" / "monkeys_spinning_monkeys.mp3"
DEFAULT_MID_ZOOM_MIN_SEGMENT_SECONDS = 5.0
DEFAULT_PUSH_IN_MIN_SEGMENT_SECONDS = 4.0
DEFAULT_PUSH_IN_MAX_ZOOM = 1.08
_VAD_DETECTOR_CACHE: SileroOnnxVad | None = None
_NVENC_AVAILABLE_CACHE: bool | None = None


class ProcessingError(RuntimeError):
    """Base error for the video cutting pipeline."""


class NoSpeechDetectedError(ProcessingError):
    """Raised when no speech was detected in the input video."""


class UserCancelledError(ProcessingError):
    """Raised when the user cancels processing."""


@dataclass(frozen=True)
class ProcessingOptions:
    vad: VadSettings = field(default_factory=VadSettings)
    crop: "CropSettings" = field(default_factory=lambda: CropSettings())
    motion: "MotionSettings" = field(default_factory=lambda: MotionSettings())
    background_music: "BackgroundMusicSettings" = field(default_factory=lambda: BackgroundMusicSettings())
    audio_boost: "AudioBoostSettings" = field(default_factory=lambda: AudioBoostSettings())
    captions: CaptionSettings = field(default_factory=CaptionSettings)
    video_crf: int = 18
    video_preset: str = "veryfast"


@dataclass(frozen=True)
class CropSettings:
    enabled: bool = False
    x: int = 0
    y: int = 0
    width: int = 0
    height: int = 0
    every_n_segments: int = 2
    min_segment_seconds: float = 1.0
    zoom_duration_seconds: float = 0.0
    mid_zoom_on_long_segments: bool = True
    mid_zoom_min_segment_seconds: float = DEFAULT_MID_ZOOM_MIN_SEGMENT_SECONDS


@dataclass(frozen=True)
class MotionSettings:
    push_in_enabled: bool = False
    push_in_min_segment_seconds: float = DEFAULT_PUSH_IN_MIN_SEGMENT_SECONDS
    push_in_max_zoom: float = DEFAULT_PUSH_IN_MAX_ZOOM


@dataclass(frozen=True)
class BackgroundMusicSettings:
    enabled: bool = False
    path: Path = DEFAULT_BACKGROUND_MUSIC_PATH
    volume: float = 0.18


@dataclass(frozen=True)
class AudioBoostSettings:
    enabled: bool = False
    gain_db: float = 6.0


@dataclass(frozen=True)
class ProcessingResult:
    input_path: Path
    output_path: Path
    input_duration: float
    kept_duration: float
    segments: list[SpeechSegment]
    caption_event_count: int = 0
    censored_word_count: int = 0

    @property
    def removed_duration(self) -> float:
        return max(self.input_duration - self.kept_duration, 0.0)


ProgressCallback = Callable[[float, str], None]
LogCallback = Callable[[str], None]
CancelCallback = Callable[[], bool]


def check_runtime() -> list[str]:
    issues: list[str] = []
    if shutil.which("ffmpeg") is None:
        issues.append("`ffmpeg` was not found in PATH.")
    if shutil.which("ffprobe") is None:
        issues.append("`ffprobe` was not found in PATH.")
    if not MODEL_PATH.exists():
        issues.append(f"Speech model is missing: {MODEL_PATH}")
    return issues


def _get_vad_detector() -> SileroOnnxVad:
    global _VAD_DETECTOR_CACHE
    if _VAD_DETECTOR_CACHE is None:
        _VAD_DETECTOR_CACHE = SileroOnnxVad(MODEL_PATH)
    return _VAD_DETECTOR_CACHE


def build_output_path(input_path: Path) -> Path:
    input_path = Path(input_path)
    return input_path.with_name(f"{input_path.stem}_speech_only.mp4")


def process_video(
    input_path: Path,
    output_path: Path,
    options: ProcessingOptions | None = None,
    *,
    progress_callback: ProgressCallback | None = None,
    log_callback: LogCallback | None = None,
    cancel_callback: CancelCallback | None = None,
) -> ProcessingResult:
    options = options or ProcessingOptions()
    input_path = Path(input_path)
    output_path = Path(output_path)

    _ensure_not_cancelled(cancel_callback)
    _emit_progress(progress_callback, 0.01, "Checking the source video...")
    _log(log_callback, "Reading video metadata.")
    metadata = probe_video_metadata(input_path)

    if not metadata["has_video"]:
        raise ProcessingError("The selected file does not contain a video stream.")
    if not metadata["has_audio"]:
        raise ProcessingError("The selected file does not contain an audio stream to analyze.")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="speech_cutter_") as temp_dir:
        temp_dir_path = Path(temp_dir)
        wav_path = temp_dir_path / "audio.wav"
        filter_script_path = temp_dir_path / "speech_concat.ffscript"
        transcription_needed = needs_transcription(options.captions)
        extraction_end = 0.16 if transcription_needed else 0.20
        detection_end = 0.42 if transcription_needed else 0.55

        _log(log_callback, "Extracting mono 16 kHz audio for speech detection.")
        _run_ffmpeg_with_progress(
            [
                "ffmpeg",
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-nostdin",
                "-i",
                str(input_path),
                "-map",
                "0:a:0",
                "-vn",
                "-ac",
                "1",
                "-ar",
                "16000",
                "-c:a",
                "pcm_s16le",
                "-progress",
                "pipe:1",
                "-nostats",
                str(wav_path),
            ],
            total_duration=max(metadata["duration"], 0.001),
            progress_start=0.02,
            progress_end=extraction_end,
            status="Extracting audio...",
            progress_callback=progress_callback,
            cancel_callback=cancel_callback,
        )

        _log(log_callback, "Running speech detection with the bundled Silero VAD model.")
        detector = _get_vad_detector()
        try:
            segments = detect_speech_in_wav(
                wav_path,
                detector,
                options.vad,
                progress_callback=lambda ratio: _emit_progress(
                    progress_callback,
                    extraction_end + ratio * (detection_end - extraction_end),
                    "Detecting spoken sections...",
                ),
                cancel_callback=cancel_callback,
            )
        except DetectionCancelledError as exc:
            raise UserCancelledError(str(exc)) from exc

        if not segments:
            raise NoSpeechDetectedError("No human speech was detected in that video.")

        frame_boundaries: list[float] = []
        if bool(metadata.get("is_vfr")):
            frame_boundaries = _probe_video_frame_boundaries(input_path, float(metadata["duration"]))
        if frame_boundaries:
            segments = _align_segments_to_frame_boundaries(
                segments,
                frame_boundaries=frame_boundaries,
                total_duration=float(metadata["duration"]),
            )
            _log(
                log_callback,
                f"Aligned segment edges to {len(frame_boundaries) - 1} exact source frame boundaries to keep sync stable.",
            )
        else:
            video_fps = metadata.get("video_fps")
            if isinstance(video_fps, float) and video_fps > 0:
                segments = _align_segments_to_video_frames(
                    segments,
                    fps=video_fps,
                    total_duration=float(metadata["duration"]),
                )
                _log(log_callback, f"Aligned segment edges to the video frame grid ({video_fps:.3f} fps) to keep sync stable.")

        crop_settings = _prepare_crop_settings(
            options.crop,
            video_width=_as_int(metadata.get("video_width")),
            video_height=_as_int(metadata.get("video_height")),
        )
        motion_settings = _prepare_motion_settings(options.motion)
        background_music_settings = _prepare_background_music_settings(options.background_music)
        audio_boost_settings = _prepare_audio_boost_settings(options.audio_boost)

        kept_duration = sum(segment.duration for segment in segments)
        _log(
            log_callback,
            f"Detected {len(segments)} speech segment(s). Keeping {kept_duration:.1f}s out of {metadata['duration']:.1f}s.",
        )
        if background_music_settings is not None:
            _log(
                log_callback,
                f"Adding looped background music at {background_music_settings.volume * 100:.0f}% volume.",
            )
        if audio_boost_settings is not None:
            _log(
                log_callback,
                f"Boosting speech volume by {audio_boost_settings.gain_db:.1f} dB.",
            )

        artifacts = CaptionArtifactResult(
            subtitle_path=None,
            profanity_intervals=[],
            caption_event_count=0,
            censored_word_count=0,
            transcribed_word_count=0,
        )
        if transcription_needed:
            _ensure_not_cancelled(cancel_callback)
            artifacts = _create_caption_artifacts(
                wav_path,
                temp_dir_path,
                options=options,
                metadata=metadata,
                segments=segments,
                kept_duration=kept_duration,
                progress_start=detection_end,
                progress_end=0.65,
                progress_callback=progress_callback,
                log_callback=log_callback,
                cancel_callback=cancel_callback,
            )

        filter_script_path.write_text(
            _build_filter_script(
                segments,
                video_width=_as_int(metadata.get("video_width")),
                video_height=_as_int(metadata.get("video_height")),
                crop_settings=crop_settings,
                motion_settings=motion_settings,
                background_music_settings=background_music_settings,
                audio_boost_settings=audio_boost_settings,
                kept_duration=kept_duration,
                caption_artifacts=artifacts,
            ),
            encoding="utf-8",
        )
        use_nvenc = _nvenc_is_available()
        if use_nvenc:
            _log(
                log_callback,
                "Rendering cuts, effects, audio, and captions in one optimized pass using NVIDIA NVENC.",
            )
        else:
            _log(
                log_callback,
                "NVIDIA NVENC is unavailable; rendering with the high-quality CPU encoder.",
            )
        render_command = [
                "ffmpeg",
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-nostdin",
                "-i",
                str(input_path.resolve()),
        ]
        if background_music_settings is not None:
            render_command.extend(
                [
                    "-stream_loop",
                    "-1",
                    "-i",
                    str(background_music_settings.path.resolve()),
                ]
            )
        render_command.extend(
            [
                "-filter_complex_script",
                str(filter_script_path),
                "-map",
                "[outv]",
                "-map",
                "[outa]",
                *_video_encoder_args(options, use_nvenc=use_nvenc),
                "-c:a",
                "aac",
                "-b:a",
                "192k",
                "-movflags",
                "+faststart",
                "-progress",
                "pipe:1",
                "-nostats",
                str(output_path.resolve()),
            ],
        )
        try:
            _run_ffmpeg_with_progress(
                render_command,
                total_duration=max(kept_duration, 0.001),
                progress_start=0.65 if transcription_needed else detection_end,
                progress_end=1.00,
                status="Rendering output video on the GPU..." if use_nvenc else "Rendering output video...",
                progress_callback=progress_callback,
                cancel_callback=cancel_callback,
                cwd=temp_dir_path,
            )
        except ProcessingError:
            if not use_nvenc or (cancel_callback and cancel_callback()):
                raise
            _mark_nvenc_unavailable()
            _log(
                log_callback,
                "Warning: NVIDIA NVENC could not finish this render; retrying automatically on the CPU.",
            )
            cpu_render_command = _replace_video_encoder_args(
                render_command,
                _video_encoder_args(options, use_nvenc=False),
            )
            _run_ffmpeg_with_progress(
                cpu_render_command,
                total_duration=max(kept_duration, 0.001),
                progress_start=0.65 if transcription_needed else detection_end,
                progress_end=1.00,
                status="Rendering output video on the CPU...",
                progress_callback=progress_callback,
                cancel_callback=cancel_callback,
                cwd=temp_dir_path,
            )

        caption_event_count = artifacts.caption_event_count
        censored_word_count = artifacts.censored_word_count

    _emit_progress(progress_callback, 1.0, "Finished.")
    _log(log_callback, f"Saved output to {output_path}")
    return ProcessingResult(
        input_path=input_path,
        output_path=output_path,
        input_duration=metadata["duration"],
        kept_duration=kept_duration,
        segments=segments,
        caption_event_count=caption_event_count,
        censored_word_count=censored_word_count,
    )


def probe_video_metadata(input_path: Path) -> dict[str, float | bool | None]:
    completed = _run_command(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_streams",
            "-show_format",
            "-of",
            "json",
            str(input_path),
        ],
        check=True,
    )
    payload = json.loads(completed.stdout)
    streams = payload.get("streams", [])
    format_info = payload.get("format", {})

    duration = _first_float(
        format_info.get("duration"),
        *(stream.get("duration") for stream in streams),
    )

    video_stream = next((stream for stream in streams if stream.get("codec_type") == "video"), None)

    average_fps = _parse_rate(video_stream.get("avg_frame_rate") if video_stream else None)
    nominal_fps = _parse_rate(video_stream.get("r_frame_rate") if video_stream else None)
    is_vfr = bool(
        average_fps
        and nominal_fps
        and abs(average_fps - nominal_fps) / max(average_fps, nominal_fps) > 0.001
    )

    return {
        "duration": duration,
        "has_video": any(stream.get("codec_type") == "video" for stream in streams),
        "has_audio": any(stream.get("codec_type") == "audio" for stream in streams),
        "video_width": _as_int(video_stream.get("width")) if video_stream else 0,
        "video_height": _as_int(video_stream.get("height")) if video_stream else 0,
        "video_fps": average_fps or nominal_fps,
        "video_nominal_fps": nominal_fps,
        "is_vfr": is_vfr,
    }


def _align_segments_to_video_frames(
    segments: list[SpeechSegment],
    *,
    fps: float,
    total_duration: float,
) -> list[SpeechSegment]:
    if not segments or fps <= 0:
        return segments

    frame_duration = 1.0 / fps
    aligned: list[SpeechSegment] = []

    for segment in segments:
        start_frame = math.floor(segment.start * fps)
        end_frame = math.ceil(segment.end * fps)

        start = max(0.0, start_frame * frame_duration)
        end = min(total_duration, end_frame * frame_duration)
        if end <= start:
            end = min(total_duration, start + frame_duration)

        if aligned and start <= aligned[-1].end + (frame_duration / 2):
            aligned[-1] = SpeechSegment(aligned[-1].start, max(aligned[-1].end, end))
        else:
            aligned.append(SpeechSegment(start, end))

    return aligned


def _align_segments_to_frame_boundaries(
    segments: list[SpeechSegment],
    *,
    frame_boundaries: list[float],
    total_duration: float,
) -> list[SpeechSegment]:
    if len(frame_boundaries) < 2:
        return segments

    aligned: list[SpeechSegment] = []

    for segment in segments:
        start_index = max(bisect_right(frame_boundaries, segment.start) - 1, 0)
        end_index = bisect_left(frame_boundaries, segment.end)

        start = frame_boundaries[start_index]
        if end_index >= len(frame_boundaries):
            end = total_duration
        else:
            end = frame_boundaries[end_index]

        if end <= start:
            next_index = min(start_index + 1, len(frame_boundaries) - 1)
            end = frame_boundaries[next_index]
            if end <= start:
                end = min(total_duration, start)

        if aligned and start <= aligned[-1].end:
            aligned[-1] = SpeechSegment(aligned[-1].start, max(aligned[-1].end, end))
        else:
            aligned.append(SpeechSegment(start, end))

    return aligned


def _build_filter_script(
    segments: list[SpeechSegment],
    *,
    video_width: int,
    video_height: int,
    crop_settings: CropSettings | None,
    motion_settings: MotionSettings | None,
    background_music_settings: BackgroundMusicSettings | None,
    audio_boost_settings: AudioBoostSettings | None,
    kept_duration: float,
    caption_artifacts: CaptionArtifactResult | None = None,
) -> str:
    lines: list[str] = []
    clip_specs = _expand_segment_clips(
        segments,
        crop_settings=crop_settings,
        motion_settings=motion_settings,
    )
    for index, clip in enumerate(clip_specs):
        segment = clip.segment
        start = f"{segment.start:.6f}"
        end = f"{segment.end:.6f}"

        video_filters = [
            f"[0:v:0]trim=start={start}:end={end}",
            "setpts=PTS-STARTPTS",
        ]
        if clip.effect == "crop":
            assert crop_settings is not None
            video_filters.append(
                f"crop=w={crop_settings.width}:h={crop_settings.height}:x={crop_settings.x}:y={crop_settings.y}"
            )
            video_filters.append(f"scale={video_width}:{video_height}:flags=lanczos")
        elif clip.effect == "push_in":
            assert motion_settings is not None
            video_filters.extend(
                _build_push_in_filters(
                    video_width=video_width,
                    video_height=video_height,
                    segment_duration=segment.duration,
                    max_zoom=motion_settings.push_in_max_zoom,
                )
            )
        video_filters.append("setsar=1")

        lines.append(",".join(video_filters) + f"[v{index}]")
        lines.append(f"[0:a:0]atrim=start={start}:end={end},asetpts=PTS-STARTPTS[a{index}]")

    concat_inputs = "".join(f"[v{index}][a{index}]" for index in range(len(clip_specs)))
    lines.append(f"{concat_inputs}concat=n={len(clip_specs)}:v=1:a=1[video_concat][speech_concat]")

    if caption_artifacts is not None and caption_artifacts.subtitle_path is not None:
        lines.append(f"[video_concat]ass={caption_artifacts.subtitle_path.name}[outv]")
    else:
        lines.append("[video_concat]null[outv]")

    if background_music_settings is None:
        lines.append("[speech_concat]anull[audio_mix]")
    else:
        lines.append(
            (
                f"[1:a:0]atrim=duration={kept_duration:.6f},asetpts=PTS-STARTPTS,"
                f"volume={background_music_settings.volume:.6f}[bga]"
            )
        )
        lines.append("[speech_concat][bga]amix=inputs=2:duration=first:dropout_transition=0:normalize=0[audio_mix]")

    if audio_boost_settings is not None:
        # A limiter after the gain bump keeps already-loud audio from clipping/distorting.
        lines.append(
            f"[audio_mix]volume={audio_boost_settings.gain_db:.3f}dB,"
            "alimiter=limit=0.97:attack=5:release=50[audio_processed]"
        )
    else:
        lines.append("[audio_mix]anull[audio_processed]")

    if caption_artifacts is not None and caption_artifacts.profanity_intervals:
        lines.append(
            _build_audio_profanity_filter(
                caption_artifacts.profanity_intervals,
                input_label="audio_processed",
            )
        )
    else:
        lines.append("[audio_processed]anull[outa]")
    return ";\n".join(lines) + "\n"


@dataclass(frozen=True)
class ClipSpec:
    segment: SpeechSegment
    effect: str = "none"


def _expand_segment_clips(
    segments: list[SpeechSegment],
    *,
    crop_settings: CropSettings | None,
    motion_settings: MotionSettings | None,
) -> list[ClipSpec]:
    clips: list[ClipSpec] = []
    for index, segment in enumerate(segments, start=1):
        should_crop = _should_crop_segment(segment, index, crop_settings)
        clips.extend(
            _split_segment_for_effects(
                segment,
                should_crop=should_crop,
                crop_settings=crop_settings,
                motion_settings=motion_settings,
            )
        )
    return clips


def _split_segment_for_effects(
    segment: SpeechSegment,
    *,
    should_crop: bool,
    crop_settings: CropSettings | None,
    motion_settings: MotionSettings | None,
) -> list[ClipSpec]:
    if crop_settings is not None:
        if should_crop:
            return _build_zoom_clips(segment, crop_settings=crop_settings, alignment="start")

        if _should_apply_middle_zoom(segment, crop_settings):
            return _build_zoom_clips(segment, crop_settings=crop_settings, alignment="middle")

    if _should_apply_push_in(segment, motion_settings):
        return [ClipSpec(segment, "push_in")]

    return [ClipSpec(segment)]


def _should_apply_middle_zoom(
    segment: SpeechSegment,
    crop_settings: CropSettings,
) -> bool:
    if not crop_settings.mid_zoom_on_long_segments:
        return False
    if crop_settings.zoom_duration_seconds <= 0:
        return False
    return segment.duration >= max(
        crop_settings.min_segment_seconds,
        crop_settings.mid_zoom_min_segment_seconds,
    )


def _build_zoom_clips(
    segment: SpeechSegment,
    *,
    crop_settings: CropSettings,
    alignment: str,
) -> list[ClipSpec]:
    zoom_duration = crop_settings.zoom_duration_seconds
    if zoom_duration <= 0 or segment.duration <= zoom_duration:
        return [ClipSpec(segment, "crop")]

    if alignment == "middle":
        padding = max((segment.duration - zoom_duration) / 2.0, 0.0)
        zoom_start = segment.start + padding
    else:
        zoom_start = segment.start

    zoom_end = min(segment.end, zoom_start + zoom_duration)
    if zoom_end <= zoom_start:
        return [ClipSpec(segment)]

    clips: list[ClipSpec] = []
    if zoom_start > segment.start:
        clips.append(ClipSpec(SpeechSegment(segment.start, zoom_start)))
    clips.append(ClipSpec(SpeechSegment(zoom_start, zoom_end), "crop"))
    if zoom_end < segment.end:
        clips.append(ClipSpec(SpeechSegment(zoom_end, segment.end)))
    return clips


def _should_apply_push_in(
    segment: SpeechSegment,
    motion_settings: MotionSettings | None,
) -> bool:
    if motion_settings is None:
        return False
    return segment.duration >= motion_settings.push_in_min_segment_seconds


def _build_push_in_filters(
    *,
    video_width: int,
    video_height: int,
    segment_duration: float,
    max_zoom: float,
) -> list[str]:
    zoom_delta = max(0.0, max_zoom - 1.0)
    duration = max(segment_duration, 0.001)
    zoom_expr = f"1+{zoom_delta:.6f}*(1-cos(PI*min(t/{duration:.6f},1)))/2"
    return [
        (
            "scale="
            f"w='trunc(iw*({zoom_expr})/2)*2':"
            f"h='trunc(ih*({zoom_expr})/2)*2':"
            "eval=frame:flags=lanczos"
        ),
        f"crop=w={video_width}:h={video_height}:x='(iw-ow)/2':y='(ih-oh)/2'",
    ]


def _create_caption_artifacts(
    transcription_audio_path: Path,
    temp_dir_path: Path,
    *,
    options: ProcessingOptions,
    metadata: dict[str, float | bool | None],
    segments: list[SpeechSegment],
    kept_duration: float,
    progress_start: float,
    progress_end: float,
    progress_callback: ProgressCallback | None,
    log_callback: LogCallback | None,
    cancel_callback: CancelCallback | None,
) -> CaptionArtifactResult:
    video_width = _as_int(metadata.get("video_width"))
    video_height = _as_int(metadata.get("video_height"))
    if video_width <= 0 or video_height <= 0:
        raise ProcessingError("Could not read the source video size needed for burned-in subtitles.")

    try:
        artifacts = create_caption_artifacts(
            transcription_audio_path,
            temp_dir_path,
            video_width=video_width,
            video_height=video_height,
            total_duration=kept_duration,
            settings=options.captions,
            source_duration=float(metadata.get("duration") or kept_duration),
            timeline_segments=[(segment.start, segment.end) for segment in segments],
            input_is_prepared_audio=True,
            progress_callback=lambda ratio, status: _emit_progress(
                progress_callback,
                progress_start + max(0.0, min(ratio, 1.0)) * (progress_end - progress_start),
                status,
            ),
            log_callback=log_callback,
            cancel_callback=cancel_callback,
        )
    except RuntimeError as exc:
        if str(exc) == "Processing was cancelled.":
            raise UserCancelledError(str(exc)) from exc
        raise ProcessingError(str(exc)) from exc

    if artifacts.transcribed_word_count:
        _log(
            log_callback,
            (
                f"Transcribed {artifacts.transcribed_word_count} word(s), built "
                f"{artifacts.caption_event_count} caption cue(s), and flagged {artifacts.censored_word_count} profane word(s)."
            ),
        )
    else:
        _log(log_callback, "No subtitle words were recovered from the trimmed output.")

    return artifacts


def _render_postprocessed_video(
    base_render_path: Path,
    output_path: Path,
    *,
    options: ProcessingOptions,
    artifacts: CaptionArtifactResult,
    temp_dir_path: Path,
    kept_duration: float,
    progress_callback: ProgressCallback | None,
    cancel_callback: CancelCallback | None,
) -> None:
    filter_script_path = temp_dir_path / "subtitle_burn.ffscript"
    filter_lines: list[str] = []
    map_video = "0:v:0"
    map_audio = "0:a:0"
    copy_video = True
    copy_audio = True

    if artifacts.subtitle_path is not None:
        filter_lines.append(f"[0:v:0]ass={artifacts.subtitle_path.name}[outv]")
        map_video = "[outv]"
        copy_video = False
    if artifacts.profanity_intervals:
        filter_lines.append(_build_audio_profanity_filter(artifacts.profanity_intervals))
        map_audio = "[outa]"
        copy_audio = False

    filter_script_path.write_text(";\n".join(filter_lines) + "\n", encoding="utf-8")

    command = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-i",
        str(base_render_path),
        "-filter_complex_script",
        str(filter_script_path),
        "-map",
        map_video,
        "-map",
        map_audio,
    ]

    if copy_video:
        command.extend(["-c:v", "copy"])
    else:
        command.extend(["-c:v", "libx264", "-preset", options.video_preset, "-crf", str(options.video_crf)])

    if copy_audio:
        command.extend(["-c:a", "copy"])
    else:
        command.extend(["-c:a", "aac", "-b:a", "192k"])

    command.extend(
        [
            "-movflags",
            "+faststart",
            "-progress",
            "pipe:1",
            "-nostats",
            str(output_path),
        ]
    )

    _run_ffmpeg_with_progress(
        command,
        total_duration=max(kept_duration, 0.001),
        progress_start=0.94,
        progress_end=1.00,
        status="Rendering subtitles and profanity filter...",
        progress_callback=progress_callback,
        cancel_callback=cancel_callback,
        cwd=temp_dir_path,
    )


def _build_audio_profanity_filter(
    intervals: list[tuple[float, float]],
    *,
    input_label: str = "0:a:0",
) -> str:
    filters = ["asetpts=PTS-STARTPTS"]
    for start, end in intervals:
        filters.append(f"volume=enable='between(t,{start:.3f},{end:.3f})':volume=0")
    return f"[{input_label}]{','.join(filters)}[outa]"


def _video_encoder_args(options: ProcessingOptions, *, use_nvenc: bool) -> list[str]:
    if not use_nvenc:
        return [
            "-c:v",
            "libx264",
            "-preset",
            options.video_preset,
            "-crf",
            str(options.video_crf),
            "-pix_fmt",
            "yuv420p",
        ]

    if options.video_crf <= 0:
        return [
            "-c:v",
            "h264_nvenc",
            "-preset",
            "p7",
            "-tune",
            "lossless",
            "-rc",
            "constqp",
            "-qp",
            "0",
            "-pix_fmt",
            "yuv420p",
        ]

    nvenc_quality = max(1, min(51, options.video_crf - 2))
    return [
        "-c:v",
        "h264_nvenc",
        "-preset",
        "p6",
        "-tune",
        "hq",
        "-rc",
        "vbr",
        "-cq",
        str(nvenc_quality),
        "-b:v",
        "0",
        "-multipass",
        "fullres",
        "-spatial-aq",
        "1",
        "-temporal-aq",
        "1",
        "-aq-strength",
        "8",
        "-rc-lookahead",
        "32",
        "-bf",
        "3",
        "-b_ref_mode",
        "middle",
        "-profile:v",
        "high",
        "-pix_fmt",
        "yuv420p",
    ]


def _replace_video_encoder_args(command: list[str], encoder_args: list[str]) -> list[str]:
    try:
        video_start = command.index("-c:v")
        audio_start = command.index("-c:a", video_start + 1)
    except ValueError as exc:
        raise ProcessingError("The FFmpeg render command is missing its video encoder settings.") from exc
    return command[:video_start] + encoder_args + command[audio_start:]


def _nvenc_is_available() -> bool:
    global _NVENC_AVAILABLE_CACHE
    if _NVENC_AVAILABLE_CACHE is not None:
        return _NVENC_AVAILABLE_CACHE
    try:
        completed = _run_command(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-f",
                "lavfi",
                "-i",
                "color=c=black:s=64x64:d=0.05",
                "-frames:v",
                "1",
                "-c:v",
                "h264_nvenc",
                "-preset",
                "p6",
                "-f",
                "null",
                "-",
            ],
            check=False,
        )
        _NVENC_AVAILABLE_CACHE = completed.returncode == 0
    except (OSError, subprocess.SubprocessError):
        _NVENC_AVAILABLE_CACHE = False
    return _NVENC_AVAILABLE_CACHE


def _mark_nvenc_unavailable() -> None:
    global _NVENC_AVAILABLE_CACHE
    _NVENC_AVAILABLE_CACHE = False


def _run_ffmpeg_with_progress(
    command: list[str],
    *,
    total_duration: float,
    progress_start: float,
    progress_end: float,
    status: str,
    progress_callback: ProgressCallback | None,
    cancel_callback: CancelCallback | None,
    cwd: Path | None = None,
) -> None:
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        cwd=str(cwd) if cwd is not None else None,
        **_startup_kwargs(),
    )

    try:
        saw_end = False
        while True:
            _ensure_not_cancelled(cancel_callback, process)

            assert process.stdout is not None
            line = process.stdout.readline()
            if not line:
                if process.poll() is not None:
                    break
                time.sleep(0.05)
                continue

            line = line.strip()
            if line.startswith("out_time="):
                current = _parse_ffmpeg_timecode(line.partition("=")[2])
                if current is not None:
                    ratio = min(current / total_duration, 1.0)
                    progress = progress_start + ratio * (progress_end - progress_start)
                    _emit_progress(progress_callback, progress, status)
            elif line == "progress=end":
                _emit_progress(progress_callback, progress_end, status)
                saw_end = True
                break

        stdout_tail, stderr_output = process.communicate(timeout=30)
        return_code = process.returncode
    finally:
        if process.stdout:
            process.stdout.close()
        if process.stderr:
            process.stderr.close()

    if return_code != 0:
        if cancel_callback and cancel_callback():
            raise UserCancelledError("Processing was cancelled.")
        message = stderr_output.strip() or "ffmpeg exited with an unknown error."
        raise ProcessingError(message)

    if not saw_end:
        _emit_progress(progress_callback, progress_end, status)


def _run_command(command: list[str], *, check: bool) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=check,
        **_startup_kwargs(),
    )


def _ensure_not_cancelled(
    cancel_callback: CancelCallback | None,
    process: subprocess.Popen[str] | None = None,
) -> None:
    if cancel_callback and cancel_callback():
        if process and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                process.kill()
        raise UserCancelledError("Processing was cancelled.")


def _emit_progress(callback: ProgressCallback | None, value: float, status: str) -> None:
    if callback:
        callback(max(0.0, min(value, 1.0)), status)


def _log(callback: LogCallback | None, message: str) -> None:
    if callback:
        callback(message)


def _parse_ffmpeg_timecode(value: str) -> float | None:
    parts = value.split(":")
    if len(parts) != 3:
        return None

    hours, minutes, seconds = parts
    try:
        return int(hours) * 3600 + int(minutes) * 60 + float(seconds)
    except ValueError:
        return None


def _first_float(*values: object) -> float:
    for value in values:
        if value in (None, "N/A", ""):
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return 0.0


def _as_int(value: object) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _parse_rate(*values: object) -> float | None:
    for value in values:
        if value in (None, "N/A", "", "0/0"):
            continue
        try:
            return float(Fraction(str(value)))
        except (ValueError, ZeroDivisionError):
            continue
    return None


def _probe_video_frame_boundaries(input_path: Path, total_duration: float) -> list[float]:
    completed = _run_command(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "frame=best_effort_timestamp_time,pkt_duration_time",
            "-of",
            "json",
            str(input_path),
        ],
        check=True,
    )
    payload = json.loads(completed.stdout)
    frames = payload.get("frames", [])
    if not frames:
        return []

    boundaries: list[float] = []
    last_duration: float | None = None
    for frame in frames:
        timestamp = _first_float(frame.get("best_effort_timestamp_time"))
        duration = _first_float(frame.get("pkt_duration_time"))
        boundaries.append(timestamp)
        if duration > 0:
            last_duration = duration

    boundaries = sorted(set(boundaries))
    if not boundaries:
        return []

    if boundaries[0] > 0:
        boundaries.insert(0, 0.0)

    final_boundary = total_duration
    if last_duration and boundaries:
        final_boundary = max(final_boundary, boundaries[-1] + last_duration)

    if final_boundary > boundaries[-1]:
        boundaries.append(final_boundary)
    elif total_duration > boundaries[-1]:
        boundaries.append(total_duration)

    return boundaries


def _prepare_crop_settings(
    crop: CropSettings,
    *,
    video_width: int,
    video_height: int,
) -> CropSettings | None:
    if not crop.enabled:
        return None

    if video_width <= 0 or video_height <= 0:
        raise ProcessingError("Could not read the source video size needed for crop/zoom.")
    if crop.width <= 0 or crop.height <= 0:
        raise ProcessingError("Crop width and height must be greater than zero.")
    if crop.x < 0 or crop.y < 0:
        raise ProcessingError("Crop X and Y must be zero or positive.")
    if crop.x + crop.width > video_width or crop.y + crop.height > video_height:
        raise ProcessingError(
            f"Crop rectangle must stay inside the source video ({video_width}x{video_height})."
        )
    aspect_error = abs((crop.width * video_height) - (crop.height * video_width))
    if aspect_error > max(video_width, video_height):
        raise ProcessingError(
            "Crop must keep the same aspect ratio as the source video so zoomed segments do not stretch."
        )
    if crop.every_n_segments < 1:
        raise ProcessingError("Crop cadence must be at least 1.")
    if crop.min_segment_seconds < 0:
        raise ProcessingError("Minimum crop segment length cannot be negative.")
    if crop.zoom_duration_seconds < 0:
        raise ProcessingError("Zoom duration cannot be negative.")
    if crop.mid_zoom_min_segment_seconds < 0:
        raise ProcessingError("Long-segment middle zoom threshold cannot be negative.")

    return crop


def _prepare_motion_settings(motion: MotionSettings) -> MotionSettings | None:
    if not motion.push_in_enabled:
        return None
    if motion.push_in_min_segment_seconds < 0:
        raise ProcessingError("Push-in segment length cannot be negative.")
    if motion.push_in_max_zoom <= 1.0:
        raise ProcessingError("Push-in zoom must be greater than 1.0.")
    if motion.push_in_max_zoom > 1.5:
        raise ProcessingError("Push-in zoom must be 1.5x or lower.")
    return motion


def _prepare_background_music_settings(background_music: BackgroundMusicSettings) -> BackgroundMusicSettings | None:
    if not background_music.enabled:
        return None
    path = Path(background_music.path)
    if not path.exists():
        raise ProcessingError(f"Background music file is missing: {path}")
    if background_music.volume < 0:
        raise ProcessingError("Background music volume cannot be negative.")
    if background_music.volume > 2.0:
        raise ProcessingError("Background music volume must be 200% or lower.")
    return BackgroundMusicSettings(
        enabled=True,
        path=path,
        volume=float(background_music.volume),
    )


def _prepare_audio_boost_settings(audio_boost: AudioBoostSettings) -> AudioBoostSettings | None:
    if not audio_boost.enabled:
        return None
    if audio_boost.gain_db < 0:
        raise ProcessingError("Audio boost cannot be negative.")
    if audio_boost.gain_db > 24.0:
        raise ProcessingError("Audio boost must be 24 dB or lower.")
    return AudioBoostSettings(enabled=True, gain_db=float(audio_boost.gain_db))


def _should_crop_segment(
    segment: SpeechSegment,
    segment_number: int,
    crop_settings: CropSettings | None,
) -> bool:
    if not crop_settings:
        return False
    if segment.duration < crop_settings.min_segment_seconds:
        return False
    return segment_number % crop_settings.every_n_segments == 0


def _startup_kwargs() -> dict[str, object]:
    kwargs: dict[str, object] = {}
    if os.name == "nt":
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        kwargs["startupinfo"] = startupinfo
        kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
    return kwargs
