from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable
import math
import os
import re
import subprocess

from .gemini import DEFAULT_GEMINI_MODEL, correct_transcript_words
from .groq import (
    DEFAULT_GROQ_CORRECTION_MODEL,
    DEFAULT_GROQ_WHISPER_MODEL,
    correct_transcript_chunk as correct_transcript_chunk_with_groq,
    transcribe_audio_words,
)


LogCallback = Callable[[str], None]
ProgressCallback = Callable[[float, str], None]
CancelCallback = Callable[[], bool]

DEFAULT_MODEL_NAME = "medium"
ENHANCED_POLISH_MODEL_NAME = "large-v3-turbo"
_MODEL_CACHE: dict[tuple[str, str], object] = {}
_WHISPER_GPU_DISABLED = False

_PROFANITY_PREFIXES = (
    "asshole",
    "bastard",
    "bitch",
    "bullshit",
    "cock",
    "cunt",
    "dick",
    "fuck",
    "motherfuck",
    "nigger",
    "nigga",
    "pussy",
    "shit",
    "slut",
    "whore",
    "chuj",
    "cipa",
    "jeb",
    "kurw",
    "pierdol",
    "pizd",
    "skurw",
    "spierdal",
    "wkurw",
    "zajeb",
)


@dataclass(frozen=True)
class CaptionSettings:
    enabled: bool = False
    profanity_filter: bool = False
    model_name: str = DEFAULT_MODEL_NAME
    enhanced_polish_recognition: bool = False
    transcription_provider: str = "local"
    groq_api_key: str = ""
    groq_model: str = DEFAULT_GROQ_WHISPER_MODEL
    ai_correction_enabled: bool = False
    gemini_api_key: str = ""
    gemini_model: str = DEFAULT_GEMINI_MODEL
    groq_correction_api_key: str = ""
    groq_correction_api_keys: tuple[str, ...] = ()
    groq_correction_model: str = DEFAULT_GROQ_CORRECTION_MODEL
    max_words_per_caption: int = 4
    max_caption_seconds: float = 2.2
    max_word_gap_seconds: float = 0.45


@dataclass(frozen=True)
class CaptionWord:
    start: float
    end: float
    text: str
    display_text: str
    is_profanity: bool


@dataclass(frozen=True)
class CaptionArtifactResult:
    subtitle_path: Path | None
    profanity_intervals: list[tuple[float, float]]
    caption_event_count: int
    censored_word_count: int
    transcribed_word_count: int


@dataclass(frozen=True)
class _CaptionStyleMetrics:
    font_size: int
    active_font_size: int
    outline: int
    margin_h: int
    margin_v: int
    max_line_chars: int
    usable_width: int


def needs_transcription(settings: CaptionSettings) -> bool:
    return bool(settings.enabled or settings.profanity_filter)


def create_caption_artifacts(
    input_path: Path,
    output_dir: Path,
    *,
    video_width: int,
    video_height: int,
    total_duration: float,
    settings: CaptionSettings,
    source_duration: float | None = None,
    timeline_segments: list[tuple[float, float]] | None = None,
    input_is_prepared_audio: bool = False,
    progress_callback: ProgressCallback | None = None,
    log_callback: LogCallback | None = None,
    cancel_callback: CancelCallback | None = None,
) -> CaptionArtifactResult:
    if not needs_transcription(settings):
        return CaptionArtifactResult(
            subtitle_path=None,
            profanity_intervals=[],
            caption_event_count=0,
            censored_word_count=0,
            transcribed_word_count=0,
        )

    if input_is_prepared_audio and settings.transcription_provider.casefold() != "groq":
        transcription_input = input_path
        if log_callback:
            log_callback("Reusing the speech-detection audio for local transcription.")
    else:
        transcription_input = _prepare_transcription_audio(
            input_path,
            output_dir=output_dir,
            total_duration=max(source_duration or total_duration, 0.001),
            settings=settings,
            progress_callback=progress_callback,
            log_callback=log_callback,
        )

    words = _transcribe_words(
        transcription_input,
        total_duration=max(source_duration or total_duration, 0.001),
        settings=settings,
        progress_callback=progress_callback,
        log_callback=log_callback,
        cancel_callback=cancel_callback,
    )

    if timeline_segments is not None:
        words = _remap_words_to_timeline(words, timeline_segments)
        if log_callback:
            log_callback(f"Mapped {len(words)} transcript word(s) onto the trimmed video timeline.")

    if settings.ai_correction_enabled and words:
        groq_correction_keys = _unique_api_keys(
            (settings.groq_correction_api_key, *settings.groq_correction_api_keys)
        )
        groq_fallback = _build_groq_correction_fallback(
            groq_correction_keys,
            model=settings.groq_correction_model,
            cancel_callback=cancel_callback,
            log_callback=log_callback,
        )
        if log_callback:
            log_callback(
                f"Checking transcript logic and recognition errors with Gemini ({settings.gemini_model}); "
                "long transcripts may take a few minutes."
            )
        try:
            corrected_text = correct_transcript_words(
                [word.text for word in words],
                api_key=settings.gemini_api_key,
                model=settings.gemini_model,
                cancel_callback=cancel_callback,
                progress_callback=(
                    (lambda done, total: log_callback(f"Gemini checked transcript section {done}/{total}."))
                    if log_callback
                    else None
                ),
                warning_callback=log_callback,
                quota_fallback=groq_fallback,
            )
        except RuntimeError as exc:
            if cancel_callback and cancel_callback():
                raise
            if log_callback:
                log_callback(
                    f"Warning: Gemini transcript correction was skipped ({exc}). "
                    "Continuing with the original transcript."
                )
        else:
            changed_count = 0
            corrected_words: list[CaptionWord] = []
            for word, text in zip(words, corrected_text):
                is_profanity = _is_profanity(text)
                display_text = _mask_profanity(text) if is_profanity and settings.profanity_filter else text
                changed_count += int(text != word.text)
                corrected_words.append(
                    CaptionWord(
                        start=word.start,
                        end=word.end,
                        text=text,
                        display_text=display_text,
                        is_profanity=is_profanity,
                    )
                )
            words = corrected_words
            if log_callback:
                log_callback(f"AI transcript check finished; corrected {changed_count} word token(s).")

    subtitle_path: Path | None = None
    caption_event_count = 0
    if settings.enabled and words:
        subtitle_path = output_dir / "captions.ass"
        caption_event_count = _write_ass_subtitles(
            subtitle_path,
            words,
            video_width=video_width,
            video_height=video_height,
            settings=settings,
        )

    profanity_intervals = _collect_profanity_intervals(words) if settings.profanity_filter else []
    censored_word_count = sum(1 for word in words if word.is_profanity) if settings.profanity_filter else 0
    return CaptionArtifactResult(
        subtitle_path=subtitle_path,
        profanity_intervals=profanity_intervals,
        caption_event_count=caption_event_count,
        censored_word_count=censored_word_count,
        transcribed_word_count=len(words),
    )


def _unique_api_keys(keys: tuple[str, ...]) -> tuple[str, ...]:
    unique: list[str] = []
    for key in keys:
        value = key.strip()
        if value and value not in unique:
            unique.append(value)
    return tuple(unique)


def _build_groq_correction_fallback(
    api_keys: tuple[str, ...],
    *,
    model: str,
    cancel_callback: CancelCallback | None,
    log_callback: LogCallback | None,
) -> Callable[[list[str], list[str], list[str]], list[str]] | None:
    if not api_keys:
        return None
    next_key_index = 0

    def correct_chunk(words: list[str], context_before: list[str], context_after: list[str]) -> list[str]:
        nonlocal next_key_index
        last_error: RuntimeError | None = None
        for offset in range(len(api_keys)):
            key_index = (next_key_index + offset) % len(api_keys)
            try:
                corrected = correct_transcript_chunk_with_groq(
                    words,
                    context_before=context_before,
                    context_after=context_after,
                    api_key=api_keys[key_index],
                    model=model,
                    cancel_callback=cancel_callback,
                )
            except RuntimeError as exc:
                last_error = exc
                if log_callback and len(api_keys) > 1:
                    log_callback(
                        f"Groq fallback key {key_index + 1}/{len(api_keys)} failed; trying the next key."
                    )
                continue
            next_key_index = (key_index + 1) % len(api_keys)
            if log_callback:
                log_callback(f"Groq fallback completed this section with key {key_index + 1}/{len(api_keys)}.")
            return corrected
        raise last_error or RuntimeError("No Groq fallback API key is available.")

    return correct_chunk


def _remap_words_to_timeline(
    words: list[CaptionWord],
    timeline_segments: list[tuple[float, float]],
) -> list[CaptionWord]:
    """Map source-media word timestamps onto the concatenated output timeline."""
    if not words or not timeline_segments:
        return []

    normalized_segments = [
        (max(0.0, float(start)), max(0.0, float(end)))
        for start, end in timeline_segments
        if float(end) > float(start)
    ]
    if not normalized_segments:
        return []

    offsets: list[float] = []
    elapsed = 0.0
    for start, end in normalized_segments:
        offsets.append(elapsed)
        elapsed += end - start

    remapped: list[CaptionWord] = []
    segment_index = 0
    for word in sorted(words, key=lambda item: (item.start, item.end)):
        while segment_index < len(normalized_segments) and normalized_segments[segment_index][1] <= word.start:
            segment_index += 1
        if segment_index >= len(normalized_segments):
            break

        best_index: int | None = None
        best_overlap = 0.0
        candidate_index = segment_index
        while candidate_index < len(normalized_segments):
            segment_start, segment_end = normalized_segments[candidate_index]
            if segment_start >= word.end:
                break
            overlap = max(0.0, min(word.end, segment_end) - max(word.start, segment_start))
            if overlap > best_overlap:
                best_overlap = overlap
                best_index = candidate_index
            candidate_index += 1

        if best_index is None:
            continue
        segment_start, segment_end = normalized_segments[best_index]
        mapped_start = offsets[best_index] + max(word.start, segment_start) - segment_start
        mapped_end = offsets[best_index] + min(word.end, segment_end) - segment_start
        if mapped_end <= mapped_start:
            continue
        remapped.append(
            CaptionWord(
                start=mapped_start,
                end=mapped_end,
                text=word.text,
                display_text=word.display_text,
                is_profanity=word.is_profanity,
            )
        )
    return remapped


def _transcribe_words(
    input_path: Path,
    *,
    total_duration: float,
    settings: CaptionSettings,
    progress_callback: ProgressCallback | None,
    log_callback: LogCallback | None,
    cancel_callback: CancelCallback | None,
) -> list[CaptionWord]:
    if cancel_callback and cancel_callback():
        raise RuntimeError("Processing was cancelled.")

    if settings.transcription_provider.casefold() == "groq":
        return _transcribe_words_with_groq(
            input_path,
            total_duration=total_duration,
            settings=settings,
            progress_callback=progress_callback,
            log_callback=log_callback,
            cancel_callback=cancel_callback,
        )

    model_name = ENHANCED_POLISH_MODEL_NAME if settings.enhanced_polish_recognition else settings.model_name
    model, transcription_device = _load_model(model_name, log_callback)
    if log_callback:
        if settings.enhanced_polish_recognition:
            log_callback(
                "Transcribing with enhanced Polish recognition (Polish language, larger model, wider search)."
            )
        else:
            log_callback("Transcribing the kept speech for captions and profanity timing.")

    transcription_options: dict[str, object] = {
        "beam_size": 4,
        "best_of": 4,
        "word_timestamps": True,
        "vad_filter": True,
        "condition_on_previous_text": False,
        "temperature": 0.0,
    }
    if settings.enhanced_polish_recognition:
        transcription_options.update(
            language="pl",
            task="transcribe",
            beam_size=8,
            best_of=8,
            patience=1.2,
        )

    segments, _info = model.transcribe(str(input_path), **transcription_options)

    words: list[CaptionWord] = []
    try:
        segment_list = list(segments)
    except (OSError, RuntimeError, ValueError) as exc:
        if transcription_device != "cuda":
            raise
        _disable_whisper_gpu()
        if log_callback:
            log_callback(
                f"Warning: local Whisper could not transcribe on the GPU ({exc}); retrying automatically on the CPU."
            )
        model, _transcription_device = _load_model(model_name, log_callback, force_cpu=True)
        segments, _info = model.transcribe(str(input_path), **transcription_options)
        segment_list = list(segments)
    for segment in segment_list:
        if cancel_callback and cancel_callback():
            raise RuntimeError("Processing was cancelled.")

        for token in _extract_segment_words(segment):
            start = max(0.0, min(float(token.start), total_duration))
            end = max(start + 0.02, min(float(token.end), total_duration))
            text = token.text.strip()
            if not text:
                continue

            is_profanity = _is_profanity(text)
            display_text = _mask_profanity(text) if is_profanity and settings.profanity_filter else text
            words.append(
                CaptionWord(
                    start=start,
                    end=end,
                    text=text,
                    display_text=display_text,
                    is_profanity=is_profanity,
                )
            )

        if progress_callback:
            segment_end = max(0.0, min(float(getattr(segment, "end", 0.0) or 0.0), total_duration))
            progress_callback(segment_end / total_duration, "Transcribing speech for captions...")

    if progress_callback:
        progress_callback(1.0, "Transcribing speech for captions...")
    return [word for word in words if word.end > word.start]


def _transcribe_words_with_groq(
    input_path: Path,
    *,
    total_duration: float,
    settings: CaptionSettings,
    progress_callback: ProgressCallback | None,
    log_callback: LogCallback | None,
    cancel_callback: CancelCallback | None,
) -> list[CaptionWord]:
    if log_callback:
        log_callback(f"Uploading compressed audio to Groq ({settings.groq_model}) for transcription.")
    if progress_callback:
        progress_callback(0.12, "Uploading audio to Groq for transcription...")

    remote_words = transcribe_audio_words(
        input_path,
        api_key=settings.groq_api_key,
        model=settings.groq_model,
        language="pl" if settings.enhanced_polish_recognition else None,
        cancel_callback=cancel_callback,
    )
    words: list[CaptionWord] = []
    for remote_word in remote_words:
        start = max(0.0, min(remote_word.start, total_duration))
        end = max(start + 0.02, min(remote_word.end, total_duration))
        text = remote_word.text.strip()
        if not text:
            continue
        is_profanity = _is_profanity(text)
        display_text = _mask_profanity(text) if is_profanity and settings.profanity_filter else text
        words.append(CaptionWord(start, end, text, display_text, is_profanity))

    if progress_callback:
        progress_callback(1.0, "Transcribing speech with Groq...")
    if log_callback:
        log_callback(f"Groq returned {len(words)} timestamped word(s).")
    return [word for word in words if word.end > word.start]


def _prepare_transcription_audio(
    input_path: Path,
    *,
    output_dir: Path,
    total_duration: float,
    settings: CaptionSettings,
    progress_callback: ProgressCallback | None,
    log_callback: LogCallback | None,
) -> Path:
    use_groq = settings.transcription_provider.casefold() == "groq"
    audio_path = output_dir / ("captions_audio.flac" if use_groq else "captions_audio.wav")
    if log_callback:
        if use_groq:
            log_callback("Extracting compressed mono FLAC audio for Groq transcription.")
        else:
            log_callback("Extracting clean mono audio for more stable subtitle timing.")
    if progress_callback:
        progress_callback(0.03, "Preparing audio for captions...")

    command = [
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
        "flac" if use_groq else "pcm_s16le",
        str(audio_path),
    ]
    try:
        subprocess.run(command, check=True, **_subprocess_kwargs())
    except subprocess.CalledProcessError as exc:
        raise RuntimeError("Could not prepare audio for subtitles.") from exc

    if progress_callback:
        progress_callback(min(0.10, max(0.03, 2.0 / max(total_duration, 0.001))), "Preparing audio for captions...")
    return audio_path


def _extract_segment_words(segment: object) -> list[CaptionWord]:
    raw_words = getattr(segment, "words", None) or []
    extracted: list[CaptionWord] = []
    for word in raw_words:
        start = getattr(word, "start", None)
        end = getattr(word, "end", None)
        text = getattr(word, "word", None) or getattr(word, "text", None) or ""
        if start is None or end is None or not str(text).strip():
            continue
        extracted.append(
            CaptionWord(
                start=float(start),
                end=float(end),
                text=str(text).strip(),
                display_text=str(text).strip(),
                is_profanity=False,
            )
        )

    if extracted:
        return extracted

    segment_text = str(getattr(segment, "text", "") or "").strip()
    segment_start = float(getattr(segment, "start", 0.0) or 0.0)
    segment_end = float(getattr(segment, "end", segment_start) or segment_start)
    tokens = [token for token in segment_text.split() if token.strip()]
    if not tokens:
        return []

    duration = max(segment_end - segment_start, 0.01)
    slice_duration = duration / len(tokens)
    fallback: list[CaptionWord] = []
    for index, token in enumerate(tokens):
        start = segment_start + slice_duration * index
        end = segment_start + slice_duration * (index + 1)
        fallback.append(
            CaptionWord(
                start=start,
                end=end,
                text=token,
                display_text=token,
                is_profanity=False,
            )
        )
    return fallback


def _write_ass_subtitles(
    subtitle_path: Path,
    words: list[CaptionWord],
    *,
    video_width: int,
    video_height: int,
    settings: CaptionSettings,
) -> int:
    metrics = _build_caption_style_metrics(video_width=video_width, video_height=video_height)
    events = _build_caption_events(words, settings, metrics=metrics)
    subtitle_path.write_text(
        _build_ass_document(events, video_width=video_width, video_height=video_height, metrics=metrics),
        encoding="utf-8",
    )
    return len(events)


def _build_caption_events(
    words: list[CaptionWord],
    settings: CaptionSettings,
    *,
    metrics: _CaptionStyleMetrics | None = None,
) -> list[tuple[float, float, str]]:
    metrics = metrics or _build_caption_style_metrics(video_width=1080, video_height=1920)
    groups: list[list[CaptionWord]] = []
    current: list[CaptionWord] = []
    for word in words:
        if not current:
            current = [word]
            continue

        previous = current[-1]
        group_duration = word.end - current[0].start
        punctuation_break = previous.text.rstrip().endswith((".", "!", "?", ";", ":"))
        gap = max(0.0, word.start - previous.end)
        candidate = [*current, word]
        should_split = (
            len(current) >= max(settings.max_words_per_caption, 1)
            or group_duration > max(settings.max_caption_seconds, 0.1)
            or gap > max(settings.max_word_gap_seconds, 0.05)
            or punctuation_break
            or not _caption_group_fits(candidate, metrics)
        )
        if should_split:
            groups.append(current)
            current = [word]
        else:
            current.append(word)

    if current:
        groups.append(current)

    events: list[tuple[float, float, str]] = []
    for group_index, group in enumerate(groups):
        next_group_start = groups[group_index + 1][0].start if group_index + 1 < len(groups) else None
        for index, word in enumerate(group):
            start = word.start
            if index < len(group) - 1:
                end = max(word.end, group[index + 1].start)
            else:
                end = max(word.end, group[-1].end + 0.08)
                if next_group_start is not None:
                    end = min(end, max(word.end, next_group_start - 0.001))
            if end <= start:
                end = start + 0.01
            events.append((start, end, _format_event_text(group, index, metrics)))
    return _normalize_caption_events(events)


def _is_short_form(video_width: int, video_height: int) -> bool:
    """Portrait videos (TikTok, YouTube Shorts, Reels) are taller than they are wide."""
    return video_height > video_width


def _build_caption_style_metrics(*, video_width: int, video_height: int) -> _CaptionStyleMetrics:
    safe_width = max(int(video_width), 1)
    safe_height = max(int(video_height), 1)
    margin_h = max(18, min(int(round(safe_width * 0.08)), safe_width // 5))

    if _is_short_form(safe_width, safe_height):
        # Bottom-of-screen captions get covered by the TikTok/Shorts UI (like/share
        # buttons, native captions, etc.), so push these down from the vertical
        # center instead of hugging the bottom edge.
        margin_v = max(28, int(round(safe_height * 0.40)))
    else:
        margin_v = max(28, min(int(round(safe_height * 0.10)), safe_height // 5))

    usable_width = max(safe_width - (margin_h * 2), max(safe_width // 2, 1))

    base_font_size = int(round(safe_height * 0.055))
    width_limited_font_size = int(round(usable_width / 11.0))
    font_size = max(18, min(base_font_size, width_limited_font_size))
    active_font_size = max(font_size + 4, int(round(font_size * 1.10)))
    outline = max(2, int(round(font_size * 0.10)))

    # Captions are uppercased and bold, so allow more room than a typical
    # proportional-font average. The final formatter also measures each actual
    # line and scales exceptionally wide text to guarantee it stays in bounds.
    max_line_chars = max(7, int(usable_width / max(active_font_size * 0.70, 1.0)))
    return _CaptionStyleMetrics(
        font_size=font_size,
        active_font_size=active_font_size,
        outline=outline,
        margin_h=margin_h,
        margin_v=margin_v,
        max_line_chars=max_line_chars,
        usable_width=usable_width,
    )


def _caption_group_fits(words: list[CaptionWord], metrics: _CaptionStyleMetrics) -> bool:
    lines = _split_caption_lines([word.display_text for word in words], metrics.max_line_chars)
    return len(lines) <= 2


def _build_ass_document(
    events: list[tuple[float, float, str]],
    *,
    video_width: int,
    video_height: int,
    metrics: _CaptionStyleMetrics,
) -> str:
    lines = [
        "[Script Info]",
        "ScriptType: v4.00+",
        f"PlayResX: {video_width}",
        f"PlayResY: {video_height}",
        "WrapStyle: 2",
        "ScaledBorderAndShadow: yes",
        "",
        "[V4+ Styles]",
        "Format: Name,Fontname,Fontsize,PrimaryColour,SecondaryColour,OutlineColour,BackColour,"
        "Bold,Italic,Underline,StrikeOut,ScaleX,ScaleY,Spacing,Angle,BorderStyle,Outline,Shadow,"
        "Alignment,MarginL,MarginR,MarginV,Encoding",
        (
            "Style: TikTok,Arial,"
            f"{metrics.font_size},&H00FFFFFF,&H00FFFFFF,&H00000000,&H64000000,-1,0,0,0,100,100,0,0,1,"
            f"{metrics.outline},0,2,{metrics.margin_h},{metrics.margin_h},{metrics.margin_v},1"
        ),
        (
            "Style: TikTokActive,Arial,"
            f"{metrics.active_font_size},&H0000FFFF,&H0000FFFF,&H00000000,&H64000000,-1,0,0,0,100,100,0,0,1,"
            f"{metrics.outline},0,2,{metrics.margin_h},{metrics.margin_h},{metrics.margin_v},1"
        ),
        "",
        "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
    ]

    for start, end, text in events:
        lines.append(
            f"Dialogue: 0,{_format_ass_time_start(start)},{_format_ass_time_end(end)},TikTok,,0,0,0,,{{\\an2\\blur0.7}}{text}"
        )
    return "\n".join(lines) + "\n"


def _format_event_text(words: list[CaptionWord], active_index: int, metrics: _CaptionStyleMetrics) -> str:
    plain_words = [word.display_text.upper() for word in words]
    plain_lines = _split_caption_lines(plain_words, metrics.max_line_chars)
    horizontal_scale = _caption_horizontal_scale(plain_lines, metrics)
    scale_override = rf"\fscx{horizontal_scale}" if horizontal_scale < 100 else ""

    rendered_words: list[str] = []
    for index, word in enumerate(words):
        display = _escape_ass(word.display_text.upper())
        if index == active_index:
            rendered_words.append(
                r"{\rTikTokActive" + scale_override + "}" + display + r"{\rTikTok" + scale_override + "}"
            )
        else:
            rendered_words.append(display)

    lines = _split_caption_lines(rendered_words, metrics.max_line_chars)
    prefix = "{" + scale_override + "}" if scale_override else ""
    return prefix + r"\N".join(" ".join(line) for line in lines)


def _caption_horizontal_scale(lines: list[list[str]], metrics: _CaptionStyleMetrics) -> int:
    """Return an ASS ScaleX percentage that keeps every caption line on screen."""
    if not lines:
        return 100

    # Reserve room for the outline on both sides. Measure every word at the
    # highlighted font size so the caption does not change width as highlighting
    # advances from word to word.
    content_width = max(metrics.usable_width - (metrics.outline * 2) - 4, 1)
    widest_line = max(
        _estimate_caption_line_width(line, font_size=metrics.active_font_size)
        for line in lines
    )
    if widest_line <= content_width:
        return 100
    return max(1, min(100, int(math.floor((content_width / widest_line) * 100))))


def _estimate_caption_line_width(words: list[str], *, font_size: int) -> float:
    if not words:
        return 0.0

    # Approximate Arial Bold glyph advances. This deliberately errs wide; the
    # resulting ScaleX override is only used when a line is near the safe edge.
    width_units = sum(_estimate_caption_text_units(_strip_ass_overrides(word)) for word in words)
    width_units += max(len(words) - 1, 0) * 0.34
    return width_units * max(font_size, 1)


def _estimate_caption_text_units(text: str) -> float:
    units = 0.0
    for character in text:
        if character.isspace():
            units += 0.34
        elif character in "ilI.,'`!|:;":
            units += 0.34
        elif character in "MW@#%&QO0":
            units += 0.92
        elif ord(character) >= 0x2E80:
            units += 1.0
        elif character.isupper():
            units += 0.70
        elif character.isdigit():
            units += 0.62
        else:
            units += 0.58
    return units


def _split_caption_lines(words: list[str], max_line_chars: int) -> list[list[str]]:
    if not words:
        return []

    lines: list[list[str]] = [[]]
    current_length = 0
    for word in words:
        visible_length = _ass_visible_length(word)
        next_length = visible_length if not lines[-1] else current_length + 1 + visible_length
        if lines[-1] and next_length > max_line_chars and len(lines) < 2:
            lines.append([word])
            current_length = visible_length
        else:
            lines[-1].append(word)
            current_length = next_length
    return lines


def _ass_visible_length(text: str) -> int:
    return len(_strip_ass_overrides(text))


def _strip_ass_overrides(text: str) -> str:
    return re.sub(r"\{[^{}]*\}", "", text)


def _collect_profanity_intervals(words: list[CaptionWord]) -> list[tuple[float, float]]:
    intervals = [
        (max(0.0, word.start - 0.04), max(word.end + 0.04, word.start + 0.06))
        for word in words
        if word.is_profanity
    ]
    if not intervals:
        return []

    merged: list[tuple[float, float]] = [intervals[0]]
    for start, end in intervals[1:]:
        prev_start, prev_end = merged[-1]
        if start <= prev_end + 0.03:
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))
    return merged


def _is_profanity(text: str) -> bool:
    core = re.sub(r"[\W_]+", "", text.casefold(), flags=re.UNICODE)
    if not core:
        return False
    return any(core == prefix or core.startswith(prefix) for prefix in _PROFANITY_PREFIXES)


def _mask_profanity(text: str) -> str:
    match = re.match(r"^([^\w]*)([\w']+)([^\w]*)$", text, flags=re.UNICODE)
    if not match:
        return "***"
    prefix, core, suffix = match.groups()
    if len(core) <= 1:
        masked = "*"
    elif len(core) == 2:
        masked = core[0] + "*"
    else:
        masked = core[0] + "*" * (len(core) - 1)
    return prefix + masked + suffix


def _escape_ass(text: str) -> str:
    return text.replace("\\", r"\\").replace("{", r"\{").replace("}", r"\}")


def _normalize_caption_events(events: list[tuple[float, float, str]]) -> list[tuple[float, float, str]]:
    if not events:
        return []

    normalized: list[tuple[float, float, str]] = []
    previous_end = -1.0
    min_duration = 0.01

    for start, end, text in events:
        safe_start = max(start, previous_end if previous_end >= 0 else start)
        safe_end = max(end, safe_start + min_duration)
        normalized.append((safe_start, safe_end, text))
        previous_end = safe_end

    return normalized


def _format_ass_time_start(seconds: float) -> str:
    total_centiseconds = max(0, int(math.ceil(seconds * 100)))
    return _format_ass_centiseconds(total_centiseconds)


def _format_ass_time_end(seconds: float) -> str:
    total_centiseconds = max(0, int(math.floor(seconds * 100)))
    return _format_ass_centiseconds(total_centiseconds)


def _format_ass_centiseconds(total_centiseconds: int) -> str:
    centiseconds = total_centiseconds % 100
    total_seconds = total_centiseconds // 100
    seconds_part = total_seconds % 60
    total_minutes = total_seconds // 60
    minutes = total_minutes % 60
    hours = total_minutes // 60
    return f"{hours}:{minutes:02d}:{seconds_part:02d}.{centiseconds:02d}"


def _load_model(
    model_name: str,
    log_callback: LogCallback | None,
    *,
    force_cpu: bool = False,
) -> tuple[object, str]:
    device, compute_type = _preferred_whisper_backend(force_cpu=force_cpu)
    cache_key = (device, model_name)
    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key], device

    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:  # pragma: no cover - handled at runtime
        raise RuntimeError(
            "Subtitles and profanity filtering need the `faster-whisper` package. "
            "Install the updated requirements first."
        ) from exc

    if log_callback:
        log_callback(f"Loading speech-to-text model '{model_name}'. The first run may download it.")

    download_root = Path.home() / ".cache" / "speech_cutter" / "whisper"
    download_root.mkdir(parents=True, exist_ok=True)

    cpu_threads = max((os.cpu_count() or 4) - 1, 1)
    if device == "cuda":
        try:
            model = WhisperModel(
                model_name,
                device="cuda",
                device_index=0,
                compute_type=compute_type,
                download_root=str(download_root),
                cpu_threads=cpu_threads,
            )
            if log_callback:
                log_callback(f"Using the NVIDIA GPU for local Whisper ({compute_type}).")
        except (OSError, RuntimeError, ValueError) as exc:
            _disable_whisper_gpu()
            if log_callback:
                log_callback(
                    f"Warning: local Whisper could not start on the GPU ({exc}); using the CPU instead."
                )
            return _load_model(model_name, log_callback, force_cpu=True)
    else:
        try:
            model = WhisperModel(
                model_name,
                device="cpu",
                compute_type="int8",
                download_root=str(download_root),
                cpu_threads=cpu_threads,
            )
        except ValueError:
            model = WhisperModel(
                model_name,
                device="cpu",
                compute_type="float32",
                download_root=str(download_root),
                cpu_threads=cpu_threads,
            )
    _MODEL_CACHE[cache_key] = model
    return model, device


def _preferred_whisper_backend(*, force_cpu: bool = False) -> tuple[str, str]:
    if force_cpu or _WHISPER_GPU_DISABLED:
        return "cpu", "int8"
    try:
        import ctranslate2

        if ctranslate2.get_cuda_device_count() < 1:
            return "cpu", "int8"
        supported_types = ctranslate2.get_supported_compute_types("cuda", device_index=0)
    except (ImportError, OSError, RuntimeError, ValueError):
        return "cpu", "int8"

    for compute_type in ("float16", "int8_float16", "int8_float32", "float32", "int8"):
        if compute_type in supported_types:
            return "cuda", compute_type
    return "cpu", "int8"


def _disable_whisper_gpu() -> None:
    global _WHISPER_GPU_DISABLED
    _WHISPER_GPU_DISABLED = True


def _subprocess_kwargs() -> dict[str, object]:
    kwargs: dict[str, object] = {}
    if os.name == "nt":
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        kwargs["startupinfo"] = startupinfo
        kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
    return kwargs
