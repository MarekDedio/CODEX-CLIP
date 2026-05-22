from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from queue import Empty, Queue
from tkinter import filedialog, messagebox, scrolledtext, ttk
import math
import os
import subprocess
import tempfile
import threading
import tkinter as tk

from .captions import CaptionSettings
from .pipeline import (
    CropSettings,
    DEFAULT_MID_ZOOM_MIN_SEGMENT_SECONDS,
    DEFAULT_PUSH_IN_MAX_ZOOM,
    DEFAULT_PUSH_IN_MIN_SEGMENT_SECONDS,
    MotionSettings,
    NoSpeechDetectedError,
    ProcessingOptions,
    ProcessingResult,
    UserCancelledError,
    build_output_path,
    check_runtime,
    probe_video_metadata,
    process_video,
)
from .presets import PRESETS, build_options


NATURAL_SETTINGS = PRESETS["natural"].settings
PREVIEW_FRAME_TIMEOUT_SECONDS = 20.0
APP_BG = "#0f172a"
APP_PANEL = "#111827"
CARD_BG = "#f8fafc"
CARD_INNER_BG = "#f8fafc"
TEXT_DARK = "#0f172a"
TEXT_MUTED = "#526070"
TEXT_LIGHT = "#e5eefb"
TEXT_LIGHT_MUTED = "#94a3b8"
ACCENT = "#2563eb"
ACCENT_HOVER = "#1d4ed8"
ACCENT_SOFT = "#dbeafe"
BORDER = "#d7dde8"
DANGER = "#dc2626"


@dataclass(frozen=True)
class QueueEvent:
    kind: str
    payload: object


@dataclass
class QueuedVideo:
    path: Path
    metadata: dict[str, int | float | bool | None] = field(default_factory=dict)
    crop_rect: tuple[int, int, int, int] | None = None


@dataclass(frozen=True)
class ProcessingJob:
    input_path: Path
    output_path: Path
    options: ProcessingOptions


@dataclass(frozen=True)
class BatchFailure:
    input_path: Path
    message: str


@dataclass(frozen=True)
class BatchRunResult:
    results: list[ProcessingResult]
    failures: list[BatchFailure]


class PreviewLoadingDialog(tk.Toplevel):
    def __init__(self, parent: tk.Misc, *, video_name: str) -> None:
        super().__init__(parent)
        self.title("Loading Preview")
        self.configure(bg=CARD_BG)
        self.resizable(False, False)
        self.transient(parent)

        frame = ttk.Frame(self, style="Card.TFrame", padding=20)
        frame.pack(fill="both", expand=True)

        ttk.Label(
            frame,
            text=f"Loading crop preview for:\n{video_name}",
            justify="left",
            wraplength=420,
            style="Section.TLabel",
        ).pack(anchor="w")
        ttk.Label(
            frame,
            text="The app is still working. Some videos just take longer to open a frame preview.",
            wraplength=420,
            style="Body.TLabel",
        ).pack(anchor="w", pady=(10, 0))

        self._progress = ttk.Progressbar(frame, mode="indeterminate", length=360, style="Accent.Horizontal.TProgressbar")
        self._progress.pack(fill="x", pady=(14, 0))
        self._progress.start(10)

        self.protocol("WM_DELETE_WINDOW", lambda: None)
        self.update_idletasks()
        self.after_idle(self._activate_dialog)

    def destroy(self) -> None:
        try:
            self._progress.stop()
        except Exception:
            pass
        super().destroy()

    def _activate_dialog(self) -> None:
        try:
            self.grab_set()
            self.lift()
            self.focus_force()
        except tk.TclError:
            pass


def _extract_preview_frame(
    *,
    input_path: Path,
    source_duration: float,
    output_dir: Path,
) -> Path:
    preview_candidates: list[float] = []
    quarter_point = min(max(source_duration * 0.25, 0.0), max(source_duration - 0.1, 0.0))
    preview_candidates.append(round(quarter_point, 3))
    preview_candidates.append(round(min(max(source_duration * 0.05, 0.0), max(source_duration - 0.1, 0.0)), 3))
    preview_candidates.extend([1.0, 0.0])

    unique_candidates: list[float] = []
    for candidate in preview_candidates:
        if candidate not in unique_candidates:
            unique_candidates.append(candidate)

    last_error: Exception | None = None
    for index, preview_time in enumerate(unique_candidates):
        preview_path = output_dir / f"preview_{index}.png"
        command = [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-ss",
            f"{preview_time:.3f}",
            "-i",
            str(input_path),
            "-frames:v",
            "1",
            "-vf",
            "scale=960:540:force_original_aspect_ratio=decrease",
            str(preview_path),
        ]
        try:
            subprocess.run(
                command,
                check=True,
                timeout=PREVIEW_FRAME_TIMEOUT_SECONDS,
                **_subprocess_kwargs(),
            )
        except subprocess.TimeoutExpired as exc:
            last_error = RuntimeError(
                f"Timed out while loading a frame preview for {input_path.name}."
            )
            last_error.__cause__ = exc
            continue
        except subprocess.CalledProcessError as exc:
            last_error = RuntimeError(
                f"Could not load a frame preview for {input_path.name}."
            )
            last_error.__cause__ = exc
            continue

        if preview_path.exists():
            return preview_path

    if last_error is not None:
        raise last_error
    raise RuntimeError(f"Could not load a frame preview for {input_path.name}.")


class CropSelectionDialog(tk.Toplevel):
    def __init__(
        self,
        parent: tk.Misc,
        *,
        preview_path: Path,
        preview_temp_dir: tempfile.TemporaryDirectory | None,
        source_width: int,
        source_height: int,
        initial_crop: tuple[int, int, int, int] | None,
    ) -> None:
        super().__init__(parent)
        self.title("Choose Crop")
        self.configure(bg=CARD_BG)
        self.resizable(False, False)
        self.transient(parent)

        self.result: tuple[int, int, int, int] | None = None
        self._preview_temp_dir = preview_temp_dir
        self._source_width = source_width
        self._source_height = source_height
        self._aspect_gcd = max(math.gcd(self._source_width, self._source_height), 1)
        self._aspect_width_unit = self._source_width // self._aspect_gcd
        self._aspect_height_unit = self._source_height // self._aspect_gcd
        self._aspect_step_units = 2 if (self._aspect_width_unit % 2 == 1 or self._aspect_height_unit % 2 == 1) else 1
        self._image = tk.PhotoImage(file=str(preview_path))
        self._preview_width = self._image.width()
        self._preview_height = self._image.height()

        self._start_x = 0
        self._start_y = 0
        self._current_rect: tuple[int, int, int, int] | None = None
        self._rect_id: int | None = None
        self._overlay_id: int | None = None

        self.summary_var = tk.StringVar(value=self._default_summary_message())

        self._build_layout()
        self._bind_events()

        if initial_crop:
            self._current_rect = self._source_crop_to_preview(initial_crop)
            self._redraw_selection()

        self.protocol("WM_DELETE_WINDOW", self._cancel)
        self.update_idletasks()
        self.after_idle(self._activate_dialog)

    def destroy(self) -> None:
        try:
            if self._preview_temp_dir is not None:
                self._preview_temp_dir.cleanup()
        except Exception:
            pass
        super().destroy()

    def _activate_dialog(self) -> None:
        try:
            self.grab_set()
            self.lift()
            self.focus_force()
        except tk.TclError:
            pass

    def _build_layout(self) -> None:
        frame = ttk.Frame(self, style="Card.TFrame", padding=18)
        frame.pack(fill="both", expand=True)

        ttk.Label(
            frame,
            text=(
                "Drag a box over the part of the frame you want to zoom into. "
                f"The crop is locked to the original aspect ratio ({self._aspect_width_unit}:{self._aspect_height_unit})."
            ),
            wraplength=860,
            style="Body.TLabel",
        ).pack(anchor="w")

        self.canvas = tk.Canvas(
            frame,
            width=self._preview_width,
            height=self._preview_height,
            highlightthickness=1,
            highlightbackground=BORDER,
            bg="#000000",
            cursor="crosshair",
        )
        self.canvas.pack(pady=(12, 10))
        self.canvas.create_image(0, 0, image=self._image, anchor="nw")
        self._overlay_id = self.canvas.create_rectangle(
            0,
            0,
            self._preview_width,
            self._preview_height,
            outline="",
            fill="",
        )

        ttk.Label(frame, textvariable=self.summary_var, wraplength=860, style="Body.TLabel").pack(anchor="w")

        buttons = ttk.Frame(frame, style="CardInner.TFrame")
        buttons.pack(fill="x", pady=(12, 0))

        ttk.Button(buttons, text="Clear Selection", command=self._clear_selection).pack(side="left")
        ttk.Button(buttons, text="Cancel", style="Danger.TButton", command=self._cancel).pack(side="right")
        ttk.Button(buttons, text="Use This Crop", style="Primary.TButton", command=self._apply).pack(side="right", padx=(0, 8))

    def _bind_events(self) -> None:
        self.canvas.bind("<ButtonPress-1>", self._on_press)
        self.canvas.bind("<B1-Motion>", self._on_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_release)

    def _on_press(self, event: tk.Event[tk.Canvas]) -> None:
        self._start_x, self._start_y = self._clamp_preview_point(event.x, event.y)
        self._current_rect = (self._start_x, self._start_y, self._start_x, self._start_y)
        self._redraw_selection()

    def _on_drag(self, event: tk.Event[tk.Canvas]) -> None:
        if self._current_rect is None:
            return
        end_x, end_y = self._clamp_preview_point(event.x, event.y)
        self._current_rect = self._build_locked_preview_rect(end_x, end_y)
        self._redraw_selection()

    def _on_release(self, event: tk.Event[tk.Canvas]) -> None:
        if self._current_rect is None:
            return
        end_x, end_y = self._clamp_preview_point(event.x, event.y)
        self._current_rect = self._build_locked_preview_rect(end_x, end_y)
        self._redraw_selection()

    def _redraw_selection(self) -> None:
        if self._rect_id is not None:
            self.canvas.delete(self._rect_id)

        if not self._current_rect:
            self.summary_var.set(self._default_summary_message())
            return

        x0, y0, x1, y1 = self._normalize_preview_rect(self._current_rect)
        self._rect_id = self.canvas.create_rectangle(
            x0,
            y0,
            x1,
            y1,
            outline="#ffb347",
            width=2,
        )

        crop = self._preview_to_source_crop((x0, y0, x1, y1))
        if crop is None:
            self.summary_var.set(
                f"Selection is too small. Drag a larger {self._aspect_width_unit}:{self._aspect_height_unit} box."
            )
            return

        x, y, width, height = crop
        self.summary_var.set(
            f"Crop selected: x={x}, y={y}, width={width}, height={height} "
            f"(locked to {self._aspect_width_unit}:{self._aspect_height_unit})."
        )

    def _clear_selection(self) -> None:
        self._current_rect = None
        if self._rect_id is not None:
            self.canvas.delete(self._rect_id)
            self._rect_id = None
        self.summary_var.set(self._default_summary_message())

    def _apply(self) -> None:
        if not self._current_rect:
            messagebox.showerror("Missing crop", "Draw a crop box first.", parent=self)
            return

        crop = self._preview_to_source_crop(self._normalize_preview_rect(self._current_rect))
        if crop is None:
            messagebox.showerror("Invalid crop", "The crop box is too small.", parent=self)
            return

        self.result = crop
        self.destroy()

    def _cancel(self) -> None:
        self.result = None
        self.destroy()

    def _source_crop_to_preview(self, crop: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
        x, y, width, height = crop
        scale_x = self._preview_width / self._source_width
        scale_y = self._preview_height / self._source_height
        return (
            int(round(x * scale_x)),
            int(round(y * scale_y)),
            int(round((x + width) * scale_x)),
            int(round((y + height) * scale_y)),
        )

    def _build_locked_preview_rect(self, end_x: int, end_y: int) -> tuple[int, int, int, int]:
        delta_x = end_x - self._start_x
        delta_y = end_y - self._start_y
        if delta_x == 0 or delta_y == 0:
            return self._start_x, self._start_y, self._start_x, self._start_y

        direction_x = 1 if delta_x >= 0 else -1
        direction_y = 1 if delta_y >= 0 else -1
        available_width = self._preview_width - self._start_x if direction_x > 0 else self._start_x
        available_height = self._preview_height - self._start_y if direction_y > 0 else self._start_y
        max_width = min(abs(delta_x), available_width)
        max_height = min(abs(delta_y), available_height)
        if max_width <= 0 or max_height <= 0:
            return self._start_x, self._start_y, self._start_x, self._start_y

        scale = min(
            max_width / max(self._preview_width, 1),
            max_height / max(self._preview_height, 1),
        )
        locked_width = int(math.floor(self._preview_width * scale))
        locked_height = int(math.floor(self._preview_height * scale))
        if locked_width <= 0 or locked_height <= 0:
            return self._start_x, self._start_y, self._start_x, self._start_y

        return (
            self._start_x,
            self._start_y,
            self._start_x + direction_x * locked_width,
            self._start_y + direction_y * locked_height,
        )

    def _preview_to_source_crop(
        self,
        rect: tuple[int, int, int, int],
    ) -> tuple[int, int, int, int] | None:
        x0, y0, x1, y1 = rect
        if x1 - x0 < 6 or y1 - y0 < 6:
            return None

        scale_x = self._source_width / self._preview_width
        scale_y = self._source_height / self._preview_height

        left = max(0, int(math.floor(x0 * scale_x)))
        top = max(0, int(math.floor(y0 * scale_y)))
        right = min(self._source_width, int(math.ceil(x1 * scale_x)))
        bottom = min(self._source_height, int(math.ceil(y1 * scale_y)))

        rough_width = right - left
        rough_height = bottom - top
        if rough_width < 2 or rough_height < 2:
            return None

        scale_units = int(
            math.floor(
                min(
                    rough_width / max(self._aspect_width_unit, 1),
                    rough_height / max(self._aspect_height_unit, 1),
                )
            )
        )
        if self._aspect_step_units > 1:
            scale_units -= scale_units % self._aspect_step_units
        if scale_units < max(self._aspect_step_units, 1):
            return None

        width = self._aspect_width_unit * scale_units
        height = self._aspect_height_unit * scale_units

        if left + width > self._source_width:
            left = max(0, self._source_width - width)
        if top + height > self._source_height:
            top = max(0, self._source_height - height)

        return left, top, width, height

    def _default_summary_message(self) -> str:
        return f"No crop selected. Aspect is locked to {self._aspect_width_unit}:{self._aspect_height_unit}."

    def _normalize_preview_rect(self, rect: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
        x0, y0, x1, y1 = rect
        left, right = sorted((x0, x1))
        top, bottom = sorted((y0, y1))
        return left, top, right, bottom

    def _clamp_preview_point(self, x: int, y: int) -> tuple[int, int]:
        return max(0, min(x, self._preview_width)), max(0, min(y, self._preview_height))

class SpeechCutterApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Speech Cutter")
        self.geometry("1120x820")
        self.minsize(980, 700)
        self.configure(bg=APP_BG)

        self._worker: threading.Thread | None = None
        self._cancel_event = threading.Event()
        self._events: Queue[QueueEvent] = Queue()
        self._last_result: ProcessingResult | None = None
        self._crop_widgets: list[tk.Widget] = []
        self._queue_refreshing = False
        self._queued_videos: list[QueuedVideo] = []
        self._selected_queue_index = 0
        self._source_path: Path | None = None
        self._source_metadata: dict[str, int | float | bool | None] = {}
        self._crop_rect: tuple[int, int, int, int] | None = None

        self.input_var = tk.StringVar()
        self.queue_enabled_var = tk.BooleanVar(value=False)
        self.queue_summary_var = tk.StringVar(value="Queue mode is off.")
        self.source_info_var = tk.StringVar(value="Pick a video to load its source size.")
        self.crop_summary_var = tk.StringVar(value="No crop selected.")
        self.status_var = tk.StringVar(value="Pick a video and press Start.")
        self.progress_var = tk.DoubleVar(value=0.0)
        self.padding_seconds_var = tk.DoubleVar(value=NATURAL_SETTINGS.speech_pad_ms / 1000)
        self.padding_value_var = tk.StringVar()

        self.crop_enabled_var = tk.BooleanVar(value=True)
        self.crop_every_var = tk.DoubleVar(value=2.0)
        self.crop_every_value_var = tk.StringVar()
        self.crop_min_seconds_var = tk.DoubleVar(value=1.2)
        self.crop_min_seconds_value_var = tk.StringVar()
        self.crop_zoom_seconds_var = tk.DoubleVar(value=2.0)
        self.crop_zoom_seconds_value_var = tk.StringVar()
        self.crop_middle_long_var = tk.BooleanVar(value=True)
        self.crop_middle_long_threshold_var = tk.DoubleVar(value=DEFAULT_MID_ZOOM_MIN_SEGMENT_SECONDS)
        self.crop_middle_long_threshold_value_var = tk.StringVar()
        self.push_in_enabled_var = tk.BooleanVar(value=True)
        self.push_in_min_seconds_var = tk.DoubleVar(value=DEFAULT_PUSH_IN_MIN_SEGMENT_SECONDS)
        self.push_in_min_seconds_value_var = tk.StringVar()
        self.push_in_zoom_percent_var = tk.DoubleVar(value=(DEFAULT_PUSH_IN_MAX_ZOOM - 1.0) * 100.0)
        self.push_in_zoom_percent_value_var = tk.StringVar()
        self.captions_enabled_var = tk.BooleanVar(value=True)
        self.profanity_filter_var = tk.BooleanVar(value=True)

        self.padding_seconds_var.trace_add("write", self._refresh_setting_labels)
        self.crop_every_var.trace_add("write", self._refresh_setting_labels)
        self.crop_min_seconds_var.trace_add("write", self._refresh_setting_labels)
        self.crop_zoom_seconds_var.trace_add("write", self._refresh_setting_labels)
        self.crop_middle_long_threshold_var.trace_add("write", self._refresh_setting_labels)
        self.push_in_min_seconds_var.trace_add("write", self._refresh_setting_labels)
        self.push_in_zoom_percent_var.trace_add("write", self._refresh_setting_labels)

        self._configure_styles()
        self._refresh_setting_labels()
        self._build_layout()
        self._update_queue_mode_ui()
        self._update_crop_state()
        self.after(100, self._drain_queue)

        issues = check_runtime()
        if issues:
            messagebox.showwarning(
                "Missing dependencies",
                "Some requirements are missing:\n\n" + "\n".join(f"- {issue}" for issue in issues),
            )

    def _configure_styles(self) -> None:
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure(".", font=("Segoe UI", 10), borderwidth=0, focusthickness=0)
        style.configure("Root.TFrame", background=APP_BG)
        style.configure("AppPanel.TFrame", background=APP_PANEL)
        style.configure("Card.TFrame", background=CARD_BG, relief="flat", borderwidth=1)
        style.configure("CardInner.TFrame", background=CARD_INNER_BG)
        style.configure("Title.TLabel", background=APP_BG, foreground=TEXT_LIGHT, font=("Segoe UI Semibold", 28))
        style.configure("Subtitle.TLabel", background=APP_BG, foreground=TEXT_LIGHT_MUTED, font=("Segoe UI", 11))
        style.configure("Section.TLabel", background=CARD_BG, foreground=TEXT_DARK, font=("Segoe UI Semibold", 11))
        style.configure("Body.TLabel", background=CARD_BG, foreground=TEXT_MUTED, font=("Segoe UI", 10))
        style.configure("Status.TLabel", background=CARD_BG, foreground=TEXT_DARK, font=("Segoe UI Semibold", 10))
        style.configure(
            "TButton",
            font=("Segoe UI Semibold", 10),
            padding=(14, 8),
            background="#e8edf5",
            foreground=TEXT_DARK,
            bordercolor=BORDER,
            lightcolor="#e8edf5",
            darkcolor="#e8edf5",
        )
        style.map(
            "TButton",
            background=[("active", "#dce5f1"), ("disabled", "#edf1f7")],
            foreground=[("disabled", "#94a3b8")],
        )
        style.configure(
            "Primary.TButton",
            font=("Segoe UI Semibold", 10),
            padding=(18, 9),
            background=ACCENT,
            foreground="#ffffff",
            bordercolor=ACCENT,
            lightcolor=ACCENT,
            darkcolor=ACCENT,
        )
        style.map(
            "Primary.TButton",
            background=[("active", ACCENT_HOVER), ("disabled", "#93c5fd")],
            foreground=[("disabled", "#eff6ff")],
        )
        style.configure(
            "Danger.TButton",
            font=("Segoe UI Semibold", 10),
            padding=(14, 8),
            background="#fee2e2",
            foreground=DANGER,
            bordercolor="#fecaca",
            lightcolor="#fee2e2",
            darkcolor="#fee2e2",
        )
        style.map("Danger.TButton", background=[("active", "#fecaca"), ("disabled", "#f1f5f9")])
        style.configure(
            "TEntry",
            padding=(10, 8),
            fieldbackground="#ffffff",
            bordercolor=BORDER,
            lightcolor=BORDER,
            darkcolor=BORDER,
            foreground=TEXT_DARK,
        )
        style.configure("TCheckbutton", background=CARD_BG, foreground=TEXT_DARK, font=("Segoe UI", 10))
        style.map("TCheckbutton", background=[("active", CARD_BG)], foreground=[("disabled", "#94a3b8")])
        style.configure(
            "Accent.Horizontal.TProgressbar",
            troughcolor="#e2e8f0",
            background=ACCENT,
            bordercolor="#e2e8f0",
            lightcolor=ACCENT,
            darkcolor=ACCENT,
        )
        style.configure(
            "Accent.Horizontal.TScale",
            background=CARD_BG,
            troughcolor=ACCENT_SOFT,
            bordercolor=ACCENT_SOFT,
            lightcolor=ACCENT,
            darkcolor=ACCENT,
        )

    def _refresh_setting_labels(self, *_args: object) -> None:
        self.padding_value_var.set(f"{float(self.padding_seconds_var.get()):.2f}s")
        self.crop_every_value_var.set(str(max(1, int(round(float(self.crop_every_var.get()))))))
        self.crop_min_seconds_value_var.set(f"{float(self.crop_min_seconds_var.get()):.1f}s")
        zoom_seconds = float(self.crop_zoom_seconds_var.get())
        self.crop_zoom_seconds_value_var.set("Full segment" if zoom_seconds <= 0.0 else f"{zoom_seconds:.1f}s")
        self.crop_middle_long_threshold_value_var.set(f"{float(self.crop_middle_long_threshold_var.get()):.1f}s")
        self.push_in_min_seconds_value_var.set(f"{float(self.push_in_min_seconds_var.get()):.1f}s")
        self.push_in_zoom_percent_value_var.set(f"{float(self.push_in_zoom_percent_var.get()):.0f}%")

    def _on_padding_scale(self, value: str) -> None:
        snapped = min(3.0, max(0.0, round(float(value) / 0.05) * 0.05))
        snapped = round(snapped, 2)
        if abs(float(self.padding_seconds_var.get()) - snapped) > 1e-9:
            self.padding_seconds_var.set(snapped)

    def _on_crop_every_scale(self, value: str) -> None:
        snapped = max(1, min(12, int(round(float(value)))))
        if max(1, int(round(float(self.crop_every_var.get())))) != snapped:
            self.crop_every_var.set(float(snapped))

    def _on_crop_min_scale(self, value: str) -> None:
        snapped = min(10.0, max(0.0, round(float(value) / 0.1) * 0.1))
        snapped = round(snapped, 1)
        if abs(float(self.crop_min_seconds_var.get()) - snapped) > 1e-9:
            self.crop_min_seconds_var.set(snapped)

    def _on_crop_zoom_scale(self, value: str) -> None:
        snapped = min(10.0, max(0.0, round(float(value) / 0.1) * 0.1))
        snapped = round(snapped, 1)
        if abs(float(self.crop_zoom_seconds_var.get()) - snapped) > 1e-9:
            self.crop_zoom_seconds_var.set(snapped)

    def _on_crop_middle_long_threshold_scale(self, value: str) -> None:
        snapped = min(30.0, max(4.0, round(float(value) / 0.5) * 0.5))
        snapped = round(snapped, 1)
        if abs(float(self.crop_middle_long_threshold_var.get()) - snapped) > 1e-9:
            self.crop_middle_long_threshold_var.set(snapped)

    def _on_push_in_min_scale(self, value: str) -> None:
        snapped = min(30.0, max(1.0, round(float(value) / 0.5) * 0.5))
        snapped = round(snapped, 1)
        if abs(float(self.push_in_min_seconds_var.get()) - snapped) > 1e-9:
            self.push_in_min_seconds_var.set(snapped)

    def _on_push_in_zoom_scale(self, value: str) -> None:
        snapped = min(25.0, max(2.0, round(float(value))))
        if abs(float(self.push_in_zoom_percent_var.get()) - snapped) > 1e-9:
            self.push_in_zoom_percent_var.set(snapped)

    def _on_scroll_frame_configure(self, _event: tk.Event[tk.Widget] | None = None) -> None:
        self._scroll_canvas.configure(scrollregion=self._scroll_canvas.bbox("all"))

    def _on_scroll_canvas_configure(self, event: tk.Event[tk.Canvas]) -> None:
        self._scroll_canvas.itemconfigure(self._scroll_window_id, width=event.width)

    def _bind_mousewheel(self, _event: tk.Event[tk.Widget] | None = None) -> None:
        self.bind_all("<MouseWheel>", self._on_mousewheel)

    def _unbind_mousewheel(self, _event: tk.Event[tk.Widget] | None = None) -> None:
        self.unbind_all("<MouseWheel>")

    def _on_mousewheel(self, event: tk.Event[tk.Widget]) -> None:
        if event.delta == 0:
            return
        self._scroll_canvas.yview_scroll(-int(event.delta / 120), "units")

    def _build_layout(self) -> None:
        shell = ttk.Frame(self, style="Root.TFrame")
        shell.pack(fill="both", expand=True)
        shell.columnconfigure(0, weight=1)
        shell.rowconfigure(0, weight=1)

        self._scroll_canvas = tk.Canvas(
            shell,
            bg=APP_BG,
            highlightthickness=0,
            borderwidth=0,
        )
        self._scroll_canvas.grid(row=0, column=0, sticky="nsew")
        self._scrollbar = ttk.Scrollbar(shell, orient="vertical", command=self._scroll_canvas.yview)
        self._scrollbar.grid(row=0, column=1, sticky="ns")
        self._scroll_canvas.configure(yscrollcommand=self._scrollbar.set)

        root = ttk.Frame(self._scroll_canvas, style="Root.TFrame", padding=24)
        self._scroll_window_id = self._scroll_canvas.create_window((0, 0), window=root, anchor="nw")
        root.bind("<Configure>", self._on_scroll_frame_configure)
        self._scroll_canvas.bind("<Configure>", self._on_scroll_canvas_configure)
        root.bind("<Enter>", self._bind_mousewheel)
        root.bind("<Leave>", self._unbind_mousewheel)
        self._scroll_canvas.bind("<Enter>", self._bind_mousewheel)
        self._scroll_canvas.bind("<Leave>", self._unbind_mousewheel)

        root.columnconfigure(0, weight=1, uniform="main")
        root.columnconfigure(1, weight=1, uniform="main")
        root.rowconfigure(4, weight=1)

        hero = ttk.Frame(root, style="Root.TFrame")
        hero.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 18))
        ttk.Label(hero, text="Speech Cutter", style="Title.TLabel").pack(anchor="w")
        ttk.Label(
            hero,
            text="Trim speech, add captions, apply crop zooms, and give longer clips a subtle push-in.",
            style="Subtitle.TLabel",
        ).pack(anchor="w", pady=(4, 0))

        files_card = ttk.Frame(root, style="Card.TFrame", padding=20)
        files_card.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(0, 16))
        files_card.columnconfigure(1, weight=1)

        ttk.Label(files_card, text="Source video", style="Section.TLabel").grid(row=0, column=0, sticky="w")
        self.input_entry = ttk.Entry(files_card, textvariable=self.input_var)
        self.input_entry.grid(row=0, column=1, sticky="ew", padx=10)
        self.browse_button = ttk.Button(files_card, text="Browse...", command=self._browse_input)
        self.browse_button.grid(row=0, column=2, padx=(2, 0))
        self.queue_mode_check = ttk.Checkbutton(
            files_card,
            text="Queue multiple videos and process them one by one",
            variable=self.queue_enabled_var,
            command=self._toggle_queue_mode,
        )
        self.queue_mode_check.grid(row=1, column=1, columnspan=2, sticky="w", padx=10, pady=(10, 0))

        self.queue_frame = ttk.Frame(files_card, style="CardInner.TFrame")
        self.queue_frame.grid(row=2, column=1, columnspan=2, sticky="ew", padx=10, pady=(10, 0))
        self.queue_frame.columnconfigure(0, weight=1)

        queue_header = ttk.Frame(self.queue_frame, style="CardInner.TFrame")
        queue_header.grid(row=0, column=0, sticky="ew")
        queue_header.columnconfigure(0, weight=1)
        ttk.Label(queue_header, text="Queued Videos", style="Section.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(queue_header, textvariable=self.queue_summary_var, style="Body.TLabel").grid(row=0, column=1, sticky="e")

        self.queue_listbox = tk.Listbox(
            self.queue_frame,
            height=5,
            activestyle="none",
            exportselection=False,
            relief="flat",
            bg="#ffffff",
            fg=TEXT_DARK,
            highlightthickness=1,
            highlightbackground=BORDER,
            highlightcolor=ACCENT,
            selectbackground=ACCENT_SOFT,
            selectforeground=TEXT_DARK,
            font=("Segoe UI", 10),
            borderwidth=0,
        )
        self.queue_listbox.grid(row=1, column=0, sticky="ew", pady=(10, 0))
        self.queue_listbox.bind("<<ListboxSelect>>", self._on_queue_selection)

        queue_actions = ttk.Frame(self.queue_frame, style="CardInner.TFrame")
        queue_actions.grid(row=2, column=0, sticky="w", pady=(10, 0))
        self.remove_queue_button = ttk.Button(queue_actions, text="Remove Selected", command=self._remove_selected_video)
        self.remove_queue_button.grid(row=0, column=0, sticky="w")
        self.clear_queue_button = ttk.Button(queue_actions, text="Clear Queue", command=self._clear_queue)
        self.clear_queue_button.grid(row=0, column=1, sticky="w", padx=(10, 0))
        ttk.Label(files_card, textvariable=self.source_info_var, style="Body.TLabel", wraplength=920).grid(
            row=3,
            column=1,
            columnspan=2,
            sticky="w",
            padx=10,
            pady=(10, 0),
        )

        trim_card = ttk.Frame(root, style="Card.TFrame", padding=20)
        trim_card.grid(row=2, column=0, sticky="new", padx=(0, 8), pady=(0, 16))
        trim_card.columnconfigure(0, weight=1)

        trim_header = ttk.Frame(trim_card, style="CardInner.TFrame")
        trim_header.grid(row=0, column=0, sticky="ew")
        trim_header.columnconfigure(0, weight=1)
        ttk.Label(trim_header, text="Keep Around Speech (seconds)", style="Section.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(trim_header, textvariable=self.padding_value_var, style="Body.TLabel").grid(row=0, column=1, sticky="e")

        self.padding_scale = ttk.Scale(
            trim_card,
            from_=0.0,
            to=3.0,
            variable=self.padding_seconds_var,
            command=self._on_padding_scale,
            style="Accent.Horizontal.TScale",
        )
        self.padding_scale.grid(row=1, column=0, sticky="ew", pady=(8, 0))
        ttk.Label(
            trim_card,
            text="Natural trimming is always used. This setting keeps a little extra time before and after speech so cuts feel less abrupt.",
            style="Body.TLabel",
            wraplength=470,
        ).grid(row=2, column=0, sticky="w", pady=(12, 0))

        crop_card = ttk.Frame(root, style="Card.TFrame", padding=20)
        crop_card.grid(row=3, column=0, sticky="new", padx=(0, 8), pady=(0, 16))
        crop_card.columnconfigure(2, weight=1)

        ttk.Checkbutton(
            crop_card,
            text="Enable crop / zoom on selected segments",
            variable=self.crop_enabled_var,
            command=self._update_crop_state,
        ).grid(row=0, column=0, columnspan=3, sticky="w")

        crop_actions = ttk.Frame(crop_card, style="CardInner.TFrame")
        crop_actions.grid(row=1, column=0, columnspan=2, sticky="w", pady=(12, 0))
        self.pick_crop_button = ttk.Button(crop_actions, text="Choose Crop Visually...", command=self._choose_crop_visually)
        self.pick_crop_button.grid(row=0, column=0, sticky="w")
        self.pick_queue_crops_button = ttk.Button(
            crop_actions,
            text="Choose Crops For Queue...",
            command=self._choose_crops_for_queue,
        )
        self.pick_queue_crops_button.grid(row=0, column=1, sticky="w", padx=(10, 0))
        self.clear_crop_button = ttk.Button(crop_actions, text="Clear Crop", command=self._clear_crop_selection)
        self.clear_crop_button.grid(row=1, column=0, sticky="w", pady=(10, 0))
        ttk.Label(crop_card, textvariable=self.crop_summary_var, style="Body.TLabel", wraplength=470).grid(
            row=2,
            column=0,
            columnspan=3,
            sticky="w",
            pady=(12, 0),
        )

        crop_every_header = ttk.Frame(crop_card, style="CardInner.TFrame")
        crop_every_header.grid(row=3, column=0, columnspan=3, sticky="ew", pady=(12, 0))
        crop_every_header.columnconfigure(0, weight=1)
        ttk.Label(crop_every_header, text="Apply Crop Every Nth Segment", style="Section.TLabel").grid(
            row=0,
            column=0,
            sticky="w",
        )
        ttk.Label(crop_every_header, textvariable=self.crop_every_value_var, style="Body.TLabel").grid(
            row=0,
            column=1,
            sticky="e",
        )
        self.crop_every_scale = ttk.Scale(
            crop_card,
            from_=1,
            to=12,
            variable=self.crop_every_var,
            command=self._on_crop_every_scale,
            style="Accent.Horizontal.TScale",
        )
        self.crop_every_scale.grid(row=4, column=0, columnspan=3, sticky="ew")

        crop_min_header = ttk.Frame(crop_card, style="CardInner.TFrame")
        crop_min_header.grid(row=5, column=0, columnspan=3, sticky="ew", pady=(12, 0))
        crop_min_header.columnconfigure(0, weight=1)
        ttk.Label(crop_min_header, text="Skip Crop Below (seconds)", style="Section.TLabel").grid(
            row=0,
            column=0,
            sticky="w",
        )
        ttk.Label(crop_min_header, textvariable=self.crop_min_seconds_value_var, style="Body.TLabel").grid(
            row=0,
            column=1,
            sticky="e",
        )
        self.crop_min_scale = ttk.Scale(
            crop_card,
            from_=0.0,
            to=10.0,
            variable=self.crop_min_seconds_var,
            command=self._on_crop_min_scale,
            style="Accent.Horizontal.TScale",
        )
        self.crop_min_scale.grid(row=6, column=0, columnspan=3, sticky="ew")

        crop_zoom_header = ttk.Frame(crop_card, style="CardInner.TFrame")
        crop_zoom_header.grid(row=7, column=0, columnspan=3, sticky="ew", pady=(12, 0))
        crop_zoom_header.columnconfigure(0, weight=1)
        ttk.Label(crop_zoom_header, text="Keep Zoom For (seconds)", style="Section.TLabel").grid(
            row=0,
            column=0,
            sticky="w",
        )
        ttk.Label(crop_zoom_header, textvariable=self.crop_zoom_seconds_value_var, style="Body.TLabel").grid(
            row=0,
            column=1,
            sticky="e",
        )
        self.crop_zoom_scale = ttk.Scale(
            crop_card,
            from_=0.0,
            to=10.0,
            variable=self.crop_zoom_seconds_var,
            command=self._on_crop_zoom_scale,
            style="Accent.Horizontal.TScale",
        )
        self.crop_zoom_scale.grid(row=8, column=0, columnspan=3, sticky="ew")

        self.crop_middle_long_check = ttk.Checkbutton(
            crop_card,
            text="Also zoom the middle of long non-zoomed segments",
            variable=self.crop_middle_long_var,
        )
        self.crop_middle_long_check.grid(row=9, column=0, columnspan=3, sticky="w", pady=(12, 0))

        crop_middle_threshold_header = ttk.Frame(crop_card, style="CardInner.TFrame")
        crop_middle_threshold_header.grid(row=10, column=0, columnspan=3, sticky="ew", pady=(12, 0))
        crop_middle_threshold_header.columnconfigure(0, weight=1)
        ttk.Label(
            crop_middle_threshold_header,
            text="Long Non-Zoomed Segment Threshold (seconds)",
            style="Section.TLabel",
        ).grid(row=0, column=0, sticky="w")
        ttk.Label(
            crop_middle_threshold_header,
            textvariable=self.crop_middle_long_threshold_value_var,
            style="Body.TLabel",
        ).grid(row=0, column=1, sticky="e")

        self.crop_middle_long_threshold_scale = ttk.Scale(
            crop_card,
            from_=4.0,
            to=30.0,
            variable=self.crop_middle_long_threshold_var,
            command=self._on_crop_middle_long_threshold_scale,
            style="Accent.Horizontal.TScale",
        )
        self.crop_middle_long_threshold_scale.grid(row=11, column=0, columnspan=3, sticky="ew")

        ttk.Label(
            crop_card,
            text=(
                "The crop box is picked on a frame preview and is locked to the original video aspect ratio, "
                "so the zoom never stretches. Cropped segments are scaled back up, short segments can be skipped "
                "to avoid fast jump cuts, setting zoom length to 0 keeps the zoom for the full segment, and the "
                "optional long-segment rule adds a centered zoom on long segments that were not picked by the main cadence."
            ),
            style="Body.TLabel",
            wraplength=470,
        ).grid(row=12, column=0, columnspan=3, sticky="w", pady=(12, 0))

        self._crop_widgets = [
            self.pick_crop_button,
            self.pick_queue_crops_button,
            self.clear_crop_button,
            self.crop_every_scale,
            self.crop_min_scale,
            self.crop_zoom_scale,
            self.crop_middle_long_check,
            self.crop_middle_long_threshold_scale,
        ]

        motion_card = ttk.Frame(root, style="Card.TFrame", padding=20)
        motion_card.grid(row=2, column=1, sticky="new", padx=(8, 0), pady=(0, 16))
        motion_card.columnconfigure(0, weight=1)

        ttk.Checkbutton(
            motion_card,
            text="Slow push-in on longer speech segments",
            variable=self.push_in_enabled_var,
        ).grid(row=0, column=0, columnspan=2, sticky="w")

        push_in_min_header = ttk.Frame(motion_card, style="CardInner.TFrame")
        push_in_min_header.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(12, 0))
        push_in_min_header.columnconfigure(0, weight=1)
        ttk.Label(push_in_min_header, text="Apply Push-In Above (seconds)", style="Section.TLabel").grid(
            row=0,
            column=0,
            sticky="w",
        )
        ttk.Label(push_in_min_header, textvariable=self.push_in_min_seconds_value_var, style="Body.TLabel").grid(
            row=0,
            column=1,
            sticky="e",
        )
        self.push_in_min_scale = ttk.Scale(
            motion_card,
            from_=1.0,
            to=30.0,
            variable=self.push_in_min_seconds_var,
            command=self._on_push_in_min_scale,
            style="Accent.Horizontal.TScale",
        )
        self.push_in_min_scale.grid(row=2, column=0, columnspan=2, sticky="ew")

        push_in_zoom_header = ttk.Frame(motion_card, style="CardInner.TFrame")
        push_in_zoom_header.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(12, 0))
        push_in_zoom_header.columnconfigure(0, weight=1)
        ttk.Label(push_in_zoom_header, text="Final Push-In Amount", style="Section.TLabel").grid(
            row=0,
            column=0,
            sticky="w",
        )
        ttk.Label(push_in_zoom_header, textvariable=self.push_in_zoom_percent_value_var, style="Body.TLabel").grid(
            row=0,
            column=1,
            sticky="e",
        )
        self.push_in_zoom_scale = ttk.Scale(
            motion_card,
            from_=2.0,
            to=25.0,
            variable=self.push_in_zoom_percent_var,
            command=self._on_push_in_zoom_scale,
            style="Accent.Horizontal.TScale",
        )
        self.push_in_zoom_scale.grid(row=4, column=0, columnspan=2, sticky="ew")
        ttk.Label(
            motion_card,
            text=(
                "Long speech clips get a gentle center zoom that eases in over the whole segment. "
                "Manual crop/zoom segments keep their chosen crop instead."
            ),
            style="Body.TLabel",
            wraplength=470,
        ).grid(row=5, column=0, columnspan=2, sticky="w", pady=(12, 0))

        caption_card = ttk.Frame(root, style="Card.TFrame", padding=20)
        caption_card.grid(row=3, column=1, sticky="new", padx=(8, 0), pady=(0, 16))
        caption_card.columnconfigure(0, weight=1)

        ttk.Label(caption_card, text="Captions & Filter", style="Section.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Checkbutton(
            caption_card,
            text="Burn in TikTok-style subtitles",
            variable=self.captions_enabled_var,
        ).grid(row=1, column=0, sticky="w", pady=(12, 0))
        ttk.Checkbutton(
            caption_card,
            text="Mute profanity automatically",
            variable=self.profanity_filter_var,
        ).grid(row=2, column=0, sticky="w", pady=(8, 0))
        ttk.Label(
            caption_card,
            text=(
                "Subtitles are baked into the video with bold lower-third captions and a live highlighted word. "
                "The first run may download the speech-to-text model."
            ),
            style="Body.TLabel",
            wraplength=470,
        ).grid(row=3, column=0, sticky="w", pady=(12, 0))

        activity_card = ttk.Frame(root, style="Card.TFrame", padding=20)
        activity_card.grid(row=4, column=0, columnspan=2, sticky="nsew")
        activity_card.columnconfigure(0, weight=1)
        activity_card.rowconfigure(4, weight=1)

        actions = ttk.Frame(activity_card, style="CardInner.TFrame")
        actions.grid(row=0, column=0, sticky="ew")
        actions.columnconfigure(3, weight=1)

        self.start_button = ttk.Button(actions, text="Start", style="Primary.TButton", command=self._start_processing)
        self.start_button.grid(row=0, column=0, sticky="w")
        self.cancel_button = ttk.Button(
            actions,
            text="Cancel",
            style="Danger.TButton",
            command=self._cancel_processing,
            state="disabled",
        )
        self.cancel_button.grid(row=0, column=1, sticky="w", padx=(10, 0))
        self.open_folder_button = ttk.Button(actions, text="Open Output Folder", command=self._open_output_folder, state="disabled")
        self.open_folder_button.grid(row=0, column=2, sticky="w", padx=(10, 0))

        ttk.Label(activity_card, text="Progress", style="Section.TLabel").grid(row=1, column=0, sticky="w", pady=(14, 6))
        ttk.Progressbar(
            activity_card,
            style="Accent.Horizontal.TProgressbar",
            variable=self.progress_var,
            maximum=100,
        ).grid(row=2, column=0, sticky="ew")

        ttk.Label(activity_card, textvariable=self.status_var, style="Status.TLabel").grid(
            row=3,
            column=0,
            sticky="nw",
            pady=(12, 6),
        )

        self.log_box = scrolledtext.ScrolledText(
            activity_card,
            wrap="word",
            height=11,
            relief="flat",
            bg="#0b1220",
            fg="#e5eefb",
            insertbackground="#e5eefb",
            selectbackground=ACCENT,
            selectforeground="#ffffff",
            font=("Cascadia Mono", 10),
            padx=14,
            pady=12,
            borderwidth=0,
        )
        self.log_box.grid(row=4, column=0, sticky="nsew", pady=(4, 0))
        self.log_box.insert("end", "Ready.\n")
        self.log_box.configure(state="disabled")

    def _browse_input(self) -> None:
        filetypes = [
            ("Video files", "*.mp4 *.mov *.mkv *.webm *.avi *.m4v *.flv"),
            ("All files", "*.*"),
        ]
        if self.queue_enabled_var.get():
            paths = filedialog.askopenfilenames(title="Choose videos", filetypes=filetypes)
            if paths:
                self._set_queued_videos([Path(path) for path in paths])
            return

        path = filedialog.askopenfilename(title="Choose a video", filetypes=filetypes)
        if path:
            self.input_var.set(path)
            self._set_queued_videos([Path(path)])

    def _toggle_queue_mode(self) -> None:
        if self.queue_enabled_var.get():
            input_text = self.input_var.get().strip()
            if input_text and not self._queued_videos:
                input_path = Path(input_text)
                if input_path.exists():
                    self._set_queued_videos([input_path])
        elif len(self._queued_videos) > 1:
            keep_video = self._get_selected_video() or self._queued_videos[0]
            should_reduce = messagebox.askyesno(
                "Leave queue mode?",
                (
                    f"Queue mode is turning off.\n\n"
                    f"Keep only this video?\n\n{keep_video.path.name}\n\n"
                    f"The other {len(self._queued_videos) - 1} queued video(s) will be removed."
                ),
            )
            if not should_reduce:
                self.queue_enabled_var.set(True)
                return
            self._queued_videos = [
                QueuedVideo(
                    path=keep_video.path,
                    metadata=dict(keep_video.metadata),
                    crop_rect=keep_video.crop_rect,
                )
            ]
            self._selected_queue_index = 0

        self._update_queue_mode_ui()
        self._refresh_queue_list()
        self._apply_selected_video()

    def _update_queue_mode_ui(self) -> None:
        is_queue_mode = bool(self.queue_enabled_var.get())
        self.input_entry.configure(state="readonly" if is_queue_mode else "normal")
        self.pick_crop_button.configure(
            text="Choose Crop For Selected Video..." if is_queue_mode else "Choose Crop Visually..."
        )
        self.clear_crop_button.configure(text="Clear Selected Crop" if is_queue_mode else "Clear Crop")
        if is_queue_mode:
            self.queue_frame.grid()
            self.pick_queue_crops_button.grid()
        else:
            self.queue_frame.grid_remove()
            self.pick_queue_crops_button.grid_remove()
        self._update_crop_state()

    def _set_queued_videos(self, paths: list[Path]) -> None:
        unique_paths: list[Path] = []
        seen_paths: set[str] = set()
        for raw_path in paths:
            path = Path(raw_path)
            key = os.path.normcase(str(path))
            if key in seen_paths:
                continue
            seen_paths.add(key)
            unique_paths.append(path)

        if not unique_paths:
            return

        previous_by_path = {
            os.path.normcase(str(video.path)): video
            for video in self._queued_videos
        }
        current_video = self._get_selected_video()
        current_key = os.path.normcase(str(current_video.path)) if current_video else None

        self._queued_videos = []
        for path in unique_paths:
            existing = previous_by_path.get(os.path.normcase(str(path)))
            if existing:
                self._queued_videos.append(
                    QueuedVideo(
                        path=path,
                        metadata=dict(existing.metadata),
                        crop_rect=existing.crop_rect,
                    )
                )
            else:
                self._queued_videos.append(QueuedVideo(path=path))

        if current_key:
            for index, video in enumerate(self._queued_videos):
                if os.path.normcase(str(video.path)) == current_key:
                    self._selected_queue_index = index
                    break
            else:
                self._selected_queue_index = 0
        else:
            self._selected_queue_index = 0

        self._refresh_queue_list()
        self._update_queue_mode_ui()
        self._select_queue_index(self._selected_queue_index, ensure_metadata=True)

    def _refresh_queue_list(self) -> None:
        self._queue_refreshing = True
        try:
            self.queue_listbox.delete(0, "end")
            for index, video in enumerate(self._queued_videos):
                self.queue_listbox.insert("end", self._format_queue_entry(video, index))

            if self._queued_videos:
                self._selected_queue_index = max(0, min(self._selected_queue_index, len(self._queued_videos) - 1))
                self.queue_listbox.selection_clear(0, "end")
                self.queue_listbox.selection_set(self._selected_queue_index)
                self.queue_listbox.activate(self._selected_queue_index)
            self.remove_queue_button.configure(state="normal" if self._queued_videos else "disabled")
            self.clear_queue_button.configure(state="normal" if self._queued_videos else "disabled")
        finally:
            self._queue_refreshing = False

        if not self._queued_videos:
            self.queue_summary_var.set("No queued videos.")
            return

        crop_count = sum(1 for video in self._queued_videos if video.crop_rect is not None)
        video_count = len(self._queued_videos)
        video_word = "video" if video_count == 1 else "videos"
        crop_word = "crop" if crop_count == 1 else "crops"
        self.queue_summary_var.set(f"{video_count} {video_word} queued, {crop_count} {crop_word} ready.")

    def _format_queue_entry(self, video: QueuedVideo, index: int) -> str:
        crop_status = "crop ready" if video.crop_rect else "no crop"
        return f"{index + 1}. {video.path.name}  [{crop_status}]"

    def _on_queue_selection(self, _event: tk.Event[tk.Listbox]) -> None:
        if self._queue_refreshing:
            return
        selection = self.queue_listbox.curselection()
        if not selection:
            return
        self._select_queue_index(int(selection[0]), ensure_metadata=True)

    def _remove_selected_video(self) -> None:
        if not self._queued_videos:
            return

        selection = self.queue_listbox.curselection()
        if selection:
            remove_index = int(selection[0])
        else:
            remove_index = self._selected_queue_index

        del self._queued_videos[remove_index]
        if not self._queued_videos:
            self._selected_queue_index = 0
            self._refresh_queue_list()
            self._clear_active_video_state(clear_input=True)
            return

        self._selected_queue_index = min(remove_index, len(self._queued_videos) - 1)
        self._refresh_queue_list()
        self._select_queue_index(self._selected_queue_index, ensure_metadata=True)

    def _clear_queue(self) -> None:
        self._queued_videos = []
        self._selected_queue_index = 0
        self._refresh_queue_list()
        self._clear_active_video_state(clear_input=True)

    def _select_queue_index(self, index: int, *, ensure_metadata: bool = False) -> None:
        if not self._queued_videos:
            self._clear_active_video_state(clear_input=not self.queue_enabled_var.get())
            return

        self._selected_queue_index = max(0, min(index, len(self._queued_videos) - 1))
        if self.queue_listbox.size():
            self._queue_refreshing = True
            try:
                self.queue_listbox.selection_clear(0, "end")
                self.queue_listbox.selection_set(self._selected_queue_index)
                self.queue_listbox.activate(self._selected_queue_index)
                self.queue_listbox.see(self._selected_queue_index)
            finally:
                self._queue_refreshing = False

        self._apply_selected_video()
        if ensure_metadata:
            self._ensure_metadata_for_video(self._queued_videos[self._selected_queue_index])

    def _get_selected_video(self) -> QueuedVideo | None:
        if not self._queued_videos:
            return None
        safe_index = max(0, min(self._selected_queue_index, len(self._queued_videos) - 1))
        return self._queued_videos[safe_index]

    def _apply_selected_video(self) -> None:
        video = self._get_selected_video()
        if video is None:
            self._clear_active_video_state(clear_input=not self.queue_enabled_var.get())
            return

        self.input_var.set(str(video.path))
        self._source_path = video.path
        self._source_metadata = dict(video.metadata)
        self._crop_rect = video.crop_rect
        self._update_source_info_for_video(video)
        self._update_crop_summary_for_video(video)
        self._refresh_queue_list()

    def _update_source_info_for_video(self, video: QueuedVideo) -> None:
        position_prefix = ""
        if self.queue_enabled_var.get() and len(self._queued_videos) > 1:
            position_prefix = f"Video {self._selected_queue_index + 1} of {len(self._queued_videos)}. "

        if not video.metadata:
            self.source_info_var.set(position_prefix + "Pick a video to load its source size.")
            return

        width = int(video.metadata.get("video_width") or 0)
        height = int(video.metadata.get("video_height") or 0)
        if width > 0 and height > 0:
            self.source_info_var.set(
                position_prefix + f"Source size: {width} x {height}. Crop is chosen visually from a frame preview."
            )
        else:
            self.source_info_var.set(position_prefix + "Source size could not be read from this file.")

    def _update_crop_summary_for_video(self, video: QueuedVideo | None) -> None:
        if video is None:
            self.crop_summary_var.set("No crop selected.")
            return

        prefix = ""
        if self.queue_enabled_var.get() and len(self._queued_videos) > 1:
            prefix = f"Video {self._selected_queue_index + 1} of {len(self._queued_videos)}. "

        if video.crop_rect is None:
            self.crop_summary_var.set(prefix + "No crop selected.")
            return

        x, y, crop_width, crop_height = video.crop_rect
        self.crop_summary_var.set(
            prefix + f"Crop selected: x={x}, y={y}, width={crop_width}, height={crop_height}"
        )

    def _clear_active_video_state(self, *, clear_input: bool) -> None:
        if clear_input:
            self.input_var.set("")
        self._source_path = None
        self._source_metadata = {}
        self._crop_rect = None
        self.source_info_var.set("Pick a video to load its source size.")
        self.crop_summary_var.set("No crop selected.")
        self._update_crop_state()

    def _ensure_single_video_selection(self) -> QueuedVideo | None:
        input_path = self._get_input_path_or_warn()
        if input_path is None:
            return None

        current_video = self._get_selected_video()
        needs_refresh = (
            current_video is None
            or len(self._queued_videos) != 1
            or current_video.path != input_path
        )
        if needs_refresh:
            self._set_queued_videos([input_path])
        else:
            self._select_queue_index(0, ensure_metadata=not bool(current_video.metadata))
        return self._get_selected_video()

    def _ensure_metadata_for_video(self, video: QueuedVideo) -> bool:
        if video.metadata:
            if self._source_path == video.path:
                self._source_metadata = dict(video.metadata)
                self._update_source_info_for_video(video)
            return True

        try:
            metadata = probe_video_metadata(video.path)
        except Exception as exc:  # noqa: BLE001
            if self._source_path == video.path:
                self.source_info_var.set("Could not read the source video size.")
            self._append_log(f"Could not inspect source video: {exc}")
            return False

        video.metadata = metadata
        if self._source_path == video.path:
            self._source_metadata = dict(metadata)
            self._update_source_info_for_video(video)
        return True

    def _prepare_crop_preview(
        self,
        *,
        input_path: Path,
        source_duration: float,
    ) -> tuple[tempfile.TemporaryDirectory, Path]:
        preview_temp_dir = tempfile.TemporaryDirectory(prefix="speech_cutter_crop_")
        result_queue: Queue[tuple[str, object]] = Queue(maxsize=1)

        def worker() -> None:
            try:
                preview_path = _extract_preview_frame(
                    input_path=input_path,
                    source_duration=source_duration,
                    output_dir=Path(preview_temp_dir.name),
                )
            except Exception as exc:  # noqa: BLE001
                result_queue.put(("error", exc))
            else:
                result_queue.put(("ok", preview_path))

        loading_dialog = PreviewLoadingDialog(self, video_name=input_path.name)
        thread = threading.Thread(target=worker, daemon=True)
        thread.start()

        def poll_worker() -> None:
            if thread.is_alive():
                if loading_dialog.winfo_exists():
                    loading_dialog.after(100, poll_worker)
                return
            if loading_dialog.winfo_exists():
                loading_dialog.destroy()

        poll_worker()
        self.wait_window(loading_dialog)

        try:
            status, payload = result_queue.get_nowait()
        except Empty as exc:
            preview_temp_dir.cleanup()
            raise RuntimeError(f"Could not finish loading a preview for {input_path.name}.") from exc

        if status != "ok":
            preview_temp_dir.cleanup()
            raise RuntimeError(str(payload))

        return preview_temp_dir, payload  # type: ignore[return-value]

    def _update_crop_state(self) -> None:
        state = "normal" if self.crop_enabled_var.get() else "disabled"
        for widget in self._crop_widgets:
            widget.configure(state=state)
        queue_crop_state = state if self.queue_enabled_var.get() and len(self._queued_videos) > 1 else "disabled"
        self.pick_queue_crops_button.configure(state=queue_crop_state)

    def _choose_crop_visually(self) -> None:
        if self.queue_enabled_var.get():
            video = self._get_selected_video()
            if video is None:
                messagebox.showerror("Missing video", "Add one or more videos to the queue first.")
                return
        else:
            video = self._ensure_single_video_selection()
            if video is None:
                return

        self._choose_crop_for_video(video, position=self._selected_queue_index + 1, total=max(len(self._queued_videos), 1))

    def _clear_crop_selection(self, update_message: bool = True) -> None:
        selected_video = self._get_selected_video()
        if selected_video is not None:
            selected_video.crop_rect = None
        self._crop_rect = None
        if update_message:
            self._update_crop_summary_for_video(selected_video)
            self._refresh_queue_list()

    def _choose_crops_for_queue(self) -> None:
        if not self.queue_enabled_var.get() or len(self._queued_videos) <= 1:
            self._choose_crop_visually()
            return

        total = len(self._queued_videos)
        for index, video in enumerate(self._queued_videos, start=1):
            self._select_queue_index(index - 1, ensure_metadata=True)
            chosen = self._choose_crop_for_video(video, position=index, total=total, batch_mode=True)
            if not chosen:
                self.status_var.set(f"Crop setup stopped at video {index} of {total}.")
                self._append_log(f"Stopped crop setup at {video.path.name}.")
                return

        self.crop_enabled_var.set(True)
        self._update_crop_state()
        self.status_var.set(f"Crops ready for {total} queued videos.")
        self._append_log(f"Crops selected for {total} queued videos.")

    def _choose_crop_for_video(
        self,
        video: QueuedVideo,
        *,
        position: int,
        total: int,
        batch_mode: bool = False,
    ) -> bool:
        if not self._ensure_metadata_for_video(video):
            messagebox.showerror("Missing video size", f"Could not read the source video size for:\n\n{video.path}")
            return False

        width = int(video.metadata.get("video_width") or 0)
        height = int(video.metadata.get("video_height") or 0)
        duration = float(video.metadata.get("duration") or 0.0)
        if width <= 0 or height <= 0:
            messagebox.showerror("Missing video size", f"Could not read the source video size for:\n\n{video.path}")
            return False

        self.status_var.set(f"Loading crop preview for video {position} of {total}: {video.path.name}")
        self.update_idletasks()
        preview_temp_dir: tempfile.TemporaryDirectory | None = None
        try:
            preview_temp_dir, preview_path = self._prepare_crop_preview(
                input_path=video.path,
                source_duration=duration,
            )
            dialog = CropSelectionDialog(
                self,
                preview_path=preview_path,
                preview_temp_dir=preview_temp_dir,
                source_width=width,
                source_height=height,
                initial_crop=video.crop_rect,
            )
            self.wait_window(dialog)
        except Exception as exc:  # noqa: BLE001
            if preview_temp_dir is not None:
                try:
                    preview_temp_dir.cleanup()
                except Exception:
                    pass
            messagebox.showerror("Could not open crop picker", str(exc))
            return False

        if not dialog.result:
            return False if batch_mode else True

        video.crop_rect = dialog.result
        self._crop_rect = dialog.result if self._source_path == video.path else self._crop_rect
        self.crop_enabled_var.set(True)
        self._update_crop_state()
        self._update_crop_summary_for_video(video)
        self._refresh_queue_list()
        self.status_var.set(f"Crop saved for video {position} of {total}: {video.path.name}")
        return True

    def _start_processing(self) -> None:
        jobs = self._build_processing_jobs()
        if not jobs:
            return

        job_count = len(jobs)

        self._last_result = None
        self._cancel_event.clear()
        self._set_running(True)
        self.progress_var.set(0.0)
        self.status_var.set(f"Preparing {job_count} video(s)...")
        if job_count == 1:
            self._append_log(f"Starting: {jobs[0].input_path.name}")
        else:
            self._append_log(f"Starting queue: {job_count} videos.")

        self._worker = threading.Thread(
            target=self._worker_main,
            args=(jobs,),
            daemon=True,
        )
        self._worker.start()

    def _build_processing_jobs(self) -> list[ProcessingJob] | None:
        try:
            padding_seconds = max(0.0, float(self.padding_seconds_var.get()))
            crop_every = max(1, int(round(float(self.crop_every_var.get()))))
            crop_min_seconds = max(0.0, float(self.crop_min_seconds_var.get()))
            crop_zoom_seconds = max(0.0, float(self.crop_zoom_seconds_var.get()))
            crop_middle_long_threshold_seconds = max(0.0, float(self.crop_middle_long_threshold_var.get()))
            push_in_min_seconds = max(0.0, float(self.push_in_min_seconds_var.get()))
            push_in_max_zoom = 1.0 + max(0.0, float(self.push_in_zoom_percent_var.get())) / 100.0
        except (tk.TclError, ValueError):
            messagebox.showerror("Invalid settings", "One of the trim, crop, or motion settings is not valid.")
            return None

        if self.queue_enabled_var.get():
            videos = list(self._queued_videos)
            if not videos:
                messagebox.showerror("Missing videos", "Add one or more videos to the queue first.")
                return None
        else:
            selected_video = self._ensure_single_video_selection()
            if selected_video is None:
                return None
            videos = [selected_video]

        missing_files = [video.path for video in videos if not video.path.exists()]
        if missing_files:
            messagebox.showerror("Missing file", f"The selected source video does not exist:\n\n{missing_files[0]}")
            return None

        if self.crop_enabled_var.get():
            missing_crop_index = next((index for index, video in enumerate(videos) if video.crop_rect is None), None)
            if missing_crop_index is not None:
                if self.queue_enabled_var.get() and self._queued_videos:
                    self._select_queue_index(missing_crop_index, ensure_metadata=False)
                    messagebox.showerror(
                        "Missing crop",
                        "Enable crop is on, but at least one queued video still has no crop box. "
                        "Use 'Choose Crops For Queue...' first.",
                    )
                else:
                    messagebox.showerror("Missing crop", "Enable crop is on, but no crop box has been chosen yet.")
                return None

        output_paths = [build_output_path(video.path) for video in videos]
        existing_outputs = [output_path for output_path in output_paths if output_path.exists()]
        if existing_outputs and not self._confirm_output_overwrite(existing_outputs):
            return None

        caption_settings = CaptionSettings(
            enabled=bool(self.captions_enabled_var.get()),
            profanity_filter=bool(self.profanity_filter_var.get()),
        )

        jobs: list[ProcessingJob] = []
        for video, output_path in zip(videos, output_paths):
            crop_rect = video.crop_rect if self.crop_enabled_var.get() else None
            crop_settings = CropSettings(
                enabled=bool(self.crop_enabled_var.get() and crop_rect),
                x=crop_rect[0] if crop_rect else 0,
                y=crop_rect[1] if crop_rect else 0,
                width=crop_rect[2] if crop_rect else 0,
                height=crop_rect[3] if crop_rect else 0,
                every_n_segments=crop_every,
                min_segment_seconds=crop_min_seconds,
                zoom_duration_seconds=crop_zoom_seconds,
                mid_zoom_on_long_segments=bool(self.crop_middle_long_var.get()),
                mid_zoom_min_segment_seconds=crop_middle_long_threshold_seconds,
            )
            motion_settings = MotionSettings(
                push_in_enabled=bool(self.push_in_enabled_var.get()),
                push_in_min_segment_seconds=push_in_min_seconds,
                push_in_max_zoom=push_in_max_zoom,
            )
            jobs.append(
                ProcessingJob(
                    input_path=video.path,
                    output_path=output_path,
                    options=build_options(
                        "natural",
                        padding_ms=int(round(padding_seconds * 1000)),
                        merge_gap_ms=NATURAL_SETTINGS.merge_gap_ms,
                        crop=crop_settings,
                        motion=motion_settings,
                        captions=caption_settings,
                    ),
                )
            )
        return jobs

    def _confirm_output_overwrite(self, existing_outputs: list[Path]) -> bool:
        if not existing_outputs:
            return True
        if len(existing_outputs) == 1:
            return messagebox.askyesno(
                "Replace existing file?",
                f"This output already exists:\n\n{existing_outputs[0]}\n\nDo you want to replace it?",
            )

        preview_lines = "\n".join(str(path) for path in existing_outputs[:4])
        extra_count = len(existing_outputs) - 4
        extra_text = f"\n...and {extra_count} more." if extra_count > 0 else ""
        return messagebox.askyesno(
            "Replace existing files?",
            (
                f"{len(existing_outputs)} output files already exist and would be replaced:\n\n"
                f"{preview_lines}{extra_text}\n\nContinue?"
            ),
        )

    def _get_input_path_or_warn(self) -> Path | None:
        input_text = self.input_var.get().strip()
        if not input_text:
            messagebox.showerror("Missing video", "Choose a source video first.")
            return None

        input_path = Path(input_text)
        if not input_path.exists():
            messagebox.showerror("Missing file", "The selected source video does not exist.")
            return None
        return input_path

    def _worker_main(
        self,
        jobs: list[ProcessingJob],
    ) -> None:
        total_jobs = max(len(jobs), 1)
        results: list[ProcessingResult] = []
        failures: list[BatchFailure] = []
        try:
            for job_index, job in enumerate(jobs, start=1):
                if self._cancel_event.is_set():
                    raise UserCancelledError()

                prefix = f"[{job_index}/{total_jobs}]"
                self._events.put(QueueEvent("log", f"{prefix} Starting {job.input_path.name}"))
                try:
                    result = process_video(
                        job.input_path,
                        job.output_path,
                        options=job.options,
                        progress_callback=lambda value, status, current_index=job_index: self._events.put(
                            QueueEvent(
                                "progress",
                                (
                                    ((current_index - 1) + float(value)) / total_jobs,
                                    f"[{current_index}/{total_jobs}] {status}",
                                ),
                            )
                        ),
                        log_callback=lambda message, current_index=job_index: self._events.put(
                            QueueEvent("log", f"[{current_index}/{total_jobs}] {message}")
                        ),
                        cancel_callback=self._cancel_event.is_set,
                    )
                except UserCancelledError:
                    raise
                except NoSpeechDetectedError as exc:
                    failures.append(BatchFailure(input_path=job.input_path, message=str(exc)))
                    self._events.put(QueueEvent("log", f"{prefix} Skipped: {exc}"))
                    self._events.put(
                        QueueEvent("progress", (job_index / total_jobs, f"{prefix} Skipped after no speech was found."))
                    )
                    continue
                except Exception as exc:  # noqa: BLE001
                    failures.append(BatchFailure(input_path=job.input_path, message=str(exc)))
                    self._events.put(QueueEvent("log", f"{prefix} Failed: {exc}"))
                    self._events.put(QueueEvent("progress", (job_index / total_jobs, f"{prefix} Stopped on this file.")))
                    continue

                results.append(result)
                self._events.put(QueueEvent("log", f"{prefix} Saved {result.output_path.name}"))
        except UserCancelledError:
            self._events.put(QueueEvent("cancelled", None))
        else:
            self._events.put(QueueEvent("done_batch", BatchRunResult(results=results, failures=failures)))

    def _cancel_processing(self) -> None:
        if self._worker and self._worker.is_alive():
            self._cancel_event.set()
            self.status_var.set("Cancelling...")
            self._append_log("Cancellation requested.")

    def _drain_queue(self) -> None:
        try:
            while True:
                event = self._events.get_nowait()
                self._handle_event(event)
        except Empty:
            pass
        finally:
            self.after(100, self._drain_queue)

    def _handle_event(self, event: QueueEvent) -> None:
        if event.kind == "progress":
            value, status = event.payload  # type: ignore[misc]
            self.progress_var.set(float(value) * 100)
            self.status_var.set(str(status))
            return

        if event.kind == "log":
            self._append_log(str(event.payload))
            return

        if event.kind == "done_batch":
            batch = event.payload  # type: ignore[assignment]
            self._set_running(False)
            self.progress_var.set(100 if (batch.results or batch.failures) else 0)

            if batch.results:
                self._last_result = batch.results[-1]
                self.open_folder_button.configure(state="normal")

            success_count = len(batch.results)
            failure_count = len(batch.failures)
            total_count = success_count + failure_count
            kept_total = sum(result.kept_duration for result in batch.results)
            removed_total = sum(result.removed_duration for result in batch.results)

            if success_count:
                summary = (
                    f"Finished {success_count} of {total_count} video(s). "
                    f"Kept {kept_total:.1f}s of speech and removed {removed_total:.1f}s."
                )
            else:
                summary = f"No videos were exported successfully. {failure_count} file(s) failed."

            if failure_count:
                summary += f" {failure_count} file(s) had errors."

            self.status_var.set(summary)
            self._append_log(summary)

            if batch.failures:
                for failure in batch.failures:
                    self._append_log(f"Failed: {failure.input_path.name} - {failure.message}")

            if success_count and not failure_count:
                if success_count == 1:
                    messagebox.showinfo("Done", f"Processed video saved to:\n\n{batch.results[0].output_path}")
                else:
                    messagebox.showinfo(
                        "Done",
                        (
                            f"Finished {success_count} videos.\n\n"
                            "Each output was saved next to its source video."
                        ),
                    )
            elif success_count:
                failure_lines = "\n".join(
                    f"{failure.input_path.name}: {failure.message}"
                    for failure in batch.failures[:5]
                )
                extra_failures = failure_count - min(failure_count, 5)
                extra_text = f"\n...and {extra_failures} more." if extra_failures > 0 else ""
                messagebox.showwarning(
                    "Finished With Some Issues",
                    (
                        f"Finished {success_count} video(s), but {failure_count} file(s) had problems.\n\n"
                        f"{failure_lines}{extra_text}"
                    ),
                )
            else:
                failure_lines = "\n".join(
                    f"{failure.input_path.name}: {failure.message}"
                    for failure in batch.failures[:5]
                )
                extra_failures = failure_count - min(failure_count, 5)
                extra_text = f"\n...and {extra_failures} more." if extra_failures > 0 else ""
                messagebox.showerror(
                    "Could Not Finish Batch",
                    f"No videos were exported successfully.\n\n{failure_lines}{extra_text}",
                )
            return

        if event.kind == "cancelled":
            self._set_running(False)
            self.progress_var.set(0)
            self.status_var.set("Cancelled.")
            self._append_log("Processing cancelled.")
            return

        if event.kind == "error":
            self._set_running(False)
            self.progress_var.set(0)
            self.status_var.set("Stopped because of an error.")
            self._append_log(f"Error: {event.payload}")
            messagebox.showerror("Could not finish", str(event.payload))

    def _set_running(self, is_running: bool) -> None:
        self.start_button.configure(state="disabled" if is_running else "normal")
        self.cancel_button.configure(state="normal" if is_running else "disabled")
        if is_running:
            self.open_folder_button.configure(state="disabled")

    def _append_log(self, message: str) -> None:
        self.log_box.configure(state="normal")
        self.log_box.insert("end", message.rstrip() + "\n")
        self.log_box.see("end")
        self.log_box.configure(state="disabled")

    def _open_output_folder(self) -> None:
        if not self._last_result:
            return
        folder = self._last_result.output_path.parent
        if os.name == "nt":
            os.startfile(folder)  # type: ignore[attr-defined]
        else:
            messagebox.showinfo("Output folder", str(folder))


def _subprocess_kwargs() -> dict[str, object]:
    kwargs: dict[str, object] = {}
    if os.name == "nt":
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        kwargs["startupinfo"] = startupinfo
        kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
    return kwargs


def launch_gui() -> None:
    app = SpeechCutterApp()
    app.mainloop()
