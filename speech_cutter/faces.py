"""Face detection and face-targeted zoom helpers.

The detector uses the bundled YuNet ONNX model through OpenCV. Detections are
sampled across the source video and linked into tracks, so each track represents
one distinct face appearance that the user can select in the GUI.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence
import math

FACE_MODEL_PATH = Path(__file__).resolve().parent / "assets" / "face_detection_yunet_2023mar.onnx"
MANUAL_ZOOM_LABEL = "Kadr ręczny"
FACE_ZOOM_LABEL = "Twarze"
FACE_COUNT_OPTIONS = ("Auto", "1", "2", "3", "4", "5", "6")
MIN_FACE_SCORE = 0.65
MAX_FACE_SAMPLES = 480
DEFAULT_FACE_PADDING = 0.35
DEFAULT_FACE_MAX_ZOOM = 1.8
_MIN_FACE_SIDE_PX = 24
_THUMBNAIL_SIDE_PX = 160

FaceProgressCallback = Callable[[float, str], None]
FaceCancelCallback = Callable[[], bool]
FaceStopCallback = Callable[[], bool]
FaceCountCallback = Callable[[int], None]


class FaceDetectionError(RuntimeError):
    """Raised when the face detector cannot analyze a video."""


class FaceDetectionCancelledError(RuntimeError):
    """Raised when face detection is cancelled."""


@dataclass(frozen=True)
class FaceBox:
    """A face rectangle in source-video pixels at a given timestamp."""

    time: float
    x: int
    y: int
    width: int
    height: int
    score: float = 0.0

    @property
    def center(self) -> tuple[float, float]:
        return (self.x + self.width / 2.0, self.y + self.height / 2.0)

    @property
    def area(self) -> int:
        return max(0, self.width) * max(0, self.height)


@dataclass(frozen=True)
class DetectedFace:
    """One distinct face appearance found in a video."""

    face_id: int
    boxes: tuple[FaceBox, ...]
    thumbnail_png: bytes = b""
    screen_time: float = 0.0
    first_seen: float = 0.0
    last_seen: float = 0.0

    @property
    def representative_box(self) -> FaceBox:
        return max(self.boxes, key=lambda box: box.area)


@dataclass(frozen=True)
class FaceAnalysis:
    """Result of scanning a video for distinct faces."""

    video_width: int
    video_height: int
    duration: float
    sample_interval: float
    faces: tuple[DetectedFace, ...]
    target_face_count: int = 0
    stopped_early: bool = False
    stopped_by_user: bool = False
    samples_scanned: int = 0

    @property
    def has_faces(self) -> bool:
        return bool(self.faces)


@dataclass
class _FaceTrack:
    boxes: list[FaceBox]
    last_box: FaceBox
    last_seen: float
    best_area: int
    thumbnail_png: bytes = b""


def detect_faces(
    video_path: Path,
    *,
    duration: float | None = None,
    sample_interval: float | None = None,
    max_samples: int = MAX_FACE_SAMPLES,
    target_face_count: int = 0,
    min_score: float = MIN_FACE_SCORE,
    progress_callback: FaceProgressCallback | None = None,
    count_callback: FaceCountCallback | None = None,
    cancel_callback: FaceCancelCallback | None = None,
    stop_callback: FaceStopCallback | None = None,
) -> FaceAnalysis:
    """Scan a video and return distinct face tracks.

    Sampling is spread evenly across the video (at most ``max_samples`` frames),
    so the detector never analyzes every frame. When ``target_face_count`` is
    greater than zero, scanning stops once that many tracks are confirmed.
    ``stop_callback`` can request an early finish that keeps the faces found so
    far, while ``cancel_callback`` aborts the scan without results.
    """

    try:
        import cv2
        import numpy as np
    except ImportError as exc:  # pragma: no cover - depends on the local environment
        raise FaceDetectionError(
            "Face detection needs OpenCV. Run: .venv\\Scripts\\python -m pip install -r requirements.txt"
        ) from exc

    if not FACE_MODEL_PATH.exists():
        raise FaceDetectionError(f"Face detection model is missing: {FACE_MODEL_PATH}")

    video_path = Path(video_path)
    if not video_path.exists():
        raise FaceDetectionError(f"Video file does not exist: {video_path}")

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise FaceDetectionError(f"Could not open the video for face detection: {video_path}")

    try:
        video_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        video_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

        if duration is None or duration <= 0:
            if fps > 0 and frame_count > 0:
                duration = frame_count / fps
            else:
                duration = 0.0
        duration = float(duration or 0.0)

        if video_width <= 0 or video_height <= 0:
            raise FaceDetectionError("Could not read the video size for face detection.")
        if duration <= 0:
            raise FaceDetectionError("Could not read the video duration for face detection.")

        if sample_interval is None or sample_interval <= 0:
            sample_count = max(1, min(max_samples, int(math.ceil(duration / 0.5))))
        else:
            sample_count = max(1, min(max_samples, int(math.ceil(duration / sample_interval))))
        actual_interval = duration / sample_count
        sample_times = [
            min(duration - 0.001, duration * (index + 0.5) / sample_count)
            for index in range(sample_count)
        ]

        detector = cv2.FaceDetectorYN.create(
            str(FACE_MODEL_PATH),
            "",
            (video_width, video_height),
            min_score,
            0.3,
            5000,
        )

        target_count = max(0, int(target_face_count or 0))
        minimum_hits = _minimum_track_hits(actual_interval)
        min_samples_before_stop = min(
            sample_count,
            max(2, int(math.ceil(1.0 / max(actual_interval, 0.001)))),
        )
        tracks: list[_FaceTrack] = []
        samples_scanned = 0
        stopped_early = False
        stopped_by_user = False
        confirmed_count = 0

        def update_confirmed_count() -> int:
            nonlocal confirmed_count
            total = sum(
                1 for track in tracks if len(track.boxes) >= minimum_hits
            )
            if target_count > 0:
                total = min(total, target_count)
            if total != confirmed_count:
                confirmed_count = total
                _emit_count(count_callback, confirmed_count)
            return confirmed_count

        _emit(progress_callback, 0.0, "Scanning the video for faces...")

        for sample_index, sample_time in enumerate(sample_times):
            if cancel_callback is not None and cancel_callback():
                raise FaceDetectionCancelledError("Face detection was cancelled.")
            if stop_callback is not None and stop_callback():
                stopped_early = True
                stopped_by_user = True
                break

            samples_scanned = sample_index + 1
            capture.set(cv2.CAP_PROP_POS_MSEC, sample_time * 1000.0)
            read_ok, frame = capture.read()
            if not read_ok or frame is None:
                continue

            detections, detection_scale = _detect_frame_faces(
                cv2,
                detector,
                frame,
                min_score=min_score,
            )
            _track_detections(
                cv2,
                np,
                frame,
                detections,
                sample_time,
                tracks,
                max_gap=max(1.5, actual_interval * 3.0),
            )

            confirmed = update_confirmed_count()
            if (
                target_count > 0
                and samples_scanned >= min_samples_before_stop
                and confirmed >= target_count
            ):
                stopped_early = True
                break

            _emit(
                progress_callback,
                min(0.99, (sample_index + 1) / sample_count),
                "Scanning the video for faces...",
            )

        faces = _finalize_tracks(tracks, actual_interval)
        if target_count > 0:
            faces = faces[:target_count]
        if len(faces) != confirmed_count:
            _emit_count(count_callback, len(faces))
        if stopped_by_user:
            _emit(
                progress_callback,
                1.0,
                f"Scan stopped; keeping {len(faces)} face track(s).",
            )
        elif stopped_early:
            _emit(
                progress_callback,
                1.0,
                f"Found {target_count} face track(s); stopping the scan early.",
            )
        else:
            _emit(progress_callback, 1.0, "Face scan finished.")
        return FaceAnalysis(
            video_width=video_width,
            video_height=video_height,
            duration=duration,
            sample_interval=actual_interval,
            faces=faces,
            target_face_count=target_count,
            stopped_early=stopped_early,
            stopped_by_user=stopped_by_user,
            samples_scanned=samples_scanned,
        )
    finally:
        capture.release()


def choose_face_for_range(
    faces: Sequence[DetectedFace],
    start: float,
    end: float,
    *,
    selected_face_ids: Sequence[int] = (),
    tolerance: float = 1.5,
) -> DetectedFace | None:
    """Pick the most prominent visible face in a time range.

    When ``selected_face_ids`` is empty, every detected face is considered and
    the largest face covering the range wins. Selected faces take priority.
    """

    if not faces:
        return None

    selected = {int(face_id) for face_id in selected_face_ids}
    candidates = [face for face in faces if not selected or face.face_id in selected]
    if not candidates:
        return None

    range_start = min(start, end)
    range_end = max(start, end)
    range_duration = max(range_end - range_start, 0.001)
    midpoint = (range_start + range_end) / 2.0

    best_face: DetectedFace | None = None
    best_score = -1.0
    for face in candidates:
        inside = [
            box
            for box in face.boxes
            if range_start - 0.001 <= box.time <= range_end + 0.001
        ]
        if inside:
            visible_span = max(box.time for box in inside) - min(box.time for box in inside)
            coverage = min(1.0, max(visible_span, 0.001) / range_duration)
            median_area = _median([box.area for box in inside])
            score = median_area * (0.5 + coverage)
        else:
            nearest = min(face.boxes, key=lambda box: abs(box.time - midpoint))
            distance = abs(nearest.time - midpoint)
            if distance > tolerance:
                continue
            score = nearest.area * 0.2 * max(0.0, 1.0 - distance / tolerance)

        if score > best_score:
            best_score = score
            best_face = face

    return best_face


def face_box_for_range(face: DetectedFace, start: float, end: float) -> FaceBox:
    """Return a representative face box (median geometry) for a time range."""

    range_start = min(start, end)
    range_end = max(start, end)
    midpoint = (range_start + range_end) / 2.0
    inside = [
        box
        for box in face.boxes
        if range_start - 0.001 <= box.time <= range_end + 0.001
    ]
    if not inside:
        inside = [min(face.boxes, key=lambda box: abs(box.time - midpoint))]
    return FaceBox(
        time=midpoint,
        x=int(round(_median([box.x for box in inside]))),
        y=int(round(_median([box.y for box in inside]))),
        width=int(round(_median([box.width for box in inside]))),
        height=int(round(_median([box.height for box in inside]))),
        score=max(box.score for box in inside),
    )


def compute_face_crop(
    box: FaceBox,
    *,
    frame_width: int,
    frame_height: int,
    padding: float = DEFAULT_FACE_PADDING,
    max_zoom: float = DEFAULT_FACE_MAX_ZOOM,
) -> tuple[int, int, int, int] | None:
    """Turn a face box into a source-aspect crop rectangle.

    The crop keeps the original video aspect ratio and includes breathing room
    around the face. ``max_zoom`` limits how aggressively the frame is enlarged.
    Returns ``None`` when no meaningful zoom would remain.
    """

    if frame_width <= 0 or frame_height <= 0 or box.width <= 0 or box.height <= 0:
        return None

    padding = min(1.5, max(0.0, float(padding)))
    max_zoom = min(4.0, max(1.01, float(max_zoom)))
    aspect = frame_width / frame_height

    crop_height = box.height * (2.0 + 2.0 * padding)
    crop_width = crop_height * aspect

    min_width_for_face = box.width * (1.6 + 2.0 * padding)
    if crop_width < min_width_for_face:
        crop_width = min_width_for_face
        crop_height = crop_width / aspect

    max_crop_width = frame_width / max_zoom
    if crop_width < max_crop_width:
        crop_width = max_crop_width
        crop_height = crop_width / aspect

    crop_width = min(float(frame_width), crop_width)
    crop_height = min(float(frame_height), crop_height)

    if crop_width >= frame_width - 2 or crop_height >= frame_height - 2:
        return None

    center_x, center_y = box.center
    x = int(round(center_x - crop_width / 2.0))
    y = int(round(center_y - crop_height / 2.0))
    width = max(2, int(round(crop_width)))
    height = max(2, int(round(crop_height)))

    x = max(0, min(frame_width - width, x))
    y = max(0, min(frame_height - height, y))
    return x, y, width, height


def _detect_frame_faces(cv2: object, detector: object, frame: object, *, min_score: float) -> tuple[list[FaceBox], float]:
    frame_height, frame_width = frame.shape[:2]
    detection_width = 640
    scale = min(1.0, detection_width / max(frame_width, 1))
    if scale < 1.0:
        small = cv2.resize(frame, (int(round(frame_width * scale)), int(round(frame_height * scale))))
    else:
        small = frame

    detector.setInputSize((small.shape[1], small.shape[0]))
    result = detector.detect(small)[1]
    if result is None:
        return [], scale

    boxes: list[FaceBox] = []
    for row in result:
        score = float(row[-1])
        if score < min_score:
            continue
        x = int(round(float(row[0]) / scale))
        y = int(round(float(row[1]) / scale))
        width = int(round(float(row[2]) / scale))
        height = int(round(float(row[3]) / scale))
        if width < _MIN_FACE_SIDE_PX or height < _MIN_FACE_SIDE_PX:
            continue
        if width / max(height, 1) < 0.45 or width / max(height, 1) > 2.2:
            continue
        x = max(0, min(frame_width - 1, x))
        y = max(0, min(frame_height - 1, y))
        width = min(width, frame_width - x)
        height = min(height, frame_height - y)
        boxes.append(FaceBox(time=0.0, x=x, y=y, width=width, height=height, score=score))

    boxes.sort(key=lambda box: box.area, reverse=True)
    return boxes, scale


def _track_detections(
    cv2: object,
    np: object,
    frame: object,
    detections: Iterable[FaceBox],
    time: float,
    tracks: list[_FaceTrack],
    *,
    max_gap: float,
) -> None:
    used_tracks: set[int] = set()
    for detection in detections:
        detection = FaceBox(
            time=time,
            x=detection.x,
            y=detection.y,
            width=detection.width,
            height=detection.height,
            score=detection.score,
        )
        best_index = -1
        best_score = -1.0
        for index, track in enumerate(tracks):
            if index in used_tracks:
                continue
            if time - track.last_seen > max_gap:
                continue
            match = _match_score(detection, track.last_box)
            if match > best_score:
                best_score = match
                best_index = index

        if best_index >= 0 and best_score >= 0.15:
            track = tracks[best_index]
            used_tracks.add(best_index)
            track.boxes.append(detection)
            track.last_box = detection
            track.last_seen = time
            if detection.area > track.best_area:
                track.best_area = detection.area
                track.thumbnail_png = _make_thumbnail(cv2, np, frame, detection)
        else:
            thumbnail = _make_thumbnail(cv2, np, frame, detection)
            tracks.append(
                _FaceTrack(
                    boxes=[detection],
                    last_box=detection,
                    last_seen=time,
                    best_area=detection.area,
                    thumbnail_png=thumbnail,
                )
            )
            used_tracks.add(len(tracks) - 1)


def _match_score(detection: FaceBox, previous: FaceBox) -> float:
    iou = _iou(detection, previous)
    if iou >= 0.25:
        return 0.5 + iou
    detection_center = detection.center
    previous_center = previous.center
    distance = math.hypot(
        detection_center[0] - previous_center[0],
        detection_center[1] - previous_center[1],
    )
    reference = max(detection.width, detection.height, previous.width, previous.height, 1)
    normalized = distance / reference
    if normalized <= 1.2:
        return 0.5 - normalized / 3.0
    return -1.0


def _iou(first: FaceBox, second: FaceBox) -> float:
    left = max(first.x, second.x)
    top = max(first.y, second.y)
    right = min(first.x + first.width, second.x + second.width)
    bottom = min(first.y + first.height, second.y + second.height)
    intersection = max(0, right - left) * max(0, bottom - top)
    if intersection <= 0:
        return 0.0
    union = first.area + second.area - intersection
    return intersection / max(union, 1)


def _make_thumbnail(cv2: object, np: object, frame: object, box: FaceBox) -> bytes:
    frame_height, frame_width = frame.shape[:2]
    margin = 0.25
    x = int(max(0, box.x - box.width * margin))
    y = int(max(0, box.y - box.height * margin))
    right = int(min(frame_width, box.x + box.width + box.width * margin))
    bottom = int(min(frame_height, box.y + box.height + box.height * margin))
    if right <= x or bottom <= y:
        return b""

    crop = frame[y:bottom, x:right]
    crop_height, crop_width = crop.shape[:2]
    side = max(crop_height, crop_width, 1)
    canvas = np.full((side, side, 3), 18, dtype=np.uint8)
    offset_x = (side - crop_width) // 2
    offset_y = (side - crop_height) // 2
    canvas[offset_y : offset_y + crop_height, offset_x : offset_x + crop_width] = crop
    thumbnail = cv2.resize(canvas, (_THUMBNAIL_SIDE_PX, _THUMBNAIL_SIDE_PX), interpolation=cv2.INTER_AREA)
    encoded_ok, encoded = cv2.imencode(".png", thumbnail)
    return encoded.tobytes() if encoded_ok else b""


def _minimum_track_hits(sample_interval: float) -> int:
    return 2 if sample_interval < 4.0 else 1


def _finalize_tracks(tracks: Sequence[_FaceTrack], sample_interval: float) -> tuple[DetectedFace, ...]:
    minimum_hits = _minimum_track_hits(sample_interval)
    kept = [track for track in tracks if len(track.boxes) >= minimum_hits]
    kept.sort(
        key=lambda track: (
            len(track.boxes) * sample_interval,
            max(box.area for box in track.boxes),
        ),
        reverse=True,
    )

    faces: list[DetectedFace] = []
    for index, track in enumerate(kept, start=1):
        faces.append(
            DetectedFace(
                face_id=index,
                boxes=tuple(track.boxes),
                thumbnail_png=track.thumbnail_png,
                screen_time=len(track.boxes) * sample_interval,
                first_seen=min(box.time for box in track.boxes),
                last_seen=max(box.time for box in track.boxes),
            )
        )
    return tuple(faces)


def _median(values: Sequence[float | int]) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return 0.0
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2.0


def _emit(progress_callback: FaceProgressCallback | None, value: float, status: str) -> None:
    if progress_callback is not None:
        progress_callback(max(0.0, min(1.0, value)), status)


def _emit_count(count_callback: FaceCountCallback | None, count: int) -> None:
    if count_callback is not None:
        count_callback(max(0, int(count)))
