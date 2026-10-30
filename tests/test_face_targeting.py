from __future__ import annotations

from pathlib import Path
from unittest.mock import patch
import importlib.util
import tempfile
import unittest

from speech_cutter.faces import (
    DetectedFace,
    FaceBox,
    _detect_anime_frame_faces,
    _merge_face_detections,
    choose_face_for_range,
    compute_face_crop,
    detect_faces,
    face_box_for_range,
)
from speech_cutter.pipeline import (
    CropSettings,
    FaceZoomSettings,
    ProcessingError,
    _build_face_zoom_clips,
    _build_filter_script,
    _build_push_in_filters,
    _expand_segment_clips,
    _prepare_crop_settings,
)
from speech_cutter.vad import SpeechSegment


def make_face(face_id: int, x: int, y: int, width: int, height: int, *, start: float = 0.0, end: float = 4.0) -> DetectedFace:
    boxes = tuple(
        FaceBox(time=time, x=x, y=y, width=width, height=height, score=0.9)
        for time in (start, (start + end) / 2, end)
    )
    return DetectedFace(
        face_id=face_id,
        boxes=boxes,
        screen_time=end - start,
        first_seen=start,
        last_seen=end,
    )


class _FakeCapture:
    def __init__(self, frame: object) -> None:
        self.frame = frame
        self.read_calls = 0

    def isOpened(self) -> bool:
        return True

    def get(self, prop: int) -> float:
        import cv2

        return {
            cv2.CAP_PROP_FRAME_WIDTH: float(self.frame.shape[1]),
            cv2.CAP_PROP_FRAME_HEIGHT: float(self.frame.shape[0]),
            cv2.CAP_PROP_FPS: 10.0,
            cv2.CAP_PROP_FRAME_COUNT: 100.0,
        }.get(prop, 0.0)

    def set(self, prop: int, value: float) -> bool:
        return True

    def read(self) -> tuple[bool, object]:
        self.read_calls += 1
        return True, self.frame

    def release(self) -> None:
        pass


class _FakeFaceDetector:
    def __init__(self, detections: object) -> None:
        self.detections = detections

    def setInputSize(self, size: tuple[int, int]) -> None:
        pass

    def detect(self, frame: object) -> tuple[int, object]:
        return 1, self.detections


class _FakeFaceDetectorFactory:
    def __init__(self, detections: object) -> None:
        self.detections = detections

    def create(self, *args: object, **kwargs: object) -> _FakeFaceDetector:
        return _FakeFaceDetector(self.detections)


class _FakeOnnxInput:
    name = "images"


class _FakeOnnxDetector:
    """Minimal stand-in for an ONNX Runtime session."""

    def __init__(self, output: object) -> None:
        self.output = output
        self.tensors: list[object] = []

    def get_inputs(self) -> list[object]:
        return [_FakeOnnxInput()]

    def run(self, output_names: object, feed: dict[str, object]) -> list[object]:
        self.tensors.append(feed["images"])
        return [self.output]


_CV2_AVAILABLE = importlib.util.find_spec("cv2") is not None


class FaceScanTests(unittest.TestCase):
    def setUp(self) -> None:
        import numpy as np

        empty = _FakeOnnxDetector(np.zeros((1, 5, 0), dtype=np.float32))
        anime_patch = patch("speech_cutter.faces._create_anime_detector", lambda _ort: empty)
        anime_patch.start()
        self.addCleanup(anime_patch.stop)

    @unittest.skipUnless(_CV2_AVAILABLE, "OpenCV is not installed")
    def test_scan_stops_after_requested_face_count(self) -> None:
        import cv2
        import numpy as np

        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        detections = np.array(
            [[100, 50, 80, 80, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.95]],
            dtype=np.float32,
        )
        capture = _FakeCapture(frame)
        factory = _FakeFaceDetectorFactory(detections)

        with tempfile.NamedTemporaryFile(suffix=".mp4") as handle:
            video_path = Path(handle.name)
            with (
                patch.object(cv2, "VideoCapture", lambda *_args, **_kwargs: capture),
                patch.object(cv2, "FaceDetectorYN", factory),
            ):
                analysis = detect_faces(
                    video_path,
                    duration=10.0,
                    sample_interval=0.5,
                    target_face_count=1,
                )

        self.assertTrue(analysis.stopped_early)
        self.assertEqual(analysis.target_face_count, 1)
        self.assertEqual(analysis.samples_scanned, 2)
        self.assertEqual(capture.read_calls, 2)
        self.assertEqual(len(analysis.faces), 1)

    @unittest.skipUnless(_CV2_AVAILABLE, "OpenCV is not installed")
    def test_scan_without_target_reads_all_samples(self) -> None:
        import cv2
        import numpy as np

        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        detections = np.array(
            [[100, 50, 80, 80, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.95]],
            dtype=np.float32,
        )
        capture = _FakeCapture(frame)
        factory = _FakeFaceDetectorFactory(detections)

        with tempfile.NamedTemporaryFile(suffix=".mp4") as handle:
            video_path = Path(handle.name)
            with (
                patch.object(cv2, "VideoCapture", lambda *_args, **_kwargs: capture),
                patch.object(cv2, "FaceDetectorYN", factory),
            ):
                analysis = detect_faces(
                    video_path,
                    duration=10.0,
                    sample_interval=0.5,
                )

        self.assertFalse(analysis.stopped_early)
        self.assertEqual(analysis.samples_scanned, 20)
        self.assertEqual(capture.read_calls, 20)
        self.assertEqual(len(analysis.faces), 1)

    @unittest.skipUnless(_CV2_AVAILABLE, "OpenCV is not installed")
    def test_scan_reports_live_face_count(self) -> None:
        import cv2
        import numpy as np

        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        detections = np.array(
            [[100, 50, 80, 80, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.95]],
            dtype=np.float32,
        )
        capture = _FakeCapture(frame)
        factory = _FakeFaceDetectorFactory(detections)
        counts: list[int] = []

        with tempfile.NamedTemporaryFile(suffix=".mp4") as handle:
            video_path = Path(handle.name)
            with (
                patch.object(cv2, "VideoCapture", lambda *_args, **_kwargs: capture),
                patch.object(cv2, "FaceDetectorYN", factory),
            ):
                detect_faces(
                    video_path,
                    duration=10.0,
                    sample_interval=0.5,
                    count_callback=counts.append,
                )

        self.assertEqual(counts, [1])

    @unittest.skipUnless(_CV2_AVAILABLE, "OpenCV is not installed")
    def test_user_stop_keeps_already_detected_faces(self) -> None:
        import cv2
        import numpy as np

        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        detections = np.array(
            [[100, 50, 80, 80, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.95]],
            dtype=np.float32,
        )
        capture = _FakeCapture(frame)
        factory = _FakeFaceDetectorFactory(detections)
        state = {"stop": False}
        counts: list[int] = []

        def on_count(count: int) -> None:
            counts.append(count)
            state["stop"] = True

        with tempfile.NamedTemporaryFile(suffix=".mp4") as handle:
            video_path = Path(handle.name)
            with (
                patch.object(cv2, "VideoCapture", lambda *_args, **_kwargs: capture),
                patch.object(cv2, "FaceDetectorYN", factory),
            ):
                analysis = detect_faces(
                    video_path,
                    duration=10.0,
                    sample_interval=0.5,
                    count_callback=on_count,
                    stop_callback=lambda: state["stop"],
                )

        self.assertTrue(analysis.stopped_early)
        self.assertTrue(analysis.stopped_by_user)
        self.assertEqual(analysis.samples_scanned, 2)
        self.assertEqual(capture.read_calls, 2)
        self.assertEqual(counts, [1])
        self.assertEqual(len(analysis.faces), 1)

    @unittest.skipUnless(_CV2_AVAILABLE, "OpenCV is not installed")
    def test_scan_counts_anime_face_when_real_detector_finds_none(self) -> None:
        import cv2
        import numpy as np

        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        anime = np.zeros((1, 5, 1), dtype=np.float32)
        anime[0, :, 0] = (240, 240, 200, 200, 0.86)
        anime_detector = _FakeOnnxDetector(anime)
        capture = _FakeCapture(frame)
        factory = _FakeFaceDetectorFactory(None)

        with tempfile.NamedTemporaryFile(suffix=".mp4") as handle:
            video_path = Path(handle.name)
            with (
                patch.object(cv2, "VideoCapture", lambda *_args, **_kwargs: capture),
                patch.object(cv2, "FaceDetectorYN", factory),
                patch("speech_cutter.faces._create_anime_detector", lambda _ort: anime_detector),
            ):
                analysis = detect_faces(video_path, duration=2.0, sample_interval=1.0)

        self.assertEqual(len(analysis.faces), 1)
        self.assertEqual(len(anime_detector.tensors), 2)
        self.assertEqual(tuple(anime_detector.tensors[0].shape), (1, 3, 480, 480))
        box = analysis.faces[0].representative_box
        self.assertEqual((box.x, box.y, box.width, box.height), (93, 70, 134, 100))
        self.assertAlmostEqual(box.score, 0.86, places=5)


class AnimeFaceDetectionTests(unittest.TestCase):
    @unittest.skipUnless(_CV2_AVAILABLE, "OpenCV is not installed")
    def test_maps_yolo_output_to_source_pixels(self) -> None:
        import cv2
        import numpy as np

        frame = np.zeros((480, 480, 3), dtype=np.uint8)
        output = np.zeros((1, 5, 1), dtype=np.float32)
        output[0, :, 0] = (240, 240, 120, 120, 0.9)
        detector = _FakeOnnxDetector(output)

        faces = _detect_anime_frame_faces(cv2, np, detector, frame)

        self.assertEqual(len(faces), 1)
        self.assertEqual((faces[0].x, faces[0].y, faces[0].width, faces[0].height), (180, 180, 120, 120))
        self.assertAlmostEqual(faces[0].score, 0.9, places=5)
        self.assertEqual(tuple(detector.tensors[0].shape), (1, 3, 480, 480))
        self.assertGreaterEqual(float(detector.tensors[0].min()), 0.0)
        self.assertLessEqual(float(detector.tensors[0].max()), 1.0)

    @unittest.skipUnless(_CV2_AVAILABLE, "OpenCV is not installed")
    def test_accepts_transposed_yolo_output(self) -> None:
        import cv2
        import numpy as np

        frame = np.zeros((480, 480, 3), dtype=np.uint8)
        output = np.zeros((1, 1, 5), dtype=np.float32)
        output[0, 0] = (240, 240, 120, 120, 0.9)
        detector = _FakeOnnxDetector(output)

        faces = _detect_anime_frame_faces(cv2, np, detector, frame)

        self.assertEqual(len(faces), 1)
        self.assertEqual((faces[0].x, faces[0].y, faces[0].width, faces[0].height), (180, 180, 120, 120))

    @unittest.skipUnless(_CV2_AVAILABLE, "OpenCV is not installed")
    def test_suppresses_overlapping_duplicate_boxes(self) -> None:
        import cv2
        import numpy as np

        frame = np.zeros((480, 480, 3), dtype=np.uint8)
        output = np.zeros((1, 5, 2), dtype=np.float32)
        output[0, :, 0] = (240, 240, 120, 120, 0.9)
        output[0, :, 1] = (244, 244, 116, 116, 0.8)
        detector = _FakeOnnxDetector(output)

        faces = _detect_anime_frame_faces(cv2, np, detector, frame)

        self.assertEqual(len(faces), 1)
        self.assertAlmostEqual(faces[0].score, 0.9, places=5)

    @unittest.skipUnless(_CV2_AVAILABLE, "OpenCV is not installed")
    def test_ignores_low_scores_tiny_faces_and_flat_boxes(self) -> None:
        import cv2
        import numpy as np

        frame = np.zeros((480, 480, 3), dtype=np.uint8)
        output = np.zeros((1, 5, 4), dtype=np.float32)
        output[0, :, 0] = (240, 240, 120, 120, 0.2)  # below score threshold
        output[0, :, 1] = (100, 100, 10, 10, 0.9)  # too small
        output[0, :, 2] = (240, 100, 300, 40, 0.9)  # too flat
        output[0, :, 3] = (240, 240, 120, 120, 0.8)  # valid face
        detector = _FakeOnnxDetector(output)

        faces = _detect_anime_frame_faces(cv2, np, detector, frame)

        self.assertEqual(len(faces), 1)
        self.assertAlmostEqual(faces[0].score, 0.8, places=5)

    @unittest.skipUnless(_CV2_AVAILABLE, "OpenCV is not installed")
    def test_clamps_boxes_at_frame_edges(self) -> None:
        import cv2
        import numpy as np

        frame = np.zeros((480, 480, 3), dtype=np.uint8)
        output = np.zeros((1, 5, 1), dtype=np.float32)
        output[0, :, 0] = (10, 10, 100, 100, 0.9)
        detector = _FakeOnnxDetector(output)

        faces = _detect_anime_frame_faces(cv2, np, detector, frame)

        self.assertEqual(len(faces), 1)
        self.assertEqual((faces[0].x, faces[0].y), (0, 0))
        self.assertEqual((faces[0].width, faces[0].height), (60, 60))


class FaceMergeTests(unittest.TestCase):
    def test_drops_anime_box_that_overlaps_real_face(self) -> None:
        real = FaceBox(time=0.0, x=100, y=100, width=100, height=100, score=0.9)
        anime = FaceBox(time=0.0, x=108, y=108, width=96, height=96, score=0.8)

        merged = _merge_face_detections([real], [anime])

        self.assertEqual(merged, [real])

    def test_drops_anime_box_contained_in_real_face(self) -> None:
        real = FaceBox(time=0.0, x=100, y=100, width=200, height=200, score=0.9)
        anime = FaceBox(time=0.0, x=150, y=150, width=80, height=80, score=0.8)

        merged = _merge_face_detections([real], [anime])

        self.assertEqual(merged, [real])

    def test_keeps_distinct_anime_face(self) -> None:
        real = FaceBox(time=0.0, x=100, y=100, width=100, height=100, score=0.9)
        anime = FaceBox(time=0.0, x=700, y=100, width=100, height=100, score=0.8)

        merged = _merge_face_detections([real], [anime])

        self.assertEqual(merged, [real, anime])

    def test_keeps_all_real_faces_and_adds_new_anime_tracks(self) -> None:
        first = FaceBox(time=0.0, x=100, y=100, width=100, height=100, score=0.9)
        second = FaceBox(time=0.0, x=500, y=100, width=80, height=80, score=0.85)
        duplicate = FaceBox(time=0.0, x=105, y=105, width=95, height=95, score=0.8)
        new_face = FaceBox(time=0.0, x=900, y=300, width=120, height=120, score=0.75)

        merged = _merge_face_detections([first, second], [duplicate, new_face])

        self.assertEqual(merged, [first, second, new_face])


class FaceCropTests(unittest.TestCase):
    def test_builds_aspect_locked_crop_around_face(self) -> None:
        box = FaceBox(time=1.0, x=860, y=400, width=200, height=200)
        crop = compute_face_crop(box, frame_width=1920, frame_height=1080, padding=0.35, max_zoom=1.8)

        self.assertIsNotNone(crop)
        x, y, width, height = crop  # type: ignore[misc]
        self.assertAlmostEqual(width / height, 1920 / 1080, places=2)
        self.assertLessEqual(width, 1920)
        self.assertLessEqual(height, 1080)
        self.assertGreaterEqual(x, 0)
        self.assertGreaterEqual(y, 0)
        self.assertLessEqual(x + width, 1920)
        self.assertLessEqual(y + height, 1080)
        self.assertGreater(width, 0)

    def test_returns_none_when_face_already_fills_frame(self) -> None:
        box = FaceBox(time=1.0, x=700, y=300, width=560, height=500)
        self.assertIsNone(
            compute_face_crop(box, frame_width=1920, frame_height=1080, padding=0.35, max_zoom=1.8)
        )

    def test_selected_face_wins_over_larger_automatic_face(self) -> None:
        large = make_face(1, 100, 100, 400, 400)
        small = make_face(2, 900, 300, 120, 120)

        automatic = choose_face_for_range((large, small), 0.0, 4.0)
        selected = choose_face_for_range((large, small), 0.0, 4.0, selected_face_ids=(2,))

        self.assertEqual(automatic.face_id, 1)  # type: ignore[union-attr]
        self.assertEqual(selected.face_id, 2)  # type: ignore[union-attr]

    def test_uses_nearest_visible_box_for_range(self) -> None:
        face = make_face(1, 100, 100, 200, 200)
        box = face_box_for_range(face, 0.5, 1.5)
        self.assertEqual(box.width, 200)
        self.assertEqual(box.height, 200)


class FaceFilterTests(unittest.TestCase):
    def test_face_mode_allows_crop_settings_without_manual_rect(self) -> None:
        crop = CropSettings(
            enabled=True,
            every_n_segments=1,
            min_segment_seconds=0.0,
            zoom_duration_seconds=0.0,
            mid_zoom_on_long_segments=False,
        )

        prepared = _prepare_crop_settings(
            crop,
            video_width=1920,
            video_height=1080,
            allow_missing_rect=True,
        )
        self.assertIsNotNone(prepared)

        with self.assertRaises(ProcessingError):
            _prepare_crop_settings(crop, video_width=1920, video_height=1080)

    def test_face_mode_without_manual_rect_builds_face_clips(self) -> None:
        face = make_face(1, 860, 400, 200, 200, start=0.0, end=4.0)
        clips = _expand_segment_clips(
            [SpeechSegment(0.0, 4.0)],
            crop_settings=CropSettings(
                enabled=True,
                every_n_segments=1,
                min_segment_seconds=0.0,
                zoom_duration_seconds=0.0,
                mid_zoom_on_long_segments=False,
            ),
            motion_settings=None,
            face_settings=FaceZoomSettings(enabled=True, faces=(face,), padding=0.35, max_zoom=1.8),
            video_width=1920,
            video_height=1080,
        )

        self.assertEqual(len(clips), 1)
        self.assertEqual(clips[0].effect, "face_crop")
        self.assertIsNotNone(clips[0].crop_rect)

    def test_face_zoom_builds_face_crop_clip(self) -> None:
        face = make_face(1, 860, 400, 200, 200, start=0.0, end=4.0)
        settings = FaceZoomSettings(enabled=True, faces=(face,), padding=0.35, max_zoom=1.8)
        crop = CropSettings(
            enabled=True,
            every_n_segments=1,
            min_segment_seconds=0.0,
            zoom_duration_seconds=0.0,
            mid_zoom_on_long_segments=False,
        )

        clips = _build_face_zoom_clips(
            SpeechSegment(0.0, 4.0),
            face_settings=settings,
            crop_settings=crop,
            video_width=1920,
            video_height=1080,
            alignment="start",
        )

        self.assertIsNotNone(clips)
        self.assertEqual(len(clips), 1)  # type: ignore[arg-type]
        self.assertEqual(clips[0].effect, "face_crop")  # type: ignore[index]
        self.assertIsNotNone(clips[0].crop_rect)  # type: ignore[index]

    def test_filter_script_contains_face_crop(self) -> None:
        face = make_face(1, 860, 400, 200, 200, start=0.0, end=2.0)
        settings = FaceZoomSettings(enabled=True, faces=(face,), padding=0.35, max_zoom=1.8)
        script = _build_filter_script(
            [SpeechSegment(0.0, 2.0)],
            video_width=1920,
            video_height=1080,
            crop_settings=CropSettings(
                enabled=True,
                every_n_segments=1,
                min_segment_seconds=0.0,
                zoom_duration_seconds=0.0,
                mid_zoom_on_long_segments=False,
            ),
            motion_settings=None,
            face_settings=settings,
            background_music_settings=None,
            audio_boost_settings=None,
            kept_duration=2.0,
        )

        self.assertIn("crop=w=", script)
        self.assertIn("y=", script)
        self.assertNotIn("crop=w=0", script)

    def test_push_in_can_target_face_center(self) -> None:
        filters = _build_push_in_filters(
            video_width=1920,
            video_height=1080,
            segment_duration=5.0,
            max_zoom=1.08,
            target_center=(960, 540),
        )
        crop_filter = filters[-1]
        self.assertIn("max(0,min(iw-ow,960.000*(", crop_filter)
        self.assertIn("max(0,min(ih-oh,540.000*(", crop_filter)

    def test_push_in_stays_centered_without_face(self) -> None:
        filters = _build_push_in_filters(
            video_width=1920,
            video_height=1080,
            segment_duration=5.0,
            max_zoom=1.08,
        )
        self.assertIn("x='(iw-ow)/2'", filters[-1])
        self.assertIn("y='(ih-oh)/2'", filters[-1])


if __name__ == "__main__":
    unittest.main()
