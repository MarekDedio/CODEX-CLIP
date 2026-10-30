from __future__ import annotations

import unittest

from speech_cutter.faces import (
    DetectedFace,
    FaceBox,
    choose_face_for_range,
    compute_face_crop,
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
