# Speech Cutter

Speech Cutter is a small desktop app that loads a video, finds the parts with human speech, and exports a new MP4 next to the original file.

## What it does

- Accepts common video formats like `mp4`, `webm`, `mkv`, `mov`, and more.
- Detects spoken audio with the bundled Silero VAD ONNX model.
- Removes non-speech stretches and joins the kept parts into a single MP4.
- Saves the result as `<original_name>_speech_only.mp4` in the same folder as the source video.
- Uses the natural trim profile by default, with a simple padding control in seconds.
- Can optionally crop selected segments with a visual frame picker that stays locked to the original video aspect ratio.
- Can detect distinct faces and optionally aim the zooms at a chosen face instead of a manual crop.
- Can add a slow push-in to longer speech segments for subtle visual motion, centered on the frame or on the selected face.
- Adds looped "Monkeys Spinning Monkeys" background music by default, with a checkbox to turn it off and a remembered volume slider.
- Can burn in TikTok-style subtitles with a highlighted active word.
- Offers an optional enhanced Polish recognition mode using a larger speech-to-text model and Polish-specific decoding.
- Supports either local Whisper or Groq API transcription (`whisper-large-v3-turbo` / `whisper-large-v3`) with word timestamps.
- Can send the transcript through Gemini before subtitle generation to correct clear recognition and logical-context errors without changing word timing.
- Can automatically mute detected profanity and censor it in the subtitles.
- Uses a responsive dark interface built with CustomTkinter, with separate preparation, video, and audio workspaces.
- Runs cuts, crop/motion effects, captions, music, profanity muting, and audio limiting in one optimized FFmpeg render.
- Also supports a CLI mode.

## Requirements

- Python 3.11 or newer
- `ffmpeg` and `ffprobe` available in `PATH`

This machine already has `ffmpeg`, so the main extra step is installing Python packages from `requirements.txt`.
The first caption-enabled run also downloads a Whisper speech-to-text model.

## Run it

### Option 1: double-click launcher

Run [run.bat](run.bat).

The launcher will:

1. Create `.venv` if it does not exist.
2. Install the required Python packages.
3. Start the GUI.

### Option 2: terminal

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
.\.venv\Scripts\python app.py
```

## CLI example

```powershell
.\.venv\Scripts\python app.py input.mp4 --padding-seconds 0.18 --push-in --captions --profanity-filter --enhanced-polish-recognition
```

To enable Gemini correction, enter the key in the masked GUI field or set it for the current terminal session:

```powershell
$env:GEMINI_API_KEY="your-key"
.\.venv\Scripts\python app.py input.mp4 --captions --ai-transcript-correction
```

To use Groq instead of running Whisper locally:

```powershell
$env:GROQ_API_KEY="your-key"
.\.venv\Scripts\python app.py input.mp4 --captions --transcription-provider groq
```

To detect faces and aim zooms at the largest visible face:

```powershell
.\.venv\Scripts\python app.py input.mp4 --face-zoom
```

To follow specific detected faces, run once with `--face-zoom` to see the detected ids in the log, then pass the ids (repeat `--face-id` for several faces):

```powershell
.\.venv\Scripts\python app.py input.mp4 --face-zoom --face-id 2 --face-padding 0.4 --face-max-zoom 1.8
```

## Face-targeted zooms

1. Load a video and enable "Włącz dynamiczny kadr".
2. Set "Cel przybliżeń" to "Twarze".
3. Click "Wykryj twarze". The scan samples frames, groups detections into distinct face tracks, and shows a thumbnail per face.
4. Pick one or more faces in the dialog, or leave everything unchecked to automatically follow the largest visible face in each segment.
5. Adjust "Oddech wokół twarzy" and "Limit przybliżenia twarzy", then export.

Face mode reuses the existing zoom cadence controls ("Co który segment mowy", "Pomiń segmenty krótsze niż", "Czas przybliżenia"). In queue mode you can scan every video at once; videos without a scan are analyzed automatically during export. If no face is found in a segment, the manual crop is used when one exists; otherwise that segment is rendered without a zoom. Switching back to "Kadr ręczny" restores the manual crop picker.

## Notes

- The exported file is always MP4 for wide compatibility.
- Very noisy videos can still produce occasional false positives or false negatives.
- The first run may take a little longer while dependencies install.
- Whisper analyzes the clean source speech before music is mixed in, and word timings are remapped onto the trimmed timeline.
- Caption-enabled exports are encoded only once; subtitles and audio effects no longer require a second video render.
- Enhanced Polish recognition is opt-in. Its larger model is slower on CPU and downloads separately on first use.
- Gemini correction is opt-in and sends the transcript text (not the video or audio) to the Gemini API.
- Groq transcription is opt-in and uploads compressed mono FLAC audio. Local Whisper remains the default and works offline.
- The GUI can remember Groq and Gemini API keys securely in Windows Credential Manager; keys are not written to the app's JSON settings.
- Face detection runs locally with OpenCV and the bundled YuNet model; no video data is uploaded.

## Third-party components

This app bundles the official Silero VAD 16 kHz ONNX model under the MIT license.

- Repo: [snakers4/silero-vad](https://github.com/snakers4/silero-vad)
- License copy: [third_party_licenses/Silero-VAD-LICENSE.txt](third_party_licenses/Silero-VAD-LICENSE.txt)

It also bundles "Monkeys Spinning Monkeys" by Kevin MacLeod for optional background music.

- Source: [incompetech](https://incompetech.com/music/royalty-free/index.html?isrc=USUAN1400011)
- License copy: [third_party_licenses/Monkeys-Spinning-Monkeys-LICENSE.txt](third_party_licenses/Monkeys-Spinning-Monkeys-LICENSE.txt)

It also bundles the YuNet face detection model, loaded through OpenCV.

- Project: [opencv/opencv_zoo](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet)
- License copy: [third_party_licenses/YuNet-LICENSE.txt](third_party_licenses/YuNet-LICENSE.txt)
