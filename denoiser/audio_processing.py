from pathlib import Path

from moviepy import AudioFileClip, VideoFileClip
from numpy import ndarray as np_ndarray


def extract_audio(video_path: Path, wav_path: Path) -> None:
    video = VideoFileClip(str(video_path))
    if video.audio is None:
        raise ValueError("Video file has no audio track")

    video.audio.write_audiofile(
        str(wav_path), verbose=False, logger=None
    )


def build_noise_profile(
    y: np_ndarray, sr: int, duration: float
) -> np_ndarray:
    if sr <= 0:
        raise ValueError(
            "Sampling rate must be greater than 0."
        )
    if duration < 0:
        raise ValueError(
            "Duration must be greater than or equal to 0."
        )
    if int(sr) != sr:
        raise TypeError("Sample rate must be an integer value.")
    return y[: min(int(sr * duration), len(y))]


def mux_audio(
    video_path: Path, clean_wav: Path, output_path: Path
):
    if not isinstance(video_path, Path):
        raise TypeError("video_path should be of type Path")

    if not isinstance(clean_wav, Path):
        raise TypeError("clean_wav should be of type Path")

    if not isinstance(output_path, Path):
        raise TypeError("output_path should be of type Path")

    if not video_path.exists():
        raise FileNotFoundError("Video file does not exist")

    if not clean_wav.exists():
        raise FileNotFoundError(
            "Clean audio file does not exist"
        )

    video = None
    clean_audio = None
    try:
        video = VideoFileClip(str(video_path))
    except (OSError, IOError):
        raise ValueError("Invalid file type for video file")

    try:
        clean_audio = AudioFileClip(str(clean_wav))
    except (OSError, IOError):
        raise ValueError("Invalid file type for clean audio")

    try:
        with video, clean_audio:
            final = video.set_audio(clean_audio)
            final.write_videofile(
                str(output_path),
                codec="libx264",
                audio_codec="aac",
            )
    finally:
        if video is not None:
            video.close()
        if clean_audio is not None:
            clean_audio.close()
