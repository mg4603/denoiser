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
    video = VideoFileClip(str(video_path))
    clean_audio = AudioFileClip(str(clean_wav))
    final = video.set_audio(clean_audio)
    final.write_videofile(
        str(output_path), codec="libx264", audio_codec="aac"
    )
