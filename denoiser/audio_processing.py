from pathlib import Path

from moviepy import AudioFileClip, VideoFileClip
from numpy import ndarray as np_ndarray


def extract_audio(video_path: Path, wav_path: Path) -> None:
    video = VideoFileClip(str(video_path))
    video.audio.write_audiofile(
        str(wav_path), verbose=False, logger=None
    )


def build_noise_profile(
    y: np_ndarray, sr: int, duration: float
) -> np_ndarray:
    return y[: int(sr * duration)]


def mux_audio(
    video_path: Path, clean_wav: Path, output_path: Path
):
    video = VideoFileClip(str(video_path))
    clean_audio = AudioFileClip(str(clean_wav))
    final = video.set_audio(clean_audio)
    final.write_videofile(
        str(output_path), codec="libx264", audio_codec="aac"
    )
