from pathlib import Path
from unittest.mock import MagicMock, patch

from numpy import arange, array_equal

from denoiser.audio_processing import (
    build_noise_profile,
    extract_audio,
    mux_audio,
)


@patch("denoiser.audio_processing.VideoFileClip")
def test_extract_audio_success(mock_video_clip):
    audio = MagicMock()
    video_instance = MagicMock()
    video_instance.audio = audio
    mock_video_clip.return_value = video_instance

    video_path = Path("input.mp4")
    output_wav = Path("output.wav")
    extract_audio(video_path, output_wav)

    mock_video_clip.assert_called_once_with(str(video_path))
    audio.write_audiofile.assert_called_once_with(
        str(output_wav), verbose=False, logger=None
    )


@patch("denoiser.audio_processing.AudioFileClip")
@patch("denoiser.audio_processing.VideoFileClip")
def test_mux_audio_success(
    mock_video_clip,
    mock_audio_clip,
):
    video_instance = MagicMock()
    audio_instance = MagicMock()
    final_video = MagicMock()

    mock_video_clip.return_value = video_instance
    mock_audio_clip.return_value = audio_instance
    video_instance.set_audio.return_value = final_video

    video_path = Path("input.mp4")
    clean_wav = Path("clean.wav")
    output_path = Path("output.mp4")

    mux_audio(video_path, clean_wav, output_path)

    mock_video_clip.assert_called_once_with(str(video_path))
    mock_audio_clip.assert_called_once_with(str(clean_wav))

    video_instance.set_audio.assert_called_once_with(
        audio_instance
    )

    final_video.write_videofile.assert_called_once_with(
        str(output_path), codec="libx264", audio_codec="aac"
    )


def test_build_noise_profile_success():
    y = arange(1000)
    sr = 100
    duration = 5

    noise_profile = build_noise_profile(y, sr, duration)

    assert len(noise_profile) == 500
    assert array_equal(noise_profile, y[:500])
