from pathlib import Path
from tempfile import TemporaryDirectory

import typer
from librosa import load as load_audio
from noisereduce import reduce_noise
from soundfile import write as save_audio

from denoiser.audio_processing import (
    build_noise_profile,
    extract_audio,
    mux_audio,
)

app = typer.Typer(
    name="denoiser",
    help="Denoise video audio using noisereduce",
    no_args_is_help=True,
)


@app.command()
def denoise(
    input_file: Path = typer.Argument(
        ..., exists=True, readable=True
    ),
    output: Path = typer.Argument(...),
    noise_duration: float = typer.Option(
        2.0, help="seconds used for noise profile"
    ),
    prop_decrease: float = typer.Option(
        1.0, help="Noise reduction aggressiveness (0-1)"
    ),
):
    with Path(TemporaryDirectory()) as tmp_dir:
        tmp_wav = tmp_dir / "audio.wav"
        clean_wav = tmp_dir / "clean.wav"

        try:
            extract_audio(input_file, tmp_wav)
        except ValueError as e:
            typer.echo(e)
            typer.Abort()

        y, sr = load_audio(tmp_wav, sr=None)

        try:
            noise_sample = build_noise_profile(
                y, sr, noise_duration
            )
        except ValueError as e:
            typer.echo(e)
            raise typer.Abort()

        y_denoised = reduce_noise(
            y=y,
            sr=sr,
            y_noise=noise_sample,
            prop_decrease=prop_decrease,
        )

        save_audio(clean_wav, y_denoised, sr)

        mux_audio(input_file, clean_wav, output)

        typer.echo(f"Saved: {output}")
