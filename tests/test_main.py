from pathlib import Path
from tempfile import NamedTemporaryFile
from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from denoiser.main import (
    app,
)

runner = CliRunner()


def test_denoise_success():

    with NamedTemporaryFile(
        suffix=".mp4", delete=False
    ) as temp_input:
        input_path = temp_input.name

    with NamedTemporaryFile(
        suffix=".mp4", delete=False
    ) as temp_output:
        output_path = temp_output.name

    Path(output_path).unlink()

    try:
        with (
            patch(
                "denoiser.main.extract_audio"
            ) as mock_extract,
            patch("denoiser.main.load_audio") as mock_load,
            patch(
                "denoiser.main.build_noise_profile"
            ) as mock_build,
            patch("denoiser.main.reduce_noise") as mock_reduce,
            patch("denoiser.main.save_audio") as mock_save,
            patch("denoiser.main.mux_audio") as mock_mux,
        ):
            mock_load.return_value = (MagicMock(), 44100)
            mock_build.return_value = MagicMock()
            mock_reduce.return_value = MagicMock()

            result = runner.invoke(
                app,
                [
                    input_path,
                    output_path,
                    "--noise-duration",
                    "2",
                    "--prop-decrease",
                    "0.8",
                ],
            )

            assert result.exit_code == 0
            assert f"Saved: {output_path}" in result.output

            mock_extract.assert_called_once()
            mock_load.assert_called_once()
            mock_build.assert_called_once()
            mock_reduce.assert_called_once()
            mock_save.assert_called_once()
            mock_mux.assert_called_once()
    finally:
        Path(input_path).unlink(missing_ok=True)
        Path(output_path).unlink(missing_ok=True)


def test_overwrite_guard_negative_confirm():

    with NamedTemporaryFile(
        suffix=".mp4", delete=False
    ) as temp_input:
        input_path = temp_input.name

    with NamedTemporaryFile(
        suffix=".mp4", delete=False
    ) as temp_output:
        output_path = temp_output.name

    try:
        result = runner.invoke(
            app,
            [
                input_path,
                output_path,
                "--noise-duration",
                "2",
                "--prop-decrease",
                "0.8",
            ],
            input="n\n",
        )
        assert result.exit_code == 1
        assert (
            result.output == f"{output_path} already exists. "
            "Overwrite? [y/N]: "
            "n\nAborted.\n"
        )

    finally:
        Path(input_path).unlink(missing_ok=True)
        Path(output_path).unlink(missing_ok=True)


def test_overwrite_guard_confirm():
    with NamedTemporaryFile(
        suffix=".mp4", delete=False
    ) as temp_input:
        input_path = temp_input.name

    with NamedTemporaryFile(
        suffix=".mp4", delete=False
    ) as temp_output:
        output_path = temp_output.name

    try:
        with (
            patch(
                "denoiser.main.extract_audio"
            ) as mock_extract,
            patch("denoiser.main.load_audio") as mock_load,
            patch(
                "denoiser.main.build_noise_profile"
            ) as mock_build,
            patch("denoiser.main.reduce_noise") as mock_reduce,
            patch("denoiser.main.save_audio") as mock_save,
            patch("denoiser.main.mux_audio") as mock_mux,
        ):
            mock_load.return_value = (MagicMock(), 44100)
            mock_build.return_value = MagicMock()
            mock_reduce.return_value = MagicMock()

            result = runner.invoke(
                app,
                [
                    input_path,
                    output_path,
                    "--noise-duration",
                    "2",
                    "--prop-decrease",
                    "0.8",
                ],
                input="y\n",
            )

            assert (
                f"{output_path} already exists. "
                f"Overwrite? [y/N]: y\nSaved: {output_path}"
            ) in result.output
            assert result.exit_code == 0

            mock_extract.assert_called_once()
            mock_load.assert_called_once()
            mock_build.assert_called_once()
            mock_reduce.assert_called_once()
            mock_save.assert_called_once()
            mock_mux.assert_called_once()

    finally:
        Path(input_path).unlink(missing_ok=True)
        Path(output_path).unlink(missing_ok=True)


def test_extract_audio_raises_error():
    with NamedTemporaryFile(
        suffix=".mp4", delete=False
    ) as temp_input:
        input_path = temp_input.name

    with NamedTemporaryFile(
        suffix=".mp4", delete=False
    ) as temp_output:
        output_path = temp_output.name

    Path(output_path).unlink()
    try:
        with patch(
            "denoiser.main.extract_audio"
        ) as mock_extract:
            mock_extract.side_effect = ValueError
            result = runner.invoke(
                app,
                [
                    input_path,
                    output_path,
                    "--noise-duration",
                    "2",
                    "--prop-decrease",
                    "0.8",
                ],
            )
            assert result.exit_code == 1
    finally:
        Path(input_path).unlink(missing_ok=True)
        Path(output_path).unlink(missing_ok=True)


def test_build_noise_profile_raises_error():
    with NamedTemporaryFile(
        suffix=".mp4", delete=False
    ) as temp_input:
        input_path = temp_input.name

    with NamedTemporaryFile(
        suffix=".mp4", delete=False
    ) as temp_output:
        output_path = temp_output.name

    Path(output_path).unlink()
    try:
        with (
            patch(
                "denoiser.main.extract_audio"
            ) as mock_extract,
            patch("denoiser.main.load_audio") as mock_load,
            patch(
                "denoiser.main.build_noise_profile"
            ) as mock_build,
        ):
            mock_build.side_effect = ValueError
            result = runner.invoke(
                app,
                [
                    input_path,
                    output_path,
                    "--noise-duration",
                    "2",
                    "--prop-decrease",
                    "0.8",
                ],
            )
            assert result.exit_code == 1
            mock_extract.assert_called_once()
            mock_load.assert_called_once()
    finally:
        Path(input_path).unlink(missing_ok=True)
        Path(output_path).unlink(missing_ok=True)


def test_mux_audio_raises_error():
    with NamedTemporaryFile(
        suffix=".mp4", delete=False
    ) as temp_input:
        input_path = temp_input.name

    with NamedTemporaryFile(
        suffix=".mp4", delete=False
    ) as temp_output:
        output_path = temp_output.name

    Path(output_path).unlink()
    try:
        with (
            patch(
                "denoiser.main.extract_audio"
            ) as mock_extract,
            patch("denoiser.main.load_audio") as mock_load,
            patch(
                "denoiser.main.build_noise_profile"
            ) as mock_build,
            patch("denoiser.main.reduce_noise") as mock_reduce,
            patch("denoiser.main.save_audio") as mock_save,
            patch("denoiser.main.mux_audio") as mock_mux,
        ):
            mock_load.return_value = (MagicMock(), 44100)
            mock_build.return_value = MagicMock()
            mock_reduce.return_value = MagicMock()

            mock_mux.side_effect = ValueError
            result = runner.invoke(
                app,
                [
                    input_path,
                    output_path,
                    "--noise-duration",
                    "2",
                    "--prop-decrease",
                    "0.8",
                ],
            )
            assert result.exit_code == 1
            mock_extract.assert_called_once()
            mock_build.assert_called_once()
            mock_load.assert_called_once()
            mock_reduce.assert_called_once()
            mock_save.assert_called_once()

    finally:
        Path(input_path).unlink(missing_ok=True)
        Path(output_path).unlink(missing_ok=True)


def test_non_existent_input_file():
    with NamedTemporaryFile(
        suffix=".mp4", delete=False
    ) as temp_input:
        input_path = temp_input.name

    with NamedTemporaryFile(
        suffix=".mp4", delete=False
    ) as temp_output:
        output_path = temp_output.name

    Path(input_path).unlink()
    Path(output_path).unlink()

    try:
        result = runner.invoke(
            app,
            [
                input_path,
                output_path,
                "--noise-duration",
                "2",
                "--prop-decrease",
                "0.8",
            ],
        )

        assert result.exit_code == 2
        assert (
            "Invalid value for 'INPUT_FILE': "
            f"Path '{input_path}' does not exist."
            in result.output
        )

    finally:
        Path(input_path).unlink(missing_ok=True)
        Path(output_path).unlink(missing_ok=True)
