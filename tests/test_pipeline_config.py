import pytest
from pipeline_config import PipelineConfig

def test_pipeline_config_validate_success(tmp_path):
    video = tmp_path / "vid.mp4"
    audio = tmp_path / "audio.mp3"
    video.touch()
    audio.touch()
    config = PipelineConfig(
        video_paths=[str(video)],
        audio_path=str(audio),
        output_path="out.mp4",
        vignette_strength=0.5,
        watermark_opacity=0.4
    )
    config.validate()  # Should not raise

def test_pipeline_config_validate_missing_video():
    config = PipelineConfig(
        video_paths=["missing.mp4"],
        audio_path="audio.mp3",
        output_path="out.mp4",
    )
    with pytest.raises(FileNotFoundError):
        config.validate()

def test_pipeline_config_validate_missing_audio(tmp_path):
    video = tmp_path / "vid.mp4"
    video.touch()
    config = PipelineConfig(
        video_paths=[str(video)],
        audio_path="missing.mp3",
        output_path="out.mp4",
    )
    with pytest.raises(FileNotFoundError):
        config.validate()

def test_pipeline_config_validate_bad_extension(tmp_path):
    video = tmp_path / "vid.mp4"
    audio = tmp_path / "audio.mp3"
    video.touch()
    audio.touch()
    config = PipelineConfig(
        video_paths=[str(video)],
        audio_path=str(audio),
        output_path="out.avi",
    )
    with pytest.raises(ValueError):
        config.validate()

def test_pipeline_config_validate_bad_vignette(tmp_path):
    video = tmp_path / "vid.mp4"
    audio = tmp_path / "audio.mp3"
    video.touch()
    audio.touch()
    config = PipelineConfig(
        video_paths=[str(video)],
        audio_path=str(audio),
        output_path="out.mp4",
        vignette_strength=1.5,
    )
    with pytest.raises(ValueError):
        config.validate()
