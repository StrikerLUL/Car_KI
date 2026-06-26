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

def test_pipeline_config_validate_bad_watermark_opacity(tmp_path):
    video = tmp_path / "vid.mp4"
    audio = tmp_path / "audio.mp3"
    video.touch()
    audio.touch()
    config = PipelineConfig(
        video_paths=[str(video)],
        audio_path=str(audio),
        output_path="out.mp4",
        watermark_opacity=1.5
    )
    with pytest.raises(ValueError, match="watermark_opacity"):
        config.validate()

def test_pipeline_config_validate_bad_grade_preset(tmp_path):
    video = tmp_path / "vid.mp4"
    audio = tmp_path / "audio.mp3"
    video.touch()
    audio.touch()
    config = PipelineConfig(
        video_paths=[str(video)],
        audio_path=str(audio),
        output_path="out.mp4",
        grade_preset="unknown_preset"
    )
    with pytest.raises(ValueError, match="grade_preset"):
        config.validate()

def test_pipeline_config_validate_bad_visualizer_bars(tmp_path):
    video = tmp_path / "vid.mp4"
    audio = tmp_path / "audio.mp3"
    video.touch()
    audio.touch()
    config = PipelineConfig(
        video_paths=[str(video)],
        audio_path=str(audio),
        output_path="out.mp4",
        visualizer_bars=0
    )
    with pytest.raises(ValueError, match="visualizer_bars"):
        config.validate()

def test_pipeline_config_validate_bad_editing_mode(tmp_path):
    video = tmp_path / "vid.mp4"
    audio = tmp_path / "audio.mp3"
    video.touch()
    audio.touch()
    config = PipelineConfig(
        video_paths=[str(video)],
        audio_path=str(audio),
        output_path="out.mp4",
        editing_mode="bad_mode"
    )
    with pytest.raises(ValueError, match="editing_mode"):
        config.validate()

def test_pipeline_config_validate_bad_trend_preset(tmp_path):
    video = tmp_path / "vid.mp4"
    audio = tmp_path / "audio.mp3"
    video.touch()
    audio.touch()
    config = PipelineConfig(
        video_paths=[str(video)],
        audio_path=str(audio),
        output_path="out.mp4",
        trend_preset="unknown_trend"
    )
    with pytest.raises(ValueError, match="trend_preset"):
        config.validate()

def test_pipeline_config_validate_bad_template_path(tmp_path):
    video = tmp_path / "vid.mp4"
    audio = tmp_path / "audio.mp3"
    video.touch()
    audio.touch()
    config = PipelineConfig(
        video_paths=[str(video)],
        audio_path=str(audio),
        output_path="out.mp4",
        template_path="missing_template.json"
    )
    with pytest.raises(FileNotFoundError, match="template_path"):
        config.validate()

def test_pipeline_config_to_dict():
    config = PipelineConfig(
        video_paths=["vid.mp4"],
        audio_path="audio.mp3",
        output_path="out.mp4"
    )
    d = config.to_dict()
    assert d["video_paths"] == ["vid.mp4"]
    assert d["audio_path"] == "audio.mp3"
    assert d["output_path"] == "out.mp4"
    assert d["grade_preset"] == "teal_orange"

def test_pipeline_config_validate_empty_video_paths(tmp_path):
    audio = tmp_path / "audio.mp3"
    audio.touch()
    config = PipelineConfig(
        video_paths=[],
        audio_path=str(audio),
        output_path="out.mp4"
    )
    with pytest.raises(ValueError, match="darf nicht leer sein"):
        config.validate()
