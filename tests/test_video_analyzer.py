import sys
from unittest.mock import MagicMock

# Mock dependencies not available in the environment
sys.modules["cv2"] = MagicMock()
sys.modules["cv2.cuda"] = MagicMock()
sys.modules["ultralytics"] = MagicMock()
sys.modules["librosa"] = MagicMock()
sys.modules["moviepy.editor"] = MagicMock()
sys.modules["numpy"] = MagicMock()
sys.modules["tqdm"] = MagicMock()

import numpy as np
import pytest
from video_analyzer import _classify_clip, ClipInfo

def test_classify_clip_drift():
    """Test that high drift and motion score returns 'drift'."""
    # drift_score >= 0.60 and motion_score >= 0.50
    assert _classify_clip(score=0.5, motion_score=0.55, drift_score=0.65, audio_score=0.5, vehicle_count=1.0, cam_type="onboard") == "drift"

def test_classify_clip_action():
    """Test that high score, high vehicle count and audio score returns 'action'."""
    # score >= 0.65 and vehicle_count >= 1.5 and audio_score >= 0.55
    assert _classify_clip(score=0.7, motion_score=0.4, drift_score=0.1, audio_score=0.6, vehicle_count=2.0, cam_type="onboard") == "action"

def test_classify_clip_overtake():
    """Test that high vehicle count and motion score returns 'overtake'."""
    # vehicle_count >= 1.8 and motion_score >= 0.45
    assert _classify_clip(score=0.5, motion_score=0.46, drift_score=0.1, audio_score=0.4, vehicle_count=2.0, cam_type="onboard") == "overtake"

def test_classify_clip_corner():
    """Test that high motion score with low vehicle count returns 'corner'."""
    # motion_score >= 0.50 and vehicle_count < 1.5
    assert _classify_clip(score=0.5, motion_score=0.55, drift_score=0.1, audio_score=0.4, vehicle_count=1.0, cam_type="onboard") == "corner"

def test_classify_clip_straight():
    """Test that low motion score and decent score returns 'straight'."""
    # motion_score < 0.35 and score >= 0.40
    assert _classify_clip(score=0.45, motion_score=0.30, drift_score=0.1, audio_score=0.4, vehicle_count=1.0, cam_type="onboard") == "straight"

def test_classify_clip_calm():
    """Test that failing all other conditions returns 'calm'."""
    assert _classify_clip(score=0.3, motion_score=0.4, drift_score=0.1, audio_score=0.4, vehicle_count=1.0, cam_type="onboard") == "calm"

def test_classify_clip_edge_zero_values():
    """Test edge case with all zero values."""
    assert _classify_clip(score=0.0, motion_score=0.0, drift_score=0.0, audio_score=0.0, vehicle_count=0.0, cam_type="onboard") == "calm"

def test_classify_clip_edge_boundary_drift():
    """Test boundary values for drift classification."""
    assert _classify_clip(score=0.0, motion_score=0.50, drift_score=0.60, audio_score=0.0, vehicle_count=0.0, cam_type="onboard") == "drift"
    # Should not be drift if slightly below
    assert _classify_clip(score=0.0, motion_score=0.49, drift_score=0.60, audio_score=0.0, vehicle_count=0.0, cam_type="onboard") != "drift"
    assert _classify_clip(score=0.0, motion_score=0.50, drift_score=0.59, audio_score=0.0, vehicle_count=0.0, cam_type="onboard") != "drift"

def test_clip_info_creation():
    """Test correct instantiation of ClipInfo class."""
    clip = ClipInfo(
        timestamp=10.5,
        score=0.8,
        motion_score=0.6,
        drift_score=0.2,
        audio_score=0.7,
        telemetry_score=0.5,
        vehicle_count=2.5,
        cam_type="onboard",
        tag="action"
    )
    assert clip.timestamp == 10.5
    assert clip.score == 0.8
    assert clip.motion_score == 0.6
    assert clip.drift_score == 0.2
    assert clip.audio_score == 0.7
    assert clip.telemetry_score == 0.5
    assert clip.vehicle_count == 2.5
    assert clip.cam_type == "onboard"
    assert clip.tag == "action"

from video_analyzer import find_highlights_multi

def test_find_highlights_multi_empty():
    """Test that empty paths list returns empty dict."""
    assert find_highlights_multi([], 5) == {}

from video_analyzer import find_highlights
from unittest.mock import patch

@patch("video_analyzer.cv2")
def test_find_highlights_wrong_file_format_or_too_short(mock_cv2):
    """Test find_highlights with a file that can't be read or is too short (0 frames)."""
    mock_cap = MagicMock()
    # Mock cap.get to return 30.0 for FPS and 0 for FRAME_COUNT
    mock_cap.get.side_effect = lambda prop: 30.0 if prop == mock_cv2.CAP_PROP_FPS else 0
    mock_cv2.VideoCapture.return_value = mock_cap

    # Analyze a hypothetical invalid file
    result = find_highlights("invalid_format.xyz", num_clips=2)

    # Should fallback gracefully with a single "calm" clip at timestamp 0.0
    assert len(result) == 1
    assert result[0].timestamp == 0.0
    assert result[0].tag == "calm"
    assert result[0].source == "invalid_format.xyz"

def test_clip_info_unexpected_types():
    """Test ClipInfo handles unexpected types without throwing errors upon instantiation (since Python dataclasses don't strictly enforce types by default)."""
    clip = ClipInfo(
        timestamp="start", # string instead of float
        score=None,        # None instead of float
        motion_score=0.6,
        drift_score=0.2,
        audio_score=0.7,
        telemetry_score=0.5,
        vehicle_count=2.5,
        cam_type=123,      # int instead of string
        tag=True           # boolean instead of string
    )
    assert clip.timestamp == "start"
    assert clip.score is None
    assert clip.cam_type == 123
    assert clip.tag is True

from unittest.mock import MagicMock, patch
import video_analyzer

def test_check_cuda_exception(monkeypatch):
    """Test that _check_cuda returns False if an exception is raised."""
    mock_cv2 = MagicMock()
    mock_cv2.cuda.getCudaEnabledDeviceCount.side_effect = Exception("Simulated CUDA error")
    monkeypatch.setattr(video_analyzer, 'cv2', mock_cv2)

    assert video_analyzer._check_cuda() == False

def test_analyze_telemetry_missing_file(monkeypatch):
    """Test that _analyze_telemetry returns an array of zeros if CSV is missing."""
    monkeypatch.setattr(video_analyzer.os.path, 'exists', MagicMock(return_value=False))

    scores = video_analyzer._analyze_telemetry("fake_video.mp4", 10, 30.0)
    assert np.array_equal(scores, np.zeros(10, dtype=np.float32))

def test_analyze_telemetry_exception(monkeypatch):
    """Test that _analyze_telemetry handles exceptions and returns zeros."""
    monkeypatch.setattr(video_analyzer.os.path, 'exists', MagicMock(return_value=True))

    # Mock pandas to raise exception
    mock_pd = MagicMock()
    mock_pd.read_csv.side_effect = Exception("Simulated pandas error")

    # We patch builtins.__import__ to return our mock_pd when pandas is imported
    original_import = __import__
    def side_effect_import(name, *args, **kwargs):
        if name == 'pandas':
            return mock_pd
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr('builtins.__import__', side_effect_import)

    scores = video_analyzer._analyze_telemetry("fake_video.mp4", 10, 30.0)
    assert np.array_equal(scores, np.zeros(10, dtype=np.float32))

def test_analyze_audio_empty_audio(monkeypatch):
    """Test that _analyze_audio returns zeros when video has no audio."""
    mock_VideoFileClip = MagicMock()
    mock_video = MagicMock()
    mock_video.audio = None
    mock_VideoFileClip.return_value = mock_video
    monkeypatch.setattr(video_analyzer, 'VideoFileClip', mock_VideoFileClip)

    scores = video_analyzer._analyze_audio("fake_video.mp4", 10, 30.0)
    assert np.array_equal(scores, np.zeros(10, dtype=np.float32))
    mock_video.close.assert_called_once()

def test_run_yolo_batch_none_model():
    """Test _run_yolo_batch returns list of zeros if model is None."""
    frames = [np.zeros((100, 100, 3)), np.zeros((100, 100, 3))]
    result = video_analyzer._run_yolo_batch(None, frames, [2, 3])
    assert result == [0.0, 0.0]

def test_run_yolo_batch_empty_frames():
    """Test _run_yolo_batch returns empty list if frames are empty."""
    mock_model = MagicMock()
    result = video_analyzer._run_yolo_batch(mock_model, [], [2, 3])
    assert result == []

def test_run_yolo_batch_exception():
    """Test _run_yolo_batch handles exceptions gracefully."""
    mock_model = MagicMock()
    mock_model.predict.side_effect = Exception("Simulated YOLO error")
    frames = [np.zeros((100, 100, 3))]

    result = video_analyzer._run_yolo_batch(mock_model, frames, [2, 3])
    assert result == [0.0]
