import sys
from unittest.mock import MagicMock

# Mock dependencies not available in the environment
sys.modules["cv2"] = MagicMock()
sys.modules["cv2.cuda"] = MagicMock()
sys.modules["ultralytics"] = MagicMock()
sys.modules["librosa"] = MagicMock()
sys.modules["moviepy.editor"] = MagicMock()
sys.modules["numpy"] = MagicMock()

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
