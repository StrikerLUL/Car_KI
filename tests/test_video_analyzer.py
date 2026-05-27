import sys
from unittest.mock import MagicMock, patch
import numpy as np


import video_analyzer
from video_analyzer import _classify_clip, ClipInfo, find_highlights_multi, find_highlights

def test_classify_clip_drift():
    assert _classify_clip(score=0.5, motion_score=0.55, drift_score=0.65, audio_score=0.5, vehicle_count=1.0, cam_type="onboard") == "drift"

def test_classify_clip_action():
    assert _classify_clip(score=0.7, motion_score=0.4, drift_score=0.1, audio_score=0.6, vehicle_count=2.0, cam_type="onboard") == "action"

def test_classify_clip_overtake():
    assert _classify_clip(score=0.5, motion_score=0.46, drift_score=0.1, audio_score=0.4, vehicle_count=2.0, cam_type="onboard") == "overtake"

def test_classify_clip_corner():
    assert _classify_clip(score=0.5, motion_score=0.55, drift_score=0.1, audio_score=0.4, vehicle_count=1.0, cam_type="onboard") == "corner"

def test_classify_clip_straight():
    assert _classify_clip(score=0.45, motion_score=0.30, drift_score=0.1, audio_score=0.4, vehicle_count=1.0, cam_type="onboard") == "straight"

def test_classify_clip_calm():
    assert _classify_clip(score=0.3, motion_score=0.4, drift_score=0.1, audio_score=0.4, vehicle_count=1.0, cam_type="onboard") == "calm"

def test_classify_clip_edge_zero_values():
    assert _classify_clip(score=0.0, motion_score=0.0, drift_score=0.0, audio_score=0.0, vehicle_count=0.0, cam_type="onboard") == "calm"

def test_classify_clip_edge_boundary_drift():
    assert _classify_clip(score=0.0, motion_score=0.50, drift_score=0.60, audio_score=0.0, vehicle_count=0.0, cam_type="onboard") == "drift"
    assert _classify_clip(score=0.0, motion_score=0.49, drift_score=0.60, audio_score=0.0, vehicle_count=0.0, cam_type="onboard") != "drift"
    assert _classify_clip(score=0.0, motion_score=0.50, drift_score=0.59, audio_score=0.0, vehicle_count=0.0, cam_type="onboard") != "drift"

def test_clip_info_creation():
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

def test_find_highlights_multi_empty():
    assert find_highlights_multi([], 5) == {}

@patch("video_analyzer.cv2")
def test_find_highlights_wrong_file_format_or_too_short(mock_cv2):
    mock_cap = MagicMock()
    mock_cap.get.side_effect = lambda prop: 30.0 if prop == mock_cv2.CAP_PROP_FPS else 0
    mock_cv2.VideoCapture.return_value = mock_cap
    result = find_highlights("invalid_format.xyz", num_clips=2)
    assert len(result) == 1
    assert result[0].timestamp == 0.0

def test_clip_info_unexpected_types():
    clip = ClipInfo(
        timestamp="start",
        score=None,
        motion_score=0.6,
        drift_score=0.2,
        audio_score=0.7,
        telemetry_score=0.5,
        vehicle_count=2.5,
        cam_type=123,
        tag=True
    )
    assert clip.timestamp == "start"

def test_check_cuda_exception():
    with patch.object(video_analyzer, 'cv2') as mock_cv2:
        mock_cv2.cuda.getCudaEnabledDeviceCount.side_effect = Exception("Simulated CUDA failure")
        result = video_analyzer._check_cuda()
        assert result is False

def test_analyze_audio_file_not_found():
    with patch.object(video_analyzer, 'VideoFileClip', side_effect=Exception("File not found")):
        result = video_analyzer._analyze_audio("missing.mp4", 100, 30.0)
        assert len(result) == 100
        assert np.array_equal(result, np.zeros(100, dtype=np.float32))

def test_analyze_audio_no_audio_track():
    mock_clip = MagicMock()
    mock_clip.audio = None
    with patch.object(video_analyzer, 'VideoFileClip', return_value=mock_clip):
        result = video_analyzer._analyze_audio("no_audio.mp4", 100, 30.0)
        assert len(result) == 100
        assert np.array_equal(result, np.zeros(100, dtype=np.float32))
        mock_clip.close.assert_called_once()

def test_compute_optical_flow_empty_fallback():
    mock_cap = MagicMock()
    mock_cap.read.return_value = (False, None)
    result = video_analyzer._compute_optical_flow(mock_cap, total_frames=10, fps=30.0)
    assert len(result) == 1
    assert result[0] == {"motion_score": 0.0, "drift_score": 0.0, "cam_type": "external"}

def test_compute_optical_flow_cpu_short():
    mock_cap = MagicMock()
    frame1 = np.zeros((100, 100, 3), dtype=np.uint8)
    frame2 = np.ones((100, 100, 3), dtype=np.uint8) * 255
    read_side_effect = [(True, frame1), (True, frame2), (False, None)]
    mock_cap.read.side_effect = read_side_effect
    result = video_analyzer._compute_optical_flow_cpu(mock_cap, total_frames=60, fps=30.0, sample_interval=1.0)
    assert isinstance(result, list)
    assert len(result) >= 1
