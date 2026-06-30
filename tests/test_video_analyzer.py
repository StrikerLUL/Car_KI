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

from unittest.mock import patch, MagicMock

def test_run_yolo_batch_empty_frames():
    """Test _run_yolo_batch with empty frames list."""
    from video_analyzer import _run_yolo_batch
    assert _run_yolo_batch(MagicMock(), [], None) == []

def test_run_yolo_batch_none_model():
    """Test _run_yolo_batch with None model."""
    from video_analyzer import _run_yolo_batch
    assert _run_yolo_batch(None, ["dummy_frame"], None) == [0.0]

def test_analyze_telemetry_missing_file():
    """Test _analyze_telemetry returns zeros when file is missing."""
    import video_analyzer
    # Set up mock np to return an array of zeros and assert length is right
    mock_np = MagicMock()
    mock_np.zeros.return_value = [0.0] * 100
    with patch.object(video_analyzer, 'np', mock_np):
        res = video_analyzer._analyze_telemetry("nonexistent_video.mp4", 100, 30.0)
        assert len(res) == 100
        assert res == [0.0] * 100

def test_analyze_audio_no_audio():
    """Test _analyze_audio when video has no audio track."""
    import video_analyzer

    mock_video = MagicMock()
    mock_video.audio = None

    mock_np = MagicMock()
    mock_np.zeros.return_value = [0.0] * 100

    with patch.object(video_analyzer, 'np', mock_np), \
         patch.object(video_analyzer, 'VideoFileClip', return_value=mock_video):

        res = video_analyzer._analyze_audio("fake_video.mp4", 100, 30.0)
        assert len(res) == 100
        assert res == [0.0] * 100

def test_analyze_audio_with_exception():
    """Test _analyze_audio when an exception occurs during processing."""
    import video_analyzer

    mock_video = MagicMock()
    mock_video.audio.write_audiofile.side_effect = Exception("Simulated audio error")

    mock_np = MagicMock()
    mock_np.zeros.return_value = [0.0] * 50

    with patch.object(video_analyzer, 'np', mock_np), \
         patch.object(video_analyzer, 'VideoFileClip', return_value=mock_video):

        res = video_analyzer._analyze_audio("fake_video.mp4", 50, 30.0)
        assert len(res) == 50
        assert res == [0.0] * 50

def test_find_highlights_multi_multiple_videos():
    """Test find_highlights_multi with multiple videos."""
    from video_analyzer import find_highlights_multi, ClipInfo

    mock_clip = ClipInfo(0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, "external", "action", "fake.mp4")

    with patch("video_analyzer.find_highlights", return_value=[mock_clip, mock_clip]):
        res = find_highlights_multi(["vid1.mp4", "vid2.mp4"], 3, 2.0)
        assert len(res) == 2
        assert "vid1.mp4" in res
        assert "vid2.mp4" in res
        assert len(res["vid1.mp4"]) == 2
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


def test_find_highlights_invalid_file():
    """Test find_highlights with a non-existent file."""
    clips = video_analyzer.find_highlights("non_existent_file_12345.mp4", num_clips=5)
    assert len(clips) == 1
    assert clips[0].timestamp == 0.0
    assert clips[0].source == "non_existent_file_12345.mp4"

def test_find_highlights_zero_clips(tmp_path):
    """Test find_highlights with num_clips=0. Should not raise error and handle gracefully."""
    # Create a dummy video file
    dummy_video = tmp_path / "dummy.mp4"
    dummy_video.write_bytes(b"dummy data")

    # Mocking cap to act like a valid but very short video to avoid actual cv2 processing
    from unittest.mock import MagicMock
    mock_cap = MagicMock()
    mock_cap.get.side_effect = lambda prop: 30.0 if prop == video_analyzer.cv2.CAP_PROP_FPS else (10 if prop == video_analyzer.cv2.CAP_PROP_FRAME_COUNT else 0)
    mock_cap.read.return_value = (True, video_analyzer.np.zeros((100, 100, 3)))
    mock_cap.isOpened.return_value = True

    original_cap = video_analyzer.cv2.VideoCapture
    video_analyzer.cv2.VideoCapture = lambda x: mock_cap
    try:
        clips = video_analyzer.find_highlights(str(dummy_video), num_clips=0)
        assert len(clips) == 1
        assert clips[0].timestamp == 0.0
    finally:
        video_analyzer.cv2.VideoCapture = original_cap

def test_find_highlights_invalid_format(tmp_path):
    """Test find_highlights with a non-video text file."""
    text_file = tmp_path / "invalid.txt"
    text_file.write_text("This is not a video file.")

    clips = video_analyzer.find_highlights(str(text_file), num_clips=5)
    assert len(clips) == 1
    assert clips[0].timestamp == 0.0
    assert clips[0].source == str(text_file)

def test_find_highlights_video_too_short(tmp_path, monkeypatch):
    """Test find_highlights fallback when video duration is shorter than clip duration."""
    import video_analyzer
    dummy_video = tmp_path / "short.mp4"
    dummy_video.write_bytes(b"dummy data")

    mock_cap = MagicMock()
    mock_cap.get.side_effect = lambda prop: 30.0 if prop == video_analyzer.cv2.CAP_PROP_FPS else (10 if prop == video_analyzer.cv2.CAP_PROP_FRAME_COUNT else 0)
    mock_cap.isOpened.return_value = True

    monkeypatch.setattr(video_analyzer.cv2, 'VideoCapture', lambda x: mock_cap)

    clips = video_analyzer.find_highlights(str(dummy_video), num_clips=5, clip_duration=2.0)

    assert len(clips) == 1
    assert clips[0].timestamp == 0.0
    assert clips[0].source == str(dummy_video)
    mock_cap.release.assert_called_once()

def test_find_highlights_yolo_exception(tmp_path, monkeypatch):
    """Test find_highlights when YOLO model instantiation throws an Exception."""
    import video_analyzer
    dummy_video = tmp_path / "vid.mp4"
    dummy_video.write_bytes(b"dummy data")

    mock_cap = MagicMock()
    mock_cap.get.side_effect = lambda prop: 30.0 if prop == video_analyzer.cv2.CAP_PROP_FPS else (300 if prop == video_analyzer.cv2.CAP_PROP_FRAME_COUNT else 0)
    mock_cap.read.return_value = (True, video_analyzer.np.zeros((100, 100, 3)))
    mock_cap.isOpened.return_value = True

    monkeypatch.setattr(video_analyzer.cv2, 'VideoCapture', lambda x: mock_cap)
    monkeypatch.setattr(video_analyzer, 'YOLO', MagicMock(side_effect=Exception("Simulated YOLO failure")))

    # Mocking out the ThreadPoolExecutor logic to avoid complex async stuff in this unit test
    monkeypatch.setattr(video_analyzer, '_analyze_audio', MagicMock(return_value=video_analyzer.np.zeros(300)))
    monkeypatch.setattr(video_analyzer, '_analyze_telemetry', MagicMock(return_value=video_analyzer.np.zeros(300)))
    monkeypatch.setattr(video_analyzer, '_compute_optical_flow', MagicMock(return_value=[{"motion_score": 0.0, "drift_score": 0.0, "cam_type": "external"} for _ in range(300)]))

    clips = video_analyzer.find_highlights(str(dummy_video), num_clips=1, clip_duration=2.0)

    assert len(clips) >= 1

def test_find_highlights_empty_raw_scores(tmp_path, monkeypatch):
    """Test find_highlights fallback when raw_scores is empty."""
    import video_analyzer
    dummy_video = tmp_path / "vid.mp4"
    dummy_video.write_bytes(b"dummy data")

    mock_cap = MagicMock()
    mock_cap.get.side_effect = lambda prop: 30.0 if prop == video_analyzer.cv2.CAP_PROP_FPS else (300 if prop == video_analyzer.cv2.CAP_PROP_FRAME_COUNT else 0)
    mock_cap.read.return_value = (False, None)  # Fail immediately on read, raw_scores will be empty
    mock_cap.isOpened.return_value = True

    monkeypatch.setattr(video_analyzer.cv2, 'VideoCapture', lambda x: mock_cap)
    monkeypatch.setattr(video_analyzer, '_analyze_audio', MagicMock(return_value=video_analyzer.np.zeros(300)))
    monkeypatch.setattr(video_analyzer, '_analyze_telemetry', MagicMock(return_value=video_analyzer.np.zeros(300)))
    monkeypatch.setattr(video_analyzer, '_compute_optical_flow', MagicMock(return_value=[]))
    monkeypatch.setattr(video_analyzer, 'YOLO', MagicMock())

    clips = video_analyzer.find_highlights(str(dummy_video), num_clips=1, clip_duration=2.0)

    assert len(clips) == 1
    assert clips[0].timestamp == 0.0

def test_find_highlights_overlap_filtering(tmp_path, monkeypatch):
    """Test that clips within clip_duration are correctly filtered."""
    import video_analyzer
    dummy_video = tmp_path / "vid.mp4"
    dummy_video.write_bytes(b"dummy data")

    mock_cap = MagicMock()
    mock_cap.get.side_effect = lambda prop: 30.0 if prop == video_analyzer.cv2.CAP_PROP_FPS else (300 if prop == video_analyzer.cv2.CAP_PROP_FRAME_COUNT else 0)

    frames = []
    for _ in range(15):
        frames.append((True, video_analyzer.np.zeros((10, 10, 3))))
    frames.append((False, None))
    mock_cap.read.side_effect = frames
    mock_cap.isOpened.return_value = True

    monkeypatch.setattr(video_analyzer.cv2, 'VideoCapture', lambda x: mock_cap)
    monkeypatch.setattr(video_analyzer, '_analyze_audio', MagicMock(return_value=video_analyzer.np.zeros(300)))
    monkeypatch.setattr(video_analyzer, '_analyze_telemetry', MagicMock(return_value=video_analyzer.np.zeros(300)))

    flow = [{"motion_score": 0.9, "drift_score": 0.0, "cam_type": "external"} for _ in range(300)]
    monkeypatch.setattr(video_analyzer, '_compute_optical_flow', MagicMock(return_value=flow))
    monkeypatch.setattr(video_analyzer, 'YOLO', MagicMock())
    monkeypatch.setattr(video_analyzer, '_run_yolo_batch', MagicMock(return_value=[1.0]*100))

    clips = video_analyzer.find_highlights(str(dummy_video), num_clips=2, clip_duration=2.0)

    # We should have at least 1 clip, but overlaps should be filtered, maybe filled by gap filling.
    assert len(clips) >= 1
    if len(clips) > 1:
        # If there are multiple clips, verify they are at least clip_duration/2 apart (due to gap filling)
        assert abs(clips[0].timestamp - clips[1].timestamp) >= 1.0

def test_find_highlights_gap_filling(tmp_path, monkeypatch):
    """Test gap filling logic when len(selected) < num_clips."""
    import video_analyzer
    dummy_video = tmp_path / "vid.mp4"
    dummy_video.write_bytes(b"dummy data")

    mock_cap = MagicMock()
    mock_cap.get.side_effect = lambda prop: 30.0 if prop == video_analyzer.cv2.CAP_PROP_FPS else (300 if prop == video_analyzer.cv2.CAP_PROP_FRAME_COUNT else 0)

    frames = []
    for _ in range(20):
        frames.append((True, video_analyzer.np.zeros((10, 10, 3))))
    frames.append((False, None))
    mock_cap.read.side_effect = frames
    mock_cap.isOpened.return_value = True

    monkeypatch.setattr(video_analyzer.cv2, 'VideoCapture', lambda x: mock_cap)
    monkeypatch.setattr(video_analyzer, '_analyze_audio', MagicMock(return_value=video_analyzer.np.zeros(300)))
    monkeypatch.setattr(video_analyzer, '_analyze_telemetry', MagicMock(return_value=video_analyzer.np.zeros(300)))

    flow = [{"motion_score": 0.9, "drift_score": 0.0, "cam_type": "external"} for _ in range(300)]
    monkeypatch.setattr(video_analyzer, '_compute_optical_flow', MagicMock(return_value=flow))
    monkeypatch.setattr(video_analyzer, 'YOLO', MagicMock())
    monkeypatch.setattr(video_analyzer, '_run_yolo_batch', MagicMock(return_value=[1.0]*100))

    # Ask for 10 clips, but there are only ~20 sample frames, they will overlap a lot.
    # Gap filling should trigger to fill up the list using a smaller threshold (0.5 * clip_duration)
    clips = video_analyzer.find_highlights(str(dummy_video), num_clips=10, clip_duration=2.0)

    # Check that gap filling added at least some clips, and they respect the 0.5 * clip_duration threshold
    for i in range(len(clips)):
        for j in range(i + 1, len(clips)):
            assert abs(clips[i].timestamp - clips[j].timestamp) >= 1.0

def test_compute_optical_flow_gpu_empty_fallback():
    mock_cap = MagicMock()
    mock_cap.read.return_value = (False, None)
    with patch.object(video_analyzer, 'cv2') as mock_cv2:
        result = video_analyzer._compute_optical_flow_gpu(mock_cap, total_frames=10, fps=30.0)
        assert result == []

def test_compute_optical_flow_gpu_success():
    mock_cap = MagicMock()
    frame1 = np.zeros((100, 100, 3), dtype=np.uint8)
    frame2 = np.ones((100, 100, 3), dtype=np.uint8) * 255
    read_side_effect = [(True, frame1), (True, frame2), (False, None)]
    mock_cap.read.side_effect = read_side_effect

    with patch.object(video_analyzer, 'cv2') as mock_cv2:
        mock_flow_gpu = MagicMock()
        mock_flow_gpu.calc.return_value = MagicMock()
        mock_flow_gpu.calc.return_value.download.return_value = np.zeros((50, 50, 2), dtype=np.float32)
        mock_cv2.cuda.FarnebackOpticalFlow.create.return_value = mock_flow_gpu
        mock_cv2.cartToPolar.return_value = (np.ones((50, 50), dtype=np.float32), None)

        # Ensure that other cv2 operations don't fail
        mock_cv2.resize.side_effect = lambda img, *args, **kwargs: np.zeros((50, 50, 3), dtype=np.uint8) if len(img.shape) == 3 else np.zeros((50, 50), dtype=np.uint8)
        mock_cv2.cvtColor.side_effect = lambda img, *args, **kwargs: np.zeros((50, 50), dtype=np.uint8)

        mock_cuda_GpuMat = MagicMock()
        mock_cv2.cuda_GpuMat.return_value = mock_cuda_GpuMat

        result = video_analyzer._compute_optical_flow_gpu(mock_cap, total_frames=60, fps=30.0, sample_interval=1.0)

        assert isinstance(result, list)
        assert len(result) >= 1
        assert "motion_score" in result[0]
        assert "drift_score" in result[0]
        assert "cam_type" in result[0]

def test_compute_optical_flow_normalization():
    mock_cap = MagicMock()
    mock_raw_data = [
        {"motion_score": 10.0, "drift_score": 5.0, "cam_type": "helmet"},
        {"motion_score": 20.0, "drift_score": 10.0, "cam_type": "external"},
        {"motion_score": 5.0, "drift_score": 2.5, "cam_type": "helmet"}
    ]

    with patch.object(video_analyzer, '_CUDA_AVAILABLE', False):
        with patch.object(video_analyzer, '_compute_optical_flow_cpu', return_value=mock_raw_data):
            result = video_analyzer._compute_optical_flow(mock_cap, total_frames=10, fps=30.0)

            assert len(result) == 3
            # Percentile 95 logic
            m_scores = [d["motion_score"] for d in result]
            d_scores = [d["drift_score"] for d in result]
            assert all(0.0 <= score <= 1.0 for score in m_scores)
            assert all(0.0 <= score <= 1.0 for score in d_scores)

def test_analyze_audio_valid_normalization():
    mock_clip = MagicMock()
    mock_audio = MagicMock()
    mock_clip.audio = mock_audio

    with patch.object(video_analyzer, 'VideoFileClip', return_value=mock_clip):
        with patch.object(video_analyzer, 'librosa') as mock_librosa:
            mock_librosa.load.return_value = (np.zeros(22050), 22050)
            mock_librosa.feature.rms.return_value = np.array([[0.1, 0.5, 1.0, 0.2]])
            mock_librosa.frames_to_time.return_value = np.array([0.0, 1.0, 2.0, 3.0])

            with patch('os.remove'):
                result = video_analyzer._analyze_audio("dummy.mp4", total_frames=10, fps=30.0)

                assert len(result) == 10
                assert np.max(result) <= 1.0
                assert np.min(result) >= 0.0

def test_analyze_telemetry_valid_normalization():
    import pandas as pd

    mock_df = pd.DataFrame({
        "Time": [0.0, 1.0, 2.0, 3.0],
        "G_Lat": [0.0, 0.5, 1.0, 0.2],
        "G_Long": [0.0, -0.5, -1.0, -0.2]
    })

    with patch('os.path.exists', return_value=True):
        with patch('pandas.read_csv', return_value=mock_df):
            result = video_analyzer._analyze_telemetry("dummy.mp4", total_frames=10, fps=30.0)

            assert len(result) == 10
            assert np.max(result) <= 1.0
            assert np.min(result) >= 0.0
