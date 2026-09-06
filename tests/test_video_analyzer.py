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

def test_compute_optical_flow_gpu_empty(mocker):
    """Test _compute_optical_flow_gpu when cap.read() returns False immediately."""
    from video_analyzer import _compute_optical_flow_gpu
    mock_cap = MagicMock()
    mock_cap.read.return_value = (False, None)

    # Mock cv2.cuda
    mock_cuda = MagicMock()
    mocker.patch('video_analyzer.cv2.cuda', mock_cuda)

    res = _compute_optical_flow_gpu(mock_cap, total_frames=100, fps=30.0)
    assert res == []


def test_analyze_audio_librosa_error(mocker, tmp_path):
    import numpy as np
    """Test _analyze_audio when librosa.load raises an Exception."""
    from video_analyzer import _analyze_audio

    video_path = tmp_path / "test.mp4"
    video_path.write_text("dummy")

    # Mock moviepy VideoFileClip
    mock_clip = MagicMock()
    mock_clip.audio = MagicMock()
    mocker.patch('video_analyzer.VideoFileClip', return_value=mock_clip)

    # Mock librosa.load to raise an exception
    mocker.patch('video_analyzer.librosa.load', side_effect=Exception("librosa error"))

    # We also mock os.remove to avoid errors when it tries to clean up the temp audio file
    mocker.patch('video_analyzer.os.remove')

    # Since there's an exception, it should return a zero-array of size total_frames
    res = _analyze_audio(str(video_path), total_frames=100, fps=30.0)

    assert isinstance(res, np.ndarray)
    assert len(res) == 100
    assert np.all(res == 0)


def test_analyze_telemetry_corrupted_csv(mocker, tmp_path):
    import numpy as np
    """Test _analyze_telemetry with a CSV missing required columns."""
    from video_analyzer import _analyze_telemetry
    import pandas as pd

    video_path = tmp_path / "test.mp4"
    csv_path = tmp_path / "test_telemetry.csv"

    # Create CSV missing G_Lat and G_Long
    df = pd.DataFrame({"Time": [0.0, 1.0, 2.0], "Speed": [100, 110, 120]})
    df.to_csv(csv_path, index=False)

    res = _analyze_telemetry(str(video_path), total_frames=60, fps=30.0)

    # Since columns are missing, it should just return the zero array
    assert isinstance(res, np.ndarray)
    assert len(res) == 60
    assert np.all(res == 0)


def test_find_highlights_gap_filling_fallback(tmp_path, mocker):
    """Test find_highlights where gap filling is needed to reach num_clips."""
    from video_analyzer import find_highlights, ClipInfo

    video_path = tmp_path / "test.mp4"
    video_path.write_text("dummy")

    # Mock OpenCV cap
    mock_cap = MagicMock()
    # 5 frames total
    import cv2
    mock_cap.get.side_effect = lambda prop: 5 if prop == cv2.CAP_PROP_FRAME_COUNT else 30.0
    mock_cap.read.return_value = (False, None)
    mocker.patch('video_analyzer.cv2.VideoCapture', return_value=mock_cap)

    # We mock _compute_optical_flow_gpu and _analyze_audio to return values
    mocker.patch('video_analyzer._compute_optical_flow', return_value=[
        {"motion_score": 0.5, "drift_score": 0.5, "cam_type": "onboard"}
    ] * 5)

    mocker.patch('video_analyzer._analyze_audio', return_value=np.array([0.5]*5))
    mocker.patch('video_analyzer._analyze_telemetry', return_value=np.array([0.5]*5))

    mocker.patch('video_analyzer._run_yolo_batch', return_value=[])

    # For the fallback we want multiple unique timestamps, but we only have 5 frames.
    # We will mock the `raw_scores` inside `find_highlights` by mocking `_compute_optical_flow` output
    # Actually, find_highlights computes `raw_scores` internally using the `flow_data`.
    # `flow_data` comes from `_compute_optical_flow`.

    # Let's mock a larger flow_data so we have more than 5 clips.
    # 60 frames -> 2 seconds at 30 fps.
    mock_cap.get.side_effect = lambda prop: 60 if prop == cv2.CAP_PROP_FRAME_COUNT else 30.0

    # Create 4 clips with distinct scores, so they can be sorted.
    # Timestamps will be 0.0, 0.25, 0.5, ...
    # We'll need `flow_data` of length (60 frames / sample_interval frames).
    # sample_interval default is 0.25. (30 * 0.25 = 7.5 frames, so 8 frames).
    # 60 / 8 = 7 items in flow_data.

    mocker.patch('video_analyzer._compute_optical_flow', return_value=[
        {"motion_score": 0.9, "drift_score": 0.1, "cam_type": "onboard"},  # score high
        {"motion_score": 0.2, "drift_score": 0.1, "cam_type": "onboard"},  # score low
        {"motion_score": 0.8, "drift_score": 0.1, "cam_type": "onboard"},  # score high
        {"motion_score": 0.1, "drift_score": 0.1, "cam_type": "onboard"},  # score low
        {"motion_score": 0.7, "drift_score": 0.1, "cam_type": "onboard"},  # score high
        {"motion_score": 0.1, "drift_score": 0.1, "cam_type": "onboard"},  # score low
        {"motion_score": 0.6, "drift_score": 0.1, "cam_type": "onboard"},  # score high
    ])

    mocker.patch('video_analyzer._analyze_audio', return_value=np.array([0.5]*60))
    mocker.patch('video_analyzer._analyze_telemetry', return_value=np.array([0.0]*60))

    # Ask for 5 clips, but clip_duration is 1.0s.
    # Total video is 2s.
    # Initial selection with 0.5s overlap will pick maybe 2-3 clips.
    # Then it will gap-fill to try and reach 5, but maybe it won't reach 5 if overlap conditions prevent it.

    res = find_highlights(str(video_path), num_clips=5, clip_duration=0.5, )

    # As long as it executed and returned a list of ClipInfo, we covered the fallback logic.
    assert isinstance(res, list)
    assert len(res) > 0


def test_analyze_audio_no_exception(mocker, tmp_path):
    """Test _analyze_audio successful execution."""
    from video_analyzer import _analyze_audio
    import numpy as np

    video_path = tmp_path / "test.mp4"
    video_path.write_text("dummy")

    mock_clip = MagicMock()
    mock_clip.audio = MagicMock()
    mocker.patch('video_analyzer.VideoFileClip', return_value=mock_clip)

    # Mock librosa components
    mocker.patch('video_analyzer.librosa.load', return_value=(np.zeros(22050), 22050))
    mocker.patch('video_analyzer.librosa.feature.rms', return_value=np.array([[0.1, 0.2, 0.3]]))
    mocker.patch('video_analyzer.librosa.frames_to_time', return_value=np.array([0.0, 0.5, 1.0]))
    mocker.patch('video_analyzer.os.remove')

    res = _analyze_audio(str(video_path), total_frames=30, fps=30.0)

    assert isinstance(res, np.ndarray)
    assert len(res) == 30
    assert np.max(res) <= 1.0


def test_analyze_telemetry_valid_csv(mocker, tmp_path):
    """Test _analyze_telemetry with a valid CSV."""
    from video_analyzer import _analyze_telemetry
    import pandas as pd

    video_path = tmp_path / "test.mp4"
    csv_path = tmp_path / "test_telemetry.csv"

    df = pd.DataFrame({
        "Time": [0.0, 0.5, 1.0],
        "G_Lat": [0.1, 0.5, 0.2],
        "G_Long": [0.2, 0.3, 0.1]
    })
    df.to_csv(csv_path, index=False)

    res = _analyze_telemetry(str(video_path), total_frames=30, fps=30.0)

    assert isinstance(res, np.ndarray)
    assert len(res) == 30
    assert np.max(res) <= 1.0
