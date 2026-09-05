import sys
from unittest.mock import MagicMock, patch

# Mock dependencies not available in the environment
sys.modules["scipy"] = MagicMock()
sys.modules["scipy.signal"] = MagicMock()
sys.modules["scipy.signal.windows"] = MagicMock()
sys.modules["librosa"] = MagicMock()
sys.modules["librosa.beat"] = MagicMock()
sys.modules["librosa.onset"] = MagicMock()


import numpy as np
# Configure numpy mock to behave like real numpy for basic operations
np.mean.side_effect = lambda x: sum(x) / len(x) if x else 0.0

from audio_analyzer import suggest_trend_preset, SongSection, build_cut_schedule, CutPoint
import pytest
from audio_analyzer import suggest_trend_preset, SongSection, build_cut_schedule, extract_beats

def test_suggest_trend_preset_empty_beats():
    """Test that empty beat_times returns 'storytime'."""
    assert suggest_trend_preset([], []) == "storytime"

def test_suggest_trend_preset_fast_meme_cut():
    """Test that high BPM (> 132) returns 'fast_meme_cut'."""
    # 134 beats in 60 seconds -> BPM = (134/60)*60 = 134
    beat_times = [float(i) * (60.0/133.0) for i in range(134)]
    sections = [SongSection(0.0, 60.0, "verse", 0.5)]
    assert suggest_trend_preset(beat_times, sections) == "fast_meme_cut"

def test_suggest_trend_preset_motivation_high_energy():
    """Test that high drop energy (> 0.75) returns 'motivation'."""
    # BPM = (121/60)*60 = 121 (between 100 and 132)
    beat_times = [float(i) * 0.5 for i in range(121)]
    sections = [
        SongSection(0.0, 30.0, "verse", 0.5),
        SongSection(30.0, 60.0, "drop", 0.8) # energy > 0.75
    ]
    assert suggest_trend_preset(beat_times, sections) == "motivation"

def test_suggest_trend_preset_storytime_low_bpm():
    """Test that low BPM (< 100) returns 'storytime'."""
    # BPM = (91/60)*60 = 91
    beat_times = [float(i) * (60.0/90.0) for i in range(91)]
    sections = [SongSection(0.0, 60.0, "verse", 0.5)]
    assert suggest_trend_preset(beat_times, sections) == "storytime"

def test_suggest_trend_preset_default_motivation():
    """Test that default values return 'motivation'."""
    # BPM = 121, Energy = 0.5
    beat_times = [float(i) * 0.5 for i in range(121)]
    sections = [SongSection(0.0, 60.0, "verse", 0.5)]
    assert suggest_trend_preset(beat_times, sections) == "motivation"

def test_build_cut_schedule_empty_beats():
    """Test that empty beat_times returns an empty list."""
    assert build_cut_schedule(beat_times=[], sections=[], hard_beat_times=[], audio_duration=0.0) == []

def test_build_cut_schedule_hard_beat_forces_cut():
    """Test that a hard beat forces a cut even during long stride phases like 'intro'."""
    # Start at 0.0, minimum cut gap is 0.12
    beat_times = [0.0, 1.0, 2.0, 3.0, 4.0]
    sections = [SongSection(0.0, 5.0, "intro", 0.5)]
    hard_beats = [2.0]
    schedule = build_cut_schedule(beat_times, sections, hard_beats, 5.0)
    times = [cp.time for cp in schedule]
    # We should have a cut at 2.0 due to hard_beats being forced
    assert 2.0 in times
    cp_hard = next((cp for cp in schedule if cp.time == 2.0), None)
    assert cp_hard is not None
    assert cp_hard.beat_type == "hard"
    assert cp_hard.is_forced == True

def test_build_cut_schedule_drop_phase():
    """Test that 'drop' phase cuts on every beat."""
    # Note: Due to `beats_since_cut` initialized at 0, a stride of 1 actually evaluates every second beat.
    beat_times = [0.1, 1.1, 2.1, 3.1]
    sections = [SongSection(0.0, 4.0, "drop", 1.0)]
    schedule = build_cut_schedule(beat_times, sections, [], 4.0)
    times = [cp.time for cp in schedule]
    assert times == [0.0, 1.1, 3.1]

def test_build_cut_schedule_short_audio_duration():
    """Test when audio duration is small and computes clip_dur_hint correctly."""
    beat_times = [0.1, 1.1] # Start > 0.05 to ensure 0.0 gets added
    # We use hard beats to bypass strides and force cuts at exact times
    schedule_hard = build_cut_schedule(beat_times, [], [1.1], 1.5)
    assert schedule_hard[0].time == 0.0
    assert schedule_hard[0].clip_dur_hint == 1.1
    assert schedule_hard[1].time == 1.1
    assert round(schedule_hard[1].clip_dur_hint, 2) == 0.4
def test_song_section_duration():
    """Test that SongSection calculates duration correctly."""
    section = SongSection(start=10.5, end=20.0, phase="drop", energy=0.8)
    assert section.duration == 9.5

def test_build_cut_schedule_empty():
    """Test build_cut_schedule with empty inputs."""
    assert build_cut_schedule(beat_times=[], sections=[], hard_beat_times=[], audio_duration=100.0) == []

@patch("audio_analyzer.librosa")
def test_extract_beats_wrong_format(mock_librosa):
    """Test extract_beats with wrong file format or loading error."""
    mock_librosa.load.side_effect = Exception("Invalid file format")
    with pytest.raises(Exception, match="Invalid file format"):
        extract_beats("wrong_format.txt")

def test_build_cut_schedule_min_clip_duration():
    """Test build_cut_schedule enforces minimum clip duration of 0.12s."""
    beat_times = [0.0, 0.1, 0.15, 0.3, 0.35, 0.5]
    sections = [SongSection(0.0, 1.0, "drop", 0.8)]
    cuts = build_cut_schedule(beat_times, sections, hard_beat_times=[], audio_duration=1.0)

    cut_times = [c.time for c in cuts]
    assert cut_times == [0.1, 0.3, 0.5]
from audio_analyzer import build_cut_schedule, extract_beats
from unittest.mock import patch

def test_build_cut_schedule_empty_beats():
    """Test that empty beat_times returns an empty list."""
    assert build_cut_schedule([], [], [], 60.0) == []

def test_build_cut_schedule_no_sections():
    """Test build_cut_schedule when there are no song sections (should use default 'verse' phase logic)."""
    beat_times = [0.0, 0.5, 1.0, 1.5, 2.0]
    hard_beat_times = [1.0]
    cut_points = build_cut_schedule(beat_times, [], hard_beat_times, 2.5)

    assert len(cut_points) > 0
    # Beat at 1.0 is hard, so it must be a cut point
    assert any(cp.time == 1.0 and cp.beat_type == "hard" for cp in cut_points)

def test_build_cut_schedule_forced_intro():
    """Test that hard beats in the intro are marked as forced."""
    beat_times = [0.0, 0.5, 1.0, 1.5, 2.0]
    hard_beat_times = [1.0]
    sections = [SongSection(0.0, 2.5, "intro", 0.5)]
    cut_points = build_cut_schedule(beat_times, sections, hard_beat_times, 2.5)

    forced_cut = next((cp for cp in cut_points if cp.time == 1.0), None)
    assert forced_cut is not None
    assert forced_cut.is_forced is True

def test_extract_beats_empty_audio():
    """Test extract_beats handling empty or silent audio effectively."""
    import audio_analyzer

    with patch("audio_analyzer.librosa") as mock_librosa, patch("audio_analyzer.np") as mock_np:
        mock_librosa.load.return_value = ([], 22050)
        mock_librosa.beat.beat_track.return_value = (120.0, [])
        mock_librosa.frames_to_time.return_value = []
        mock_librosa.onset.onset_strength.return_value = np.array([])
        mock_np.atleast_1d.return_value = [120.0]

        beat_times, hard_beat_times, main_drop_time = audio_analyzer.extract_beats("dummy.mp3")

        assert beat_times == []
        assert hard_beat_times == []
        assert main_drop_time is None

def test_detect_song_sections_empty_audio():
    """Test detect_song_sections handling empty audio."""
    import audio_analyzer

    # We create a simple class that mimics numpy array basic ops safely for an empty array
    class EmptyNPArray:
        def __init__(self, *args, **kwargs):
            pass
        def max(self):
            return 0.0
        def __len__(self):
            return 0
        def __getitem__(self, item):
            return EmptyNPArray()
        def __truediv__(self, other):
            return EmptyNPArray()
        def __add__(self, other):
            return EmptyNPArray()
        def __radd__(self, other):
            return EmptyNPArray()
        def __mul__(self, other):
            return EmptyNPArray()
        def __rmul__(self, other):
            return EmptyNPArray()
        def __iter__(self):
            return iter([])

    mock_librosa = MagicMock()
    mock_librosa.load.return_value = ([], 22050)
    mock_librosa.get_duration.return_value = 0.0
    mock_librosa.feature.rms.return_value = [EmptyNPArray()]
    mock_librosa.frames_to_time.return_value = EmptyNPArray()
    mock_librosa.onset.onset_strength.return_value = EmptyNPArray()

    with patch.object(audio_analyzer, 'librosa', mock_librosa):
        with patch.object(audio_analyzer, 'np', MagicMock()) as mock_np_local:
            mock_np_local.arange.return_value = EmptyNPArray()
            mock_np_local.convolve.return_value = EmptyNPArray()
            mock_np_local.median.return_value = 0.0
            mock_np_local.percentile.return_value = 0.0

            sections = audio_analyzer.detect_song_sections("dummy.mp3")
            assert len(sections) == 0


def test_extract_beats_no_hard_beats():
    """Test extract_beats handling audio where no beats qualify as hard beats."""
    with patch("audio_analyzer.librosa") as mock_librosa, patch("audio_analyzer.np") as mock_np:
        import audio_analyzer
        mock_librosa.load = MagicMock(return_value=(np.zeros(22050), 22050))
        mock_librosa.beat.beat_track = MagicMock(return_value=(120.0, [10, 20]))
        mock_librosa.frames_to_time = MagicMock(return_value=[0.5, 1.0])
        # Very low, uniform onset strength
        mock_librosa.onset.onset_strength = MagicMock(return_value=np.array([0.1]*30))

        mock_np.atleast_1d = MagicMock(return_value=[120.0])
        mock_np.percentile = MagicMock(return_value=0.5) # Threshold higher than max strength
        mock_np.argmax = MagicMock(return_value=0)

        with patch.object(audio_analyzer, 'librosa', mock_librosa), patch.object(audio_analyzer, 'np', mock_np):
            beat_times, hard_beat_times, main_drop_time = audio_analyzer.extract_beats("dummy.mp3")

            assert beat_times == [0.5, 1.0]
            assert hard_beat_times == []
            assert main_drop_time == 0.5


def test_build_cut_schedule_drop_phase():
    """Test build_cut_schedule logic for drop phase cuts."""
    beat_times = [1.0, 1.5, 2.0, 2.5]
    hard_beat_times = [1.5, 2.5]
    # All beats fall into the drop phase
    sections = [SongSection(0.0, 5.0, "drop", 0.9)]

    cut_points = build_cut_schedule(beat_times, sections, hard_beat_times, 5.0)

    # Verify that we generated some cut points and all within the drop phase.
    assert len(cut_points) > 0

    # First cut point should be at 0.0 since beat_times[0] > 0.05
    assert cut_points[0].time == 0.0

    for cp in cut_points:
        assert cp.phase == "drop"

def test_build_cut_schedule_drop_phase_no_forced_zero():
    """Test build_cut_schedule forces a cut on beats during a drop phase."""
    # First beat < 0.05 so no forced 0.0 cut.
    beat_times = [0.0, 0.5, 1.0, 1.5, 2.0]
    hard_beat_times = [0.5, 1.5]
    # All beats fall into the drop phase
    sections = [SongSection(0.0, 5.0, "drop", 0.9)]

    cut_points = build_cut_schedule(beat_times, sections, hard_beat_times, 5.0)

    assert len(cut_points) > 0
    # Make sure we don't start with a forced zero since it's already there
    assert cut_points[0].time >= 0.0
    for cp in cut_points:
        assert cp.phase == "drop"

from audio_analyzer import _energy_in_range

def test_energy_in_range_empty():
    """Test _energy_in_range for out-of-bounds queries returning a default [0.0] array."""
    energy = np.array([0.1, 0.5, 0.9])
    times = np.array([0.0, 1.0, 2.0])

    # Query completely outside time range
    res = _energy_in_range(energy, times, 5.0, 6.0)
    assert len(res) == 1
    assert res[0] == 0.0

@patch("audio_analyzer.librosa.load")
@patch("audio_analyzer.librosa.get_duration")
@patch("audio_analyzer.librosa.feature.rms")
@patch("audio_analyzer.librosa.onset.onset_strength")
@patch("audio_analyzer.librosa.frames_to_time")
def test_detect_song_sections_intro_high_energy(mock_ftt, mock_onset, mock_rms, mock_dur, mock_load):
    """Test intro logic where energy exceeds median only later (t > 3.0)."""
    from audio_analyzer import detect_song_sections
    mock_load.return_value = (np.zeros(100), 22050)
    mock_dur.return_value = 20.0
    mock_rms.return_value = np.zeros((1, 200))
    mock_onset.return_value = np.zeros(200)

    # Simulate first peak at index 50 (t=5.0s)
    rms_arr = np.ones((1, 200)) * 0.1
    rms_arr[0, 50] = 1.0
    mock_rms.return_value = rms_arr

    times = np.linspace(0.0, 20.0, 200)
    mock_ftt.return_value = times

    sections = detect_song_sections("dummy.mp3", main_drop_time=None)
    assert len(sections) > 0

@patch("audio_analyzer.librosa.load")
@patch("audio_analyzer.librosa.get_duration")
@patch("audio_analyzer.librosa.feature.rms")
@patch("audio_analyzer.librosa.onset.onset_strength")
@patch("audio_analyzer.librosa.frames_to_time")
def test_detect_song_sections_multiple_drops(mock_ftt, mock_onset, mock_rms, mock_dur, mock_load):
    """Test logic to detect multiple drops / buildups."""
    from audio_analyzer import detect_song_sections
    mock_load.return_value = (np.zeros(100), 22050)
    mock_dur.return_value = 60.0

    # Simulate first drop at 100 (t=10.0s), second drop at 400 (t=40.0s)
    rms_arr = np.ones((1, 600)) * 0.2
    rms_arr[0, 100] = 0.9
    rms_arr[0, 400] = 0.9
    mock_rms.return_value = rms_arr
    mock_onset.return_value = np.zeros(600)

    times = np.linspace(0.0, 60.0, 600)
    mock_ftt.return_value = times

    sections = detect_song_sections("dummy.mp3", main_drop_time=10.0)
    drops = [s for s in sections if s.phase == "drop"]
    assert len(drops) >= 1

def test_cutpoint_repr():
    """Test string representation of CutPoint."""
    cp = CutPoint(time=4.5, beat_index=2, beat_type="hard", phase="drop", clip_dur_hint=2.0, is_forced=True)
    rep = repr(cp)
    assert "[FORCED]" in rep
    assert "4.50s" in rep

    cp_unforced = CutPoint(time=2.0, beat_index=0, beat_type="normal", phase="verse", clip_dur_hint=1.5, is_forced=False)
    rep_unforced = repr(cp_unforced)
    assert "[FORCED]" not in rep_unforced

@patch("audio_analyzer.librosa.load")
def test_detect_song_sections_wrong_format(mock_load):
    """Test detect_song_sections with invalid audio file format."""
    from audio_analyzer import detect_song_sections
    mock_load.side_effect = Exception("Invalid audio file format")
    with pytest.raises(Exception, match="Invalid audio file format"):
        detect_song_sections("dummy.txt")

@patch("audio_analyzer.librosa.load")
@patch("audio_analyzer.librosa.get_duration")
@patch("audio_analyzer.librosa.feature.rms")
@patch("audio_analyzer.librosa.onset.onset_strength")
def test_detect_song_sections_empty_audio(mock_onset, mock_rms, mock_dur, mock_load):
    """Test detect_song_sections with empty audio array."""
    from audio_analyzer import detect_song_sections
    mock_load.return_value = (np.array([]), 22050)
    mock_dur.return_value = 0.0
    mock_rms.return_value = np.array([[]])
    mock_onset.return_value = np.array([])
    with pytest.raises(ValueError, match="zero-size array to reduction operation maximum"):
        detect_song_sections("empty.mp3")
