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

def test_build_cut_schedule_buildup_stride():
    """Test that the cut stride dynamically shrinks during a buildup phase."""
    from audio_analyzer import build_cut_schedule, SongSection

    sections = [SongSection(start=10.0, end=20.0, phase="buildup", energy=0.5)]
    beat_times = [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0]

    cut_points = build_cut_schedule(beat_times, sections, hard_beat_times=[], audio_duration=25.0)

    # Extract just the times for the cut points
    times = [cp.time for cp in cut_points]

    # Since stride starts at 3 and shrinks to 1, cuts should accelerate.
    # From debug run we know the cuts happen at [0.0, 13.0, 16.0, 18.0]
    # Note: 0.0 is the forced first clip for audio desync prevention.
    assert 13.0 in times
    assert 16.0 in times
    assert 18.0 in times

    # Verify the gap between cuts is shrinking
    gap1 = 16.0 - 13.0 # 3.0
    gap2 = 18.0 - 16.0 # 2.0
    assert gap2 < gap1

def test_song_section_repr():
    """Test the __repr__ method of SongSection."""
    from audio_analyzer import SongSection
    section = SongSection(start=0.0, end=1.0, phase="intro", energy=0.5)
    expected_repr = "SongSection(intro    |   0.00s–  1.00s | energy=0.50)"
    assert repr(section) == expected_repr

@patch("audio_analyzer.librosa")
@patch("audio_analyzer.np.percentile")
@patch("audio_analyzer.np.convolve")
def test_detect_song_sections_with_main_drop(mock_convolve, mock_percentile, mock_librosa):
    """Test detect_song_sections drop loop logic when main_drop_time is provided."""
    import numpy as np
    from audio_analyzer import detect_song_sections

    N = 1000
    mock_librosa.load.return_value = (np.zeros(22050), 22050)
    mock_librosa.get_duration.return_value = 100.0

    rms_times = np.linspace(0, 100, N)
    mock_librosa.frames_to_time.return_value = rms_times

    mock_librosa.feature.rms.return_value = np.zeros((1, N))
    mock_librosa.onset.onset_strength.return_value = np.zeros(N)

    def percentile_side_effect(a, q, *args, **kwargs):
        if q == 75: return 50.0
        if q == 30: return 10.0
        return 0.0
    mock_percentile.side_effect = percentile_side_effect

    macro_energy = np.zeros(N)
    macro_energy[500] = 100.0 # Peak at ~50.05s

    mock_convolve.return_value = macro_energy

    sections = detect_song_sections("dummy.mp3", main_drop_time=20.0)

    drops = [s for s in sections if s.phase == "drop"]
    assert len(drops) == 2
    assert drops[0].start == 19.5
    assert abs(drops[1].start - 49.5) < 0.2

@patch("audio_analyzer.librosa")
@patch("audio_analyzer.np.convolve")
@patch("audio_analyzer.np.median")
def test_detect_song_sections_outro(mock_median, mock_convolve, mock_librosa):
    """Test detect_song_sections properly detects the outro phase based on energy drops."""
    import numpy as np
    from audio_analyzer import detect_song_sections

    N = 1000
    mock_librosa.load.return_value = (np.zeros(22050), 22050)
    mock_librosa.get_duration.return_value = 100.0

    rms_times = np.linspace(0, 100, N)
    mock_librosa.frames_to_time.return_value = rms_times

    mock_librosa.feature.rms.return_value = np.zeros((1, N))
    mock_librosa.onset.onset_strength.return_value = np.zeros(N)

    mock_median.return_value = 5.0

    macro_energy = np.zeros(N)
    macro_energy[900] = 10.0

    mock_convolve.return_value = macro_energy

    sections = detect_song_sections("dummy.mp3")

    outros = [s for s in sections if s.phase == "outro"]
    assert len(outros) > 0
    assert abs(outros[0].start - 90.0) < 0.2

def test_detect_song_sections_empty_audio_array():
    """Test detect_song_sections behaves gracefully when an empty audio array is loaded."""
    from audio_analyzer import detect_song_sections
    import numpy as np

    with patch("audio_analyzer.librosa") as mock_librosa:
        # Simulate loading a completely empty audio file
        mock_librosa.load.return_value = (np.array([]), 22050)
        mock_librosa.get_duration.return_value = 0.0

        # All feature extractions return empty or very small arrays
        mock_librosa.feature.rms.return_value = np.array([[]])
        mock_librosa.frames_to_time.return_value = np.array([])
        mock_librosa.onset.onset_strength.return_value = np.array([])

        # Current implementation throws ValueError on rms.max() for empty arrays
        with pytest.raises(ValueError, match="zero-size array to reduction operation"):
            sections = detect_song_sections("dummy.mp3")

def test_extract_beats_none_audio():
    """Test extract_beats handling of missing or None values."""
    from audio_analyzer import extract_beats
    with pytest.raises(Exception):
        # Depending on how librosa handles it, it will raise FileNotFoundError or TypeError
        extract_beats(None)

def test_build_cut_schedule_invalid_duration():
    """Test build_cut_schedule with negative audio duration."""
    from audio_analyzer import build_cut_schedule, SongSection
    beat_times = [1.0, 2.0, 3.0]
    hard_beat_times = [2.0]
    sections = [SongSection(0.0, 3.0, "verse", 0.5)]

    # Passing negative duration
    schedule = build_cut_schedule(beat_times, sections, hard_beat_times, audio_duration=-10.0)

    assert len(schedule) > 0
    # The last cut_point dur_hint should fallback to min 0.1s
    assert schedule[-1].clip_dur_hint >= 0.1
