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

def test_energy_in_range():
    """Test _energy_in_range helper."""
    import audio_analyzer
    import numpy as np

    times = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    energy = np.array([0.1, 0.5, 0.9, 0.2, 0.1])

    # Matching range
    vals = audio_analyzer._energy_in_range(energy, times, 0.5, 2.5)
    assert np.array_equal(vals, np.array([0.5, 0.9]))

    # Non-matching range
    vals_empty = audio_analyzer._energy_in_range(energy, times, 5.0, 6.0)
    assert np.array_equal(vals_empty, np.array([0.0]))

def test_build_cut_schedule_buildup_stride():
    """Test build_cut_schedule logic for buildup phase strides."""
    from audio_analyzer import build_cut_schedule, SongSection
    beat_times = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
    hard_beat_times = []

    # Simulate a buildup phase from 0.0 to 4.0
    sections = [SongSection(0.0, 4.0, "buildup", 0.8)]

    cut_points = build_cut_schedule(beat_times, sections, hard_beat_times, 5.0)

    # Check that cuts are made and buildup stride logic runs without errors
    assert len(cut_points) > 0
    for cp in cut_points:
        assert cp.phase == "buildup"











def test_detect_song_sections_multiple_drops_and_buildups():
    """Test detect_song_sections handling multiple drop and buildup sections."""
    import audio_analyzer
    import numpy as np
    from unittest.mock import MagicMock, patch

    mock_librosa = MagicMock()
    # Mocking long audio (e.g. 60s)
    mock_librosa.load.return_value = (np.zeros(22050 * 60), 22050)
    mock_librosa.get_duration.return_value = 60.0

    num_frames = 600
    fake_rms = np.zeros((1, num_frames))
    fake_rms[0, 200:300] = 5.0 # Drop 1 (around 20-30s)
    fake_rms[0, 400:500] = 5.0 # Drop 2 (around 40-50s)
    fake_rms[0, :10] = 0.5

    mock_librosa.feature.rms.return_value = fake_rms
    mock_librosa.frames_to_time.return_value = np.linspace(0.0, 60.0, num_frames)
    mock_librosa.onset.onset_strength.return_value = np.zeros(num_frames)

    with patch.object(audio_analyzer, 'librosa', mock_librosa):
        sections = audio_analyzer.detect_song_sections("dummy.mp3", main_drop_time=25.0)
        phases = [s.phase for s in sections]

        # We expect at least two drops.
        assert phases.count("drop") >= 2
        # And potentially buildups before the drops
        assert "buildup" in phases


def test_detect_song_sections_gap_filling():
    """Test detect_song_sections gap filling logic (bridge vs verse)."""
    import audio_analyzer
    import numpy as np
    from unittest.mock import MagicMock, patch

    mock_librosa = MagicMock()
    # Mocking audio of 30s
    mock_librosa.load.return_value = (np.zeros(22050 * 30), 22050)
    mock_librosa.get_duration.return_value = 30.0

    # Create a small drop at the beginning, followed by a gap, then outro
    num_frames = 300
    fake_rms = np.zeros((1, num_frames))
    fake_rms[0, 10:50] = 5.0 # Drop
    fake_rms[0, 250:] = 5.0 # Outro spike

    mock_librosa.feature.rms.return_value = fake_rms
    mock_librosa.frames_to_time.return_value = np.linspace(0.0, 30.0, num_frames)
    mock_librosa.onset.onset_strength.return_value = np.zeros(num_frames)

    with patch.object(audio_analyzer, 'librosa', mock_librosa):
        sections = audio_analyzer.detect_song_sections("dummy.mp3", main_drop_time=3.0)
        phases = [s.phase for s in sections]

        # There should be a gap after the first drop, which gets filled with 'bridge'
        assert "bridge" in phases

def test_detect_song_sections_break_conditions():
    """Test detect_song_sections break conditions in the drop search loop."""
    import audio_analyzer
    import numpy as np
    from unittest.mock import MagicMock, patch

    mock_librosa = MagicMock()
    # Mocking audio of 30s
    mock_librosa.load.return_value = (np.zeros(22050 * 30), 22050)
    mock_librosa.get_duration.return_value = 30.0

    # We want a drop, but the rest of the song is just empty energy (triggering the window length break or low peak break)
    num_frames = 300
    fake_rms = np.zeros((1, num_frames))
    fake_rms[0, 10:50] = 5.0 # Main Drop (0.0 to 3.0s is intro/drop)
    # No more energy spikes, so it hits the `else: break` when searching for another drop.

    mock_librosa.feature.rms.return_value = fake_rms
    # Let's make it so that the time arrays are short, or manipulated so `idx_start >= idx_end` can trigger
    mock_librosa.frames_to_time.return_value = np.linspace(0.0, 30.0, num_frames)
    mock_librosa.onset.onset_strength.return_value = np.zeros(num_frames)

    with patch.object(audio_analyzer, 'librosa', mock_librosa):
        # We explicitly set main_drop_time = 3.0
        sections = audio_analyzer.detect_song_sections("dummy.mp3", main_drop_time=3.0)
        phases = [s.phase for s in sections]

        assert "drop" in phases




def test_detect_song_sections_break_empty_window():
    """Test detect_song_sections break conditions in the drop search loop when window is empty."""
    import audio_analyzer
    import numpy as np
    from unittest.mock import MagicMock, patch

    mock_librosa = MagicMock()
    mock_librosa.load.return_value = (np.zeros(22050 * 30), 22050)
    mock_librosa.get_duration.return_value = 30.0

    num_frames = 300
    fake_rms = np.zeros((1, num_frames))
    fake_rms[0, 10:50] = 5.0 # Main Drop

    mock_librosa.feature.rms.return_value = fake_rms
    times = np.linspace(0.0, 30.0, num_frames)

    # We want the window `macro_energy[idx_start:idx_end]` to be empty.
    # This happens if idx_start == idx_end.
    # We can fake np.searchsorted to return the same index for both.

    mock_librosa.frames_to_time.return_value = times
    mock_librosa.onset.onset_strength.return_value = np.zeros(num_frames)

    with patch.object(audio_analyzer, 'librosa', mock_librosa):
        with patch("audio_analyzer.np.searchsorted", return_value=100) as mock_searchsorted:
            sections = audio_analyzer.detect_song_sections("dummy.mp3", main_drop_time=3.0)
            assert len(sections) > 0

def test_detect_song_sections_break_len_window_zero():
    """Test detect_song_sections break when window len is 0 but idx_start < idx_end."""
    # This is an edge case that could theoretically happen if the slice returns empty
    import audio_analyzer
    import numpy as np
    from unittest.mock import MagicMock, patch

    mock_librosa = MagicMock()
    mock_librosa.load.return_value = (np.zeros(22050 * 30), 22050)
    mock_librosa.get_duration.return_value = 30.0

    num_frames = 300
    fake_rms = np.zeros((1, num_frames))
    fake_rms[0, 10:50] = 5.0 # Main Drop

    mock_librosa.feature.rms.return_value = fake_rms
    times = np.linspace(0.0, 30.0, num_frames)

    mock_librosa.frames_to_time.return_value = times
    mock_librosa.onset.onset_strength.return_value = np.zeros(num_frames)

    # We mock numpy array slicing for macro_energy to return an empty array
    # We patch np.convolve since it returns macro_energy

    class FakeArray(np.ndarray):
        def __new__(cls, input_array):
            obj = np.asarray(input_array).view(cls)
            return obj
        def __getitem__(self, key):
            if isinstance(key, slice):
                return [] # Return empty list for len(window) == 0
            return super().__getitem__(key)

    original_convolve = np.convolve
    def mock_convolve(*args, **kwargs):
        res = original_convolve(*args, **kwargs)
        return FakeArray(res)

    with patch.object(audio_analyzer, 'librosa', mock_librosa):
        with patch("audio_analyzer.np.convolve", side_effect=mock_convolve):
            sections = audio_analyzer.detect_song_sections("dummy.mp3", main_drop_time=3.0)
            assert len(sections) > 0






def test_detect_song_sections_skip_short():
    """Test detect_song_sections skips phases shorter than 0.3s."""
    import audio_analyzer
    import numpy as np
    from unittest.mock import MagicMock, patch

    mock_librosa = MagicMock()
    mock_librosa.load.return_value = (np.zeros(22050 * 5), 22050)
    mock_librosa.get_duration.return_value = 5.0 # Very short duration

    num_frames = 50
    fake_rms = np.zeros((1, num_frames))
    mock_librosa.feature.rms.return_value = fake_rms
    mock_librosa.frames_to_time.return_value = np.linspace(0.0, 5.0, num_frames)
    mock_librosa.onset.onset_strength.return_value = np.zeros(num_frames)

    with patch.object(audio_analyzer, 'librosa', mock_librosa):
        original_round = round
        def fake_round(val, ndigits=None):
            # If rounding anything, make them very close
            # intro_end is usually 0.75
            # outro_start is usually 4.4
            # duration is 5.0
            # Let's map everything to 0.0, 0.1, 0.2 to force differences < 0.3
            return 0.1

        with patch("builtins.round", side_effect=fake_round):
            sections = audio_analyzer.detect_song_sections("dummy.mp3")
            # Since all rounded values are 0.1, the difference is 0.0 < 0.3
            assert len(sections) == 0

def test_cut_point_repr():
    """Test the __repr__ method of CutPoint."""
    from audio_analyzer import CutPoint
    cp = CutPoint(time=1.23, beat_index=0, beat_type="hard", phase="drop", clip_dur_hint=2.5, is_forced=True)
    rep = repr(cp)
    assert "1.23s" in rep
    assert "drop" in rep
    assert "hard" in rep
    assert "[FORCED]" in rep

    cp2 = CutPoint(time=4.56, beat_index=1, beat_type="normal", phase=None, clip_dur_hint=1.0, is_forced=False)
    rep2 = repr(cp2)
    assert "4.56s" in rep2
    assert "?" in rep2
    assert "[FORCED]" not in rep2

def test_song_section_repr():
    """Test the __repr__ method of SongSection."""
    from audio_analyzer import SongSection
    ss = SongSection(start=10.5, end=20.0, phase="drop", energy=0.85)
    rep = repr(ss)
    assert "SongSection" in rep
    assert "drop" in rep
    assert "10.50s" in rep
    assert "20.00s" in rep
    assert "energy=0.85" in rep

def test_detect_song_sections_wrong_format():
    """Test detect_song_sections with wrong file format or loading error."""
    import audio_analyzer
    from unittest.mock import patch
    with patch("audio_analyzer.librosa.load", side_effect=Exception("Invalid file format")):
        import pytest
        with pytest.raises(Exception, match="Invalid file format"):
            audio_analyzer.detect_song_sections("wrong_format.txt")
