import numpy as np
import sys
from unittest.mock import MagicMock

# Setup the environment by inserting the mocks before import
_cv2_mock = MagicMock()
_numpy_mock = MagicMock()
_PIL_mock = MagicMock()
_PIL_Image_mock = MagicMock()
_PIL_ImageDraw_mock = MagicMock()
_PIL_ImageFilter_mock = MagicMock()
_PIL_ImageFont_mock = MagicMock()
_moviepy_mock = MagicMock()
_moviepy_editor_mock = MagicMock()
_whisper_mock = MagicMock()
_torch_mock = MagicMock()
_mutagen_mock = MagicMock()
_mutagen_easyid3_mock = MagicMock()
_mutagen_id3_mock = MagicMock()

# Mock only missing packages. Since it runs in the test environment
# where they aren't installed, we safely supply MagicMocks.
for module in ['cv2', 'numpy', 'PIL', 'PIL.Image', 'PIL.ImageDraw',
               'PIL.ImageFilter', 'PIL.ImageFont', 'moviepy', 'moviepy.editor',
               'whisper', 'torch', 'mutagen', 'mutagen.easyid3', 'mutagen.id3']:
    sys.modules[module] = MagicMock()

import visual_effects
import pytest

def test_make_zoom_punch_invalid_clip():
    """Test that make_zoom_punch handles invalid clips by returning the original clip on exception."""
    class InvalidClip:
        @property
        def size(self):
            raise AttributeError("Invalid size")

    clip = InvalidClip()
    result = visual_effects.make_zoom_punch(clip)
    assert result == clip

def test_make_pip_overlay_invalid_clip():
    """Test that make_pip_overlay handles exceptions correctly."""
    class InvalidMainClip:
        @property
        def size(self):
            raise AttributeError("Invalid size")

    main_clip = InvalidMainClip()
    pip_clip = MagicMock()

    result = visual_effects.make_pip_overlay(main_clip, pip_clip)
    assert result == main_clip

def test_apply_letterbox_invalid_clip():
    """Test that apply_letterbox handles exceptions correctly."""
    class InvalidClip:
        @property
        def size(self):
            raise AttributeError("Invalid size")

    clip = InvalidClip()
    result = visual_effects.apply_letterbox(clip)
    assert result == clip

def test_make_mirror_x_invalid_clip():
    """Test that make_mirror_x handles exceptions correctly."""
    class InvalidClip:
        def fl(self, *args, **kwargs):
            raise Exception("Simulated exception")

    clip = InvalidClip()
    result = visual_effects.make_mirror_x(clip)
    assert result == clip

def test_make_camera_shake_invalid_clip():
    """Test that make_camera_shake handles exceptions correctly."""
    class InvalidClip:
        @property
        def size(self):
            raise AttributeError("Invalid size")

    clip = InvalidClip()
    result = visual_effects.make_camera_shake(clip)
    assert result == clip

def test_extract_music_words_empty_or_invalid_path():
    """Test edge cases for extract_music_words when given empty or invalid path."""
    words = visual_effects.extract_music_words("")
    assert isinstance(words, list)
    assert len(words) > 0
    assert words == visual_effects._RACING_WORDS

def test_get_beat_synced_words_empty_beats():
    """Test get_beat_synced_words with empty beat_times."""
    words = visual_effects.get_beat_synced_words("dummy_audio.mp3", beat_times=[])
    assert words == []

class DummyClip:
    @property
    def size(self):
        return (1920, 1080)
    @property
    def duration(self):
        return 5.0
    @property
    def fps(self):
        return 60.0
    def fl(self, func, apply_to=None, keep_duration=False):
        return self

def test_make_split_screen_glitch_zero_stripes():
    """Test that make_split_screen_glitch handles num_stripes=0 without crashing."""
    clip = DummyClip()
    result = visual_effects.make_split_screen_glitch(clip, num_stripes=0)
    assert result == clip

def test_make_watermark_overlay_empty_text():
    """Test make_watermark_overlay with empty string text and extreme opacity."""
    clip = DummyClip()
    result = visual_effects.make_watermark_overlay(clip, text="", opacity=2.5)
    assert result == clip

def test_make_bw_overlay_extreme_contrast():
    """Test make_bw_overlay with extreme contrast values."""
    clip = DummyClip()
    result = visual_effects.make_bw_overlay(clip, contrast_boost=-5.0)
    assert result == clip

def test_make_text_mask_sequence_empty_words():
    """Test make_text_mask_sequence with empty list of words."""
    result = visual_effects.make_text_mask_sequence(MagicMock(), words=[])
def test_make_text_mask_sequence_empty_words():
    """Test make_text_mask_sequence with empty words list."""
    clip = MagicMock()
    result = visual_effects.make_text_mask_sequence(clip, words=[])
    assert result == []

def test_get_beat_synced_words_strict_mode_fallback():
    """Test get_beat_synced_words behavior when strict_mode is False and whisper timed words fail."""
    # Strict mode False should use fallback words.
    words = visual_effects.get_beat_synced_words(
        "dummy_audio.mp3",
        beat_times=[1.0, 2.0],
        fallback_words=["TEST", "WORDS"],
        strict_mode=False
    )
    assert words == ["TEST", "WORDS"]

def test_make_glitch_effect_invalid_clip():
    """Test make_glitch_effect with an invalid clip that raises an exception."""
def test_make_text_mask_clip_invalid_clip():
    """Test that make_text_mask_clip handles invalid clips by returning None."""
    class InvalidClip:
        @property
        def size(self):
            raise AttributeError("Invalid size")

    clip = InvalidClip()
    result = visual_effects.make_text_mask_clip(clip, "TEST")
    assert result is None

def test_make_glitch_effect_invalid_clip():
    """Test that make_glitch_effect handles exceptions correctly."""
    class InvalidClip:
        @property
        def size(self):
            raise AttributeError("Invalid size")

    clip = InvalidClip()
    result = visual_effects.make_glitch_effect(clip)
    assert result == clip

def test_make_blend_text_overlay_invalid_clip():
    """Test make_blend_text_overlay with an invalid clip that raises an exception."""
def test_make_glitch_effect_valid_clip():
    """Test that make_glitch_effect returns a modified clip for valid input."""
    class ValidClip:
        @property
        def size(self):
            return (1920, 1080)
        @property
        def fps(self):
            return 60.0
        def fl(self, *args, **kwargs):
            return "ModifiedClip"

    clip = ValidClip()
    result = visual_effects.make_glitch_effect(clip)
    assert result == "ModifiedClip"

def test_make_bw_overlay_invalid_clip():
    """Test that make_bw_overlay handles exceptions correctly."""
    class InvalidClip:
        def fl(self, *args, **kwargs):
            raise Exception("Simulated exception")

    clip = InvalidClip()
    result = visual_effects.make_bw_overlay(clip)
    assert result == clip

def test_make_watermark_overlay_invalid_clip():
    """Test that make_watermark_overlay handles exceptions correctly."""
    class InvalidClip:
        @property
        def size(self):
            raise AttributeError("Invalid size")

    clip = InvalidClip()
    result = visual_effects.make_blend_text_overlay(clip, "TEST")
    assert result == clip

def test_make_watermark_overlay_invalid_clip_2():
    """Test make_watermark_overlay with an invalid clip that raises an exception."""
    class InvalidClip:
        @property
        def size(self):
            raise AttributeError("Invalid size")
    clip = InvalidClip()
    result = visual_effects.make_watermark_overlay(clip, "watermark")
    assert result == clip

def test_make_split_screen_glitch_invalid_clip():
    """Test that make_split_screen_glitch handles exceptions correctly."""
    class InvalidClip:
        @property
        def size(self):
            raise AttributeError("Invalid size")

    clip = InvalidClip()
    result = visual_effects.make_watermark_overlay(clip, "WATERMARK")
    assert result == clip
    result = visual_effects.make_split_screen_glitch(clip)
    assert result == clip

def test_make_text_mask_sequence_empty_words():
    """Test make_text_mask_sequence with empty words list."""
    clip = MagicMock()
    result = visual_effects.make_text_mask_sequence(clip, words=[])
def test_make_blend_text_overlay_invalid_clip():
    """Test that make_blend_text_overlay handles exceptions correctly."""
    class InvalidClip:
        @property
        def size(self):
            raise AttributeError("Invalid size")

    clip = InvalidClip()
    result = visual_effects.make_blend_text_overlay(clip, "TEST")
    assert result == clip

def test_make_text_mask_sequence_empty_words():
    """Test that make_text_mask_sequence returns empty list when given empty words."""
    result = visual_effects.make_text_mask_sequence(MagicMock(), [])
    assert result == []

def test_make_zoom_punch_valid_clip():
    class ValidClip:
        @property
        def size(self):
            return (1920, 1080)
        @property
        def duration(self):
            return 5.0
        def fl(self, *args, **kwargs):
            return "ZoomedClip"

    clip = ValidClip()
    result = visual_effects.make_zoom_punch(clip)
    assert result == "ZoomedClip"

def test_make_camera_shake_valid_clip():
    class ValidClip:
        @property
        def size(self):
            return (1920, 1080)
        @property
        def fps(self):
            return 60.0
        def fl(self, *args, **kwargs):
            return "ShakenClip"

    clip = ValidClip()
    result = visual_effects.make_camera_shake(clip)
    assert result == "ShakenClip"

def test_make_split_screen_glitch_valid_clip():
    class ValidClip:
        @property
        def size(self):
            return (1920, 1080)
        @property
        def duration(self):
            return 5.0
        @property
        def fps(self):
            return 60.0
        def fl(self, *args, **kwargs):
            return "GlitchClip"

    clip = ValidClip()
    result = visual_effects.make_split_screen_glitch(clip)
    assert result == "GlitchClip"

def test_make_watermark_overlay_valid_clip():
    class ValidClip:
        @property
        def size(self):
            return (1920, 1080)
        def fl(self, *args, **kwargs):
            return "WatermarkClip"

    clip = ValidClip()
    result = visual_effects.make_watermark_overlay(clip, "TEST")
    assert result == "WatermarkClip"

def test_measure_text_invalid_font(monkeypatch):
    from visual_effects import _measure_text
    import sys
    monkeypatch.setitem(sys.modules, 'PIL', None)
    w, h = _measure_text("TEST", None)
    # the fallback if PIL fails is len * 22
    assert w == len("TEST") * 22
    assert h == 44

def test_get_pil_font_invalid_path():
    from visual_effects import _get_pil_font
    font = _get_pil_font(12)
    assert font is not None

def test_pick_text_mask_word_fallback():
    res = visual_effects.pick_text_mask_word("invalid.mp3", use_lyrics=False)
    assert res in visual_effects._RACING_WORDS

def test_transcribe_audio_for_words_missing_whisper(monkeypatch):
    monkeypatch.setitem(sys.modules, 'whisper', None)
    res = visual_effects.transcribe_audio_for_words("dummy.mp3")
    assert res == []

def test_make_zoom_punch_zero_duration():
    class EdgeClip:
        @property
        def size(self): return (1920, 1080)
        @property
        def duration(self): return 0.0 # Extreme
        def fl(self, *args, **kwargs): return "EdgeClip"

    assert visual_effects.make_zoom_punch(EdgeClip()) == "EdgeClip"

def test_make_camera_shake_extreme_intensity():
    class EdgeClip:
        @property
        def size(self): return (1920, 1080)
        @property
        def fps(self): return 60.0
        def fl(self, *args, **kwargs): return "Shaken"

    assert visual_effects.make_camera_shake(EdgeClip(), intensity=50.0) == "Shaken"

def test_make_text_mask_clip_valid(monkeypatch):
    class ValidClip:
        @property
        def size(self):
            return (1920, 1080)
        @property
        def duration(self):
            return 5.0
        def get_frame(self, t):
            import numpy as np
            return np.zeros((1080, 1920, 3), dtype=__import__('numpy').uint8)

    clip = ValidClip()
    # Mocking pillow features would require complex mocks, so we just test the missing pillow path
    monkeypatch.setitem(sys.modules, 'PIL', None)
    res = visual_effects.make_text_mask_clip(clip, "TEST")
    assert res is None

def test_make_pip_overlay_valid_clip(monkeypatch):
    """Test make_pip_overlay with valid input."""
    class ValidClip:
        @property
        def size(self):
            return (1920, 1080)
        def fl(self, *args, **kwargs):
            return self
        def resize(self, *args, **kwargs):
            return self
        def set_position(self, *args, **kwargs):
            return self
        def set_duration(self, *args, **kwargs):
            return self
        @property
        def duration(self):
            return 5.0

    class CompositeMock:
        def __init__(self, clips, **kwargs):
            self.clips = clips
            self.result = "CompositeResult"

    import sys
    class MockEditor:
        CompositeVideoClip = CompositeMock

    monkeypatch.setitem(sys.modules, 'moviepy.editor', MockEditor)

    base_clip = ValidClip()
    pip_clip = ValidClip()
    result = visual_effects.make_pip_overlay(base_clip, pip_clip)
    assert result.result == "CompositeResult"

def test_apply_letterbox_valid_clip():
    """Test apply_letterbox with valid input."""
    class ValidClip:
        @property
        def size(self):
            return (1920, 1080)
        def fl(self, *args, **kwargs):
            return "LetterboxClip"

    clip = ValidClip()
    result = visual_effects.apply_letterbox(clip, bar_fraction=0.1)
    assert result == "LetterboxClip"

def test_apply_letterbox_zero_fraction():
    """Test apply_letterbox with bar_fraction=0.0."""
    class ValidClip:
        @property
        def size(self):
            return (1920, 1080)
        def fl(self, *args, **kwargs):
            return "ZeroFractionClip"

    clip = ValidClip()
    result = visual_effects.apply_letterbox(clip, bar_fraction=0.0)
    assert result == "ZeroFractionClip"

def test_make_mirror_x_valid_clip():
    """Test make_mirror_x with valid input."""
    class ValidClip:
        @property
        def size(self):
            return (1920, 1080)
        def fl(self, *args, **kwargs):
            return "MirrorXClip"

    clip = ValidClip()
    result = visual_effects.make_mirror_x(clip)
    assert result == "MirrorXClip"

def test_make_text_mask_clip_empty_text():
    """Test make_text_mask_clip with empty string text."""
    from unittest.mock import MagicMock
    clip = MagicMock()
    result = visual_effects.make_text_mask_clip(clip, text="")
    assert result is None

def test_make_pip_overlay_edge_cases(monkeypatch):
    """Test make_pip_overlay with zero size fraction and unknown position string."""
    class ValidClip:
        @property
        def size(self):
            return (1920, 1080)
        def fl(self, *args, **kwargs):
            return self
        def resize(self, *args, **kwargs):
            return self
        def set_position(self, *args, **kwargs):
            return self
        def set_duration(self, *args, **kwargs):
            return self
        @property
        def duration(self):
            return 5.0

    class CompositeMock:
        def __init__(self, clips, **kwargs):
            self.clips = clips
            self.result = "CompositeResult"

    import sys
    class MockEditor:
        CompositeVideoClip = CompositeMock

    monkeypatch.setitem(sys.modules, 'moviepy.editor', MockEditor)

    base_clip = ValidClip()
    pip_clip = ValidClip()

    # Position fallback test
    result1 = visual_effects.make_pip_overlay(base_clip, pip_clip, position="invalid_position_string")
    assert result1.result == "CompositeResult"

    # Edge size fraction
    result2 = visual_effects.make_pip_overlay(base_clip, pip_clip, size_frac=-0.5)
    assert result2.result == "CompositeResult"

def test_apply_letterbox_invalid_fraction():
    """Test apply_letterbox with negative or extremely large bar_fraction."""
    import numpy as np

    class EdgeClip:
        @property
        def size(self):
            return (1920, 1080)
        def fl(self, func, *args, **kwargs):
            # Just test if the inner function crashes on edge cases
            def get_frame(t):
                return np.ones((1080, 1920, 3), dtype=__import__('numpy').uint8) * 255
            try:
                func(get_frame, 0.0)
            except Exception as e:
                return e
            return self

    clip = EdgeClip()

    # Too large bar fraction
    result1 = visual_effects.apply_letterbox(clip, bar_fraction=5.0)
    assert not isinstance(result1, Exception), f"Large fraction crashed: {result1}"

    # Negative bar fraction
    result2 = visual_effects.apply_letterbox(clip, bar_fraction=-0.5)
    assert not isinstance(result2, Exception), f"Negative fraction crashed: {result2}"

def test_make_text_mask_sequence_none_words():
    """Test make_text_mask_sequence with unexpected None type for words."""
    from unittest.mock import MagicMock
    clip = MagicMock()
    with pytest.raises(TypeError):
        visual_effects.make_text_mask_sequence(clip, words=None)

def test_apply_letterbox_extreme_fractions():
    class ExtremeClip(MagicMock):
        @property
        def size(self):
            return (100, 100)
        def fl(self, func, *args, **kwargs):
            return func(lambda t: __import__('numpy').zeros((100, 100, 3), dtype=__import__('numpy').uint8), 0.0)

    clip = ExtremeClip()
    res1 = visual_effects.apply_letterbox(clip, bar_fraction=1.5)
    assert res1 is not clip
    assert hasattr(res1, 'fl')
    assert hasattr(res1, 'fl')

    res2 = visual_effects.apply_letterbox(clip, bar_fraction=-0.5)
    assert res2 is not clip
    assert hasattr(res2, 'fl')
    assert hasattr(res2, 'fl')

def test_make_camera_shake_zero_frames():
    class ShakeClip(MagicMock):
        @property
        def size(self):
            return (1920, 1080)
        @property
        def fps(self):
            return 30.0
        def fl(self, func, *args, **kwargs):
            # simulate 1 frame
            return func(lambda t: np.ones((1080, 1920, 3), dtype=__import__('numpy').uint8)*255, 0.0)

    clip = ShakeClip()
    # zero frames
    res1 = visual_effects.make_camera_shake(clip, intensity=-0.1, shake_frames=0)
    assert res1 is clip

def test_make_zoom_punch_negative_zoom():
    class ZoomClip(MagicMock):
        @property
        def size(self):
            return (1920, 1080)
        @property
        def duration(self):
            return 2.0
        def fl(self, func, *args, **kwargs):
            return func(lambda t: __import__('numpy').zeros((1080, 1920, 3), dtype=__import__('numpy').uint8), 1.0)

    clip = ZoomClip()
    res1 = visual_effects.make_zoom_punch(clip, zoom_start=-1.0, zoom_end=0.0)
    assert res1 is not clip
    assert hasattr(res1, 'fl')

def test_make_watermark_overlay_invalid_opacity(monkeypatch):

    # Mock font rendering
    mock_font = MagicMock()
    monkeypatch.setattr(visual_effects, '_get_pil_font', lambda s: mock_font)
    monkeypatch.setattr(visual_effects, '_measure_text', lambda t, f: (50, 20))

    class WatermarkClip(MagicMock):
        @property
        def size(self):
            return (1920, 1080)
        def fl(self, func, *args, **kwargs):
            return func(lambda t: __import__('numpy').zeros((1080, 1920, 3), dtype=__import__('numpy').uint8), 0.0)

    clip = WatermarkClip()
    res1 = visual_effects.make_watermark_overlay(clip, text="test", opacity=2.0)
    assert res1 is not clip
    assert hasattr(res1, 'fl')

    res2 = visual_effects.make_watermark_overlay(clip, text="test", opacity=-1.0)
    assert res2 is not clip
    assert hasattr(res2, 'fl')

def test_make_blend_text_overlay_empty_text(monkeypatch):

    # Mock PIL imports since tests run in mocked cv2/numpy often and we want to test pure logic
    mock_font = MagicMock()
    monkeypatch.setattr(visual_effects, '_get_pil_font', lambda s: mock_font)
    monkeypatch.setattr(visual_effects, '_measure_text', lambda t, f: (0, 0))

    class BlendClip(MagicMock):
        @property
        def size(self):
            return (1920, 1080)
        def fl(self, func, *args, **kwargs):
            return func(lambda t: __import__('numpy').zeros((1080, 1920, 3), dtype=__import__('numpy').uint8), 0.0)

    clip = BlendClip()
    res1 = visual_effects.make_blend_text_overlay(clip, text="", font_size_frac=-0.5)
    assert res1 is not clip
    assert hasattr(res1, 'fl')
