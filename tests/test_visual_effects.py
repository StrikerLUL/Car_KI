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
