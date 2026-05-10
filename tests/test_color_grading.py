import pytest
from unittest.mock import MagicMock, patch
import sys

# Mock modules for testing color_grading without its dependencies
mock_modules = {
    'numpy': MagicMock(),
    'cv2': MagicMock(),
    'torch': MagicMock(),
}

with patch.dict('sys.modules', mock_modules):
    import color_grading
    from color_grading import ClipGrade, build_per_clip_grade, _get_vignette_cpu, grade_frame


def test_clip_grade_defaults():
    grade = ClipGrade()
    assert grade.contrast == 1.20
    assert grade.saturation == 1.18
    assert grade.brightness == 0.0
    assert grade.lut_preset == "teal_orange"


def test_build_per_clip_grade_unknown_preset():
    grade = build_per_clip_grade(base_preset="unknown_preset", clip_tag="action", randomize=False)
    assert grade.contrast == 1.27
    assert grade.lut_preset == "teal_orange"


def test_build_per_clip_grade_calm_tag():
    grade_default = build_per_clip_grade(base_preset="cinematic", clip_tag="", randomize=False)
    grade_calm = build_per_clip_grade(base_preset="cinematic", clip_tag="calm", randomize=False)

    assert grade_calm.saturation == 1.00
    assert grade_default.saturation == 1.10


def test_build_per_clip_grade_randomize_bounds():
    grades = [build_per_clip_grade(base_preset="cinematic", clip_tag="calm", randomize=True) for _ in range(50)]
    for g in grades:
        assert 0.80 <= g.contrast <= 1.50
        assert 0.70 <= g.saturation <= 1.60
        assert -15.0 <= g.brightness <= 15.0


def test_grade_frame_torch_fallback(monkeypatch):
    frame_mock = MagicMock()
    grade = ClipGrade()

    # We patch the module-level variable
    monkeypatch.setattr(color_grading, '_TORCH_OK', True)

    # Mock _grade_frame_torch to raise exception
    monkeypatch.setattr(color_grading, '_grade_frame_torch', MagicMock(side_effect=Exception("Torch Failed")))

    # Mock _grade_frame_cpu to return success
    mock_cpu = MagicMock(return_value="CPU_SUCCESS")
    monkeypatch.setattr(color_grading, '_grade_frame_cpu', mock_cpu)

    result = grade_frame(frame_mock, grade)

    mock_cpu.assert_called_once_with(frame_mock, grade)
    assert result == "CPU_SUCCESS"


def test_make_grade_filter():
    grade = ClipGrade()
    filter_func = color_grading.make_grade_filter(grade)
    assert callable(filter_func)


def test_get_vignette_cpu_caching(monkeypatch):
    # Clear the cache
    color_grading._vignette_cache_cpu = {}

    mock_make = MagicMock(return_value="mock_vignette")
    monkeypatch.setattr(color_grading, '_make_vignette_mask_np', mock_make)

    vig1 = color_grading._get_vignette_cpu(100, 100, 0.5, 0.5)
    vig2 = color_grading._get_vignette_cpu(100, 100, 0.5, 0.5)

    mock_make.assert_called_once()
    assert vig1 == "mock_vignette"
    assert vig2 == "mock_vignette"


def test_grade_frame_empty_input(monkeypatch):
    """Test that an empty array/input returns a sensible fallback or handles the error gracefully."""
    grade = ClipGrade()
    # If the user passes an empty list or invalid type instead of a numpy array
    invalid_input = []

    # Normally, the system handles it gracefully if torch fails by falling back to CPU.
    # In CPU, cv2.cvtColor might fail if input is empty array or wrong shape.

    # We'll simulate `_grade_frame_cpu` raising an exception (e.g. ValueError due to unpacking or cv2 exception)
    # We want to see how the system behaves. Let's just verify it raises the exception since there's no top-level try/except catching everything in CPU mode.

    monkeypatch.setattr(color_grading, '_TORCH_OK', False)

    # Let's mock _grade_frame_cpu to simulate the cv2 exception for an invalid input format
    mock_cpu = MagicMock(side_effect=ValueError("Invalid input shape"))
    monkeypatch.setattr(color_grading, '_grade_frame_cpu', mock_cpu)

    with pytest.raises(ValueError, match="Invalid input shape"):
        grade_frame(invalid_input, grade)


def test_grade_frame_wrong_file_format(monkeypatch):
    """Test unexpected values in ClipGrade."""
    frame_mock = MagicMock()
    # Set extreme/unexpected values
    grade = ClipGrade(contrast=-1.0, saturation=5.0, lut_preset="non_existent")

    monkeypatch.setattr(color_grading, '_TORCH_OK', False)

    mock_cpu = MagicMock(return_value="SUCCESS_UNEXPECTED")
    monkeypatch.setattr(color_grading, '_grade_frame_cpu', mock_cpu)

    result = grade_frame(frame_mock, grade)
    assert result == "SUCCESS_UNEXPECTED"
    mock_cpu.assert_called_once_with(frame_mock, grade)
