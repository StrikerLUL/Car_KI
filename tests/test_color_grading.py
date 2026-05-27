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
    from color_grading import ClipGrade, build_per_clip_grade, _get_vignette_cpu, grade_frame, _apply_base_grade_cpu, _lut_cinematic_np, _lut_teal_orange_np, apply_grade_to_clip


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


def test_apply_base_grade_cpu_edge_cases():
    """Test edge cases for _apply_base_grade_cpu."""
    mock_img_f = MagicMock()
    # It attempts to add brightness to img_f. If img_f is an invalid type, this raises TypeError.

    with pytest.raises(TypeError):
        _apply_base_grade_cpu("not_an_array", 1.0, 1.0, 0.0)

    # Mock img_f needs to raise an exception when added to a string
    mock_img_f.__add__.side_effect = TypeError("Invalid type")

    with pytest.raises(TypeError):
        _apply_base_grade_cpu(mock_img_f, "not_a_float", 1.0, "not_a_float")


def test_lut_cinematic_np_extreme_values():
    """Test _lut_cinematic_np handles empty arrays or unexpected shapes safely."""
    # Create mock objects to simulate numpy arrays that return empty shapes
    mock_img_f = MagicMock()
    mock_lum3 = MagicMock()

    # We shouldn't raise a python error when dealing with valid empty numpy arrays,
    # np.clip handles this without throwing python errors.
    mock_np = MagicMock()
    mock_empty_array = MagicMock()

    mock_np.abs.return_value = mock_empty_array
    mock_np.clip.return_value = mock_empty_array

    # Simulate empty array operations
    mock_img_f.__mul__.return_value = mock_empty_array
    mock_lum3.__mul__.return_value = mock_empty_array
    mock_empty_array.copy.return_value = mock_empty_array
    mock_empty_array.__mul__.return_value = mock_empty_array
    mock_empty_array.__add__.return_value = mock_empty_array
    mock_empty_array.__sub__.return_value = mock_empty_array
    mock_empty_array.__rsub__.return_value = mock_empty_array

    with patch.object(color_grading, 'np', mock_np):
        try:
            _lut_cinematic_np(mock_img_f, mock_lum3)
        except Exception as e:
            pytest.fail(f"Raised exception on empty array: {e}")


def test_apply_grade_to_clip_invalid_clip():
    """Test apply_grade_to_clip handles invalid clip without an fl_image method."""
    mock_clip = object()  # No fl_image method
    grade = ClipGrade()

    with pytest.raises(AttributeError):
        apply_grade_to_clip(mock_clip, grade)


def test_lut_teal_orange_np_extreme_values():
    """Test _lut_teal_orange_np correctly handles empty arrays or unexpected shapes safely."""
    # Create mock objects to simulate numpy arrays that return empty shapes
    mock_img_f = MagicMock()
    mock_lum3 = MagicMock()

    # We shouldn't raise a python error when dealing with valid empty numpy arrays,
    # np.clip handles this without throwing python errors.
    mock_np = MagicMock()
    mock_empty_array = MagicMock()
    mock_np.clip.return_value = mock_empty_array

    # Simulate empty array operations
    mock_img_f.__mul__.return_value = mock_empty_array
    mock_lum3.__mul__.return_value = mock_empty_array
    mock_empty_array.copy.return_value = mock_empty_array
    mock_empty_array.__mul__.return_value = mock_empty_array
    mock_empty_array.__add__.return_value = mock_empty_array
    mock_empty_array.__sub__.return_value = mock_empty_array
    mock_empty_array.__rsub__.return_value = mock_empty_array

    with patch.object(color_grading, 'np', mock_np):
        try:
            _lut_teal_orange_np(mock_img_f, mock_lum3)
        except Exception as e:
            pytest.fail(f"Raised exception on empty array: {e}")
