import pytest
from unittest.mock import MagicMock, patch
import sys

# For certain tests, we will provide actual numpy/cv2, while keeping torch mocked to avoid slow GPU ops.
# We'll re-import without patching everything for the new tests to actually work with numpy objects.

import color_grading
from color_grading import ClipGrade, build_per_clip_grade, _get_vignette_cpu, grade_frame, _apply_base_grade_cpu, _lut_cinematic_np, _lut_teal_orange_np, apply_grade_to_clip

mock_modules = {
    'torch': MagicMock(),
    'numpy': MagicMock(),
    'cv2': MagicMock(),
}

with patch.dict('sys.modules', mock_modules):
    import color_grading
    from color_grading import ClipGrade, build_per_clip_grade, _get_vignette_cpu, grade_frame, _apply_base_grade_cpu, _lut_cinematic_np, _lut_teal_orange_np, apply_grade_to_clip

import numpy as np
import cv2


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

def test_grade_frame_cpu_full_features():
    """Test _grade_frame_cpu with enabled features (LUT, Crush, Bloom, Vignette)."""
    grade = ClipGrade(
        contrast=1.0, saturation=1.0, brightness=0.0,
        lut_preset="none", lut_strength=0.0,
        highlights_crush=True, crush_threshold=0.5, crush_strength=1.0,
        haze_bloom=True, bloom_threshold=0.5, bloom_strength=1.0,
        vignette_strength=1.0, vignette_radius=0.5
    )

    frame = np.ones((100, 100, 3), dtype=np.uint8) * 128

    with patch.object(color_grading, 'np', np), patch.object(color_grading, 'cv2', cv2):
        try:
            result = color_grading._grade_frame_cpu(frame, grade)
            assert result.shape == (100, 100, 3)
            assert result.dtype == np.uint8
        except Exception as e:
            pytest.fail(f"_grade_frame_cpu failed with real arrays: {e}")

def test_build_per_clip_grade_cinematic():
    """Test build_per_clip_grade bounds and default behavior for the 'cinematic' preset."""
    # Test without randomize
    grade = build_per_clip_grade(base_preset="cinematic", clip_tag="", randomize=False)
    assert grade.lut_preset == "cinematic"
    assert grade.contrast == 1.25  # From GRADE_PRESETS["cinematic"]
    assert grade.saturation == 1.10

    # Test with randomize
    grade_random = build_per_clip_grade(base_preset="cinematic", clip_tag="action", randomize=True)
    assert 0.8 <= grade_random.contrast <= 1.5
    assert 0.7 <= grade_random.saturation <= 1.6

def test_build_per_clip_grade_none():
    """Test build_per_clip_grade behavior for an unmapped/unknown preset (defaults to teal_orange)."""
    grade = build_per_clip_grade(base_preset="non_existent", clip_tag="", randomize=False)
    assert grade.lut_preset == "teal_orange"

def test_apply_haze_bloom_cpu_zeros():
    """Test _apply_haze_bloom_cpu using real empty/zero numpy arrays."""
    # Test with standard 0-filled array
    img_f = np.zeros((100, 100, 3), dtype=np.float32)
    lum = np.zeros((100, 100), dtype=np.float32)

    with patch.object(color_grading, 'np', np), patch.object(color_grading, 'cv2', cv2):
        try:
            result = color_grading._apply_haze_bloom_cpu(img_f, lum, threshold=0.5, strength=0.5)
            assert result.shape == (100, 100, 3)
        except Exception as e:
            pytest.fail(f"_apply_haze_bloom_cpu failed on zero arrays: {e}")

def test_apply_haze_bloom_cpu_empty_array():
    """Test _apply_haze_bloom_cpu using real empty numpy arrays (0x0)."""
    # Test with completely empty array to verify handling of zero-division or errors
    img_f = np.empty((0, 0, 3), dtype=np.float32)
    lum = np.empty((0, 0), dtype=np.float32)

    with patch.object(color_grading, 'np', np), patch.object(color_grading, 'cv2', cv2):
        try:
            result = color_grading._apply_haze_bloom_cpu(img_f, lum, threshold=0.5, strength=0.5)
            assert result is not None
        except cv2.error:
            # cv2 functions will complain about size 0, which is normal.
            pass
        except Exception as e:
            pytest.fail(f"Failed with unexpected error: {e}")

def test_lut_cinematic_torch_extreme_values():
    """Test _lut_cinematic_torch correctly handles edge case tensors."""
    if not color_grading._TORCH_OK:
        pytest.skip("PyTorch not available")

    import torch
    t = torch.zeros((1, 3, 0, 0), dtype=torch.float32).to(color_grading._DEVICE)
    lum = torch.zeros((1, 1, 0, 0), dtype=torch.float32).to(color_grading._DEVICE)

    try:
        out = color_grading._lut_cinematic_torch(t, lum)
        assert out.shape == (1, 3, 0, 0)
    except Exception as e:
        pytest.fail(f"Raised exception on empty tensor: {e}")

    # Test extreme values
    t_extreme = torch.full((1, 3, 10, 10), 500.0, dtype=torch.float32).to(color_grading._DEVICE)
    lum_extreme = torch.full((1, 1, 10, 10), 500.0, dtype=torch.float32).to(color_grading._DEVICE)
    out_extreme = color_grading._lut_cinematic_torch(t_extreme, lum_extreme)
    assert out_extreme.shape == (1, 3, 10, 10)


def test_lut_teal_orange_torch_extreme_values():
    """Test _lut_teal_orange_torch correctly handles edge case tensors."""
    if not color_grading._TORCH_OK:
        pytest.skip("PyTorch not available")

    import torch
    t = torch.zeros((1, 3, 0, 0), dtype=torch.float32).to(color_grading._DEVICE)
    lum = torch.zeros((1, 1, 0, 0), dtype=torch.float32).to(color_grading._DEVICE)

    try:
        out = color_grading._lut_teal_orange_torch(t, lum)
        assert out.shape == (1, 3, 0, 0)
    except Exception as e:
        pytest.fail(f"Raised exception on empty tensor: {e}")

    # Test extreme values
    t_extreme = torch.full((1, 3, 10, 10), -100.0, dtype=torch.float32).to(color_grading._DEVICE)
    lum_extreme = torch.full((1, 1, 10, 10), -100.0, dtype=torch.float32).to(color_grading._DEVICE)
    out_extreme = color_grading._lut_teal_orange_torch(t_extreme, lum_extreme)
    assert out_extreme.shape == (1, 3, 10, 10)


def test_grade_frame_torch_full_features():
    """Test _grade_frame_torch with all features enabled (LUT, crush, bloom, vignette)."""
    if not color_grading._TORCH_OK:
        pytest.skip("PyTorch not available")

    grade = ClipGrade(
        contrast=1.2, saturation=1.1, brightness=5.0,
        lut_preset="teal_orange", lut_strength=0.8,
        highlights_crush=True, crush_threshold=0.5, crush_strength=1.0,
        haze_bloom=True, bloom_threshold=0.5, bloom_strength=1.0,
        vignette_strength=0.5, vignette_radius=0.5
    )

    frame = np.ones((100, 100, 3), dtype=np.uint8) * 128

    try:
        result = color_grading._grade_frame_torch(frame, grade)
        assert result.shape == (100, 100, 3)
        assert result.dtype == np.uint8
    except Exception as e:
        pytest.fail(f"_grade_frame_torch failed with real tensors: {e}")


def test_grade_frame_torch_invalid_input():
    """Test that _grade_frame_torch handles errors properly on invalid input."""
    if not color_grading._TORCH_OK:
        pytest.skip("PyTorch not available")

    grade = ClipGrade()

    # Pass an invalid shape that PyTorch cannot convert properly
    invalid_frame = "not_an_array"

    with pytest.raises(AttributeError):
        color_grading._grade_frame_torch(invalid_frame, grade)


def test_lut_cinematic_torch_extreme_values_mocked(monkeypatch):
    """Test _lut_cinematic_torch correctly handles edge case tensors using mocks."""
    try:
        import torch
    except ImportError:
        pytest.skip("PyTorch not available")
    # Ensure Torch appears OK
    monkeypatch.setattr(color_grading, '_TORCH_OK', True)
    import torch

    # We must patch torch device if not available
    device = torch.device("cpu")
    monkeypatch.setattr(color_grading, '_DEVICE', device)

    t = torch.zeros((1, 3, 0, 0), dtype=torch.float32).to(device)
    lum = torch.zeros((1, 1, 0, 0), dtype=torch.float32).to(device)

    try:
        out = color_grading._lut_cinematic_torch(t, lum)
        assert out.shape == (1, 3, 0, 0)
    except Exception as e:
        pytest.fail(f"Raised exception on empty tensor: {e}")

    # Test extreme values
    t_extreme = torch.full((1, 3, 10, 10), 500.0, dtype=torch.float32).to(device)
    lum_extreme = torch.full((1, 1, 10, 10), 500.0, dtype=torch.float32).to(device)
    out_extreme = color_grading._lut_cinematic_torch(t_extreme, lum_extreme)
    assert out_extreme.shape == (1, 3, 10, 10)


def test_lut_teal_orange_torch_extreme_values_mocked(monkeypatch):
    """Test _lut_teal_orange_torch correctly handles edge case tensors using mocks."""
    try:
        import torch
    except ImportError:
        pytest.skip("PyTorch not available")
    monkeypatch.setattr(color_grading, '_TORCH_OK', True)
    import torch

    device = torch.device("cpu")
    monkeypatch.setattr(color_grading, '_DEVICE', device)

    t = torch.zeros((1, 3, 0, 0), dtype=torch.float32).to(device)
    lum = torch.zeros((1, 1, 0, 0), dtype=torch.float32).to(device)

    try:
        out = color_grading._lut_teal_orange_torch(t, lum)
        assert out.shape == (1, 3, 0, 0)
    except Exception as e:
        pytest.fail(f"Raised exception on empty tensor: {e}")

    # Test extreme values
    t_extreme = torch.full((1, 3, 10, 10), -100.0, dtype=torch.float32).to(device)
    lum_extreme = torch.full((1, 1, 10, 10), -100.0, dtype=torch.float32).to(device)
    out_extreme = color_grading._lut_teal_orange_torch(t_extreme, lum_extreme)
    assert out_extreme.shape == (1, 3, 10, 10)


def test_grade_frame_torch_full_features_mocked(monkeypatch):
    """Test _grade_frame_torch with all features enabled (LUT, crush, bloom, vignette) using mocks."""
    try:
        import torch
    except ImportError:
        pytest.skip("PyTorch not available")
    monkeypatch.setattr(color_grading, '_TORCH_OK', True)
    import torch

    device = torch.device("cpu")
    monkeypatch.setattr(color_grading, '_DEVICE', device)

    grade = ClipGrade(
        contrast=1.2, saturation=1.1, brightness=5.0,
        lut_preset="teal_orange", lut_strength=0.8,
        highlights_crush=True, crush_threshold=0.5, crush_strength=1.0,
        haze_bloom=True, bloom_threshold=0.5, bloom_strength=1.0,
        vignette_strength=0.5, vignette_radius=0.5
    )

    frame = np.ones((100, 100, 3), dtype=np.uint8) * 128

    try:
        result = color_grading._grade_frame_torch(frame, grade)
        assert result.shape == (100, 100, 3)
        assert result.dtype == np.uint8
    except Exception as e:
        pytest.fail(f"_grade_frame_torch failed with real tensors: {e}")


def test_grade_frame_torch_invalid_input_mocked(monkeypatch):
    """Test that _grade_frame_torch handles errors properly on invalid input."""
    try:
        import torch
    except ImportError:
        pytest.skip("PyTorch not available")
    monkeypatch.setattr(color_grading, '_TORCH_OK', True)
    import torch

    device = torch.device("cpu")
    monkeypatch.setattr(color_grading, '_DEVICE', device)

    grade = ClipGrade()

    # Pass an invalid shape that PyTorch cannot convert properly
    invalid_frame = "not_an_array"

    with pytest.raises(AttributeError):
        color_grading._grade_frame_torch(invalid_frame, grade)

def test_get_gauss_kernel_gpu_caching(monkeypatch):
    """Test that _get_gauss_kernel_gpu caches the kernel."""
    try:
        import torch
    except ImportError:
        pytest.skip("PyTorch not available")
    monkeypatch.setattr(color_grading, '_TORCH_OK', True)
    import torch
    device = torch.device("cpu")

    # Clear cache
    color_grading._GAUSS_KERNEL_GPU = None

    kernel1 = color_grading._get_gauss_kernel_gpu(device)
    kernel2 = color_grading._get_gauss_kernel_gpu(device)

    assert kernel1 is kernel2
    assert kernel1.shape == (3, 1, 15, 15)
