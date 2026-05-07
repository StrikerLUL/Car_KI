"""
color_grading.py – Professionelles Cinema-Color-Grading für den Simracing-Editor

GPU-Beschleunigung via PyTorch (bereits mit CUDA 12.4 vorhanden):
  - grade_frame(): alle Matrix-Ops auf GPU (float16 für maximale Geschwindigkeit)
  - _apply_haze_bloom(): GaussianBlur via torch.nn.functional.conv2d auf GPU
  - _apply_vignette(): gecachte Maske, einmalig auf GPU übertragen
  - Graceful Degradation: CPU-Fallback wenn keine GPU

PyTorch wird bevorzugt weil es bereits mit CUDA-DLLs gebündelt ist.
"""

import random
import numpy as np
import cv2
from dataclasses import dataclass
from typing import Optional, Dict, Tuple

# ---------------------------------------------------------------------------
# GPU-Check (PyTorch CUDA)
# ---------------------------------------------------------------------------

try:
    import torch
    _TORCH_OK = torch.cuda.is_available()
    _DEVICE   = torch.device("cuda") if _TORCH_OK else torch.device("cpu")
except ImportError:
    torch    = None
    _TORCH_OK = False
    _DEVICE   = None

if _TORCH_OK:
    print(f"[color_grading] GPU OK: PyTorch CUDA aktiv ({torch.cuda.get_device_name(0)}) - Grading auf GPU")
else:
    print("[color_grading] INFO: Kein PyTorch/CUDA - CPU-Fallback aktiv")


# ---------------------------------------------------------------------------
# Konfiguration pro Clip
# ---------------------------------------------------------------------------

@dataclass
class ClipGrade:
    """Enthält alle Farb-Parameter für einen einzelnen Clip."""
    contrast:   float = 1.20
    saturation: float = 1.18
    brightness: float = 0.0

    lut_preset:   str   = "teal_orange"
    lut_strength: float = 0.75

    highlights_crush: bool  = True
    crush_threshold:  float = 0.82
    crush_strength:   float = 0.55

    haze_bloom:      bool  = True
    bloom_threshold: float = 0.78
    bloom_strength:  float = 0.40

    vignette_strength: float = 0.50
    vignette_radius:   float = 0.75


# ---------------------------------------------------------------------------
# Standard-Presets
# ---------------------------------------------------------------------------

GRADE_PRESETS = {
    "teal_orange": ClipGrade(
        contrast=1.22, saturation=1.20, brightness=0.0,
        lut_preset="teal_orange", lut_strength=0.80,
        highlights_crush=True, crush_threshold=0.82, crush_strength=0.55,
        haze_bloom=True, bloom_threshold=0.78, bloom_strength=0.40,
        vignette_strength=0.50, vignette_radius=0.75,
    ),
    "cinematic": ClipGrade(
        contrast=1.25, saturation=1.10, brightness=-5.0,
        lut_preset="cinematic", lut_strength=0.70,
        highlights_crush=True, crush_threshold=0.85, crush_strength=0.45,
        haze_bloom=True, bloom_threshold=0.80, bloom_strength=0.30,
        vignette_strength=0.60, vignette_radius=0.70,
    ),
    "neutral": ClipGrade(
        contrast=1.10, saturation=1.05, brightness=0.0,
        lut_preset="none", lut_strength=0.0,
        highlights_crush=False, crush_threshold=0.90, crush_strength=0.0,
        haze_bloom=False, bloom_threshold=0.85, bloom_strength=0.0,
        vignette_strength=0.40, vignette_radius=0.80,
    ),
}


# ---------------------------------------------------------------------------
# Gauss-Kernel für GPU-Bloom (einmalig vorberechnet)
# ---------------------------------------------------------------------------

_GAUSS_KERNEL_GPU = None

def _get_gauss_kernel_gpu(device):
    """Erzeugt einen 15×15 Gauss-Kernel als PyTorch-Tensor (gecacht)."""
    global _GAUSS_KERNEL_GPU
    if _GAUSS_KERNEL_GPU is not None:
        return _GAUSS_KERNEL_GPU
    # Gauss-Kernel via OpenCV erzeugen
    k1d = cv2.getGaussianKernel(15, 0)
    k2d = (k1d @ k1d.T).astype(np.float32)
    # Shape für conv2d: (out_channels, in_channels/groups, H, W)
    # Wir machen Depthwise-Conv: groups=3, je 1 Kanal
    kernel = torch.from_numpy(k2d).to(device)
    kernel = kernel.unsqueeze(0).unsqueeze(0).repeat(3, 1, 1, 1)  # (3,1,15,15)
    _GAUSS_KERNEL_GPU = kernel
    return _GAUSS_KERNEL_GPU


# ---------------------------------------------------------------------------
# Vignette-Cache (CPU + GPU)
# ---------------------------------------------------------------------------

_vignette_cache_cpu: Dict[tuple, np.ndarray] = {}
_vignette_cache_gpu: Dict[tuple, object]     = {}


def _make_vignette_mask_np(h: int, w: int, strength: float, radius: float) -> np.ndarray:
    Y, X = np.ogrid[:h, :w]
    dist = np.sqrt(((X - w / 2) / (w / 2))**2 + ((Y - h / 2) / (h / 2))**2)
    t    = np.clip((dist - radius) / (1.0 - radius + 1e-6), 0.0, 1.0)
    vig  = 1.0 - strength * (3 * t**2 - 2 * t**3)
    return vig[..., np.newaxis].astype(np.float32)


def _get_vignette_gpu(h: int, w: int, strength: float, radius: float, device):
    key = (h, w, round(strength, 3), round(radius, 3))
    if key not in _vignette_cache_gpu:
        mask_np = _make_vignette_mask_np(h, w, strength, radius)
        # Shape: (1, 3, H, W) für Broadcasting mit Batch-Tensor
        t = torch.from_numpy(mask_np).permute(2, 0, 1).unsqueeze(0)  # (1,1,H,W)
        t = t.expand(1, 3, h, w).to(device)                           # (1,3,H,W)
        _vignette_cache_gpu[key] = t
    return _vignette_cache_gpu[key]


def _get_vignette_cpu(h: int, w: int, strength: float, radius: float) -> np.ndarray:
    key = (h, w, round(strength, 3), round(radius, 3))
    if key not in _vignette_cache_cpu:
        _vignette_cache_cpu[key] = _make_vignette_mask_np(h, w, strength, radius)
    return _vignette_cache_cpu[key]


# ---------------------------------------------------------------------------
# LUT als reine GPU-Tensor-Operationen (kein CPU-Transfer!)
# ---------------------------------------------------------------------------

def _lut_teal_orange_torch(t: "torch.Tensor", lum: "torch.Tensor") -> "torch.Tensor":
    """
    Teal & Orange LUT vollstaendig auf GPU.
    t:   (1,3,H,W) float32, Werte 0..255, Kanaele: B=0 G=1 R=2
    lum: (1,1,H,W) float32
    """
    norm     = t     * (1.0 / 255.0)          # 0..1
    lum_norm = lum   * (1.0 / 255.0)

    # Shadows -> Teal
    shadow = (1.0 - lum_norm * 2.0).clamp(0, 1)
    out = norm.clone()
    out[:, 0] += shadow[:, 0] * 0.07   # B +7%
    out[:, 1] += shadow[:, 0] * 0.04   # G +4%
    out[:, 2] -= shadow[:, 0] * 0.10   # R -10%

    # Highlights -> Orange
    hi = ((lum_norm - 0.55) * 2.5).clamp(0, 1)
    out[:, 0] -= hi[:, 0] * 0.08       # B -8%
    out[:, 1] += hi[:, 0] * 0.04       # G +4%
    out[:, 2] += hi[:, 0] * 0.12       # R +12%

    return (out * 255.0).clamp(0, 255)


def _lut_cinematic_torch(t: "torch.Tensor", lum: "torch.Tensor") -> "torch.Tensor":
    """
    Cinematic Film-Look vollstaendig auf GPU.
    t:   (1,3,H,W) float32, Werte 0..255
    lum: (1,1,H,W) float32
    """
    norm     = t   * (1.0 / 255.0)
    lum_norm = lum * (1.0 / 255.0)

    out = norm * 0.92 + 0.04           # Lift blacks

    # Cool shadows (B+, R-)
    shadow = (1.0 - lum_norm * 3.0).clamp(0, 1)
    out[:, 0] += shadow[:, 0] * 0.06
    out[:, 2] -= shadow[:, 0] * 0.04

    # Desaturate mids
    mid = (1.0 - (lum_norm - 0.5).abs() * 4.0).clamp(0, 1)
    out = out * (1.0 - mid * 0.15) + lum_norm * (mid * 0.15)

    # Warm highlights
    hi = ((lum_norm - 0.60) * 3.0).clamp(0, 1)
    out[:, 0] -= hi[:, 0] * 0.06
    out[:, 2] += hi[:, 0] * 0.08

    return (out * 255.0).clamp(0, 255)


_LUT_TORCH = {
    "teal_orange": _lut_teal_orange_torch,
    "cinematic":   _lut_cinematic_torch,
}

# CPU-LUT-Fallback bleibt fuer CPU-Pfad
def _lut_teal_orange_np(img_f: np.ndarray, lum3: np.ndarray) -> np.ndarray:
    """Teal & Orange Look. img_f: float32 BGR 0..255."""
    norm     = img_f * (1.0 / 255.0)
    lum_norm = lum3  * (1.0 / 255.0)
    shadow_mask = np.clip(1.0 - lum_norm * 2.0, 0.0, 1.0)
    out = norm.copy()
    out[..., 0] += shadow_mask[..., 0] * 0.07
    out[..., 1] += shadow_mask[..., 0] * 0.04
    out[..., 2] -= shadow_mask[..., 0] * 0.10
    hi_mask = np.clip((lum_norm - 0.55) * 2.5, 0.0, 1.0)
    out[..., 0] -= hi_mask[..., 0] * 0.08
    out[..., 1] += hi_mask[..., 0] * 0.04
    out[..., 2] += hi_mask[..., 0] * 0.12
    return np.clip(out * 255.0, 0, 255)


def _lut_cinematic_np(img_f: np.ndarray, lum3: np.ndarray) -> np.ndarray:
    """Cinematic Film-Look. img_f: float32 BGR 0..255."""
    norm     = img_f * (1.0 / 255.0)
    lum_norm = lum3  * (1.0 / 255.0)
    out = norm.copy() * 0.92 + 0.04
    shadow_mask = np.clip(1.0 - lum_norm * 3.0, 0.0, 1.0)
    out[..., 0] += shadow_mask[..., 0] * 0.06
    out[..., 2] -= shadow_mask[..., 0] * 0.04
    mid_mask = np.clip(1.0 - np.abs(lum_norm - 0.5) * 4.0, 0.0, 1.0)
    out = out * (1.0 - mid_mask * 0.15) + lum_norm * (mid_mask * 0.15)
    hi_mask = np.clip((lum_norm - 0.60) * 3.0, 0.0, 1.0)
    out[..., 0] -= hi_mask[..., 0] * 0.06
    out[..., 2] += hi_mask[..., 0] * 0.08
    return np.clip(out * 255.0, 0, 255)


_LUT_FNS = {
    "teal_orange": _lut_teal_orange_np,
    "cinematic":   _lut_cinematic_np,
}


# ---------------------------------------------------------------------------
# GPU-Grading (PyTorch)
# ---------------------------------------------------------------------------

def _grade_frame_torch(frame: np.ndarray, grade: ClipGrade) -> np.ndarray:
    """
    Vollstaendiges Color-Grading auf der RTX GPU via PyTorch.
    ALLES auf GPU – kein CPU/GPU-Transfer waehrend der Berechnung.

    Pipeline:
      1. uint8 -> float32 Tensor auf GPU  (einmaliger Upload)
      2. Helligkeit + Kontrast + Saettigung
      3. LUT vollstaendig auf GPU (torch tensor math)
      4. Highlights Crush
      5. Haze/Bloom (Depthwise-Gauss via conv2d)
      6. Vignette (gecachte GPU-Maske)
      7. GPU -> uint8 NumPy                (einmaliger Download)
    """
    device = _DEVICE
    h, w   = frame.shape[:2]

    # 1. Einmaliger Upload: (H,W,3) uint8 -> (1,3,H,W) float32 auf GPU
    t = torch.from_numpy(frame).to(device=device, dtype=torch.float32)
    t = t.permute(2, 0, 1).unsqueeze(0)  # (1,3,H,W)

    # 2a. Helligkeit & Kontrast
    t = ((t + grade.brightness) - 128.0) * grade.contrast + 128.0
    t = t.clamp_(0, 255)

    # 2b. Saettigung – Luminanz BGR: 0.114*B + 0.587*G + 0.299*R
    lum = (t[:, 0] * 0.114 + t[:, 1] * 0.587 + t[:, 2] * 0.299).unsqueeze(1)  # (1,1,H,W)
    t   = lum.add(grade.saturation * (t - lum)).clamp_(0, 255)

    # 3. LUT vollstaendig auf GPU (kein Transfer!)
    if grade.lut_preset in _LUT_TORCH and grade.lut_strength > 0.0:
        lut_fn  = _LUT_TORCH[grade.lut_preset]
        lut_out = lut_fn(t, lum)                                     # (1,3,H,W)
        t = torch.lerp(t, lut_out, grade.lut_strength).clamp_(0, 255)
        # Luminanz aktualisieren
        lum = (t[:, 0] * 0.114 + t[:, 1] * 0.587 + t[:, 2] * 0.299).unsqueeze(1)

    # 4. Highlights Crush (GPU)
    if grade.highlights_crush and grade.crush_strength > 0:
        thresh      = grade.crush_threshold * 255.0
        over        = ((lum - thresh) / (255.0 - thresh + 1e-6)).clamp_(0, 1)
        crush_curve = 3 * over * over - 2 * over * over * over      # smooth-step
        crush_f     = crush_curve * grade.crush_strength
        t = torch.lerp(t, torch.full_like(t, 255.0), crush_f).clamp_(0, 255)

    # 5. Haze/Bloom – Depthwise Gauss auf GPU
    if grade.haze_bloom and grade.bloom_strength > 0:
        thresh     = grade.bloom_threshold * 255.0
        hi_mask    = ((lum - thresh) / (255.0 - thresh + 1e-6)).clamp_(0, 1)
        highlights = t * hi_mask
        # Downscale (4x) -> Gauss -> Upscale
        small      = torch.nn.functional.interpolate(
            highlights, scale_factor=0.25, mode="bilinear", align_corners=False)
        kernel     = _get_gauss_kernel_gpu(device)
        blurred_s  = torch.nn.functional.conv2d(small, kernel, padding=7, groups=3)
        blurred    = torch.nn.functional.interpolate(
            blurred_s, size=(h, w), mode="bilinear", align_corners=False)
        t = (t + blurred * grade.bloom_strength).clamp_(0, 255)

    # 6. Vignette (gecachte GPU-Maske – kein Transfer)
    if grade.vignette_strength > 0:
        vig = _get_vignette_gpu(h, w, grade.vignette_strength, grade.vignette_radius, device)
        t   = t.mul_(vig).clamp_(0, 255)

    # 7. Einmaliger Download: GPU -> CPU -> uint8
    return t.squeeze(0).permute(1, 2, 0).to(dtype=torch.uint8).cpu().numpy()


# ---------------------------------------------------------------------------
# CPU-Fallback Grading
# ---------------------------------------------------------------------------

def _apply_base_grade_cpu(img_f: np.ndarray, contrast: float,
                           saturation: float, brightness: float) -> np.ndarray:
    img_f = img_f + brightness
    img_f = (img_f - 128.0) * contrast + 128.0
    gray  = cv2.cvtColor(img_f, cv2.COLOR_BGR2GRAY)[..., np.newaxis]
    img_f = gray + saturation * (img_f - gray)
    return np.clip(img_f, 0, 255)


def _apply_haze_bloom_cpu(img_f: np.ndarray, lum: np.ndarray,
                            threshold: float, strength: float) -> np.ndarray:
    thresh_val = threshold * 255.0
    hi_mask    = np.clip((lum - thresh_val) / (255.0 - thresh_val + 1e-6), 0.0, 1.0)
    highlights = img_f * hi_mask[..., np.newaxis]
    h, w = img_f.shape[:2]
    scale_w, scale_h = max(1, w // 4), max(1, h // 4)
    small   = cv2.resize(highlights, (scale_w, scale_h), interpolation=cv2.INTER_LINEAR)
    blurred = cv2.resize(cv2.GaussianBlur(small, (15, 15), 0),
                         (w, h), interpolation=cv2.INTER_LINEAR)
    return np.clip(img_f + blurred * strength, 0, 255)


def _grade_frame_cpu(frame: np.ndarray, grade: ClipGrade) -> np.ndarray:
    """CPU-Fallback Color-Grading (identisch zur alten Implementierung)."""
    img  = frame.astype(np.float32)
    img  = _apply_base_grade_cpu(img, grade.contrast, grade.saturation, grade.brightness)
    lum  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lum3 = lum[..., np.newaxis]

    if grade.lut_preset in _LUT_FNS and grade.lut_strength > 0.0:
        lut_out = _LUT_FNS[grade.lut_preset](img, lum3)
        img = img * (1.0 - grade.lut_strength) + lut_out * grade.lut_strength

    if grade.highlights_crush and grade.crush_strength > 0:
        thresh      = grade.crush_threshold * 255.0
        over        = np.clip((lum - thresh) / (255.0 - thresh + 1e-6), 0.0, 1.0)
        crush_curve = 3 * over**2 - 2 * over**3
        crush_f     = crush_curve[..., np.newaxis] * grade.crush_strength
        img = img * (1.0 - crush_f) + 255.0 * crush_f

    if grade.haze_bloom and grade.bloom_strength > 0:
        img = _apply_haze_bloom_cpu(img, lum, grade.bloom_threshold, grade.bloom_strength)

    if grade.vignette_strength > 0:
        vig = _get_vignette_cpu(img.shape[0], img.shape[1],
                                grade.vignette_strength, grade.vignette_radius)
        img = img * vig

    return cv2.convertScaleAbs(img)


# ---------------------------------------------------------------------------
# Haupt-Pipeline – wählt automatisch GPU oder CPU
# ---------------------------------------------------------------------------

def grade_frame(frame: np.ndarray, grade: ClipGrade) -> np.ndarray:
    """
    Wendet das gesamte Color-Grading auf ein einzelnes Frame an.
    Nutzt PyTorch GPU wenn verfügbar, sonst CPU-Fallback.
    """
    if _TORCH_OK:
        try:
            return _grade_frame_torch(frame, grade)
        except Exception as e:
            pass  # Stiller Fallback auf CPU
    return _grade_frame_cpu(frame, grade)


# ---------------------------------------------------------------------------
# MoviePy-Filter-Builder
# ---------------------------------------------------------------------------

def make_grade_filter(grade: ClipGrade):
    def _filter(frame: np.ndarray) -> np.ndarray:
        return grade_frame(frame, grade)
    return _filter


def apply_grade_to_clip(clip, grade: ClipGrade):
    """Wendet ClipGrade auf einen MoviePy-Clip an."""
    return clip.fl_image(make_grade_filter(grade))


# ---------------------------------------------------------------------------
# Per-Clip-Varianz
# ---------------------------------------------------------------------------

def build_per_clip_grade(base_preset: str = "teal_orange",
                          clip_tag: str = "action",
                          randomize: bool = True) -> ClipGrade:
    grade = GRADE_PRESETS.get(base_preset, GRADE_PRESETS["teal_orange"])
    g = ClipGrade(
        contrast=grade.contrast, saturation=grade.saturation,
        brightness=grade.brightness, lut_preset=grade.lut_preset,
        lut_strength=grade.lut_strength, highlights_crush=grade.highlights_crush,
        crush_threshold=grade.crush_threshold, crush_strength=grade.crush_strength,
        haze_bloom=grade.haze_bloom, bloom_threshold=grade.bloom_threshold,
        bloom_strength=grade.bloom_strength, vignette_strength=grade.vignette_strength,
        vignette_radius=grade.vignette_radius,
    )

    if clip_tag in ("action", "overtake"):
        g.lut_strength      = min(1.0,  g.lut_strength + 0.10)
        g.crush_strength    = min(0.85, g.crush_strength + 0.12)
        g.bloom_strength    = min(0.65, g.bloom_strength + 0.10)
        g.contrast          = min(1.40, g.contrast + 0.05)
        g.vignette_strength = min(0.70, g.vignette_strength + 0.08)
    elif clip_tag == "calm":
        g.lut_strength   = max(0.30, g.lut_strength - 0.20)
        g.crush_strength = max(0.10, g.crush_strength - 0.20)
        g.bloom_strength = max(0.05, g.bloom_strength - 0.15)
        g.saturation     = max(0.90, g.saturation - 0.10)

    if randomize:
        g.contrast        += random.uniform(-0.03, 0.03)
        g.saturation      += random.uniform(-0.04, 0.04)
        g.brightness      += random.uniform(-3.0,  3.0)
        g.lut_strength    += random.uniform(-0.05, 0.05)
        g.crush_threshold += random.uniform(-0.03, 0.03)
        g.bloom_threshold += random.uniform(-0.03, 0.03)

        g.contrast        = max(0.80, min(1.50, g.contrast))
        g.saturation      = max(0.70, min(1.60, g.saturation))
        g.brightness      = max(-15.0, min(15.0, g.brightness))
        g.lut_strength    = max(0.0,   min(1.0,  g.lut_strength))
        g.crush_threshold = max(0.60,  min(0.95, g.crush_threshold))
        g.bloom_threshold = max(0.55,  min(0.92, g.bloom_threshold))

    return g
