"""
video_editor.py – Intelligenter Beat-synchroner TikTok-Editor für Simracing

Nutzt ClipInfo-Objekte aus video_analyzer.py, um Clips inhaltlich zu verstehen
und passend zur Musik-Intensität zusammenzuschneiden.

Beat-Kontext → bevorzugter Clip-Tag
─────────────────────────────────────────────────────────────
  hard beat        →  "action"  >  "overtake"  >  "corner"
  normaler beat    →  "corner"  >  "straight"  >  "overtake"
  zwischen beats   →  "straight" > "calm"      >  "corner"
  Main Drop        →  bester "action"-Clip über alle Quellen
─────────────────────────────────────────────────────────────

Abwechslungs-Rhythmus (Transition Matrix):
  nach "action"   → bevorzuge "calm" / "straight"
  nach "overtake" → bevorzuge "corner" / "straight"
  nach "corner"   → bevorzuge "action" / "overtake"
  nach "straight" → bevorzuge "action" / "corner"
  nach "calm"     → bevorzuge "action" / "overtake"
"""

import os
import time
import json
import hashlib
import logging
import random
import copy
import logging
import numpy as np
import cv2
import logging
from tqdm import tqdm
from itertools import groupby
from moviepy.editor import VideoFileClip, AudioFileClip, concatenate_videoclips
from typing import Callable, List, Dict, Optional, Tuple, Any
from tqdm import tqdm
from video_analyzer import ClipInfo
from audio_analyzer import CutPoint, SongSection
from audio_effects import (
    normalize_audio_gain,
    apply_volume_dip,
    extract_spectrum_frames,
    save_processed_audio,
)
from proglog import ProgressBarLogger

class TqdmProgressBarLogger(ProgressBarLogger):
    def __init__(self, init_state=None, bars=None, ignored_bars=None, logged_bars='all', min_time_interval=0, ignore_bars_under=0):
        super().__init__(init_state, bars, ignored_bars, logged_bars, min_time_interval, ignore_bars_under)
        self.tqdm_bars = {}

    def bars_callback(self, bar, attr, value, old_value=None):
        if bar not in self.tqdm_bars:
            total = self.bars[bar].get('total')
            if total is not None:
                self.tqdm_bars[bar] = tqdm(total=total, desc=bar, leave=False)

        if attr == 'index' and bar in self.tqdm_bars:
            self.tqdm_bars[bar].n = value
            self.tqdm_bars[bar].refresh()
            if value == self.bars[bar].get('total'):
                self.tqdm_bars[bar].close()
                del self.tqdm_bars[bar]

def _write_videofile_with_retry(video_clip, max_retries=3, delay_sec=5.0, **kwargs):
    """
    Führt video_clip.write_videofile() aus. Schlägt der Aufruf fehl (z.B. durch
    einen ffmpeg-Crash, out-of-memory etc.), wird nach kurzer Pause erneut probiert.
    """
    if kwargs.get("logger", "bar") == "bar":
        kwargs["logger"] = TqdmProgressBarLogger()

    last_exception = None
    for attempt in range(1, max_retries + 1):
        try:
            video_clip.write_videofile(**kwargs)
            return True
        except Exception as e:
            last_exception = e
            logging.error(f"FFMPEG-Export Fehler (Versuch {attempt}/{max_retries}): {e}")
            if attempt < max_retries:
                logging.info(f"Warte {delay_sec} Sekunden vor dem nächsten Versuch...")
                time.sleep(delay_sec)

    # Wenn alle Versuche fehlgeschlagen sind
    raise last_exception

from color_grading import (
    ClipGrade,
    apply_grade_to_clip,
    build_per_clip_grade,
    GRADE_PRESETS,
)
from visual_effects import (
    make_text_mask_clip,
    make_pip_overlay,
    make_zoom_punch,
    make_glitch_effect,
    make_camera_shake,
    make_mirror_x,
    apply_letterbox,
    pick_text_mask_word,
    make_blend_text_overlay,
    make_text_mask_sequence,
    make_split_screen_glitch,
    make_bw_overlay,
    extract_music_words,
    get_beat_synced_words,
)


# ---------------------------------------------------------------------------
# (Farbkorrektur jetzt in color_grading.py – per-Clip via ClipGrade)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# White-Flash
# ---------------------------------------------------------------------------

def _make_flash_frame(t: float, clip, flash_dur: float) -> np.ndarray:
    frame = clip.get_frame(t)
    if t < flash_dur:
        alpha = 1.0 - (t / flash_dur)
        white = np.ones_like(frame, dtype=np.float32) * 255
        frame = (1 - alpha) * frame.astype(np.float32) + alpha * white
        return np.clip(frame, 0, 255).astype(np.uint8)
    return frame


# ---------------------------------------------------------------------------
# Schnitttechnik-Helfer
# ---------------------------------------------------------------------------

def _make_freeze_frame(clip, freeze_dur: float = 0.12):
    """
    Hängt am Ende des Clips ein Standbild (letztes Frame) für `freeze_dur` Sekunden an.
    Erzeugt einen dramatischen Moment-Stopp kurz vor dem nächsten Schnitt.
    """
    from moviepy.editor import ImageClip
    last_frame = clip.get_frame(clip.duration - 1 / 60.0)
    freeze = ImageClip(last_frame, duration=freeze_dur)
    freeze = freeze.set_fps(clip.fps or 60)
    from moviepy.editor import concatenate_videoclips as _cat
    return _cat([clip, freeze], method="chain")


def _make_reverse_clip(clip):
    """
    Spielt den Clip rückwärts – klassisches TikTok-Stilmittel.
    """
    return clip.fl_time(lambda t: clip.duration - t, apply_to=["video"], keep_duration=True)


def _make_overlap_transition(clip_a, clip_b, overlap_dur: float = 0.08):
    """
    Überlappungs-Übergang: Das Ende von clip_a und der Anfang von clip_b
    werden für `overlap_dur` Sekunden als Pixel-Crossfade überlagert.
    Gibt (neues_clip_a, neues_clip_b) zurück – clip_a wird um overlap_dur
    verlängert, clip_b bleibt unverändert (die Überblendung ist eingebacken).
    """
    from moviepy.editor import CompositeVideoClip

    if clip_a.duration <= overlap_dur or clip_b.duration <= overlap_dur:
        return clip_a, clip_b   # zu kurz → kein Overlap

    # Tail von clip_a (letzten overlap_dur Sekunden)
    tail_start = clip_a.duration - overlap_dur
    tail_a = clip_a.subclip(tail_start)
    # Head von clip_b
    head_b = clip_b.subclip(0, overlap_dur)

    def _blend_frame(get_frame, t):
        fa = tail_a.get_frame(t)
        fb = head_b.get_frame(t)
        alpha = t / overlap_dur   # 0 → 1 (fade in clip_b)
        blended = (1 - alpha) * fa.astype(np.float32) + alpha * fb.astype(np.float32)
        return np.clip(blended, 0, 255).astype(np.uint8)

    blended_clip = tail_a.fl(_blend_frame, apply_to=["video"], keep_duration=True)

    # clip_a ohne tail + blended tail
    clip_a_trimmed = clip_a.subclip(0, tail_start) if tail_start > 0 else clip_a
    from moviepy.editor import concatenate_videoclips as _cat
    new_a = _cat([clip_a_trimmed, blended_clip], method="chain")
    return new_a, clip_b


def _make_jump_cut_burst(
    video, start_t: float, total_dur: float,
    num_cuts: int = 5, min_micro_dur: float = 0.04
):
    """
    Erzeugt 4–6 sehr schnelle Jump-Cuts hintereinander (Drop-Moment).
    Wählt zufällig leicht versetzte Start-Punkte innerhalb eines kurzen Fensters.
    Gibt eine Liste von Subclips zurück.
    """
    micro_dur = max(min_micro_dur, total_dur / num_cuts)
    window = min(2.5, video.duration * 0.10)   # Quell-Fenster für die Jumps

    micro_clips = []
    for k in range(num_cuts):
        # Zufälliger Offset innerhalb des Fensters
        offset = random.uniform(0, max(0, window - micro_dur))
        s = start_t + offset
        e = min(s + micro_dur, video.duration)
        if e <= s:
            continue
        try:
            micro_clips.append(video.subclip(s, e))
        except Exception:
            pass

    return micro_clips


# ---------------------------------------------------------------------------
# Speed Ramping
# ---------------------------------------------------------------------------

# Geschwindigkeitsprofile: Liste von (normalisierte_Output-Position [0-1], Speed-Faktor)
# Speed < 1 → Zeitlupe, Speed > 1 → Zeitraffer
_SPEED_PROFILES: Dict[str, List[Tuple[float, float]]] = {
    # Corner-Exit: langsam an der Kurven-Spitze, dann explosiver Ramp-Up
    "slowmo_ramp": [
        (0.00, 1.00),   # Einfahrt: Normalgeschwindigkeit
        (0.20, 0.45),   # Abbremsen
        (0.40, 0.30),   # Super Slow-Mo (0.3x) – Kurven-Apex
        (0.65, 1.00),   # Ramp zurück auf Normal
        (0.85, 2.00),   # Kick: 2x Beschleunigung
        (1.00, 2.00),   # Ende: schnell
    ],
    # Action/Überholen: Start verlangsamt, Ramp-Up zum Höhepunkt
    "ramp_up": [
        (0.00, 0.50),   # Start 0.5x
        (0.25, 0.70),
        (0.60, 1.40),
        (1.00, 2.20),   # Ende 2.2x – maximale Energie
    ],
    # Gerade: kurzer Slowmo-Blitz als Überraschungs-Moment
    "hiccup": [
        (0.00, 1.20),   # leicht schnell
        (0.33, 1.20),
        (0.40, 0.40),   # Blitz-Slowmo
        (0.55, 1.20),   # Zurück
        (1.00, 1.20),
    ],
    # Einfahrt in dramatische Szene: von schnell zu sehr langsam
    "ramp_down": [
        (0.00, 2.00),   # Start: schnell
        (0.35, 1.20),
        (0.70, 0.45),   # Slow-Mo am Ende
        (1.00, 0.30),
    ],
}

# Welches Profil + Wahrscheinlichkeit für jeden Clip-Tag
_TAG_RAMP_CONFIG: Dict[str, Tuple[Optional[str], float]] = {
    "corner":   ("slowmo_ramp", 0.42),  # 42 % der Corner-Clips bekommen Speed-Ramp
    "overtake": ("slowmo_ramp", 0.40),  # 40 % Overtakes
    "action":   ("ramp_up",    0.32),   # 32 % Action-Clips
    "straight": ("hiccup",     0.18),   # 18 % Geraden – dezent
    "calm":     (None,         0.00),   # keine Manipulation bei ruhigen Clips
}


def _build_time_map(
    duration: float,
    speed_keyframes: List[Tuple[float, float]],
) -> Callable:
    """
    Erstellt eine Zeitabbildungs-Funktion aus einem Speed-Profil.

    speed_keyframes – Liste von (norm. Output-Position, Speed-Faktor).
    Gibt eine Funktion output_t → input_t zurück, die via fl_time anwendbar ist.

    Die Normalisierung stellt sicher, dass der gesamte Input-Clip verbraucht
    wird und der Output exakt `duration` Sekunden lang bleibt.
    """
    kf_t = np.array([k[0] for k in speed_keyframes], dtype=np.float64)
    kf_s = np.array([k[1] for k in speed_keyframes], dtype=np.float64)

    # Feines Raster für numerische Integration (2000 Punkte → ~0.1 ms Auflösung)
    N = 2000
    t_norm   = np.linspace(0.0, 1.0, N)
    speeds   = np.interp(t_norm, kf_t, kf_s)

    # Kumulative Integration: input_accum[i] = integral(speed, 0..t_norm[i])
    dt           = 1.0 / (N - 1)
    input_accum  = np.cumsum(speeds) * dt

    # Normalisieren: Gesamtverbrauch = duration → Clip wird vollständig abgebildet
    total        = input_accum[-1]
    input_t_arr  = (input_accum / total) * duration   # in Sekunden
    output_t_arr = t_norm * duration

    def _map(t: float) -> float:
        val = np.interp(
            np.clip(t, 0.0, duration - 1e-6),
            output_t_arr,
            input_t_arr,
        )
        return float(min(val, duration - 0.001))

    return _map


def _apply_speed_ramp(
    clip,
    tag: str,
    profile_key: Optional[str] = None,
    random_vary: bool = True,
):
    """
    Wendet ein Speed-Ramp-Profil auf einen Clip an.

    - profile_key=None → wird aus `_TAG_RAMP_CONFIG[tag]` gewählt
    - random_vary=True → ±15 % zufällige Variation der Slowmo-Tiefe
    - Bei zu kurzen Clips (<0.28s) → kein Ramp (keine Zeit für die Kurve)
    - Gibt den zeitlich remappten Clip zurück (gleiche Ausgabe-Dauer!)
    """
    if clip.duration < 0.28:
        return clip

    if profile_key is None:
        profile_key = _TAG_RAMP_CONFIG.get(tag, (None, 0.0))[0]
    if profile_key is None:
        return clip

    keyframes = list(_SPEED_PROFILES[profile_key])

    if random_vary:
        # Leichte zufällige Variation: Slowmo-Punkte variieren ±15 %
        vary = random.uniform(0.85, 1.15)
        keyframes = [
            (t, max(0.15, min(3.0, s * vary if s < 1.0 else s))
             if s != 1.0 else 1.0)
            for t, s in keyframes
        ]

    time_map = _build_time_map(clip.duration, keyframes)
    return clip.fl_time(time_map, apply_to=["video"], keep_duration=True)


# ---------------------------------------------------------------------------
# Audio-Visualizer – Bar-Zeichner
# ---------------------------------------------------------------------------

def _draw_visualizer_bars(frame: np.ndarray,
                           spectrum: np.ndarray,
                           height_frac: float = 0.13) -> np.ndarray:
    """
    Zeichnet TikTok-style Audio-Visualizer-Balken unten ins Frame.
    Gradient: Cyan (Bass) → Magenta (Höhen)
    Vollständig vektorisiert – kein Python-Loop über Balken.
    """
    h, w     = frame.shape[:2]
    num_bars = len(spectrum)
    if num_bars == 0:
        return frame

    max_bar_h    = int(h * height_frac)
    margin_bot   = max(6, int(h * 0.014))
    total_w      = int(w * 0.72)
    bar_and_gap  = total_w // num_bars
    bar_w        = max(2, int(bar_and_gap * 0.65))
    gap_w        = bar_and_gap - bar_w
    actual_total = (bar_w + gap_w) * num_bars
    x_start      = (w - actual_total) // 2
    y_base       = h - margin_bot
    panel_top    = h - max_bar_h - margin_bot - 6

    # ROI
    roi_x1, roi_y1 = max(0, x_start - 10), max(0, panel_top)
    roi_x2, roi_y2 = min(w, x_start + actual_total + 10), h
    roi = frame[roi_y1:roi_y2, roi_x1:roi_x2]

    # 1. Semi-transparentes Hintergrund-Panel (ein einziger cv2.rectangle-Aufruf)
    panel_overlay = roi.copy()
    cv2.rectangle(panel_overlay,
                  (x_start - 10 - roi_x1, panel_top - roi_y1),
                  (x_start + actual_total + 10 - roi_x1, h - roi_y1),
                  (0, 0, 0), -1)
    cv2.addWeighted(panel_overlay, 0.38, roi, 0.62, 0, roi)

    # ── Vektorisiert: Balken-Farben + Positionen vorberechnen ──────────────
    idx_arr  = np.arange(num_bars, dtype=np.float32)
    t_arr    = idx_arr / max(1, num_bars - 1)          # 0..1
    bar_h_arr = np.maximum(3, (spectrum * max_bar_h).astype(np.int32))

    # Gradient BGR: B=255, G=230→0, R=0→255
    r_arr = (t_arr * 255).astype(np.uint8)
    g_arr = ((1.0 - t_arr) * 230).astype(np.uint8)
    b_arr = np.full(num_bars, 255, dtype=np.uint8)

    # Glow-Farben
    glow_r = np.clip(r_arr.astype(np.int32) + 100, 0, 255).astype(np.uint8)
    glow_g = np.clip(g_arr.astype(np.int32) + 100, 0, 255).astype(np.uint8)
    glow_b = np.clip(b_arr.astype(np.int32) + 60,  0, 255).astype(np.uint8)

    # 2. Balken-Overlay auf ROI zeichnen (NumPy-Array-Slices statt Loop)
    bars_overlay = roi.copy()
    roi_h, roi_w = roi.shape[:2]

    for i in range(num_bars):
        bh = int(bar_h_arr[i])
        x1 = (x_start + i * (bar_w + gap_w)) - roi_x1
        x2 = min(x1 + bar_w, roi_w)
        y1 = max(0, (y_base - bh) - roi_y1)
        y2 = min((y_base - roi_y1), roi_h)
        if x1 < 0 or x1 >= roi_w or y1 >= y2:
            continue
        # Direkt als Array-Slice setzen – kein cv2.rectangle-Overhead
        bars_overlay[y1:y2, x1:x2] = (int(r_arr[i]), int(g_arr[i]), int(b_arr[i]))

    cv2.addWeighted(bars_overlay, 0.75, roi, 0.25, 0, roi)

    # 3. Glow-Streifen oben auf jedem Balken (2px, 100% deckend)
    for i in range(num_bars):
        bh = int(bar_h_arr[i])
        if bh <= 5:
            continue
        x1 = (x_start + i * (bar_w + gap_w)) - roi_x1
        x2 = min(x1 + bar_w, roi_w)
        y1 = max(0, (y_base - bh) - roi_y1)
        y2 = min(y1 + 2, roi_h)
        if x1 < 0 or x1 >= roi_w or y1 >= y2:
            continue
        roi[y1:y2, x1:x2] = (int(glow_r[i]), int(glow_g[i]), int(glow_b[i]))

    return frame


def _make_visualizer_filter(spectrum_frames: np.ndarray,
                             fps: float,
                             height_frac: float = 0.13):
    """Gibt einen MoviePy-kompatiblen fl()-Filter zurück der
    pro Frame den Visualizer zeichnet."""
    total_frames = len(spectrum_frames)

    def _filter(get_frame, t):
        frame     = get_frame(t).copy()  # make writable (MoviePy frames are read-only)
        frame_idx = min(int(t * fps), total_frames - 1)
        return _draw_visualizer_bars(frame, spectrum_frames[frame_idx], height_frac)

    return _filter


# ---------------------------------------------------------------------------
# Video vorbereiten (Crop auf 9:16 + HD-Upscale auf 1080×1920)
# ---------------------------------------------------------------------------

# Ziel-Auflösung: TikTok/Reels Standard HD (1080×1920)
_TARGET_W = 1080
_TARGET_H = 1920

_TREND_STYLE_PRESETS: Dict[str, Dict[str, Any]] = {
    "storytime": {
        "grade_preset": "neutral",
        "use_glitch": False,
        "use_camera_shake": False,
        "use_mirror_x": False,
        "use_zoom_punch": False,
        "use_split_screen_glitch": False,
        "use_intro_text_sequence": True,
        "use_blend_text": True,
        "visualizer": False,
    },
    "motivation": {
        "grade_preset": "cinematic",
        "use_glitch": True,
        "use_camera_shake": True,
        "use_mirror_x": False,
        "use_zoom_punch": True,
        "use_white_flash": True,
        "use_split_screen_glitch": True,
        "use_intro_text_sequence": True,
        "visualizer": True,
    },
    "fast_meme_cut": {
        "grade_preset": "teal_orange",
        "use_jump_cut_burst": True,
        "use_speed_ramp": True,
        "use_reverse_clip": True,
        "use_glitch": True,
        "use_camera_shake": True,
        "use_mirror_x": True,
        "use_blend_text": True,
        "use_overlap_transition": False,
        "visualizer": False,
    },
}


def save_edit_template(path: str, template_config: Dict[str, Any]) -> None:
    """Speichert ein Edit-Template als JSON."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(template_config, f, ensure_ascii=True, indent=2)


def load_edit_template(path: str) -> Dict[str, Any]:
    """Lädt ein Edit-Template aus JSON."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {}


def _merge_config(base: Dict[str, Any], overrides: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not overrides:
        return base
    out = copy.deepcopy(base)
    for k, v in overrides.items():
        if k in out:
            out[k] = v
    return out


def _estimate_focus_x(video: VideoFileClip, sample_count: int = 12) -> Optional[float]:
    """
    Schätzt die horizontale Fokusposition anhand von Bewegungsenergie.
    Fallback ist None, wenn keine robuste Schätzung möglich ist.
    """
    try:
        if video.duration <= 0.2:
            return None
        ts = np.linspace(0.0, max(0.0, video.duration - 0.06), sample_count)
        prev_gray = None
        acc_energy = None
        for t in ts:
            frame = video.get_frame(float(t))
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            if prev_gray is not None:
                diff = cv2.absdiff(gray, prev_gray).astype(np.float32)
                col_energy = diff.sum(axis=0)
                acc_energy = col_energy if acc_energy is None else (acc_energy + col_energy)
            prev_gray = gray
        if acc_energy is None or float(np.sum(acc_energy)) <= 1e-6:
            return None
        x_idx = int(np.argmax(acc_energy))
        return float(x_idx)
    except Exception:
        return None


def _detect_hook_moments(highlights_by_video: Dict[str, List[ClipInfo]],
                         beat_times: List[float]) -> List[float]:
    """
    Heuristik für Hook-Momente in den ersten 3 Sekunden.
    Kombiniert frühe Hard/Normal-Beats mit hohen Clip-Scores.
    """
    hook_candidates: List[float] = []
    early_beats = [b for b in beat_times if b <= 3.0]
    hook_candidates.extend(early_beats[:3])
    for clips in highlights_by_video.values():
        top = sorted(clips, key=lambda c: c.score, reverse=True)[:6]
        for c in top:
            if c.timestamp <= 3.0:
                hook_candidates.append(float(c.timestamp))
    hook_candidates = sorted(set(round(x, 2) for x in hook_candidates))
    return hook_candidates[:6]


def _estimate_retention_heatmap(schedule_iter: List[CutPoint],
                                cut_to_clip: Dict[int, ClipInfo]) -> List[Dict[str, Any]]:
    """
    Simuliert Retention-Risiko pro Segment.
    Höhere Scores bedeuten wahrscheinlicheren Absprung.
    """
    heatmap: List[Dict[str, Any]] = []
    for idx, cp in enumerate(schedule_iter):
        info = cut_to_clip.get(idx)
        if info is None:
            continue
        risk = 0.0
        if cp.clip_dur_hint >= 1.1:
            risk += 0.35
        if info.tag == "calm":
            risk += 0.30
        if cp.beat_type == "soft":
            risk += 0.20
        if cp.phase in ("intro", "bridge", "outro"):
            risk += 0.15
        risk = float(np.clip(risk, 0.0, 1.0))
        heatmap.append({
            "time": round(float(cp.time), 2),
            "phase": cp.phase or "unknown",
            "tag": info.tag,
            "risk": round(risk, 2),
        })
    return heatmap


def _prepare_video(video_path: str, focus_mode: str = "center") -> VideoFileClip:
    video = VideoFileClip(video_path)
    w, h = video.size

    # ── Schritt 1: Auf 9:16 croppen (vertikal zentriert) ─────────────────────
    new_w = int(h * 9 / 16)
    if new_w % 2 != 0:
        new_w += 1
    x_center = w / 2
    if focus_mode == "motion":
        focus_x = _estimate_focus_x(video)
        if focus_x is not None:
            left_bound = new_w / 2
            right_bound = max(left_bound, w - new_w / 2)
            x_center = float(np.clip(focus_x, left_bound, right_bound))
    cropped = video.crop(x_center=x_center, y_center=h / 2,
                         width=new_w, height=h).without_audio()

    # ── Schritt 2: Auf 1080×1920 skalieren (falls nötig) ─────────────────────
    cw, ch = cropped.size
    if cw != _TARGET_W or ch != _TARGET_H:
        cropped = cropped.resize((_TARGET_W, _TARGET_H))
        print(f"  [{os.path.basename(video_path)}] "
              f"{w}×{h} → crop {new_w}×{h} → resize {_TARGET_W}×{_TARGET_H} (HD)")
    else:
        print(f"  [{os.path.basename(video_path)}] "
              f"{w}×{h} → crop {new_w}×{h} (9:16 HD, bereits korrekte Größe)")

    return cropped


def _safe_cache_key_blob(payload: Dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, ensure_ascii=True)


def _cache_disabled() -> bool:
    return os.environ.get("KI_AUTO_DISABLE_CACHE", "").strip().lower() in ("1", "true", "yes", "on")


def _audio_fingerprint(audio_path: str) -> Dict[str, Any]:
    st = os.stat(audio_path)
    return {
        "path": os.path.abspath(audio_path),
        "size": int(st.st_size),
        "mtime_ns": int(st.st_mtime_ns),
    }


def _load_json_cache(path: str) -> Optional[Dict[str, Any]]:
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _save_json_cache(path: str, data: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=True, indent=2)


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Audio-Hilfsfunktionen (intern)
# ---------------------------------------------------------------------------

def _compute_gain_factor(y_mono: np.ndarray, target_rms: float = 0.12) -> float:
    """Berechnet den Gain-Faktor für Gain-Staging (begrenzt auf ±12 dB)."""
    current_rms = float(np.sqrt(np.mean(y_mono ** 2)))
    if current_rms < 1e-6:
        return 1.0
    gain = target_rms / current_rms
    return float(np.clip(gain, 0.25, 4.0))


def _apply_volume_dip_array(y: np.ndarray, sr: int,
                             drop_time: float,
                             dip_duration: float = 2.5,
                             dip_depth: float = 0.08) -> np.ndarray:
    """
    Lautstärke-Dip VOR dem Drop. Unterstützt Mono (1D) und Stereo (2D).
    Bei Stereo (shape: channels × samples) wird die Hüllkurve auf beide Kanäle angewandt.
    """
    dip_start_t = max(0.0, drop_time - dip_duration)
    start_s     = int(dip_start_t * sr)
    drop_s      = int(drop_time   * sr)

    # Hüllkurve berechnen
    length = drop_s - start_s
    if length <= 0:
        return y

    envelope = np.ones(length, dtype=np.float32)
    for i in range(length):
        t = i / length
        if t < 0.80:
            t2 = t / 0.80
            envelope[i] = 1.0 - (1.0 - dip_depth) * (3 * t2**2 - 2 * t2**3)
        else:
            t2 = (t - 0.80) / 0.20
            envelope[i] = dip_depth + (1.0 - dip_depth) * (3 * t2**2 - 2 * t2**3)

    y = y.copy()
    n = y.shape[-1] if y.ndim > 1 else len(y)
    end_s = min(drop_s, n)
    seg_len = end_s - start_s

    if y.ndim == 1:
        y[start_s:end_s] *= envelope[:seg_len]
    else:
        y[:, start_s:end_s] *= envelope[:seg_len]

    print(f"  Volume-Dip: {dip_start_t:.2f}s → {drop_time:.2f}s  (Tiefe={dip_depth:.0%})")
    return y


# ---------------------------------------------------------------------------
# Beat-Kontext-Präferenz
# ---------------------------------------------------------------------------

# Welche Tags passen zu welchem Beat-Typ?  (in Prioritäts-Reihenfolge)
_BEAT_PREFERENCE: Dict[str, List[str]] = {
    "hard":    ["action", "overtake", "corner", "straight", "calm"],
    "normal":  ["corner", "straight", "overtake", "action", "calm"],
    "soft":    ["straight", "calm", "corner", "overtake", "action"],
}

# Phase-spezifische Tag-Präferenz (überschreibt Beat-Präferenz für Tag-Reihenfolge)
_PHASE_TAG_PREFERENCE: Dict[str, List[str]] = {
    "intro":   ["calm", "straight", "corner", "overtake", "action"],
    "verse":   ["straight", "corner", "calm", "overtake", "action"],
    "buildup": ["corner", "overtake", "straight", "action", "calm"],
    "drop":    ["action", "overtake", "corner", "straight", "calm"],
    "bridge":  ["calm", "straight", "corner", "overtake", "action"],
    "outro":   ["calm", "straight", "corner", "overtake", "action"],
}

# Clip-Dauer-Multiplikator pro Phase (relativ zur Beat-Interval-basierten Dauer)
_PHASE_DURATION_MULT: Dict[str, float] = {
    "intro":   2.8,   # sehr lange, ruhige Kamerafahrten
    "verse":   1.6,   # mittellang
    "buildup": 0.85,  # etwas schneller, Spannung aufbauen
    "drop":    0.50,  # maximale Schnelligkeit
    "bridge":  2.2,   # dramatische Pause
    "outro":   2.0,   # auslaufend
}

# Effekt-Erlaubnis pro Phase: {phase: {effekt: (erlaubt, Wahrscheinlichkeit)}}
# Wahrscheinlichkeit 0.0 = nie, 1.0 = immer (wenn Feature global an ist)
_PHASE_EFFECT_CONFIG: Dict[str, Dict[str, float]] = {
    #              flash  ramp   reverse freeze overlap  burst
    "intro":   {"flash": 0.00, "ramp": 0.00, "reverse": 0.00, "freeze": 0.00, "overlap": 0.60, "burst": 0.00},
    "verse":   {"flash": 0.05, "ramp": 0.30, "reverse": 0.00, "freeze": 0.10, "overlap": 0.35, "burst": 0.00},
    "buildup": {"flash": 0.15, "ramp": 0.45, "reverse": 0.05, "freeze": 0.15, "overlap": 0.30, "burst": 0.00},
    "drop":    {"flash": 1.00, "ramp": 0.32, "reverse": 0.08, "freeze": 0.12, "overlap": 0.30, "burst": 1.00},
    "bridge":  {"flash": 0.00, "ramp": 0.00, "reverse": 0.00, "freeze": 0.20, "overlap": 0.50, "burst": 0.00},
    "outro":   {"flash": 0.00, "ramp": 0.00, "reverse": 0.00, "freeze": 0.10, "overlap": 0.60, "burst": 0.00},
}

# Abwechslungs-Matrix: nach welchem Tag wird was bevorzugt?
_TRANSITION_PREFERENCE: Dict[str, List[str]] = {
    "action":   ["calm", "straight", "corner", "overtake", "action"],
    "overtake": ["corner", "straight", "calm", "action", "overtake"],
    "corner":   ["action", "overtake", "straight", "calm", "corner"],
    "straight": ["action", "corner", "overtake", "calm", "straight"],
    "calm":     ["action", "overtake", "corner", "straight", "calm"],
}


# ---------------------------------------------------------------------------
# IntelligentPool – kontextbewusster Clip-Selektor (Anti-Repetition v2)
# ---------------------------------------------------------------------------

class IntelligentPool:
    """
    Verwaltet eine Liste von ClipInfo-Objekten und wählt beim `pick()`-Aufruf
    denjenigen aus, der am besten zum Beat-Typ und zum vorherigen Clip passt.

    Anti-Wiederholungs-Strategie:
      - Globales `_used_timestamps`-Set: dauerhaft gespeichert, kein Reset beim Refill
      - Stufenweiser Cooldown: erst kurzer Cooldown-Check, dann weicher Fallback
      - `preview()`: liest den besten Clip OHNE ihn zu konsumieren
      - Beim Refill: bereits benutzte Timestamps bekommen hohe Penalty statt Ausschluss
    """

    def __init__(self, clips: List[ClipInfo], video_duration: float,
                 jitter: float = 0.25, cooldown: float = 15.0):
        self._base: List[ClipInfo] = list(clips)
        self._duration = video_duration
        self._jitter = jitter
        self._cooldown = cooldown          # Mindeststunden zwischen gleichem Clip
        self._pool: List[ClipInfo] = []
        self._used_timestamps: List[float] = []   # wächst global, kein Reset!
        self._use_count: Dict[float, int] = {}    # wie oft jeder Timestamp benutzt
        self._refill()

    def _refill(self):
        """
        Pool neu befüllen. Clips die bereits oft benutzt wurden
        kommen hinten in den Pool (hohe Penalty), werden aber nicht
        komplett ausgeschlossen (Fallback-Sicherheit).
        """
        # Alle Basis-Clips nach Benutzungs-Häufigkeit sortieren
        candidates = sorted(
            self._base,
            key=lambda c: self._use_count.get(round(c.timestamp, 2), 0)
        )
        # Leicht zufällig mischen innerhalb gleicher Nutzungsstufen
        shuffled: List[ClipInfo] = []
        for _, group in groupby(candidates,
                                 key=lambda c: self._use_count.get(round(c.timestamp, 2), 0)):
            g = list(group)
            random.shuffle(g)
            shuffled.extend(g)
        self._pool = shuffled

    def _clip_priority(self, c: ClipInfo, beat_pref: List[str],
                       trans_pref: List[str], strict_cooldown: bool) -> float:
        """Berechnet den Prioritäts-Score eines Clips (niedrig = besser)."""
        beat_rank  = beat_pref.index(c.tag)  if c.tag in beat_pref  else len(beat_pref)
        trans_rank = trans_pref.index(c.tag) if c.tag in trans_pref else len(trans_pref)

        # Wie oft wurde dieser Timestamp bereits benutzt?
        use_cnt = self._use_count.get(round(c.timestamp, 2), 0)
        use_penalty = use_cnt * 8.0   # jede Benutzung kostet 8 Prioritätspunkte

        # Cooldown-Check
        if strict_cooldown:
            in_cooldown = any(abs(c.timestamp - u) < self._cooldown
                              for u in self._used_timestamps[-20:])
            cooldown_penalty = 12.0 if in_cooldown else 0.0
        else:
            # Weicher Fallback: nur direkte Wiederholung (5s) penalisieren
            in_cooldown = any(abs(c.timestamp - u) < 5.0
                              for u in self._used_timestamps[-5:])
            cooldown_penalty = 6.0 if in_cooldown else 0.0

        return (beat_rank * 0.5 + trans_rank * 0.5
                - c.score * 2.0 + use_penalty + cooldown_penalty)

    def preview(self,
                beat_type: str = "normal",
                last_tag: Optional[str] = None,
                min_clip_duration: float = 0.5) -> ClipInfo:
        """
        Gibt den besten Clip zurück, OHNE ihn aus dem Pool zu entfernen.
        Wird von _pick_best_source verwendet, um Quellen vergleichen zu können.
        """
        if not self._pool:
            self._refill()

        beat_pref  = _BEAT_PREFERENCE.get(beat_type, _BEAT_PREFERENCE["normal"])
        trans_pref = _TRANSITION_PREFERENCE.get(last_tag or "calm",
                                                _TRANSITION_PREFERENCE["calm"])

        available = [c for c in self._pool
                     if c.timestamp + min_clip_duration <= self._duration]
        if not available:
            available = self._pool

        # Erst strikt filtern, dann weich
        best = min(available,
                   key=lambda c: self._clip_priority(c, beat_pref, trans_pref,
                                                     strict_cooldown=True))
        return best

    def pick(self,
             beat_type: str = "normal",
             last_tag: Optional[str] = None,
             min_clip_duration: float = 0.5) -> ClipInfo:
        """
        Wählt und konsumiert den besten verfügbaren Clip.
        Versucht zuerst mit strengem Cooldown, dann mit weichem Fallback.
        """
        if not self._pool:
            self._refill()

        beat_pref  = _BEAT_PREFERENCE.get(beat_type, _BEAT_PREFERENCE["normal"])
        trans_pref = _TRANSITION_PREFERENCE.get(last_tag or "calm",
                                                _TRANSITION_PREFERENCE["calm"])

        available = [c for c in self._pool
                     if c.timestamp + min_clip_duration <= self._duration]
        if not available:
            available = list(self._pool)

        # Besten Clip mit strenger Cooldown-Prüfung finden
        chosen = min(available,
                     key=lambda c: self._clip_priority(c, beat_pref, trans_pref,
                                                       strict_cooldown=True))

        self._pool.remove(chosen)

        # Jitter anwenden (kleiner als vorher → weniger Zufälligkeit)
        lo = max(0.0, chosen.timestamp - self._jitter)
        hi = min(self._duration - min_clip_duration, chosen.timestamp + self._jitter)
        actual_t = random.uniform(lo, hi) if hi > lo else chosen.timestamp

        # Benutzungs-Tracking
        key = round(chosen.timestamp, 2)
        self._use_count[key] = self._use_count.get(key, 0) + 1
        self._used_timestamps.append(actual_t)

        # Ergebnis-Kopie
        result = ClipInfo(
            timestamp=actual_t,
            score=chosen.score,
            motion_score=chosen.motion_score,
            drift_score=chosen.drift_score,
            audio_score=chosen.audio_score,
            telemetry_score=chosen.telemetry_score,
            vehicle_count=chosen.vehicle_count,
            cam_type=chosen.cam_type,
            tag=chosen.tag,
            source=chosen.source,
        )

        # Pool refüllen wenn leer
        if not self._pool:
            self._refill()

        return result


# ---------------------------------------------------------------------------
# Beat-Typ bestimmen
# ---------------------------------------------------------------------------

def _get_beat_type(beat_idx: int, beat_time: float,
                   hard_beat_set: set,
                   is_main_drop: bool) -> str:
    if is_main_drop:
        return "hard"
    if round(beat_time, 3) in hard_beat_set:
        return "hard"
    # Jeden 2. normalen Beat als "soft" einstufen
    if beat_idx % 2 == 0:
        return "normal"
    return "soft"


def _get_section_for_time(t: float, sections: List[SongSection]) -> Optional[SongSection]:
    """
    Gibt die SongSection zurück, in die der Zeitpunkt `t` fällt.
    Gibt None zurück wenn keine Sektionen definiert sind.
    """
    if not sections:
        return None
    for sec in sections:
        if sec.start <= t < sec.end:
            return sec
    # Fallback: letzte Sektion
    return sections[-1]


def _phase_effect_prob(phase: Optional[str], effect: str, global_flag: bool) -> float:
    """
    Gibt die effektive Wahrscheinlichkeit für einen Effekt zurück,
    unter Berücksichtigung von globalem Flag und Phasen-Konfiguration.
    Wenn keine Phase (None) → Standardverhalten (globaler Flag entscheidet).
    """
    if not global_flag:
        return 0.0
    if phase is None:
        # Kein Phasen-System → altes Standardverhalten
        defaults = {"flash": 1.0, "ramp": 1.0, "reverse": 0.08,
                    "freeze": 0.12, "overlap": 0.30, "burst": 1.0}
        return defaults.get(effect, 0.0)
    cfg = _PHASE_EFFECT_CONFIG.get(phase, _PHASE_EFFECT_CONFIG["verse"])
    return cfg.get(effect, 0.0)


def _get_beat_preference_for_phase(beat_type: str, phase: Optional[str]) -> List[str]:
    """
    Gibt die Tag-Präferenzliste zurück – bei bekannter Phase wird
    die Phasen-Präferenz mit der Beat-Präferenz gemischt.
    """
    beat_pref = _BEAT_PREFERENCE.get(beat_type, _BEAT_PREFERENCE["normal"])
    if phase is None:
        return beat_pref
    phase_pref = _PHASE_TAG_PREFERENCE.get(phase, beat_pref)
    # Mischen: harte Beats behalten Beat-Präferenz, weiche Beats nutzen Phase
    if beat_type == "hard":
        return beat_pref   # beim Drop immer Action bevorzugen
    # Für normale/soft Beats: Phase hat 70% Gewicht
    # Einfache Implementierung: Phase-Reihenfolge nehmen, aber hard-beat-Tags vorne lassen
    return phase_pref


# ---------------------------------------------------------------------------
# Multi-Cam Quellwahl
# ---------------------------------------------------------------------------

def _pick_best_source(video_paths: List[str],
                      pools: Dict[str, IntelligentPool],
                      beat_type: str,
                      last_tag: Optional[str],
                      last_source: Optional[str],
                      consecutive_same_source: int,
                      min_clip_dur: float) -> Tuple[str, ClipInfo]:
    """
    Wählt die Kamera-Quelle mit dem besten Clip für den aktuellen Beat-Kontext.

    Wichtig: Nutzt pool.preview() statt pool.pick() um Kandidaten zu vergleichen.
    Nur der GEWÄHLTE Pool konsumiert den Clip via pool.pick().
    Nach 2× gleicher Quelle → Kamerawechsel-Druck erhöhen.
    Nach 3× gleicher Quelle → zwingend wechseln.
    """
    beat_pref  = _BEAT_PREFERENCE.get(beat_type, _BEAT_PREFERENCE["normal"])
    trans_pref = _TRANSITION_PREFERENCE.get(last_tag or "calm",
                                            _TRANSITION_PREFERENCE["calm"])

    # Kandidaten aus allen Quellen ermitteln – NUR LESEN, nicht konsumieren!
    candidates: List[Tuple[float, str, ClipInfo]] = []

    for vp in video_paths:
        pool = pools[vp]
        # preview() liest den besten Clip OHNE ihn zu verbrauchen
        preview = pool.preview(beat_type=beat_type, last_tag=last_tag,
                               min_clip_duration=min_clip_dur)

        beat_rank  = beat_pref.index(preview.tag)  if preview.tag in beat_pref  else len(beat_pref)
        trans_rank = trans_pref.index(preview.tag) if preview.tag in trans_pref else len(trans_pref)

        # Kamerawechsel-Druck: steigt mit jedem aufeinanderfolgenden Clip
        if vp == last_source:
            same_source_penalty = (
                20.0 if consecutive_same_source >= 3  # zwingend wechseln
                else 4.0 * consecutive_same_source    # wachsender Druck
            )
        else:
            same_source_penalty = 0.0

        priority = (beat_rank * 0.5 + trans_rank * 0.5
                    - preview.score * 2.0 + same_source_penalty)

        candidates.append((priority, vp, preview))

    # Besten Kandidaten wählen
    candidates.sort(key=lambda x: x[0])
    chosen_priority, chosen_vp, _ = candidates[0]

    # Jetzt erst wirklich konsumieren!
    chosen_clip = pools[chosen_vp].pick(
        beat_type=beat_type, last_tag=last_tag, min_clip_duration=min_clip_dur
    )
    return chosen_vp, chosen_clip


# ---------------------------------------------------------------------------
# Haupt-Funktion: TikTok-Edit erstellen
# ---------------------------------------------------------------------------

def create_tiktok_edit(
        video_paths,                        # str ODER Liste von str
        audio_path: str,
        beat_times: list,
        hard_beat_times: list,
        main_drop_time,
        highlight_times,                    # dict {path: [ClipInfo]} ODER Liste
        output_path: str,
        # ── Color Grading ───────────────────────────────────────────────────
        grade_preset: str = "teal_orange",  # "teal_orange" | "cinematic" | "neutral"
        grade_randomize: bool = True,       # Leichte Variation pro Clip
        vignette_strength: float = 0.50,
        vignette_radius: float = 0.75,
        # ── Legacy-Parameter (für Rückwärtskompatibilität, optional) ────────
        color_contrast: float = 1.20,
        color_saturation: float = 1.18,
        color_brightness: float = 0.0,
        # ── Audio-reaktive Effekte ──────────────────────────────────────────
        gain_staging: bool = True,
        volume_dip: bool = True,
        visualizer: bool = True,
        visualizer_bars: int = 24,
        visualizer_height: float = 0.13,
        # ── Schnitt-Techniken (optional ein-/ausschalten) ────────────────────
        use_jump_cut_burst: bool = True,    # Jump-Cut-Burst beim Main Drop
        use_speed_ramp: bool = True,        # Speed Ramping (Slowmo/Zeitraffer)
        use_reverse_clip: bool = True,      # Reverse Clip (rückwärts)
        use_white_flash: bool = True,       # White-Flash auf harten Beats
        use_freeze_frame: bool = True,      # Freeze Frame vor dem Schnitt
        use_overlap_transition: bool = True, # Overlap-Überblendung zwischen Clips
        # ── Neue visuelle Effekte ────────────────────────────────────────────
        use_text_mask: bool = True,         # Text-Mask-Clip vor dem Drop
        use_pip: bool = True,               # Picture-in-Picture (Multi-Cam)
        use_zoom_punch: bool = True,        # Zoom-Punch auf harten Beats
        use_glitch: bool = True,            # Glitch-Frame auf harten Beats
        use_camera_shake: bool = True,      # Camera Shake auf harten Beats
        use_mirror_x: bool = True,          # Video spiegeln
        use_letterbox: bool = True,         # Cinematic Letterbox
        text_mask_word: Optional[str] = None,  # None = aus Musik-Metadaten
        text_mask_use_lyrics: bool = False, # Ob auch Lyrics für die Wortermittlung genutzt werden sollen
        lyrics_strict_mode: bool = True,    # True = nur exakte Wörter, False = lockerer/Fallback
        # ── TikTok-Style Upgrade (neue Effekte) ─────────────────────────────
        use_blend_text: bool = True,        # Screen-Blend-Text nach Glitch (@editdd032-Stil)
        use_intro_text_sequence: bool = True, # Schnelle Wort-Sequenz als Intro (@azmiedtz03-Stil)
        use_split_screen_glitch: bool = True, # Vertikale Streifen am Ende
        use_bw_intro: bool = False,         # Schwarz-Weiß in Intro-Phase
        watermark_text: Optional[str] = None, # Wasserzeichen Text
        watermark_opacity: float = 0.4,       # Wasserzeichen Deckkraft
        # ── UX/Produktivität ────────────────────────────────────────────────
        editing_mode: str = "pro",          # "quick" | "pro"
        trend_preset: Optional[str] = None, # storytime | motivation | fast_meme_cut
        template_overrides: Optional[Dict[str, Any]] = None,  # via load_edit_template()
        auto_reframe: bool = True,          # One-Click Reframe (9:16 Fokus)
        reframe_focus_mode: str = "motion", # motion | center
        generate_retention_report: bool = True,
        # ── Musik-Struktur (Song-Phasen) ────────────────────────────────────
        sections=None,                      # List[SongSection] aus detect_song_sections()
        cut_schedule=None,                  # List[CutPoint] aus build_cut_schedule() – empfohlen!
        preview: bool = False,              # Schnell-Export (540p, 30fps)
) -> Dict[str, Any]:
    """
    Erstellt einen beat-synchronen TikTok-Edit und gibt eine Statistik-Dict zurück.
    """
    print("\n" + "═"*60)
    print("  Intelligente Videoproduktion startet...")
    print("═"*60)

    # ── Eingaben normalisieren ───────────────────────────────────────────────
    if isinstance(video_paths, str):
        video_paths = [video_paths]
    is_multi = len(video_paths) > 1

    # ── Smart Defaults: Quick/Pro + Presets + Templates ─────────────────────
    runtime_cfg: Dict[str, Any] = {
        "grade_preset": grade_preset,
        "grade_randomize": grade_randomize,
        "visualizer": visualizer,
        "use_jump_cut_burst": use_jump_cut_burst,
        "use_speed_ramp": use_speed_ramp,
        "use_reverse_clip": use_reverse_clip,
        "use_white_flash": use_white_flash,
        "use_freeze_frame": use_freeze_frame,
        "use_overlap_transition": use_overlap_transition,
        "use_text_mask": use_text_mask,
        "use_pip": use_pip,
        "use_zoom_punch": use_zoom_punch,
        "use_glitch": use_glitch,
        "use_camera_shake": use_camera_shake,
        "use_mirror_x": use_mirror_x,
        "use_letterbox": use_letterbox,
        "use_blend_text": use_blend_text,
        "use_intro_text_sequence": use_intro_text_sequence,
        "use_split_screen_glitch": use_split_screen_glitch,
        "use_bw_intro": use_bw_intro,
        "watermark_text": watermark_text,
        "watermark_opacity": watermark_opacity,
        "visualizer_height": visualizer_height,
    }
    if editing_mode == "quick":
        runtime_cfg = _merge_config(runtime_cfg, {
            "grade_randomize": False,
            "visualizer": False,
            "use_reverse_clip": False,
            "use_overlap_transition": False,
            "use_split_screen_glitch": False,
            "use_bw_intro": False,
            "use_pip": is_multi,
            "use_jump_cut_burst": True,
            "use_speed_ramp": True,
            "use_glitch": True,
            "use_camera_shake": True,
            "use_mirror_x": False,
            "use_white_flash": True,
        })
    if trend_preset:
        runtime_cfg = _merge_config(runtime_cfg, _TREND_STYLE_PRESETS.get(trend_preset, {}))
    runtime_cfg = _merge_config(runtime_cfg, template_overrides)

    grade_preset = runtime_cfg["grade_preset"]
    grade_randomize = runtime_cfg["grade_randomize"]
    visualizer = runtime_cfg["visualizer"]
    use_jump_cut_burst = runtime_cfg["use_jump_cut_burst"]
    use_speed_ramp = runtime_cfg["use_speed_ramp"]
    use_reverse_clip = runtime_cfg["use_reverse_clip"]
    use_white_flash = runtime_cfg["use_white_flash"]
    use_freeze_frame = runtime_cfg["use_freeze_frame"]
    use_overlap_transition = runtime_cfg["use_overlap_transition"]
    use_text_mask = runtime_cfg["use_text_mask"]
    use_pip = runtime_cfg["use_pip"]
    use_zoom_punch = runtime_cfg["use_zoom_punch"]
    use_glitch = runtime_cfg["use_glitch"]
    use_camera_shake = runtime_cfg["use_camera_shake"]
    use_mirror_x = runtime_cfg["use_mirror_x"]
    use_letterbox = runtime_cfg["use_letterbox"]
    use_blend_text = runtime_cfg["use_blend_text"]
    use_intro_text_sequence = runtime_cfg["use_intro_text_sequence"]
    use_split_screen_glitch = runtime_cfg["use_split_screen_glitch"]
    use_bw_intro = runtime_cfg["use_bw_intro"]
    visualizer_height = runtime_cfg["visualizer_height"]

    # highlight_times kann sein:
    #   a) dict  {video_path: [ClipInfo, ...]}    ← neues Format
    #   b) dict  {video_path: [float, ...]}        ← altes Format (Rückwärtskompatibel)
    #   c) list  [float, ...]                      ← Single-Video, alt
    if isinstance(highlight_times, dict):
        highlights_by_video = {}
        for vp, items in highlight_times.items():
            if items and isinstance(items[0], ClipInfo):
                highlights_by_video[vp] = items
            else:
                # Altes Format: floats → ClipInfo konvertieren
                highlights_by_video[vp] = [
                    ClipInfo(timestamp=float(t), score=0.5, motion_score=0.5, drift_score=0.0,
                             audio_score=0.5, telemetry_score=0.0, vehicle_count=0.0, 
                             cam_type="external", tag="action", source=vp)
                    for t in items
                ]
    else:
        vp0 = video_paths[0]
        highlights_by_video = {
            vp0: [ClipInfo(timestamp=float(t), score=0.5, motion_score=0.5, drift_score=0.0,
                           audio_score=0.5, telemetry_score=0.0, vehicle_count=0.0, 
                           cam_type="external", tag="action", source=vp0)
                  for t in highlight_times]
        }
    hook_moments = _detect_hook_moments(highlights_by_video, beat_times)
    if hook_moments:
        print(f"Hook-Momente (0-3s): {hook_moments}")

    # ── Videos laden & croppen ──────────────────────────────────────────────
    print()
    _focus_mode = reframe_focus_mode if auto_reframe else "center"
    videos = {}
    for vp in tqdm(video_paths, desc="Lade & crop Videos", unit="Video"):
        videos[vp] = _prepare_video(vp, focus_mode=_focus_mode)

    # ── Audio-Effekte: Gain-Staging + Volume-Dip ─────────────────────────────
    import librosa
    processed_audio_path = audio_path
    if gain_staging or volume_dip:
        print("\nVerarbeite Audio (Gain-Staging / Volume-Dip)...")
        y_raw, sr_raw = librosa.load(audio_path, sr=44100, mono=False)
        # Stereo → 2D array (channels, samples) oder 1D (mono)
        if y_raw.ndim == 1:
            y_proc = y_raw
        else:
            y_proc = y_raw  # soundfile / librosa behält Stereo

        # Für Gain-Staging brauchen wir Mono-RMS
        y_mono = y_raw if y_raw.ndim == 1 else np.mean(y_raw, axis=0)
        if gain_staging:
            gain_factor = _compute_gain_factor(y_mono, target_rms=0.12)
            y_proc = np.clip(y_proc * gain_factor, -1.0, 1.0)
            print(f"  Gain angewendet: {20*np.log10(gain_factor):+.1f} dB")

        if volume_dip and main_drop_time is not None and main_drop_time > 0:
            y_proc = _apply_volume_dip_array(y_proc, sr_raw, main_drop_time)

        processed_audio_path = "_processed_audio.wav"
        import soundfile as sf
        if y_proc.ndim == 1:
            sf.write(processed_audio_path, y_proc, sr_raw, subtype="PCM_16")
        else:
            sf.write(processed_audio_path, y_proc.T, sr_raw, subtype="PCM_16")
        print(f"  Audio gespeichert → {processed_audio_path}")

    # ── Text-Mask-Wort aus Musik-Metadaten ───────────────────────────────────
    if use_text_mask and text_mask_word is None:
        text_mask_word = pick_text_mask_word(audio_path, use_lyrics=text_mask_use_lyrics)
        print(f"  Text-Mask-Wort (aus Metadaten{' + Lyrics' if text_mask_use_lyrics else ''}): '{text_mask_word}'")

    # ── Beat-synchrone Lyrics (Whisper word-timestamps) ──────────────────────
    # Einmalig: für JEDEN Beat das passende Lyric-Wort ermitteln.
    # Wird für Blend-Text, Intro-Sequenz und Text-Mask genutzt.
    print("\nErmittle beat-synchrone Lyrics (Whisper)...")
    _lyrics_max_dist = 0.10 if lyrics_strict_mode else 0.22
    print(f"  Lyrics-Modus: {'STRICT' if lyrics_strict_mode else 'LOOSE'} (max_dist={_lyrics_max_dist:.2f}s)")

    _cache_root = os.path.join(os.getcwd(), ".cache")
    os.makedirs(_cache_root, exist_ok=True)
    _lyrics_cache_key = {
        "version": "lyrics_sync_v1",
        "audio": _audio_fingerprint(audio_path),
        "beats": [round(float(b), 3) for b in beat_times],
        "lyrics_strict_mode": bool(lyrics_strict_mode),
        "text_mask_use_lyrics": bool(text_mask_use_lyrics),
        "max_dist": float(_lyrics_max_dist),
    }
    _lyrics_cache_hash = hashlib.sha1(
        _safe_cache_key_blob(_lyrics_cache_key).encode("utf-8")
    ).hexdigest()
    _lyrics_cache_path = os.path.join(_cache_root, f"lyrics_sync_{_lyrics_cache_hash}.json")
    _lyrics_cached = None if _cache_disabled() else _load_json_cache(_lyrics_cache_path)

    if _lyrics_cached:
        print("  [CACHE] Beat-synchrone Lyrics aus Cache geladen.")
        _fallback_word_pool = [str(w) for w in _lyrics_cached.get("fallback_word_pool", [])]
        _synced_words_list = [str(w) for w in _lyrics_cached.get("synced_words_list", [])]
    else:
        _fallback_word_pool = extract_music_words(audio_path, use_lyrics=text_mask_use_lyrics)
        _synced_words_list = get_beat_synced_words(
            audio_path,
            beat_times=beat_times,
            fallback_words=_fallback_word_pool if _fallback_word_pool else None,
            max_dist=_lyrics_max_dist,
            strict_mode=lyrics_strict_mode,
        )
        if not _cache_disabled():
            _save_json_cache(
                _lyrics_cache_path,
                {
                    "fallback_word_pool": _fallback_word_pool or [],
                    "synced_words_list": _synced_words_list or [],
                },
            )
    # Lookup: beat_index → Wort (robust gegen fehlende Einträge)
    _beat_word_map: Dict[int, str] = {
        i: w for i, w in enumerate(_synced_words_list)
    }
    _fallback_pool = _fallback_word_pool if _fallback_word_pool else list([
        "DRIFT", "APEX", "SPEED", "PUSH", "BURN", "RUSH", "RACE",
        "BOOST", "FIRE", "HARD", "PEAK", "DROP", "GONE", "RAGE", "FULL",
    ])

    # ── Spektrum für Visualizer vorab extrahieren ─────────────────────────────
    spectrum_frames = None
    if visualizer:
        print("\nExtrahiere Frequenzspektrum für Visualizer...")
        spectrum_frames = extract_spectrum_frames(
            audio_path,
            fps=60.0,
            num_bars=visualizer_bars,
            smoothing=0.65,
        )


    # ── Audio laden ─────────────────────────────────────────────────────────
    audio = AudioFileClip(processed_audio_path)
    audio_duration = audio.duration
    beat_times = [b for b in beat_times if b <= audio_duration]

    # ── Main Drop ───────────────────────────────────────────────────────────
    main_drop_idx = -1
    if main_drop_time is not None and beat_times:
        distances = [abs(b - main_drop_time) for b in beat_times]
        main_drop_idx = distances.index(min(distances))
        print(f"Main Drop → Beat #{main_drop_idx} ({beat_times[main_drop_idx]:.2f}s)")

    # ── Bester Action-Clip über alle Quellen (für den Drop) ─────────────────
    best_drop_clip: Optional[ClipInfo] = None
    for vp in video_paths:
        clips = highlights_by_video.get(vp, [])
        action_clips = [c for c in clips if c.tag in ("action", "overtake")]
        candidates = action_clips if action_clips else clips
        if candidates:
            top = max(candidates, key=lambda c: c.score)
            if best_drop_clip is None or top.score > best_drop_clip.score:
                best_drop_clip = top

    if best_drop_clip is None and video_paths:
        clips = highlights_by_video.get(video_paths[0], [])
        best_drop_clip = clips[0] if clips else ClipInfo(
            timestamp=0.0, score=0.0, motion_score=0.0, audio_score=0.0,
            vehicle_count=0.0, tag="action", source=video_paths[0], drift_score=0.0, telemetry_score=0.0, cam_type="external")

    print(f"Drop-Clip: {best_drop_clip}")

    # ── IntelligentPools ────────────────────────────────────────────────────
    pools = {
        vp: IntelligentPool(highlights_by_video[vp], videos[vp].duration)
        for vp in video_paths
    }

    # ── Hard-Beat-Set ────────────────────────────────────────────────────────
    hard_beat_set = set(round(b, 3) for b in hard_beat_times)

    # ── Schnitt-Planung ──────────────────────────────────────────────────────
    print("\nPlane Clip-Sequenz (IntelligentPool: Tag + Beat + Abwechslung)...")

    # Entscheide ob wir den musik-adaptiven Schnitt-Plan nutzen
    use_cut_schedule = cut_schedule is not None and len(cut_schedule) > 0
    if use_cut_schedule:
        print(f"  ✓ Musik-adaptiver Schnitt-Plan aktiv ({len(cut_schedule)} Schnittpunkte)")
    else:
        logging.warning(f"Kein Schnitt-Plan übergeben – Fallback: alle {len(beat_times)} Beats")
        logging.warning(f"  ⚠ Kein Schnitt-Plan übergeben – Fallback: alle {len(beat_times)} Beats")

    # Einheitliche Planungs-Iteration: CutPoints oder Pseudo-CutPoints aus beat_times
    if use_cut_schedule:
        schedule_iter = cut_schedule
    else:
        schedule_iter = []
        if beat_times and beat_times[0] > 0.05:
            schedule_iter.append(CutPoint(
                time=0.0,
                beat_index=-1,
                beat_type="soft",
                phase=sec.phase if (sec := _get_section_for_time(0.0, sections)) else None,
                clip_dur_hint=beat_times[0],
                is_forced=False,
            ))
        schedule_iter.extend([
            CutPoint(
                time=b,
                beat_index=i,
                beat_type=_get_beat_type(i, b, hard_beat_set, i == main_drop_idx),
                phase=sec.phase if (sec := _get_section_for_time(b, sections)) else None,
                clip_dur_hint=(beat_times[i+1] - b) if i+1 < len(beat_times) else 0.5,
            )
            for i, b in enumerate(beat_times)
        ])

    cut_to_clip: Dict[int, ClipInfo] = {}   # schedule_index → ClipInfo
    last_tag: Optional[str] = None
    last_source: Optional[str] = None
    consecutive_same_source = 0

    for sched_idx, cp in enumerate(schedule_iter):
        beat_type     = cp.beat_type

        is_drop       = (cp.beat_index == main_drop_idx)
        min_clip_dur  = max(0.06, cp.clip_dur_hint * 0.5)

        if is_drop and best_drop_clip is not None:
            cut_to_clip[sched_idx] = best_drop_clip
            last_tag = best_drop_clip.tag
            last_source = best_drop_clip.source
            consecutive_same_source = 1 if last_source == best_drop_clip.source else 0
        elif is_multi:
            chosen_vp, clip = _pick_best_source(
                video_paths, pools, beat_type, last_tag,
                last_source, consecutive_same_source,
                min_clip_dur=min_clip_dur
            )
            cut_to_clip[sched_idx] = clip
            consecutive_same_source = (consecutive_same_source + 1) if chosen_vp == last_source else 1
            last_source = chosen_vp
            last_tag = clip.tag
        else:
            clip = pools[video_paths[0]].pick(
                beat_type=beat_type, last_tag=last_tag,
                min_clip_duration=min_clip_dur
            )
            cut_to_clip[sched_idx] = clip
            last_tag = clip.tag
            last_source = video_paths[0]

    # ── Schnitt-Tabelle ausgeben ────────────────────────────────────────────
    tag_stats: Dict[str, int] = {}
    beat_to_clip: Dict[int, ClipInfo] = cut_to_clip
    final_clips = []
    # ── Text-Mask-Sequenz (Intro) vorbereiten ────────────────────────────────
    _intro_words_to_show = []
    if use_intro_text_sequence and video_paths:
        # Ermittle 3-5 Wörter aus den ersten Beats
        _n_words = random.randint(3, 5)
        _seq_words_synced = [
            _beat_word_map.get(i, "") for i in range(min(_n_words * 2, len(beat_times)))
            if _beat_word_map.get(i, "")
        ]
        # Dedupliziere
        _seen = set()
        for w in _seq_words_synced:
            if w not in _seen:
                _seen.add(w)
                _intro_words_to_show.append(w)
                if len(_intro_words_to_show) >= _n_words:
                    break
        
        if len(_intro_words_to_show) < 2:
            _seq_words_pool = _fallback_word_pool or extract_music_words(
                audio_path, use_lyrics=text_mask_use_lyrics
            )
            _intro_words_to_show = (
                random.sample(_seq_words_pool, min(_n_words, len(_seq_words_pool)))
                if len(_seq_words_pool) >= 2
                else ["RUSH", "FIRE", "APEX"]
            )
        print(f"  ★ Intro Text-Sequenz (Whisper-Sync bereitgestellt): {_intro_words_to_show}")

    _text_mask_inserted     = False   # Text-Mask-Drop: nur EINMAL
    _split_glitch_countdown = 2       # Split-Screen-Glitch: letzte N Clips
    _pip_positions          = ["bottom_right", "bottom_left", "top_right", "top_left"]
    _pip_pos_idx            = 0       # Position rotiert pro PiP-Clip

    _pending_overlap = None

    for sched_idx, cp in enumerate(tqdm(schedule_iter, desc="Generiere Clips", unit="Clip")):
        beat          = cp.time
        clip_duration = cp.clip_dur_hint

        if clip_duration <= 0.05:
            current_audio_time = beat + clip_duration
            continue

        info = cut_to_clip.get(sched_idx)
        if info is None:
            current_audio_time = beat + clip_duration
            continue

        vp = info.source
        video = videos.get(vp)
        if video is None:
            current_audio_time = beat + clip_duration
            continue

        start_t = info.timestamp
        if start_t + clip_duration > video.duration:
            start_t = max(0.0, video.duration - clip_duration)

        is_drop       = (cp.beat_index == main_drop_idx)
        beat_type_now = cp.beat_type
        cur_phase     = cp.phase
        phase_label   = f"[{cur_phase}]" if cur_phase else ""

        # Phase-spezifische Effekt-Wahrscheinlichkeiten
        FLASH_PROB_NOW   = _phase_effect_prob(cur_phase, "flash",   use_white_flash)
        RAMP_SCALE_NOW   = _phase_effect_prob(cur_phase, "ramp",    use_speed_ramp)
        REVERSE_PROB_NOW = _phase_effect_prob(cur_phase, "reverse", use_reverse_clip)
        FREEZE_PROB_NOW  = _phase_effect_prob(cur_phase, "freeze",  use_freeze_frame)
        OVERLAP_PROB_NOW = _phase_effect_prob(cur_phase, "overlap", use_overlap_transition)
        BURST_PROB_NOW   = _phase_effect_prob(cur_phase, "burst",   use_jump_cut_burst)

        try:
            # ── Text-Mask-Clip VOR dem Drop (einmalig) ────────────────────
            if is_drop and use_text_mask and not _text_mask_inserted:
                _text_mask_inserted = True
                # Wir brauchen einen kurzen Video-Ausschnitt als Quelle
                _tm_src = video.subclip(start_t, min(start_t + 2.0, video.duration))
                _tm_clip = make_text_mask_clip(
                    _tm_src,
                    text=text_mask_word or "DROP",
                    duration=min(1.6, cp.clip_dur_hint),
                )
                if _tm_clip is not None:
                    if _pending_overlap is not None:
                        final_clips.append(_pending_overlap)
                        _pending_overlap = None
                    final_clips.append(_tm_clip)
                    print(f"  ★ Text-Mask-Clip eingefügt vor Drop @{beat:.2f}s")

            # ── Jump Cut Burst beim Main Drop (nur in Drop-Phase) ─────────
            if is_drop and random.random() < BURST_PROB_NOW:
                num_bursts = random.randint(4, 6)
                print(f"  ★ Jump-Cut-Burst {phase_label}: {num_bursts} Micro-Cuts")
                burst_clips = _make_jump_cut_burst(
                    video, start_t, clip_duration, num_cuts=num_bursts
                )
                if burst_clips:
                    graded_bursts = []
                    for bc in burst_clips:
                        cg = build_per_clip_grade(grade_preset, info.tag, grade_randomize)
                        cg.vignette_strength = vignette_strength
                        cg.vignette_radius   = vignette_radius
                        graded_bursts.append(apply_grade_to_clip(bc, cg))
                    if _pending_overlap is not None:
                        final_clips.append(_pending_overlap)
                        _pending_overlap = None
                    final_clips.extend(graded_bursts)
                    tag_stats[info.tag] = tag_stats.get(info.tag, 0) + num_bursts
                    current_audio_time = beat + clip_duration
                    continue

            subclip = video.subclip(start_t, start_t + clip_duration)

            # ── Speed Ramping ──────────────────────────────────────────────
            _ramp_profile, _ramp_prob_base = _TAG_RAMP_CONFIG.get(info.tag, (None, 0.0))
            # Im Intro: nur ramp_down (langsam einfahren) statt normaler Profil
            if cur_phase == "intro" and RAMP_SCALE_NOW == 0.0:
                _ramp_profile = None   # kein Ramp im Intro
            elif cur_phase == "intro":
                _ramp_profile = "ramp_down"
            _effective_ramp_prob = _ramp_prob_base * RAMP_SCALE_NOW
            _do_speed_ramp = (
                _ramp_profile is not None
                and _effective_ramp_prob > 0.0
                and random.random() < _effective_ramp_prob
                and clip_duration >= 0.28
            )
            if _do_speed_ramp:
                subclip = _apply_speed_ramp(
                    subclip,
                    tag=info.tag,
                    profile_key=_ramp_profile,
                    random_vary=True,
                )
            # ── Reverse Clip ──────────────────────────────────────────────────
            _do_reverse = (
                random.random() < REVERSE_PROB_NOW
                and clip_duration >= 0.3
                and beat_type_now != "hard"
                and not _do_speed_ramp
            )
            if _do_reverse:
                print(f"  ↩ Reverse Clip {phase_label} @ Cut {sched_idx} ({clip_duration:.2f}s)")
                subclip = _make_reverse_clip(subclip)

            # ── Per-Clip Color Grading ────────────────────────────────────
            clip_grade = build_per_clip_grade(
                base_preset=grade_preset,
                clip_tag=info.tag,
                randomize=grade_randomize,
            )
            clip_grade.vignette_strength = vignette_strength
            clip_grade.vignette_radius   = vignette_radius
            subclip = apply_grade_to_clip(subclip, clip_grade)

            # ── Glitch-Frame auf harten Beats ────────────────────────────
            if (use_glitch
                    and beat_type_now == "hard"
                    and clip_duration >= 0.12
                    and random.random() < 0.65):
                glitch_frames = random.randint(2, 4)
                print(f"  ⚡ Glitch @ Cut {sched_idx} ({glitch_frames} Frames)")
                subclip = make_glitch_effect(subclip, glitch_frames=glitch_frames)

            # ── Camera Shake auf harten Beats ────────────────────────────
            if (use_camera_shake
                    and beat_type_now == "hard"
                    and clip_duration >= 0.16
                    and random.random() < 0.50):
                shake_frames = random.randint(4, 8)
                print(f"  📳 Camera Shake @ Cut {sched_idx} ({shake_frames} Frames)")
                subclip = make_camera_shake(subclip, intensity=random.uniform(0.04, 0.08), shake_frames=shake_frames)

            # ── Mirror X Overlay ─────────────────────────────────────────
            if use_mirror_x and random.random() < 0.15:
                print(f"  ↔️ Mirror X @ Cut {sched_idx}")
                subclip = make_mirror_x(subclip)

            # ── Zoom-Punch auf harten Beats ────────────────────────────────
            if (use_zoom_punch
                    and beat_type_now == "hard"
                    and clip_duration >= 0.16
                    and random.random() < 0.60):
                subclip = make_zoom_punch(
                    subclip,
                    zoom_start=1.00, zoom_end=random.uniform(1.05, 1.12),
                )

            # ── Blend-Text (Screen-Mode) auf harten Beats ─────────────────
            # Kommt NACH Glitch (ergänzt, nicht ersetzt) – @editdd032-Stil:
            # Wort in Screen-Blend über dem Clip → Footage-Farben scheinen durch
            # WICHTIG: Nutzt beat-synchrone Whisper-Lyrics (exakt passend zum Lied!)
            if (use_blend_text
                    and beat_type_now == "hard"
                    and clip_duration >= 0.15
                    and _beat_word_map.get(cp.beat_index, "")
                    and random.random() < 0.70):
                # Beat-synchrones Lyric-Wort aus Whisper-Ergebnis holen
                _bt_word = _beat_word_map.get(cp.beat_index, "")
                _bt_mode = random.choice(["screen", "screen", "overlay"])  # screen bevorzugt
                print(f"  ✦ Blend-Text ({_bt_mode}): '{_bt_word}' @ Beat {cp.beat_index} ({cp.time:.2f}s)")
                subclip = make_blend_text_overlay(
                    subclip,
                    text=_bt_word,
                    blend_mode=_bt_mode,
                    alpha=random.uniform(0.75, 0.92),
                )

            # ── Intro Text-Sequenz (Mask auf Schwarz, in-sync) ─────────────
            # @azmiedtz03-Stil: Wörter als Maske vor dem echten Video-Start
            _synced_intro_word = _beat_word_map.get(cp.beat_index, "")
            if (use_intro_text_sequence 
                    and cur_phase == "intro" 
                    and _synced_intro_word
                    and _synced_intro_word in _intro_words_to_show):
                print(f"  ★ Intro Text-Mask: '{_synced_intro_word}' @ Cut {sched_idx}")
                _intro_words_to_show.remove(_synced_intro_word)
                _mask_clip = make_text_mask_clip(
                    subclip,
                    text=_synced_intro_word,
                    duration=clip_duration,
                    fade_in=0.08,
                    fade_out=0.08,
                )
                if _mask_clip is not None:
                    subclip = _mask_clip
                    # Skip B&W overlay for text mask clips since they are already black
                    cur_phase = "intro_masked"

            # ── B&W-Effekt in Intro-Phase ──────────────────────────────────
            if use_bw_intro and cur_phase == "intro":
                subclip = make_bw_overlay(subclip, contrast_boost=1.18)

            # ── Split-Screen-Glitch (letzte Clips in bridge/outro) ────────
            # @azmiedtz03-Stil: am Ende des Videos – gestaffelte Paneel-Cuts
            if (use_split_screen_glitch
                    and cur_phase in ("outro", "bridge")
                    and _split_glitch_countdown > 0
                    and clip_duration >= 0.30):
                _split_glitch_countdown -= 1
                print(f"  ▦ Split-Screen-Glitch {phase_label} @ Cut {sched_idx}")
                subclip = make_split_screen_glitch(
                    subclip,
                    num_stripes=3,
                    max_offset_frac=0.022,
                    time_offset_frac=0.03,
                )

            # ── Picture-in-Picture (Multi-Cam) ────────────────────────────
            if (use_pip and is_multi and clip_duration >= 0.40
                    and random.random() < 0.30):
                _pip_sources = [v for v in video_paths if v != vp]
                if _pip_sources:
                    _pip_vp     = random.choice(_pip_sources)
                    _pip_video  = videos.get(_pip_vp)
                    if _pip_video:
                        _pip_info  = pools[_pip_vp].preview(beat_type=beat_type_now)
                        _pip_st    = _pip_info.timestamp
                        _pip_end   = min(_pip_st + clip_duration, _pip_video.duration)
                        if _pip_end > _pip_st:
                            _pip_sub   = _pip_video.subclip(_pip_st, _pip_end)
                            _pip_pos   = _pip_positions[_pip_pos_idx % len(_pip_positions)]
                            _pip_pos_idx += 1
                            print(f"  📺 PiP @ Cut {sched_idx} ({_pip_pos})")
                            subclip = make_pip_overlay(
                                subclip, _pip_sub,
                                position=_pip_pos,
                                size_frac=0.27,
                            )

            # ── White-Flash auf harten Beats ──────────────────────────────
            if random.random() < FLASH_PROB_NOW and beat_type_now == "hard" and clip_duration > 0.2:
                flash_dur = min(0.25, clip_duration * 0.4)
                subclip = subclip.fl(
                    lambda gf, t, _sd=subclip, _fd=flash_dur:
                        _make_flash_frame(t, _sd, _fd),
                    keep_duration=True
                )

            # ── Freeze Frame kurz vor dem nächsten Schnitt ─────────────────
            _do_freeze = (
                random.random() < FREEZE_PROB_NOW
                and clip_duration >= 0.25
                and not _do_reverse
            )
            if _do_freeze:
                freeze_dur = random.uniform(0.08, 0.20) if cur_phase in ("bridge", "outro") \
                             else random.uniform(0.08, 0.14)
                print(f"  ❄ Freeze Frame {phase_label} @ Cut {sched_idx} ({freeze_dur*1000:.0f} ms)")
                subclip = _make_freeze_frame(subclip, freeze_dur)

            # ── Overlap Transition ────────────────────────────────────────
            if _pending_overlap is not None:
                if random.random() < OVERLAP_PROB_NOW:
                    ov_dur = random.uniform(0.05, 0.10)
                    print(f"  ⟷ Overlap Transition {phase_label} @ Cut {sched_idx} ({ov_dur*1000:.0f} ms)")
                    new_prev, subclip = _make_overlap_transition(
                        _pending_overlap, subclip, overlap_dur=ov_dur
                    )
                    final_clips.append(new_prev)
                else:
                    final_clips.append(_pending_overlap)
                _pending_overlap = None

            # Diesen Clip als "pending" für den nächsten Overlap-Check merken
            _pending_overlap = subclip
            tag_stats[info.tag] = tag_stats.get(info.tag, 0) + 1

        except Exception as e:
            logging.error(f"Clip {sched_idx} ({os.path.basename(vp)}, t={start_t:.2f}s, dur={clip_duration:.2f}s): {e}")
            logging.error(f"  ✗ Clip {sched_idx} ({os.path.basename(vp)}, "
                          f"t={start_t:.2f}s, dur={clip_duration:.2f}s): {e}")

        current_audio_time = beat + clip_duration

    # Letzten pending Clip noch hinzufügen
    if _pending_overlap is not None:
        final_clips.append(_pending_overlap)
        _pending_overlap = None

    # ── Letzter Rest ────────────────────────────────────────────────────────
    if current_audio_time < audio_duration:
        leftover = audio_duration - current_audio_time
        if leftover > 0.1 and best_drop_clip:
            vp = best_drop_clip.source
            video = videos.get(vp)
            if video:
                start_t = best_drop_clip.timestamp
                if start_t + leftover > video.duration:
                    start_t = max(0.0, video.duration - leftover)
                try:
                    final_clips.append(video.subclip(start_t, start_t + leftover))
                except Exception as e:
                    logging.error(f"Letzter Clip: {e}")

    if not final_clips:
        logging.error("Fehler: Konnte keine Clips erstellen.")
        for v in videos.values():
            v.close()
        audio.close()
        return {}

    # ── Qualitätswarnungen ──────────────────────────────────────────────────
    _quality_check(tag_stats, video_paths, beat_to_clip)
    retention_heatmap = _estimate_retention_heatmap(schedule_iter, cut_to_clip) \
        if generate_retention_report else []
    if retention_heatmap:
        top_risk = sorted(retention_heatmap, key=lambda x: x["risk"], reverse=True)[:5]
        print(f"Retention-Hotspots: {top_risk}")

    # ── Video zusammensetzen ────────────────────────────────────────────────
    print("\nSetze finales Video zusammen...")
    final_video = concatenate_videoclips(final_clips, method="chain")

    # ── Cinematic Letterbox (globaler Filter) ────────────────────────────────
    if use_letterbox:
        print("Wende Cinematic Letterbox an...")
        final_video = apply_letterbox(final_video, bar_fraction=0.07)
    # Hinweis: Farbkorrektur wurde bereits per-Clip angewendet (s.o.)

    # ── Watermark Overlay ────────────────────────────────────────────────────
    # (Wird nach Letterbox angewendet, damit es oben/unten oder darüber liegen kann)
    w_text = runtime_cfg.get("watermark_text")
    w_opacity = runtime_cfg.get("watermark_opacity", 0.4)
    if w_text:
        from visual_effects import make_watermark_overlay
        print(f"Wende Watermark an: '{w_text}' (Opacity: {w_opacity:.1%})...")
        final_video = make_watermark_overlay(final_video, w_text, opacity=w_opacity)

    # ── Audio-Visualizer Overlay ──────────────────────────────────────────────
    if visualizer and spectrum_frames is not None:
        print("Wende Audio-Visualizer Overlay an...")
        viz_filter = _make_visualizer_filter(
            spectrum_frames, fps=60.0, height_frac=visualizer_height
        )
        final_video = final_video.fl(viz_filter, apply_to=["video"], keep_duration=True)

    # ── Audio ────────────────────────────────────────────────────────────────
    final_video = final_video.set_audio(audio)

    # ── Export Settings ─────────────────────────────────────────────────────
    _n_threads = os.cpu_count() or 8

    export_w, export_h = _TARGET_W, _TARGET_H
    export_fps = 60

    if preview:
        export_w, export_h = 540, 960
        export_fps = 30
        print(f"\n🚀 [PREVIEW MODE] Exportiere mit reduzierter Last → {output_path}")
    else:
        print(f"\n🎬 Exportiere High-Quality Video → {output_path}")

    print(f"  Auflösung: {export_w}x{export_h} | {export_fps} FPS | {_n_threads} Threads")

    # Sicherstellen dass das finale Video die korrekte Größe hat
    fvw, fvh = final_video.size
    if fvw != export_w or fvh != export_h:
        final_video = final_video.resize((export_w, export_h))

    if preview:
        # Ultra-Fast settings for preview
        nvenc_params = ["-cq", "28", "-preset", "p1", "-pix_fmt", "yuv420p"]
        x264_params  = ["-crf", "28", "-preset", "ultrafast", "-pix_fmt", "yuv420p"]
    else:
        # NVENC: CQ 14 = sehr hohe Qualität (niedriger = besser), VBR mit 80 Mbit/s Peak
        nvenc_params = [
            "-cq",         "14",
            "-rc",         "vbr",
            "-maxrate",    "80M",
            "-bufsize",    "160M",
            "-pix_fmt",    "yuv420p",
            "-spatial-aq", "1",
            "-temporal-aq","1",
            "-b:v",        "0",    # Reine CQ-Kontrolle (kein festes Bitrate-Ziel)
        ]
        # libx264: CRF 14 = fast verlustlos
        x264_params = [
            "-crf",     "14",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
        ]

    try:
        _write_videofile_with_retry(
            final_video, max_retries=3, delay_sec=3.0,
            filename=output_path, fps=export_fps, codec="h264_nvenc",
            audio_codec="aac", audio_bitrate="192k" if preview else "320k",
            preset="p1" if preview else "p4", threads=_n_threads, ffmpeg_params=nvenc_params, logger="bar"
        )
        print("  ✓ NVENC GPU-Export erfolgreich (1080×1920, 60 FPS, CQ14)")
    except Exception as e:
        logging.warning(f"\nNVENC fehlgeschlagen oder nicht verfügbar nach Retries: {e}\nFallback → CPU (libx264)...")
        _write_videofile_with_retry(
            final_video, max_retries=3, delay_sec=3.0,
            filename=output_path, fps=export_fps, codec="libx264",
            audio_codec="aac", audio_bitrate="192k" if preview else "320k",
            preset="ultrafast" if preview else "medium", threads=_n_threads,
            ffmpeg_params=x264_params, logger="bar"
        )
        print("  ✓ CPU-Export erfolgreich (1080×1920, 60 FPS, CRF14)")

    # ── Ressourcen freigeben ─────────────────────────────────────────────────
    for v in videos.values():
        v.close()
    audio.close()
    final_video.close()

    # Temp-Audio-Datei entfernen
    if processed_audio_path != audio_path and os.path.exists(processed_audio_path):
        try:
            os.remove(processed_audio_path)
        except Exception:
            pass

    print("\n" + "═"*60)
    print("  Fertig! TikTok-Edit erfolgreich erstellt.")
    print("═"*60)

    return {
        "tag_stats": tag_stats,
        "output": output_path,
        "hook_moments": hook_moments,
        "retention_heatmap": retention_heatmap,
        "active_runtime_config": runtime_cfg,
        "trend_presets": sorted(list(_TREND_STYLE_PRESETS.keys())),
    }


# ---------------------------------------------------------------------------
# Ausgabe-Hilfsfunktionen
# ---------------------------------------------------------------------------

def _print_cut_table(beat_times: list,
                     beat_to_clip: Dict[int, ClipInfo],
                     hard_beat_set: set,
                     main_drop_idx: int):
    """Gibt eine übersichtliche Tabelle der geplanten Schnitte aus."""
    print("\n" + "─"*70)
    print(f"  {'Beat':>4}  {'Zeit':>6}  {'Typ':<7}  {'Tag':<10}  "
          f"{'Score':>5}  {'Motion':>6}  {'Quelle'}")
    print("─"*70)

    for i, beat in enumerate(beat_times):
        info = beat_to_clip.get(i)
        if info is None:
            continue
        is_drop = (i == main_drop_idx)
        is_hard = round(beat, 3) in hard_beat_set
        btype = "★ DROP" if is_drop else ("HARD  " if is_hard else "normal")
        source_name = os.path.basename(info.source) if info.source else "?"

        print(f"  {i:>4}  {beat:>6.2f}s  {btype:<7}  {info.tag:<10}  "
              f"{info.score:>5.2f}  {info.motion_score:>6.2f}  {source_name}")

    print("─"*70)


def _quality_check(tag_stats: Dict[str, int],
                   video_paths: List[str],
                   beat_to_clip: Dict[int, ClipInfo]):
    """Gibt Qualitätswarnungen aus wenn das Video unausgewogen wäre."""
    total = sum(tag_stats.values())
    if total == 0:
        return

    print("\n  Qualitätsprüfung:")

    # Zu viele ruhige Clips
    calm_ratio = tag_stats.get("calm", 0) / total
    if calm_ratio > 0.40:
        logging.warning(f"  ⚠  {calm_ratio*100:.0f}% der Clips sind 'calm' "
              f"→ Video könnte langweilig wirken!")
        logging.warning(f"{calm_ratio*100:.0f}% der Clips sind 'calm' → Video könnte langweilig wirken!")
        logging.warning(f"  ⚠  {calm_ratio*100:.0f}% der Clips sind 'calm' "
                        f"→ Video könnte langweilig wirken!")
    else:
        print(f"  ✓  Calm-Anteil: {calm_ratio*100:.0f}% (OK)")

    # Quellen-Verteilung prüfen (Multi-Cam)
    if len(video_paths) > 1:
        source_counts: Dict[str, int] = {}
        for info in beat_to_clip.values():
            k = os.path.basename(info.source)
            source_counts[k] = source_counts.get(k, 0) + 1
        for src, count in source_counts.items():
            ratio = count / total
            if ratio > 0.70:
                logging.warning(f"  ⚠  '{src}' erscheint in {ratio*100:.0f}% aller Clips "
                      f"→ mehr Kamerawechsel wären besser!")
                logging.warning(f"'{src}' erscheint in {ratio*100:.0f}% aller Clips → mehr Kamerawechsel wären besser!")
                logging.warning(f"  ⚠  '{src}' erscheint in {ratio*100:.0f}% aller Clips "
                                f"→ mehr Kamerawechsel wären besser!")
            else:
                print(f"  ✓  '{src}': {ratio*100:.0f}% (OK)")

    # Tag-Übersicht
    print(f"  Tag-Verteilung: {tag_stats}")
