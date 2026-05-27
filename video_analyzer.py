"""
video_analyzer.py – Intelligente Videoanalyse mit Optischem Fluss & Clip-Klassifizierung

GPU-Beschleunigung:
  - Optischer Fluss: cv2.cuda.FarnebackOpticalFlow (GPU) mit CPU-Fallback
  - YOLO: Batch-Inferenz (8 Frames auf einmal) statt frame-by-frame
  - Audio + Flow: parallel via ThreadPoolExecutor
  - Frame-Downscale vor Flow: 50% → 4× schneller
"""

import cv2
import numpy as np
import os
import librosa
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import List, Dict
from moviepy.editor import VideoFileClip
from ultralytics import YOLO
from tqdm import tqdm

# ---------------------------------------------------------------------------
# GPU-Check (einmalig beim Import)
# ---------------------------------------------------------------------------

def _check_cuda() -> bool:
    """Gibt True zurück wenn OpenCV mit CUDA kompiliert wurde und eine GPU vorhanden ist."""
    try:
        return cv2.cuda.getCudaEnabledDeviceCount() > 0
    except Exception:
        return False

_CUDA_AVAILABLE = _check_cuda()

if _CUDA_AVAILABLE:
    print(f"[video_analyzer] GPU OK: CUDA aktiv - GPU-Optischer-Fluss aktiv")
else:
    print(f"[video_analyzer] INFO: Kein CUDA - CPU-Fallback aktiv")


# ---------------------------------------------------------------------------
# ClipInfo – Datenklasse für einen analysierten Clip
# ---------------------------------------------------------------------------

@dataclass
class ClipInfo:
    timestamp: float         # Startzeitpunkt im Video (Sekunden)
    score: float             # Gesamt-Action-Score 0–1
    motion_score: float      # Optischer Fluss (wie dynamisch, 0–1)
    drift_score: float       # Seitlicher Bewegungsfluss (0-1)
    audio_score: float       # Audio-Intensität 0–1
    telemetry_score: float   # Telemetrie-Intensität (G-Kräfte, 0-1)
    vehicle_count: float     # Ø Fahrzeuge erkannt
    cam_type: str            # "helmet" | "external"
    tag: str                 # Inhaltsklasse (s. unten)
    source: str = ""         # Quell-Videopfad (wird von außen gesetzt)

    def __repr__(self):
        return (f"ClipInfo(t={self.timestamp:.2f}s, tag={self.tag!r}, "
                f"cam={self.cam_type}, "
                f"score={self.score:.2f}, motion={self.motion_score:.2f}, "
                f"drift={self.drift_score:.2f}, "
                f"vehicles={self.vehicle_count:.1f})")


# ---------------------------------------------------------------------------
# Tagging-Logik
# ---------------------------------------------------------------------------

def _classify_clip(score: float, motion_score: float, drift_score: float, audio_score: float,
                   vehicle_count: float, cam_type: str) -> str:
    if drift_score >= 0.60 and motion_score >= 0.50:
        return "drift"
    if score >= 0.65 and vehicle_count >= 1.5 and audio_score >= 0.55:
        return "action"
    if vehicle_count >= 1.8 and motion_score >= 0.45:
        return "overtake"
    if motion_score >= 0.50 and vehicle_count < 1.5:
        return "corner"
    if motion_score < 0.35 and score >= 0.40:
        return "straight"
    return "calm"


# ---------------------------------------------------------------------------
# Optischer Fluss – GPU (CUDA) oder CPU Fallback
# ---------------------------------------------------------------------------

def _compute_optical_flow_gpu(cap: cv2.VideoCapture, total_frames: int,
                               fps: float, sample_interval: float = 0.25) -> List[Dict]:
    """
    GPU-beschleunigter Farneback-Optischer-Fluss via cv2.cuda.
    Frames werden vor der Berechnung auf 50% downgesaclt (4× schneller).
    """
    print("  ⚡ Berechne Optischen Fluss [GPU – CUDA Farneback]...")
    frames_to_skip = max(1, int(fps * sample_interval))
    flow_data = []
    SCALE = 0.5  # Downscale-Faktor – halbiert jede Dimension

    # GPU-Optical-Flow Objekt erstellen
    gpu_flow = cv2.cuda.FarnebackOpticalFlow.create(
        numLevels=3, pyrScale=0.5, fastPyramids=False,
        winSize=13, numIters=3, polyN=5, polySigma=1.2, flags=0
    )

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ret, prev_bgr = cap.read()
    if not ret:
        return []

    # Downscale + Graustufen
    prev_small = cv2.resize(prev_bgr, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_LINEAR)
    prev_gray_cpu = cv2.cvtColor(prev_small, cv2.COLOR_BGR2GRAY)
    prev_gray_gpu = cv2.cuda_GpuMat()
    prev_gray_gpu.upload(prev_gray_cpu)

    h, w = prev_gray_cpu.shape
    current_frame = frames_to_skip

    while current_frame < total_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
        ret, curr_bgr = cap.read()
        if not ret:
            break

        curr_small = cv2.resize(curr_bgr, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_LINEAR)
        curr_gray_cpu = cv2.cvtColor(curr_small, cv2.COLOR_BGR2GRAY)
        curr_gray_gpu = cv2.cuda_GpuMat()
        curr_gray_gpu.upload(curr_gray_cpu)

        # GPU-Flow berechnen
        flow_gpu = gpu_flow.calc(prev_gray_gpu, curr_gray_gpu, None)
        flow_cpu = flow_gpu.download()  # (H, W, 2) float32

        magnitude, _ = cv2.cartToPolar(flow_cpu[..., 0], flow_cpu[..., 1])
        motion_score = float(np.mean(magnitude))
        drift_score = float(np.mean(np.abs(flow_cpu[..., 0])))

        # Camera-Type Heuristik
        cockpit_region = magnitude[int(h * 0.7):, int(w * 0.3):int(w * 0.7)]
        window_region  = magnitude[int(h * 0.2):int(h * 0.5), :]
        cam_type = "helmet" if np.mean(cockpit_region) < np.mean(window_region) * 0.4 else "external"

        flow_data.append({
            "motion_score": motion_score,
            "drift_score":  drift_score,
            "cam_type":     cam_type,
        })

        prev_gray_gpu = curr_gray_gpu
        current_frame += frames_to_skip

    return flow_data


def _compute_optical_flow_cpu(cap: cv2.VideoCapture, total_frames: int,
                               fps: float, sample_interval: float = 0.25) -> List[Dict]:
    """
    CPU-Fallback Farneback Optischer Fluss.
    Auf CPU wird sample_interval auf mindestens 1.0s angehoben und der
    Downscale auf 25% gesetzt – das macht den Flow ~16-64× schneller als
    der naïve Ansatz, ohne die Score-Qualität wesentlich zu verschlechtern.
    """
    # Auf CPU mindestens 1 Sample pro Sekunde – sonst dauert es ewig
    cpu_interval = max(sample_interval, 1.0)
    frames_to_skip = max(1, int(fps * cpu_interval))
    SCALE = 0.25   # 25 % statt 50 % → 16× weniger Pixel pro Frame
    flow_data = []

    total_samples = max(1, total_frames // frames_to_skip)
    print(f"  Berechne Optischen Fluss [CPU – Farneback 25%-Downscale, "
          f"{cpu_interval:.1f}s-Intervall, ~{total_samples} Frames]...")

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ret, prev_bgr = cap.read()
    if not ret:
        return []

    prev_small = cv2.resize(prev_bgr, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_LINEAR)
    prev_gray  = cv2.cvtColor(prev_small, cv2.COLOR_BGR2GRAY)
    h, w = prev_gray.shape
    current_frame = frames_to_skip
    processed = 0
    REPORT_EVERY = max(1, total_samples // 10)  # alle ~10 % Fortschritt melden

    while current_frame < total_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
        ret, curr_bgr = cap.read()
        if not ret:
            break

        curr_small = cv2.resize(curr_bgr, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_LINEAR)
        curr_gray  = cv2.cvtColor(curr_small, cv2.COLOR_BGR2GRAY)

        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, curr_gray, None,
            pyr_scale=0.5, levels=2, winsize=11,
            iterations=2, poly_n=5, poly_sigma=1.2, flags=0
        )

        magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        motion_score = float(np.mean(magnitude))
        drift_score  = float(np.mean(np.abs(flow[..., 0])))

        cockpit_region = magnitude[int(h * 0.7):, int(w * 0.3):int(w * 0.7)]
        window_region  = magnitude[int(h * 0.2):int(h * 0.5), :]
        cam_type = "helmet" if np.mean(cockpit_region) < np.mean(window_region) * 0.4 else "external"

        flow_data.append({
            "motion_score": motion_score,
            "drift_score":  drift_score,
            "cam_type":     cam_type,
        })

        prev_gray = curr_gray
        current_frame += frames_to_skip
        processed += 1
        if processed % REPORT_EVERY == 0:
            pct = int(100 * processed / total_samples)
            print(f"    Flow-Fortschritt: {pct}% ({processed}/{total_samples} Samples)", flush=True)

    print(f"    Flow abgeschlossen – {processed} Samples verarbeitet.")
    return flow_data


def _compute_optical_flow(cap: cv2.VideoCapture, total_frames: int,
                           fps: float, sample_interval: float = 0.25) -> List[Dict]:
    """Wählt automatisch GPU oder CPU Optischen Fluss."""
    try:
        if _CUDA_AVAILABLE:
            raw = _compute_optical_flow_gpu(cap, total_frames, fps, sample_interval)
        else:
            raw = _compute_optical_flow_cpu(cap, total_frames, fps, sample_interval)
    except Exception as e:
        print(f"  ⚠ GPU-Flow fehlgeschlagen ({e}) – CPU-Fallback")
        raw = _compute_optical_flow_cpu(cap, total_frames, fps, sample_interval)

    if not raw:
        return [{"motion_score": 0.0, "drift_score": 0.0, "cam_type": "external"}]

    # Normierung
    m_scores = np.array([d["motion_score"] for d in raw])
    d_scores = np.array([d["drift_score"]  for d in raw])
    m_max = np.percentile(m_scores, 95) if len(m_scores) > 0 else 1.0
    d_max = np.percentile(d_scores, 95) if len(d_scores) > 0 else 1.0

    for d in raw:
        d["motion_score"] = float(min(1.0, max(0.0, d["motion_score"] / m_max if m_max > 0 else 0.0)))
        d["drift_score"]  = float(min(1.0, max(0.0, d["drift_score"]  / d_max if d_max > 0 else 0.0)))

    return raw


# ---------------------------------------------------------------------------
# Audio-Analyse
# ---------------------------------------------------------------------------

def _analyze_audio(video_path: str, total_frames: int, fps: float) -> np.ndarray:
    """Extrahiert Audio aus dem Video und berechnet RMS-Energie pro Frame."""
    print("  Extrahiere & analysiere Video-Audio...")
    audio_scores = np.zeros(total_frames, dtype=np.float32)
    try:
        video = VideoFileClip(video_path)
        if video.audio is None:
            video.close()
            return audio_scores
        temp_audio_path = "_temp_audio_analysis.wav"
        video.audio.write_audiofile(temp_audio_path, logger=None, fps=22050)
        video.close()

        y, sr = librosa.load(temp_audio_path, sr=22050)
        rms = librosa.feature.rms(y=y)[0]
        times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=512)
        video_times = np.arange(total_frames) / fps
        audio_scores = np.interp(video_times, times, rms).astype(np.float32)
        if np.max(audio_scores) > 0:
            audio_scores /= np.max(audio_scores)
        os.remove(temp_audio_path)
    except Exception as e:
        print(f"  Warnung: Audio-Analyse fehlgeschlagen – {e}")
    return audio_scores


# ---------------------------------------------------------------------------
# Telemetrie-Analyse (CSV)
# ---------------------------------------------------------------------------

def _analyze_telemetry(video_path: str, total_frames: int, fps: float) -> np.ndarray:
    """Lädt G-Kräfte aus einer Telemetrie-CSV, falls vorhanden."""
    base = os.path.splitext(video_path)[0]
    csv_path = base + "_telemetry.csv"
    scores = np.zeros(total_frames, dtype=np.float32)
    if not os.path.exists(csv_path):
        return scores
    print(f"  Lade Telemetrie-Daten aus {os.path.basename(csv_path)}...")
    try:
        import pandas as pd
        df = pd.read_csv(csv_path)
        if "Time" in df.columns and "G_Lat" in df.columns and "G_Long" in df.columns:
            g_total = np.sqrt(df["G_Lat"]**2 + df["G_Long"]**2).values
            times = df["Time"].values
            video_times = np.arange(total_frames) / fps
            interp_g = np.interp(video_times, times, g_total)
            if np.max(interp_g) > 0:
                interp_g /= np.max(interp_g)
            scores = interp_g.astype(np.float32)
    except Exception as e:
        print(f"  Warnung: Telemetrie konnte nicht geladen werden – {e}")
    return scores


# ---------------------------------------------------------------------------
# YOLO Batch-Inferenz
# ---------------------------------------------------------------------------

def _run_yolo_batch(model, frames: List[np.ndarray],
                    target_classes) -> List[float]:
    """
    Führt YOLO-Inferenz auf einem Batch von Frames durch.
    Nutzt die GPU automatisch (wenn YOLO auf GPU initialisiert wurde).
    Gibt eine Liste von vehicle_counts zurück.
    """
    if model is None or not frames:
        return [0.0] * len(frames)
    try:
        results = model.predict(frames, classes=target_classes, verbose=False)
        return [float(len(r.boxes)) for r in results]
    except Exception as e:
        print(f"  ⚠ YOLO Batch-Fehler: {e}")
        return [0.0] * len(frames)


# ---------------------------------------------------------------------------
# Kern-Analyse: find_highlights
# ---------------------------------------------------------------------------

def find_highlights(video_path: str, num_clips: int,
                    clip_duration: float = 2.0) -> List[ClipInfo]:
    """
    Analysiert ein Video vollständig (YOLO + Optischer Fluss + Audio) und
    gibt eine nach Score sortierte Liste von ClipInfo-Objekten zurück.

    GPU-Optimierungen:
      - Optischer Fluss auf GPU (CUDA FarnebackOpticalFlow)
      - YOLO Batch-Inferenz (BATCH_SIZE Frames gleichzeitig)
      - Audio und Flow parallel via ThreadPoolExecutor
    """
    import time
    t0 = time.perf_counter()
    print(f"\n  Analysiere: {os.path.basename(video_path)}")

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_duration = total_frames / fps

    if video_duration < clip_duration:
        print("  Warnung: Video zu kurz – gebe Start zurück.")
        cap.release()
        return [ClipInfo(timestamp=0.0, score=0.5, motion_score=0.5, drift_score=0.0,
                         audio_score=0.5, telemetry_score=0.0, vehicle_count=0.0,
                         cam_type="external", tag="calm", source=video_path)]

    sample_interval = 0.25  # Ziel: 4 Samples/s (auf CPU wird automatisch auf 1.0s erhöht)
    frames_to_skip  = max(1, int(fps * (sample_interval if _CUDA_AVAILABLE else 1.0)))

    # ── 1 + 1b: Audio & Telemetrie PARALLEL zum Rest ──────────────────────
    print("  Starte Audio-Analyse & Telemetrie parallel...")
    with ThreadPoolExecutor(max_workers=2) as pool:
        future_audio = pool.submit(_analyze_audio, video_path, total_frames, fps)
        future_telem = pool.submit(_analyze_telemetry, video_path, total_frames, fps)

        # ── 2. Optischer Fluss (läuft während Audio parallel analysiert wird) ──
        flow_data = _compute_optical_flow(cap, total_frames, fps, sample_interval)

        audio_scores     = future_audio.result()
        telemetry_scores = future_telem.result()

    # ── 3. YOLO laden ────────────────────────────────────────────────────
    print("  Lade YOLOv8 für Fahrzeug-Erkennung (GPU)...")
    model = None
    target_classes = None
    YOLO_BATCH = 8   # Frames pro Batch → GPU-Parallelisierung

    try:
        if os.path.exists("simracing_model.pt"):
            print("  Custom Sim-Racing YOLO Modell gefunden!")
            model = YOLO("simracing_model.pt")
        else:
            model = YOLO("yolov8n.pt")
            target_classes = [2, 3, 5, 7]  # car, motorcycle, bus, truck
    except Exception as e:
        print(f"  YOLO konnte nicht geladen werden: {e}")

    # ── 4. Frames sammeln + YOLO Batch-Inferenz ───────────────────────────
    print("  Analysiere Frames (YOLO-Batch + Motion + Audio + Telemetry)...")
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # Alle Sample-Frame-Positionen vorberechnen
    sample_positions = list(range(0, total_frames, frames_to_skip))

    raw_scores: List[dict] = []
    batch_frames: List[np.ndarray] = []
    batch_meta:   List[dict] = []  # meta-daten pro Frame im Batch

    def _flush_batch():
        """Schickt den aktuellen Batch durch YOLO und leert ihn."""
        counts = _run_yolo_batch(model, batch_frames, target_classes)
        for meta, vc in zip(batch_meta, counts):
            meta["vehicle_count"] = vc
        batch_frames.clear()
        batch_meta.clear()

    for sample_idx, current_frame in enumerate(tqdm(sample_positions, desc="Analysiere Frames", unit="Frame")):
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
        ret, frame = cap.read()
        if not ret:
            break

        a_score = float(audio_scores[current_frame]) if current_frame < len(audio_scores) else 0.0
        t_score = float(telemetry_scores[current_frame]) if current_frame < len(telemetry_scores) else 0.0
        f_data  = flow_data[sample_idx] if sample_idx < len(flow_data) else {
            "motion_score": 0.0, "drift_score": 0.0, "cam_type": "external"
        }

        meta = {
            "frame":         current_frame,
            "timestamp":     current_frame / fps,
            "score":         0.0,
            "motion_score":  f_data["motion_score"],
            "drift_score":   f_data["drift_score"],
            "audio_score":   a_score,
            "telemetry_score": t_score,
            "vehicle_count": 0.0,
            "cam_type":      f_data["cam_type"],
        }
        raw_scores.append(meta)
        batch_frames.append(frame)
        batch_meta.append(meta)

        if len(batch_frames) >= YOLO_BATCH:
            _flush_batch()

    # Letzten unvollständigen Batch abarbeiten
    if batch_frames:
        _flush_batch()

    cap.release()

    # ── 5. Gesamt-Score berechnen ─────────────────────────────────────────
    has_telemetry = np.max(telemetry_scores) > 0
    for r in raw_scores:
        yolo_score = min(r["vehicle_count"], 5.0) / 5.0
        if has_telemetry:
            r["score"] = (r["audio_score"]      * 0.20
                          + yolo_score           * 0.25
                          + r["motion_score"]    * 0.20
                          + r["drift_score"]     * 0.15
                          + r["telemetry_score"] * 0.20)
        else:
            r["score"] = (r["audio_score"]   * 0.30
                          + yolo_score        * 0.30
                          + r["motion_score"] * 0.25
                          + r["drift_score"]  * 0.15)

    if not raw_scores:
        return [ClipInfo(timestamp=0.0, score=0.0, motion_score=0.0, drift_score=0.0,
                         audio_score=0.0, telemetry_score=0.0, vehicle_count=0.0,
                         cam_type="external", tag="calm", source=video_path)]

    # ── 6. Scores glätten ────────────────────────────────────────────────
    scores_arr = np.array([r["score"] for r in raw_scores], dtype=np.float32)
    window = max(1, min(len(scores_arr), int(clip_duration / sample_interval)))
    smoothed = np.convolve(scores_arr, np.ones(window) / window, mode="valid")

    # ── 7. Beste Clips auswählen (keine Überlappung) ─────────────────────
    sorted_idx = np.argsort(smoothed)[::-1]
    selected: List[ClipInfo] = []

    for idx in sorted_idx:
        if idx >= len(raw_scores):
            continue
        r = raw_scores[idx]
        t = r["timestamp"]
        if any(abs(t - s.timestamp) < clip_duration for s in selected):
            continue
        if t + clip_duration > video_duration:
            continue

        tag = _classify_clip(r["score"], r["motion_score"], r["drift_score"],
                             r["audio_score"], r["vehicle_count"], r["cam_type"])
        selected.append(ClipInfo(
            timestamp=t, score=r["score"], motion_score=r["motion_score"],
            drift_score=r["drift_score"], audio_score=r["audio_score"],
            telemetry_score=r["telemetry_score"], vehicle_count=r["vehicle_count"],
            cam_type=r["cam_type"], tag=tag, source=video_path,
        ))
        if len(selected) >= num_clips:
            break

    # Lücken auffüllen
    if len(selected) < num_clips:
        for idx in sorted_idx:
            if len(selected) >= num_clips:
                break
            if idx >= len(raw_scores):
                continue
            r = raw_scores[idx]
            t = r["timestamp"]
            if any(abs(t - s.timestamp) < clip_duration * 0.5 for s in selected):
                continue
            tag = _classify_clip(r["score"], r["motion_score"], r["drift_score"],
                                 r["audio_score"], r["vehicle_count"], r["cam_type"])
            selected.append(ClipInfo(
                timestamp=t, score=r["score"], motion_score=r["motion_score"],
                drift_score=r["drift_score"], audio_score=r["audio_score"],
                telemetry_score=r["telemetry_score"], vehicle_count=r["vehicle_count"],
                cam_type=r["cam_type"], tag=tag, source=video_path,
            ))

    selected.sort(key=lambda c: c.score, reverse=True)

    elapsed = time.perf_counter() - t0
    print(f"\n  {len(selected)} Highlights gefunden  (Analysezeit: {elapsed:.1f}s):")
    tag_counts: Dict[str, int] = {}
    for c in selected:
        tag_counts[c.tag] = tag_counts.get(c.tag, 0) + 1
        print(f"    t={c.timestamp:6.2f}s  tag={c.tag:<10s}  cam={c.cam_type:<8s} "
              f"score={c.score:.2f}  drift={c.drift_score:.2f}  "
              f"vehicles={c.vehicle_count:.1f}")
    print(f"  Tag-Verteilung: {tag_counts}")

    return selected


# ---------------------------------------------------------------------------
# Multi-Video Wrapper
# ---------------------------------------------------------------------------

def find_highlights_multi(video_paths: List[str], num_clips_total: int,
                           clip_duration: float = 2.0) -> Dict[str, List[ClipInfo]]:
    """
    Analysiert mehrere Videos und gibt ein dict  { video_path: [ClipInfo, ...] } zurück.
    """
    n = len(video_paths)
    if n == 0:
        return {}

    clips_per_video = max(1, num_clips_total // n)
    remainder = num_clips_total - clips_per_video * n

    result: Dict[str, List[ClipInfo]] = {}
    for i, vp in enumerate(video_paths):
        extra = 1 if i < remainder else 0
        num = clips_per_video + extra
        print(f"\n── Analysiere Quelle {i+1}/{n}: {os.path.basename(vp)} ({num} Highlights) ──")
        result[vp] = find_highlights(vp, num_clips=num, clip_duration=clip_duration)

    return result
