"""
audio_effects.py – Audio-reaktive Effekte für den TikTok-Editor

Features:
  1. Gain-Staging     : Normalisiert Musik auf Ziel-Lautstärke (TikTok-optimiert)
  2. Volume-Dip       : Lautstärke-Senke vor dem Drop → dann BOOM
  3. Spectrum-Frames  : Frequenzspektrum pro Frame für den Audio-Visualizer
  4. save_processed   : Speichert verarbeitetes Audio als WAV
"""

import os
import numpy as np
import librosa
import soundfile as sf


# ---------------------------------------------------------------------------
# 1. Gain-Staging
# ---------------------------------------------------------------------------

def normalize_audio_gain(y: np.ndarray, sr: int,
                          target_rms: float = 0.12) -> np.ndarray:
    """
    Normalisiert die Musik auf einen Ziel-RMS-Pegel.
    target_rms = 0.12  →  ca. -18 dBFS  (optimal für TikTok/mobile Lautsprecher)
    Gain wird auf ±12 dB begrenzt um Clipping zu vermeiden.
    """
    current_rms = float(np.sqrt(np.mean(y ** 2)))
    if current_rms < 1e-6:
        return y

    gain = target_rms / current_rms
    gain = float(np.clip(gain, 0.25, 4.0))
    print(f"  Gain-Staging: RMS {current_rms:.4f} → {target_rms:.4f}  "
          f"({20 * np.log10(gain):+.1f} dB)")

    return np.clip(y * gain, -1.0, 1.0)


# ---------------------------------------------------------------------------
# 2. Volume-Dip vor dem Drop
# ---------------------------------------------------------------------------

def apply_volume_dip(y: np.ndarray, sr: int,
                     drop_time: float,
                     dip_duration: float = 2.5,
                     dip_depth: float = 0.08) -> np.ndarray:
    """
    Baut einen Lautstärke-Dip VOR dem Drop ein:

      Volle Lautstärke ───╮              ╭─── Volle Lautstärke (Drop!)
                          │  dip_depth   │
                          ╰──────────────╯
                      dip_start      drop_time

    Parameters
    ----------
    drop_time    : Zeitpunkt des Main-Drops (Sekunden)
    dip_duration : Dauer der Absenkungs-Phase (Standard 2.5 s)
    dip_depth    : Tiefste Lautstärke (0.0 = lautlos, 0.08 = fast lautlos)
    """
    if drop_time is None or drop_time <= 0:
        print("  Volume-Dip: kein Drop-Zeitpunkt → übersprungen.")
        return y

    y = y.copy()
    dip_start_t = max(0.0, drop_time - dip_duration)
    start_s     = int(dip_start_t * sr)
    drop_s      = int(drop_time   * sr)
    n           = len(y)

    print(f"  Volume-Dip: {dip_start_t:.2f}s → {drop_time:.2f}s  "
          f"(Tiefe = {dip_depth:.0%})")

    for i in range(start_s, min(drop_s, n)):
        t = (i - start_s) / max(1, drop_s - start_s)
        if t < 0.80:
            # Sanfter Fade runter (80% der Zeit)
            t2 = t / 0.80
            envelope = 1.0 - (1.0 - dip_depth) * (3 * t2**2 - 2 * t2**3)
        else:
            # Kurzer Anstieg zurück zum Drop (letzte 20%)
            t2 = (t - 0.80) / 0.20
            envelope = dip_depth + (1.0 - dip_depth) * (3 * t2**2 - 2 * t2**3)
        y[i] *= float(envelope)

    return y


# ---------------------------------------------------------------------------
# 3. Spektrum-Frames für den Audio-Visualizer
# ---------------------------------------------------------------------------

def extract_spectrum_frames(audio_path: str,
                             fps: float,
                             num_bars: int = 24,
                             smoothing: float = 0.65) -> np.ndarray:
    """
    Extrahiert das Frequenzspektrum für jeden Video-Frame.

    Returns
    -------
    np.ndarray  shape (num_frames, num_bars)  –  Werte 0.0 bis 1.0
    """
    print(f"  Extrahiere Spektrum ({num_bars} Bars @ {fps:.1f} FPS)...")
    y, sr      = librosa.load(audio_path, sr=22050, mono=True)
    duration   = len(y) / sr
    num_frames = int(np.ceil(duration * fps)) + 1
    hop_length = max(1, int(sr / fps))
    n_fft      = 2048

    stft  = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

    # Logarithmische Frequenzaufteilung (Bass bekommt mehr Platz)
    low_hz, high_hz = 30.0, 8000.0
    edges   = np.logspace(np.log10(low_hz), np.log10(high_hz), num_bars + 1)
    bar_raw = np.zeros((stft.shape[1], num_bars), dtype=np.float32)

    for k in range(num_bars):
        mask = (freqs >= edges[k]) & (freqs < edges[k + 1])
        if mask.any():
            bar_raw[:, k] = np.mean(stft[mask, :], axis=0)

    # Logarithmische Amplitudenskalierung + Normalisierung
    bar_raw = np.log1p(bar_raw * 8.0)
    p99 = np.percentile(bar_raw, 99)
    if p99 > 0:
        bar_raw = bar_raw / p99
    bar_raw = np.clip(bar_raw, 0.0, 1.0)

    # Zeitliches Smoothing
    if smoothing > 0:
        smoothed    = np.zeros_like(bar_raw)
        smoothed[0] = bar_raw[0]
        for i in range(1, len(bar_raw)):
            smoothed[i] = smoothing * smoothed[i - 1] + (1.0 - smoothing) * bar_raw[i]
        bar_raw = smoothed

    # Auf genau num_frames interpolieren
    if bar_raw.shape[0] != num_frames:
        x_old  = np.linspace(0.0, 1.0, bar_raw.shape[0])
        x_new  = np.linspace(0.0, 1.0, num_frames)
        result = np.zeros((num_frames, num_bars), dtype=np.float32)
        for b in range(num_bars):
            result[:, b] = np.interp(x_new, x_old, bar_raw[:, b])
        bar_raw = result

    print(f"  Spektrum fertig: {bar_raw.shape[0]} Frames × {num_bars} Bars.")
    return bar_raw


# ---------------------------------------------------------------------------
# 4. Verarbeitetes Audio speichern
# ---------------------------------------------------------------------------

def save_processed_audio(y: np.ndarray, sr: int, out_path: str) -> str:
    """Speichert das verarbeitete Audio (Gain + Dip) als WAV-Datei."""
    sf.write(out_path, y, sr, subtype="PCM_16")
    print(f"  Verarbeitetes Audio → {out_path}")
    return out_path
