import scipy.signal
if not hasattr(scipy.signal, 'hann'):
    import scipy.signal.windows
    scipy.signal.hann = scipy.signal.windows.hann

import librosa
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Set


# ---------------------------------------------------------------------------
# Song-Abschnitt (Phase)
# ---------------------------------------------------------------------------

@dataclass
class SongSection:
    """Beschreibt einen zeitlichen Abschnitt im Song mit erkannter Phase."""
    start: float        # Startzeit in Sekunden
    end: float          # Endzeit in Sekunden
    phase: str          # 'intro' | 'verse' | 'buildup' | 'drop' | 'bridge' | 'outro'
    energy: float       # Normalisierte Energie 0.0–1.0

    @property
    def duration(self) -> float:
        return self.end - self.start

    def __repr__(self):
        return (f"SongSection({self.phase:8s} | "
                f"{self.start:6.2f}s–{self.end:6.2f}s | "
                f"energy={self.energy:.2f})")


# ---------------------------------------------------------------------------
# Beat-Extraktion
# ---------------------------------------------------------------------------

def extract_beats(audio_path: str):
    """
    Analysiert eine Audiodatei und gibt Timestamps für Beats,
    harte Beats (Drops/Kicks) und den Main Drop zurück.
    """
    print(f"Lade Audio für Analyse: {audio_path}...")

    y, sr = librosa.load(audio_path)

    print("Analysiere Beats...")
    # Get tempo and beat frames
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)

    # Convert frames to time (seconds)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)

    # Analyze onset strength for hard beats (Drops/Kicks)
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    valid_beat_frames = [f for f in beat_frames if f < len(onset_env)]
    beat_strengths = onset_env[valid_beat_frames]

    # Define a "hard beat" as being in the top 20% of beat strengths
    main_drop_time = None
    if len(beat_strengths) > 0:
        threshold = np.percentile(beat_strengths, 80)
        hard_beat_times = [beat_times[i] for i, strength in enumerate(beat_strengths) if strength >= threshold]

        # Das stärkste Event im Song (Main Drop)
        max_idx = np.argmax(beat_strengths)
        main_drop_time = beat_times[max_idx]
    else:
        hard_beat_times = []

    tempo_val = float(np.atleast_1d(tempo)[0])
    print(f"Erkanntes Tempo: {tempo_val:.2f} BPM")
    print(f"{len(beat_times)} Beats gefunden, davon {len(hard_beat_times)} harte Beats.")
    if main_drop_time:
        print(f"Main Drop erkannt bei {main_drop_time:.2f} Sekunden.")

    return list(beat_times), list(hard_beat_times), main_drop_time


# ---------------------------------------------------------------------------
# Song-Struktur-Erkennung
# ---------------------------------------------------------------------------

def detect_song_sections(audio_path: str,
                          main_drop_time: Optional[float] = None,
                          beat_times: Optional[List[float]] = None) -> List[SongSection]:
    """
    Erkennt die Makro-Struktur des Songs (Intro, Verse, Buildup, Drop, Bridge, Outro)
    anhand von RMS-Energie, Onset-Stärke und dem bekannten Main-Drop-Zeitpunkt.

    Gibt eine sortierte Liste von SongSection-Objekten zurück.
    """
    print("\nAnalysiere Song-Struktur (Phasen)...")

    y, sr = librosa.load(audio_path)
    duration = librosa.get_duration(y=y, sr=sr)

    # ── RMS-Energie-Verlauf berechnen ──────────────────────────────────────
    # Fenster: ~0.5s, Hop: ~0.1s → gute zeitliche Auflösung
    hop_length = int(sr * 0.1)
    frame_length = int(sr * 0.5)
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    rms_times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length)

    # Normalisieren
    rms_max = rms.max() if rms.max() > 0 else 1.0
    rms_norm = rms / rms_max

    # ── Onset-Dichte (Anzahl Onsets pro Zeitfenster) ───────────────────────
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    # Auf die gleiche Länge bringen wie rms
    min_len = min(len(rms_norm), len(onset_env))
    rms_norm  = rms_norm[:min_len]
    onset_env_norm = onset_env[:min_len] / (onset_env.max() + 1e-6)
    rms_times = rms_times[:min_len]

    # ── Energie-Score: Kombination aus RMS und Onset-Dichte ───────────────
    energy_score = 0.6 * rms_norm + 0.4 * onset_env_norm

    # ── Phasen-Grenzen ermitteln ───────────────────────────────────────────
    sections: List[SongSection] = []

    # Gleitender Durchschnitt (5s Fenster) für "Makro-Energie"
    window_frames = max(1, int(5.0 / 0.1))
    macro_energy = np.convolve(energy_score,
                               np.ones(window_frames) / window_frames,
                               mode='same')

    # Schwellwerte
    median_energy = np.median(macro_energy)
    high_thresh   = np.percentile(macro_energy, 75)
    low_thresh    = np.percentile(macro_energy, 30)

    # ── Intro erkennen ──────────────────────────────────────────────────────
    # Der Intro ist entweder die erste 15% des Songs ODER
    # der Bereich bevor die Energie erstmals die mittlere Schwelle überschreitet.
    intro_end = duration * 0.15
    # Suche: wann überschreitet macro_energy erstmals median_energy?
    first_high_idx = None
    for idx in range(len(macro_energy)):
        if macro_energy[idx] >= median_energy and rms_times[idx] > 3.0:
            first_high_idx = idx
            break

    if first_high_idx is not None:
        intro_end = min(intro_end, rms_times[first_high_idx])
        intro_end = max(intro_end, min(duration * 0.05, 5.0))  # mind. 5s oder 5%
    intro_end = round(intro_end, 2)

    # ── Outro erkennen ──────────────────────────────────────────────────────
    outro_start = duration * 0.88
    # Suche: letzte Energie-Senke vor Songende
    for idx in range(len(macro_energy) - 1, -1, -1):
        if macro_energy[idx] >= median_energy and rms_times[idx] < duration * 0.95:
            outro_start = min(rms_times[idx] + 2.0, duration * 0.90)
            break
    outro_start = round(outro_start, 2)

    # ── Drop / Buildup um main_drop_time ────────────────────────────────────
    drop_sections: List[tuple] = []  # (start, end)
    if main_drop_time is not None:
        # Drop-Fenster: ±4s um den Main-Drop (erster Drop)
        drop_start = max(intro_end, main_drop_time - 0.5)
        drop_end   = min(outro_start, main_drop_time + 8.0)

        # Buildup: 8–20s vor dem Drop, sofern Energie ansteigt
        buildup_start = max(intro_end, main_drop_time - 18.0)
        buildup_end   = drop_start

        if buildup_end - buildup_start >= 2.0:
            drop_sections.append(("buildup", buildup_start, buildup_end))

        drop_sections.append(("drop", drop_start, drop_end))

        # Weitere Drop-Wiederholungen: suche weitere Energie-Spitzen
        # nach dem ersten Drop (Chorus-Wiederholungen)
        search_start_t = drop_end
        while search_start_t < outro_start - 4.0:
            # Suche nächsten starken Energie-Peak im restlichen Song
            idx_start = np.searchsorted(rms_times, search_start_t)
            idx_end   = np.searchsorted(rms_times, outro_start)
            if idx_start >= idx_end:
                break
            window = macro_energy[idx_start:idx_end]
            if len(window) == 0:
                break
            local_peak_idx = idx_start + np.argmax(window)
            peak_time = rms_times[local_peak_idx]
            peak_val  = macro_energy[local_peak_idx]

            # Nur als weiteren Drop zählen wenn Energie hoch genug
            if peak_val >= high_thresh * 0.85:
                new_drop_start = max(search_start_t, peak_time - 0.5)
                new_drop_end   = min(outro_start, peak_time + 8.0)

                # Buildup vor diesem Drop?
                new_buildup_start = max(search_start_t, new_drop_start - 12.0)
                new_buildup_end   = new_drop_start
                if new_buildup_end - new_buildup_start >= 2.0:
                    drop_sections.append(("buildup", new_buildup_start, new_buildup_end))

                drop_sections.append(("drop", new_drop_start, new_drop_end))
                search_start_t = new_drop_end
            else:
                break

    # ── Alle Phasen zusammenbauen ────────────────────────────────────────────
    # Erst spezielle Phasen eintragen, dann Lücken mit verse/bridge füllen

    named: List[tuple] = [("intro", 0.0, intro_end)]
    named.extend(drop_sections)
    named.append(("outro", outro_start, duration))
    named.sort(key=lambda x: x[1])

    # Lücken füllen
    filled: List[tuple] = []
    prev_end = 0.0
    for phase, s, e in named:
        if s > prev_end + 0.5:
            # Lücke füllen: Bridge (wenn nach einem Drop) oder Verse
            gap_energy = float(np.mean(_energy_in_range(
                energy_score, rms_times, prev_end, s)))
            filler = "bridge" if any(p == "drop" for p, _, pe in filled if pe <= prev_end + 0.1) \
                     else "verse"
            filled.append((filler, round(prev_end, 2), round(s, 2)))
        filled.append((phase, round(s, 2), round(e, 2)))
        prev_end = e

    if prev_end < duration - 0.5:
        filled.append(("outro", round(prev_end, 2), round(duration, 2)))

    # ── SongSection-Objekte erzeugen ─────────────────────────────────────────
    for phase, s, e in filled:
        if e - s < 0.3:
            continue
        avg_energy = float(np.mean(_energy_in_range(energy_score, rms_times, s, e)))
        sections.append(SongSection(
            start=s, end=e, phase=phase, energy=avg_energy
        ))

    # ── Ausgabe ────────────────────────────────────────────────────────────
    print(f"\n  Erkannte Song-Phasen ({len(sections)}):")
    print(f"  {'Phase':<10}  {'Start':>6}  {'Ende':>6}  {'Dauer':>6}  {'Energie':>7}")
    print(f"  {'─'*48}")
    _PHASE_ICONS = {
        "intro":   "🎬",
        "verse":   "🎵",
        "buildup": "📈",
        "drop":    "💥",
        "bridge":  "🌉",
        "outro":   "🎤",
    }
    for sec in sections:
        icon = _PHASE_ICONS.get(sec.phase, "  ")
        print(f"  {icon} {sec.phase:<8}  {sec.start:>6.2f}s  {sec.end:>6.2f}s  "
              f"{sec.duration:>5.1f}s  {sec.energy:>6.2f}")
    print()

    return sections


def suggest_trend_preset(beat_times: List[float], sections: List[SongSection]) -> str:
    """
    Analysiert BPM und Energie, um ein passendes Trend-Preset vorzuschlagen.

    - Hohe BPM (>128) -> fast_meme_cut (schnelle Schnitte)
    - Niedrige Energie/Ruhig -> storytime (linearer, ruhiger)
    - Hohe Energie/Viel Drop -> motivation (effektreich)
    """
    if not beat_times:
        return "storytime"

    # BPM berechnen
    dur = beat_times[-1] - beat_times[0]
    bpm = (len(beat_times) / dur) * 60 if dur > 0 else 120

    # Durchschnittliche Energie der Drop-Phasen
    drop_energies = [s.energy for s in sections if s.phase == "drop"]
    avg_drop_energy = np.mean(drop_energies) if drop_energies else np.mean([s.energy for s in sections])

    if bpm > 132:
        return "fast_meme_cut"
    if avg_drop_energy > 0.75:
        return "motivation"
    if bpm < 100:
        return "storytime"

    return "motivation" # Default


def _energy_in_range(energy: np.ndarray, times: np.ndarray,
                      t_start: float, t_end: float) -> np.ndarray:
    """Gibt die Energie-Werte im Zeitbereich [t_start, t_end] zurück."""
    mask = (times >= t_start) & (times < t_end)
    vals = energy[mask]
    return vals if len(vals) > 0 else np.array([0.0])


# ---------------------------------------------------------------------------
# CutPoint – Ein geplanter Schnittpunkt
# ---------------------------------------------------------------------------

@dataclass
class CutPoint:
    """
    Repräsentiert einen einzelnen, geplanten Schnittpunkt im Edit.

    Felder:
        time            – Zeitpunkt im Song (Sekunden)
        beat_index      – Original-Beat-Index (aus beat_times)
        beat_type       – 'hard' | 'normal' | 'soft'
        phase           – Song-Phase ('intro', 'verse', …) oder None
        clip_dur_hint   – Empfohlene Clip-Länge (= Zeit bis nächsten Schnitt)
        is_forced       – True wenn durch harten Beat erzwungen
    """
    time: float
    beat_index: int
    beat_type: str          # 'hard' | 'normal' | 'soft'
    phase: Optional[str]    # 'intro' | 'verse' | 'buildup' | 'drop' | 'bridge' | 'outro'
    clip_dur_hint: float    # Sekunden bis zum nächsten Schnittpunkt
    is_forced: bool = False # True = harter Beat hat diesen Schnitt erzwungen

    def __repr__(self):
        forced = " [FORCED]" if self.is_forced else ""
        return (f"CutPoint({self.time:6.2f}s | {self.phase or '?':8s} | "
                f"{self.beat_type:6s} | dur={self.clip_dur_hint:.2f}s{forced})")


# ---------------------------------------------------------------------------
# Schnitt-Plan: build_cut_schedule()
# ---------------------------------------------------------------------------

# Beat-Stride pro Phase: (min_stride, max_stride)
# stride = wie viele Beats zwischen zwei Schnitten liegen
_PHASE_STRIDE: dict = {
    "intro":   (4, 6),   # ruhig – lange Kamerafahrten
    "verse":   (2, 4),   # normales Erzähltempo
    "buildup": (3, 1),   # Countdown: sinkend von 3→1 (Spannung aufbauen)
    "drop":    (1, 1),   # jeden Beat – maximale Energie
    "bridge":  (4, 8),   # dramatische Pause
    "outro":   (4, 6),   # auslaufend
}


def build_cut_schedule(
    beat_times: List[float],
    sections: List["SongSection"],
    hard_beat_times: List[float],
    audio_duration: float,
) -> List[CutPoint]:
    """
    Erstellt eine musikalisch-adaptive Liste von Schnittpunkten.

    Regeln:
      1. Harte Beats (Top-20% Onset) → immer Schnittpunkt, egal welche Phase
      2. Drop-Phase               → jeder Beat ist Schnittpunkt
      3. Buildup-Phase            → Stride schrumpft linear 3→1 (Countdown)
      4. Intro / Verse / Bridge / Outro → Stride aus _PHASE_STRIDE
      5. Minimum-Clip-Dauer: 0.12s (kein Schnitt wenn Abstand zum letzten < 0.12s)
    """
    if not beat_times:
        return []

    hard_set: Set[float] = set(round(b, 3) for b in hard_beat_times)

    # Hilfs-Lookup: phase-Start/-Ende für Buildup-Fortschritt
    buildup_sections = [(s.start, s.end) for s in sections if s.phase == "buildup"]

    def _get_phase(t: float) -> Optional[str]:
        for sec in sections:
            if sec.start <= t < sec.end:
                return sec.phase
        return sections[-1].phase if sections else None

    def _buildup_progress(t: float) -> float:
        """Fortschritt innerhalb der aktuellen Buildup-Phase [0.0–1.0]."""
        for bs, be in buildup_sections:
            if bs <= t < be:
                dur = be - bs
                return (t - bs) / dur if dur > 0 else 1.0
        return 0.0

    MIN_CUT_GAP = 0.12   # Sekunden – kein doppelter Schnitt kürzer als das

    cut_points: List[CutPoint] = []
    
    # ── Clip von t=0 bis zum ersten Beat (Verhindert Audio-Desync) ────────
    if beat_times[0] > 0.05:
        cut_points.append(CutPoint(
            time=0.0,
            beat_index=-1,
            beat_type="soft",
            phase=_get_phase(0.0),
            clip_dur_hint=beat_times[0],
            is_forced=False,
        ))

    last_cut_time: float = 0.0 if beat_times[0] > 0.05 else -9999.0
    # Stride-Zähler: wie viele Beats seit dem letzten Schnitt
    beats_since_cut: int = 0

    for idx, bt in enumerate(beat_times):
        is_hard = round(bt, 3) in hard_set
        phase   = _get_phase(bt)

        # ── Stride für diese Phase berechnen ─────────────────────────────
        if phase == "buildup":
            # Stride schrumpft linear von 3 → 1 über die Buildup-Dauer
            prog = _buildup_progress(bt)
            stride = max(1, round(3 - 2 * prog))   # 3 → 2 → 1
        elif phase == "drop":
            stride = 1  # immer schneiden
        else:
            lo, hi = _PHASE_STRIDE.get(phase or "verse", (2, 4))
            # Leichte Variation innerhalb des Fensters (nicht jedes Video klingt gleich)
            stride = lo if lo == hi else (lo + (idx % (hi - lo + 1)))

        # ── Entscheiden ob hier geschnitten wird ──────────────────────────
        gap = bt - last_cut_time
        must_cut = is_hard or (beats_since_cut >= stride)

        if must_cut and gap >= MIN_CUT_GAP:
            beat_type = "hard" if is_hard else ("normal" if idx % 2 == 0 else "soft")
            # clip_dur_hint wird unten nachgetragen
            cut_points.append(CutPoint(
                time=bt,
                beat_index=idx,
                beat_type=beat_type,
                phase=phase,
                clip_dur_hint=0.0,   # wird unten berechnet
                is_forced=is_hard and phase == "intro",
            ))
            last_cut_time = bt
            beats_since_cut = 0
        else:
            beats_since_cut += 1

    # ── clip_dur_hint nachfüllen ────────────────────────────────────────
    for k in range(len(cut_points)):
        if k + 1 < len(cut_points):
            cut_points[k].clip_dur_hint = cut_points[k + 1].time - cut_points[k].time
        else:
            # Letzter Schnitt → bis Songended
            cut_points[k].clip_dur_hint = max(0.1, audio_duration - cut_points[k].time)

    # ── Ausgabe ─────────────────────────────────────────────────────────
    print(f"\n  Schnitt-Plan: {len(cut_points)} Schnittpunkte "
          f"(aus {len(beat_times)} Beats gefiltert)")
    print(f"  {'Zeit':>6}  {'Phase':^8}  {'Typ':^6}  {'Dauer':>5}  Flags")
    print(f"  {'─'*46}")
    _ICONS = {"intro": "🎬", "verse": "🎵", "buildup": "📈",
              "drop": "💥", "bridge": "🌉", "outro": "🎤"}
    for cp in cut_points:
        icon  = _ICONS.get(cp.phase or "", "  ")
        flags = " ★HARD" if cp.beat_type == "hard" else ""
        flags += " [forced-intro]" if cp.is_forced else ""
        print(f"  {cp.time:>6.2f}s  {icon}{(cp.phase or '?'):^7}  "
              f"{cp.beat_type:^6}  {cp.clip_dur_hint:>4.2f}s{flags}")
    print()

    return cut_points
