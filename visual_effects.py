"""
visual_effects.py – Professionelle visuelle Effekte für den TikTok-Editor

Enthält:
  - make_text_mask_clip()   → Video durch Buchstaben (\"TONIGHT\"-Style)
  - make_pip_overlay()      → Picture-in-Picture Overlay
  - make_zoom_punch()       → Ken-Burns Energie-Zoom
  - make_glitch_effect()    → Chromatic Aberration + Pixel-Verschiebung
  - apply_letterbox()       → Schwarze Balken oben/unten
  - pick_text_mask_word()   → Wort aus Musik-Metadaten
  - make_camera_shake()     → Wuchtiges Wackeln
  - make_mirror_x()         → Horizontale Spiegelung
"""

import os
import re
import random
import numpy as np
import cv2
from typing import Optional, Tuple, List


# ---------------------------------------------------------------------------
# Interne Hilfe: Pillow-Schriftart
# ---------------------------------------------------------------------------

def _get_pil_font(size: int):
    """Lädt Impact / Arial Bold via Pillow oder gibt None zurück."""
    try:
        from PIL import ImageFont
        candidates = [
            "C:/Windows/Fonts/impact.ttf",
            "C:/Windows/Fonts/ariblk.ttf",
            "C:/Windows/Fonts/arialbd.ttf",
        ]
        for fp in candidates:
            if os.path.exists(fp):
                return ImageFont.truetype(fp, size)
        return ImageFont.load_default()
    except Exception:
        return None


def _measure_text(text: str, font) -> Tuple[int, int]:
    try:
        from PIL import Image, ImageDraw
        img = Image.new("L", (1, 1))
        draw = ImageDraw.Draw(img)
        if hasattr(draw, "textbbox"):
            bb = draw.textbbox((0, 0), text, font=font)
            return bb[2] - bb[0], bb[3] - bb[1]
        return draw.textsize(text, font=font)
    except Exception:
        return len(text) * 22, 44


# ---------------------------------------------------------------------------
# 1. TEXT-MASK-CLIP  (Video durch Buchstaben)
# ---------------------------------------------------------------------------

def make_text_mask_clip(
    video_clip,
    text: str,
    duration: float = 1.6,
    fps: float = 60.0,
    fade_in: float = 0.12,
    fade_out: float = 0.20,
) -> Optional[object]:
    """
    Schwarzer Hintergrund + großer Text als Maske:
    Das Video ist NUR innerhalb der Buchstaben sichtbar.
    Gibt einen MoviePy-Clip zurück (selbe Auflösung).
    """
    try:
        from PIL import Image, ImageDraw, ImageFilter
        from moviepy.editor import VideoClip
    except ImportError:
        print("  ✗ Text-Mask: Pillow fehlt → pip install Pillow")
        return None

    try:
        w, h = video_clip.size

        # Schriftgröße automatisch auf ~85% der Breite anpassen
        font_size = int(h * 0.48)
        for _ in range(12):
            font = _get_pil_font(font_size)
            tw, _ = _measure_text(text, font)
            if tw < w * 0.76:
                font_size = int(font_size * 1.09)
            elif tw > w * 0.93:
                font_size = int(font_size * 0.93)
            else:
                break

        font = _get_pil_font(font_size)
        tw, th = _measure_text(text, font)
        tx, ty = (w - tw) // 2, (h - th) // 2

        # Buchstaben-Maske als numpy-Array [0..1]
        mask_img = Image.new("L", (w, h), 0)
        ImageDraw.Draw(mask_img).text((tx, ty), text, fill=255, font=font)
        mask_np = np.array(mask_img, dtype=np.float32) / 255.0

        # Leichter Glow: aufgeblasene Maske
        glow_img = mask_img.filter(ImageFilter.GaussianBlur(radius=14))
        glow_np  = np.array(glow_img, dtype=np.float32) / 255.0

        def _frame(t: float) -> np.ndarray:
            vt     = float(np.clip(t, 0.0, video_clip.duration - 1.0 / fps))
            vframe = video_clip.get_frame(vt).astype(np.float32)

            # Fade
            if t < fade_in:
                alpha = t / max(fade_in, 1e-6)
            elif t > duration - fade_out:
                alpha = (duration - t) / max(fade_out, 1e-6)
            else:
                alpha = 1.0
            alpha = float(np.clip(alpha, 0.0, 1.0))

            m      = mask_np[:, :, np.newaxis]          # (h,w,1)
            result = vframe * m * alpha                  # Video durch Buchstaben

            # Bläulicher Glow-Rand
            glow_col = np.array([160, 200, 255], dtype=np.float32)
            result  += glow_np[:, :, np.newaxis] * glow_col * 0.40 * alpha

            return np.clip(result, 0, 255).astype(np.uint8)

        clip = VideoClip(_frame, duration=duration).set_fps(fps)
        print(f"  [OK] Text-Mask: '{text}'  ({duration:.1f}s, {w}x{h})")
        return clip

    except Exception as e:
        print(f"  [ERR] Text-Mask: {e}")
        return None


# ---------------------------------------------------------------------------
# 2. PICTURE-IN-PICTURE
# ---------------------------------------------------------------------------

def make_pip_overlay(
    main_clip,
    pip_clip,
    position: str = "bottom_right",
    size_frac: float = 0.27,
    margin_frac: float = 0.03,
    border_color: Tuple[int, int, int] = (220, 220, 255),
    border_px: int = 3,
) -> object:
    """
    Legt pip_clip als kleines Fenster auf main_clip.
    position: 'top_left' | 'top_right' | 'bottom_left' | 'bottom_right'
    """
    try:
        from moviepy.editor import CompositeVideoClip

        mw, mh  = main_clip.size
        pip_w   = int(mw * size_frac)
        pip_h   = int(pip_w * mh / mw)
        margin  = int(mw * margin_frac)

        pos_map = {
            "top_left":     (margin,          margin),
            "top_right":    (mw - pip_w - margin, margin),
            "bottom_left":  (margin,          mh - pip_h - margin),
            "bottom_right": (mw - pip_w - margin, mh - pip_h - margin),
        }
        px, py = pos_map.get(position, pos_map["bottom_right"])

        pip_resized = pip_clip.resize((pip_w, pip_h))

        def _add_border(get_frame, t):
            frame = get_frame(t).copy()
            cv2.rectangle(frame, (0, 0), (pip_w - 1, pip_h - 1), border_color, border_px)
            return frame

        pip_bordered   = pip_resized.fl(_add_border, apply_to=["video"], keep_duration=True)
        pip_positioned = pip_bordered.set_position((px, py)).set_duration(main_clip.duration)

        return CompositeVideoClip([main_clip, pip_positioned], size=(mw, mh))

    except Exception as e:
        print(f"  [ERR] PiP: {e}")
        return main_clip


# ---------------------------------------------------------------------------
# 3. ZOOM-PUNCH  (Ken-Burns Energie-Zoom)
# ---------------------------------------------------------------------------

def make_zoom_punch(
    clip,
    zoom_start: float = 1.00,
    zoom_end: float   = 1.09,
) -> object:
    """Animierter Zoom-In über die Clip-Dauer (z.B. 1.0 → 1.09 = +9%)."""
    try:
        w, h = clip.size

        def _zoom_frame(get_frame, t):
            frame    = get_frame(t).copy()          # writable copy
            progress = (t / max(clip.duration, 1e-6)) ** 2   # ease-in
            zoom     = zoom_start + (zoom_end - zoom_start) * progress
            zoom     = max(zoom, 1.001)             # Mindest-Zoom > 1

            # Crop-Dimensionen: gerade Zahl, mindestens 2px
            nw = max(2, int(w / zoom))
            nh = max(2, int(h / zoom))
            nw -= nw % 2
            nh -= nh % 2

            x1 = max(0, (w - nw) // 2)
            y1 = max(0, (h - nh) // 2)
            x2 = min(w, x1 + nw)
            y2 = min(h, y1 + nh)

            cropped = frame[y1:y2, x1:x2]
            if cropped.size == 0:
                return frame
            # Sicherstellen dass dtype uint8 ist (MoviePy gibt manchmal int32)
            if cropped.dtype != np.uint8:
                cropped = np.clip(cropped, 0, 255).astype(np.uint8)
            try:
                return cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)
            except Exception:
                from PIL import Image as _Img
                pil = _Img.fromarray(cropped).resize((w, h), _Img.BILINEAR)
                return np.array(pil)

        return clip.fl(_zoom_frame, apply_to=["video"], keep_duration=True)

    except Exception as e:
        print(f"  [ERR] Zoom-Punch: {e}")
        return clip



# ---------------------------------------------------------------------------
# 4. GLITCH-FRAME  (Chromatic Aberration + Pixel-Verschiebung)
# ---------------------------------------------------------------------------

def make_glitch_effect(
    clip,
    intensity: float = 0.014,
    glitch_frames: int = 3,
) -> object:
    """
    Digitaler Glitch auf den ersten `glitch_frames` Frames:
    - RGB-Kanal-Verschiebung (Chromatic Aberration)
    - Zufällige horizontale Pixel-Streifen
    """
    try:
        w, h   = clip.size
        fps    = clip.fps or 60.0
        glitch_dur = glitch_frames / fps

        def _glitch_frame(get_frame, t):
            frame = get_frame(t).copy()
            if t > glitch_dur:
                return frame

            progress = 1.0 - (t / max(glitch_dur, 1e-6))
            sx       = max(1, int(w * intensity * progress))

            result = frame.copy()
            # Chromatic Aberration: R rechts, B links
            result[:, sx:,  0] = frame[:, :-sx, 0]
            result[:, :-sx, 2] = frame[:, sx:,  2]

            # Horizontale Verschiebungsstreifen
            for _ in range(random.randint(3, 7)):
                y0   = random.randint(0, h - 6)
                sh   = random.randint(2, 6)
                offs = random.randint(-int(w * 0.05), int(w * 0.05))
                if offs != 0:
                    strip = result[y0:y0 + sh, :, :].copy()
                    result[y0:y0 + sh, :, :] = np.roll(strip, offs, axis=1)

            return result

        return clip.fl(_glitch_frame, apply_to=["video"], keep_duration=True)

    except Exception as e:
        print(f"  [ERR] Glitch: {e}")
        return clip


# ---------------------------------------------------------------------------
# 5. CINEMATIC LETTERBOX
# ---------------------------------------------------------------------------

def apply_letterbox(clip, bar_fraction: float = 0.07) -> object:
    """Schwarze Balken oben + unten (Cinemascope-Look)."""
    try:
        w, h   = clip.size
        bar_h  = int(h * bar_fraction)

        def _lb_frame(get_frame, t):
            frame = get_frame(t).copy()
            frame[:bar_h, :]   = 0
            frame[h - bar_h:, :] = 0
            return frame

        return clip.fl(_lb_frame, apply_to=["video"], keep_duration=True)

    except Exception as e:
        print(f"  [ERR] Letterbox: {e}")
        return clip


# ---------------------------------------------------------------------------
# 6. MUSIK-WORT-EXTRAKTION  (für Text-Mask)
# ---------------------------------------------------------------------------

# Racing/Energie-Wörter als Fallback
_RACING_WORDS = [
    "DRIFT", "APEX", "SPEED", "PUSH", "BURN",
    "RUSH", "RACE", "BOOST", "FIRE", "HARD",
    "PEAK", "DROP", "GONE", "RAGE", "FULL",
    "FAST", "STORM", "KING", "PURE", "HEAT",
]

_WHISPER_WORDS = []

def transcribe_audio_for_words(audio_path: str) -> List[str]:
    """Transkribiert das Audio via Whisper und gibt geeignete Wörter zurück."""
    global _WHISPER_WORDS
    if _WHISPER_WORDS:
        return _WHISPER_WORDS
        
    try:
        import whisper
        import torch
        print("  [WHISPER] Lade KI-Sprachmodell (base) für Lyrics-Erkennung...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = whisper.load_model("base").to(device)
        print("  [WHISPER] Transkribiere Audio (das kann einen Moment dauern)...")
        result = model.transcribe(audio_path)
        
        stopwords = {"THE", "AND", "YOU", "FOR", "THAT", "THIS", "WITH", "OUT", "ALL",
                     "JUST", "YOUR", "ARE", "WAS", "HAVE", "FROM"}
        
        words = []
        for segment in result.get("segments", []):
            text = segment.get("text", "")
            for w in re.findall(r"[A-Za-z]+", text):
                w = w.upper()
                if 3 <= len(w) <= 8 and w not in stopwords:
                    words.append(w)
                    
        if words:
            print(f"  [WHISPER] {len(words)} geeignete Wörter gefunden!")
            _WHISPER_WORDS = words
        else:
            print("  [WHISPER] Keine passenden Wörter im Gesang gefunden.")
            
        return words
    except ImportError:
        print("  [ERR] openai-whisper ist nicht installiert. Nutze Fallback.")
    except Exception as e:
        print(f"  [ERR] Whisper Transkription fehlgeschlagen: {e}")
        
    return []


# ---------------------------------------------------------------------------
# Beat-synchrone Lyrics-Erkennung (Whisper word-level timestamps)
# ---------------------------------------------------------------------------

# Cache: {audio_path: [(start_sec, end_sec, word), ...]}
_WHISPER_TIMED_WORDS: dict = {}


def get_beat_synced_words(
    audio_path: str,
    beat_times: List[float],
    fallback_words: Optional[List[str]] = None,
    max_dist: float = 0.10,   # Strenger: lieber kein Wort als falsches Wort
    strict_mode: bool = True,
) -> List[str]:
    """
    Gibt für jeden Beat-Zeitpunkt das Wort zurück, das zum exakten Zeitpunkt
    im Song gesungen/gesprochen wird – via Whisper word-level Timestamps.

    Rückgabe: Liste mit len(beat_times) Einträgen (str).
    - Leerer String ("") wenn kein Wort nah genug am Beat liegt (Instrumental/Pause).
    - Falls Whisper nicht verfügbar → Fallback-Wörter aus Metadaten/Racing-Liste.

    Ablauf:
      1. Whisper transkribiert mit word_timestamps=True
      2. Für jeden Beat → finde das Wort das GERADE gesungen wird (beat im Zeitfenster)
      3. Falls kein aktives Wort: nimm das zeitlich nächste innerhalb max_dist Sekunden
      4. Falls kein Wort innerhalb max_dist → leerer String (kein Overlay!)
    """
    global _WHISPER_TIMED_WORDS

    if not beat_times:
        return []

    _fb = fallback_words if fallback_words else list(_RACING_WORDS)

    # ── Cache prüfen ──────────────────────────────────────────────────────────
    if audio_path in _WHISPER_TIMED_WORDS:
        timed = _WHISPER_TIMED_WORDS[audio_path]
    else:
        timed = []
        try:
            import whisper
            import torch
            print("  [WHISPER] Lade Modell für Beat-Sync Lyrics (word timestamps)...")
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = whisper.load_model("base").to(device)
            print("  [WHISPER] Transkribiere mit Word-Timestamps...")
            result = model.transcribe(audio_path, word_timestamps=True)

            stopwords = {"THE", "AND", "YOU", "FOR", "THAT", "THIS", "WITH",
                         "OUT", "ALL", "JUST", "YOUR", "ARE", "WAS", "HAVE",
                         "FROM", "ITS", "INTO", "LIKE", "BUT", "NOT", "WHEN"}

            for seg in result.get("segments", []):
                for w_info in seg.get("words", []):
                    raw = re.sub(r"[^A-Za-z]", "", w_info.get("word", "")).upper()
                    if len(raw) >= 2 and raw not in stopwords:
                        timed.append((
                            float(w_info.get("start", 0.0)),
                            float(w_info.get("end",   0.0)),
                            raw,
                        ))

            _WHISPER_TIMED_WORDS[audio_path] = timed
            print(f"  [WHISPER] {len(timed)} Wörter mit Timestamps erkannt")

        except ImportError:
            print("  [WHISPER] Nicht installiert – nutze Fallback-Wörter")
        except Exception as e:
            print(f"  [WHISPER] Fehler: {e} – nutze Fallback-Wörter")

    # ── Jedem Beat ein Wort zuordnen ──────────────────────────────────────────
    if not timed:
        # Kein Whisper/keine gültigen Timings
        if strict_mode:
            # Keine "erratenen" Wörter einblenden, damit nichts Off-Beat wirkt.
            print("  [BEAT-SYNC] Keine Whisper-Timestamps verfügbar – Text-Overlays werden übersprungen (strict).")
            return [""] * len(beat_times)
        print("  [BEAT-SYNC] Keine Whisper-Timestamps verfügbar – nutze Fallback-Wörter (loose).")
        return [_fb[i % len(_fb)] for i in range(len(beat_times))]

    result_words: List[str] = []
    matched = 0
    for bt in beat_times:
        # Schritt 1: Prüfe ob Beat GENAU in ein Wort-Zeitfenster fällt (aktives Wort)
        active_word = ""
        for w_start, w_end, w_text in timed:
            if w_start <= bt <= w_end:
                active_word = w_text
                break  # Erstes aktives Wort gewinnt

        if active_word:
            result_words.append(active_word)
            matched += 1
            continue

        # Schritt 2: Kein aktives Wort → suche das zeitlich nächste Wortfenster
        # Distanz wird zum Intervall [start, end] gemessen (nicht nur zur Mitte),
        # damit angrenzende Wörter zeitlich robuster zugeordnet werden.
        best_word = ""
        best_dist = float("inf")
        for w_start, w_end, w_text in timed:
            if bt < w_start:
                dist = w_start - bt
            elif bt > w_end:
                dist = bt - w_end
            else:
                dist = 0.0
            if dist < best_dist:
                best_dist = dist
                best_word = w_text

        # Leerer String wenn kein Wort nahe genug (Instrumental/Pause)
        if best_dist <= max_dist:
            result_words.append(best_word)
            matched += 1
        else:
            result_words.append("")   # → kein Overlay anzeigen!

    skipped = len(beat_times) - matched
    print(f"  [BEAT-SYNC] {matched}/{len(beat_times)} Beats mit Lyrics verknüpft"
          f"{f', {skipped} ohne Gesang (kein Overlay)' if skipped else ''}")
    return result_words



def extract_music_words(audio_path: str, use_lyrics: bool = False) -> List[str]:
    """
    Liest ID3-Tags (Titel + Artist) aus der Audiodatei.
    Gibt kurze Wörter (3–8 Buchstaben) zurück die als Text-Maske geeignet sind.
    Wenn use_lyrics=True, werden auch Songtexte (USLT/SYLT) durchsucht.
    Fallback: Racing-Wortliste.
    """
    words: List[str] = []
    combined = ""
    try:
        from mutagen import File as MutagenFile
        meta = MutagenFile(audio_path, easy=True)
        if meta:
            combined += " " + " ".join([
                meta.get("title",  [""])[0],
                meta.get("artist", [""])[0],
            ])

        if use_lyrics:
            # KI-Erkennung
            whisper_words = transcribe_audio_for_words(audio_path)
            if whisper_words:
                return whisper_words
                
            # Fallback auf ID3-Tags, falls Whisper fehlschlägt
            try:
                from mutagen.id3 import ID3
                tags = ID3(audio_path)
                for key in tags.keys():
                    if key.startswith("USLT") or key.startswith("SYLT"):
                        # tags[key].text returns string or list
                        val = tags[key].text
                        if isinstance(val, list):
                            val = " ".join(val)
                        combined += " " + str(val)
            except Exception:
                pass

        words = [w.upper() for w in re.findall(r"[A-Za-z]+", combined)
                 if 3 <= len(w) <= 8]
    except Exception:
        pass

    return words if words else list(_RACING_WORDS)


def pick_text_mask_word(audio_path: str, use_lyrics: bool = False) -> str:
    """Wählt zufällig ein passendes Wort aus den Musik-Metadaten/Lyrics."""
    return random.choice(extract_music_words(audio_path, use_lyrics=use_lyrics))


# ---------------------------------------------------------------------------
# 7. BLEND-TEXT-OVERLAY  (Screen/Multiply Blend-Mode – @editdd032-Stil)
# ---------------------------------------------------------------------------

def make_blend_text_overlay(
    clip,
    text: str,
    blend_mode: str = "screen",   # "screen" | "multiply" | "overlay"
    text_color: Tuple[int, int, int] = (255, 255, 255),
    alpha: float = 0.85,
    font_size_frac: float = 0.52,  # Schriftgröße relativ zur Höhe
) -> object:
    """
    Legt großen Text via Blend-Mode über den Clip.
    Im 'screen'-Modus scheinen die Farben des Footage durch die Buchstaben –
    exakt der Effekt aus @editdd032-Videos (lila/blau/türkis Töne).

    blend_mode:
      'screen'   – helle Bereiche des Textes lassen Footage durchscheinen (Standard)
      'multiply' – Text verdunkelt das Footage an den Buchstaben-Stellen
      'overlay'  – Mix aus beidem, kontrastreich
    """
    try:
        from PIL import Image, ImageDraw
        from moviepy.editor import VideoClip
    except ImportError:
        print("  ✗ Blend-Text: Pillow fehlt → pip install Pillow")
        return clip

    try:
        w, h = clip.size

        # Schriftgröße anpassen bis Text ~85% der Breite füllt
        font_size = int(h * font_size_frac)
        for _ in range(14):
            font = _get_pil_font(font_size)
            tw, _ = _measure_text(text, font)
            if tw < w * 0.75:
                font_size = int(font_size * 1.08)
            elif tw > w * 0.92:
                font_size = int(font_size * 0.94)
            else:
                break

        font = _get_pil_font(font_size)
        tw, th = _measure_text(text, font)
        tx, ty = (w - tw) // 2, (h - th) // 2

        # Text-Maske: weiße Buchstaben auf Schwarz
        mask_img = Image.new("RGB", (w, h), (0, 0, 0))
        draw = ImageDraw.Draw(mask_img)
        draw.text((tx, ty), text, fill=text_color, font=font)
        text_np = np.array(mask_img, dtype=np.float32) / 255.0  # (h,w,3)

        # Alpha-Maske (weiße Buchstaben = 1, Schwarz = 0)
        alpha_mask = np.max(text_np, axis=2, keepdims=True)  # (h,w,1)

        def _blend_frame(get_frame, t: float) -> np.ndarray:
            frame = get_frame(t).astype(np.float32) / 255.0  # (h,w,3)

            if blend_mode == "screen":
                # Screen: result = 1 - (1-a)*(1-b)
                blended = 1.0 - (1.0 - frame) * (1.0 - text_np)
            elif blend_mode == "multiply":
                blended = frame * (1.0 - text_np * 0.6)
            else:  # overlay
                low  = 2.0 * frame * text_np
                high = 1.0 - 2.0 * (1.0 - frame) * (1.0 - text_np)
                blended = np.where(frame < 0.5, low, high)

            # Nur wo Buchstaben sind wird geblended
            result = frame * (1.0 - alpha_mask * alpha) + blended * (alpha_mask * alpha)
            return np.clip(result * 255.0, 0, 255).astype(np.uint8)

        result_clip = clip.fl(_blend_frame, apply_to=["video"], keep_duration=True)
        print(f"  [OK] Blend-Text ({blend_mode}): '{text}'  ({w}x{h})")
        return result_clip

    except Exception as e:
        print(f"  [ERR] Blend-Text: {e}")
        return clip


# ---------------------------------------------------------------------------
# 8. TEXT-MASK-SEQUENZ  (Intro-Phase: mehrere Wörter schnell hintereinander)
# ---------------------------------------------------------------------------

def make_text_mask_sequence(
    video_clip,
    words: List[str],
    word_duration: float = 0.50,   # Sekunden pro Wort
    fps: float = 60.0,
    fade_in: float = 0.06,
    fade_out: float = 0.10,
) -> List[object]:
    """
    Erzeugt eine Liste von Text-Mask-Clips – jedes Wort als eigener Clip.
    Gibt eine Liste zurück die als Intro-Sequenz konkateniert werden kann.
    Stil: @azmiedtz03 – Video nur IN den riesigen Buchstaben auf Schwarz.
    """
    clips = []
    for word in words:
        c = make_text_mask_clip(
            video_clip,
            text=word,
            duration=word_duration,
            fps=fps,
            fade_in=fade_in,
            fade_out=fade_out,
        )
        if c is not None:
            clips.append(c)
    return clips


# ---------------------------------------------------------------------------
# 9. SPLIT-SCREEN-GLITCH  (3 vertikale Streifen zeitversetzt – @azmiedtz03-Ende)
# ---------------------------------------------------------------------------

def make_split_screen_glitch(
    clip,
    num_stripes: int = 3,
    max_offset_frac: float = 0.025,  # max horizontale Verschiebung (% der Breite)
    time_offset_frac: float = 0.04,  # Zeitversatz zwischen Streifen (% der Clip-Dauer)
) -> object:
    """
    Teilt das Bild in `num_stripes` vertikale Streifen.
    Jeder Streifen läuft leicht zeitversetzt und horizontal verschoben –
    klassischer Glitch-Paneel-Cut-Effekt wie am Ende von @azmiedtz03.
    """
    try:
        w, h = clip.size
        dur   = clip.duration
        fps   = clip.fps or 60.0

        # Statische Stripe-Parameter vorberechnen
        stripe_w = w // num_stripes
        # Horizontale Pixel-Offsets pro Streifen (fest, nicht random pro Frame)
        rng = np.random.default_rng(42)
        offsets = rng.integers(-int(w * max_offset_frac), int(w * max_offset_frac) + 1,
                               size=num_stripes).tolist()
        offsets[num_stripes // 2] = 0   # Mittel-Streifen bleibt zentriert
        t_offsets = [i * dur * time_offset_frac for i in range(num_stripes)]

        def _split_frame(get_frame, t: float) -> np.ndarray:
            result = np.zeros((h, w, 3), dtype=np.uint8)
            for i in range(num_stripes):
                # Zeitversatz anwenden
                src_t = float(np.clip(t - t_offsets[i], 0.0, dur - 1.0 / fps))
                src_frame = get_frame(src_t).copy()

                # Quell-Streifen aus dem Frame
                x1_src = i * stripe_w
                x2_src = x1_src + stripe_w if i < num_stripes - 1 else w
                stripe = src_frame[:, x1_src:x2_src, :]

                # Ziel-Position mit horizontalem Offset
                x1_dst = x1_src + offsets[i]
                x2_dst = x1_dst + stripe.shape[1]

                # Clamp auf Frame-Grenzen
                src_crop_l = max(0, -x1_dst)
                x1_dst = max(0, x1_dst)
                x2_dst = min(w, x2_dst)
                stripe_w_actual = x2_dst - x1_dst
                if stripe_w_actual <= 0:
                    continue
                result[:, x1_dst:x2_dst, :] = stripe[:, src_crop_l:src_crop_l + stripe_w_actual, :]

            return result

        return clip.fl(_split_frame, apply_to=["video"], keep_duration=True)

    except Exception as e:
        print(f"  [ERR] Split-Screen-Glitch: {e}")
        return clip


# ---------------------------------------------------------------------------
# 10. B&W OVERLAY  (Schwarz-Weiß – klinisch-cleaner Look wie @azmiedtz03)
# ---------------------------------------------------------------------------

def make_bw_overlay(clip, contrast_boost: float = 1.15) -> object:
    """
    Konvertiert den Clip zu Schwarz-Weiß via Luminanz-Formel.
    Optional mit leichtem Kontrast-Boost für den harten, cleanen B&W-Look.
    """
    try:
        def _bw_frame(get_frame, t: float) -> np.ndarray:
            frame = get_frame(t).astype(np.float32)
            # Luminanz (ITU-R BT.601)
            lum = (0.299 * frame[:, :, 0]
                 + 0.587 * frame[:, :, 1]
                 + 0.114 * frame[:, :, 2])
            # Kontrast-Boost um Mittelpunkt 128
            if contrast_boost != 1.0:
                lum = (lum - 128.0) * contrast_boost + 128.0
            lum = np.clip(lum, 0, 255).astype(np.uint8)
            return np.stack([lum, lum, lum], axis=-1)

        return clip.fl(_bw_frame, apply_to=["video"], keep_duration=True)

    except Exception as e:
        print(f"  [ERR] B&W Overlay: {e}")
        return clip


# ---------------------------------------------------------------------------
# 11. WATERMARK OVERLAY
# ---------------------------------------------------------------------------

def make_watermark_overlay(clip, text: str, opacity: float = 0.4) -> object:
    """
    Legt einen semi-transparenten Text-Watermark unten rechts über den Clip.
    """
    try:
        from PIL import Image, ImageDraw
        w, h = clip.size

        # Font-Größe basierend auf Video-Höhe (z.B. 3% der Höhe)
        font_size = max(20, int(h * 0.035))
        font = _get_pil_font(font_size)

        # Text-Maße
        tw, th = _measure_text(text, font)

        # Position: unten rechts mit Margin
        margin = int(w * 0.04)
        tx = w - tw - margin
        ty = h - th - margin

        # Watermark-Layer vorab erstellen (Performance!)
        wm_layer = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(wm_layer)
        draw.text((tx, ty), text, font=font, fill=(255, 255, 255, int(255 * opacity)))

        wm_np = np.array(wm_layer).astype(np.float32) / 255.0
        wm_rgb = wm_np[:, :, :3]
        wm_alpha = wm_np[:, :, 3:]

        def _filter(get_frame, t):
            frame = get_frame(t).astype(np.float32) / 255.0
            # Alpha-Blending: out = (1 - alpha) * frame + alpha * wm_rgb
            out = (1.0 - wm_alpha) * frame + wm_alpha * wm_rgb
            return (out * 255.0).astype(np.uint8)

        return clip.fl(_filter, apply_to=["video"], keep_duration=True)

    except Exception as e:
        print(f"  [ERR] Watermark: {e}")
        return clip

# ---------------------------------------------------------------------------
# 12. CAMERA SHAKE OVERLAY
# ---------------------------------------------------------------------------

def make_camera_shake(clip, intensity: float = 0.05, shake_frames: int = 5) -> object:
    """
    Simuliert einen wuchtigen Camera-Shake (z.B. bei einem Drop oder Crash).
    Verschiebt das Bild zufällig und zoomt leicht ein, um schwarze Ränder zu vermeiden.
    """
    try:
        w, h = clip.size
        fps = clip.fps or 60.0
        shake_dur = shake_frames / fps
        zoom = 1.0 + intensity * 2  # Etwas einzoomen, um Ränder beim Shaken zu verbergen

        # Random offsets vorab berechnen, damit es ruckelig wirkt
        rng = np.random.default_rng()
        max_offset_x = int(w * intensity)
        max_offset_y = int(h * intensity)

        # Generiere offsets für jeden der shake frames
        offsets = []
        for _ in range(shake_frames):
            ox = rng.integers(-max_offset_x, max_offset_x + 1)
            oy = rng.integers(-max_offset_y, max_offset_y + 1)
            offsets.append((ox, oy))

        def _shake_frame(get_frame, t: float) -> np.ndarray:
            frame = get_frame(t).copy()
            if t > shake_dur:
                return frame

            # Finde den aktuellen frame index
            frame_idx = min(int(t * fps), shake_frames - 1)
            ox, oy = offsets[frame_idx]

            # Zoom crop
            nw = max(2, int(w / zoom))
            nh = max(2, int(h / zoom))
            nw -= nw % 2
            nh -= nh % 2

            # Center crop + offset
            x1 = max(0, min(w - nw, (w - nw) // 2 + ox))
            y1 = max(0, min(h - nh, (h - nh) // 2 + oy))
            x2 = x1 + nw
            y2 = y1 + nh

            cropped = frame[y1:y2, x1:x2]
            if cropped.size == 0:
                return frame

            if cropped.dtype != np.uint8:
                cropped = np.clip(cropped, 0, 255).astype(np.uint8)

            try:
                import cv2
                return cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)
            except Exception:
                from PIL import Image as _Img
                pil = _Img.fromarray(cropped).resize((w, h), _Img.BILINEAR)
                return np.array(pil)

        return clip.fl(_shake_frame, apply_to=["video"], keep_duration=True)

    except Exception as e:
        print(f"  [ERR] Camera Shake: {e}")
        return clip

# ---------------------------------------------------------------------------
# 13. MIRROR X OVERLAY
# ---------------------------------------------------------------------------

def make_mirror_x(clip) -> object:
    """
    Spiegelt das Video horizontal (links <-> rechts).
    """
    try:
        def _mirror_frame(get_frame, t: float) -> np.ndarray:
            frame = get_frame(t)
            return np.fliplr(frame)

        return clip.fl(_mirror_frame, apply_to=["video"], keep_duration=True)
    except Exception as e:
        print(f"  [ERR] Mirror X: {e}")
        return clip
