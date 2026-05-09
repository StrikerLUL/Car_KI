from typing import List
import os
import sys
import argparse
import shutil
from audio_analyzer import build_cut_schedule
from video_editor import create_tiktok_edit, load_edit_template, save_edit_template
from analysis_cache import get_audio_analysis_cached, get_highlights_cached
from pipeline_config import PipelineConfig


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Intelligenter Simracing TikTok Editor",
    )
    parser.add_argument(
        "--clear-cache",
        action="store_true",
        help="Loescht den .cache-Ordner vor dem Start des Runs.",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Ignoriert Cache fuer diesen Run (ohne Cache-Dateien zu loeschen).",
    )
    parser.add_argument(
        "--mode",
        choices=["quick", "pro"],
        default="pro",
        help="Bedienmodus fuer den Edit-Builder.",
    )
    parser.add_argument(
        "--preset",
        choices=["storytime", "motivation", "fast_meme_cut"],
        default=None,
        help="Trend-Preset fuer Look und Effekte.",
    )
    parser.add_argument(
        "--template",
        type=str,
        default=None,
        help="JSON-Template mit Feature-Overrides laden.",
    )
    parser.add_argument(
        "--save-template",
        type=str,
        default=None,
        help="Aktuelle Einstellungen als JSON-Template speichern.",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Schnell-Export: Reduzierte Auflösung (540p), 30 FPS und schnellere Encodierung für schnelle Iteration.",
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        help="Startet die grafische Benutzeroberfläche.",
    )
    parser.add_argument(
        "--watermark",
        type=str,
        default=None,
        help="Wasserzeichen-Text unten rechts.",
    )
    parser.add_argument(
        "--watermark-opacity",
        type=float,
        default=0.4,
        help="Deckkraft des Wasserzeichens (0.0 - 1.0).",
    )
    return parser.parse_args()


def _clear_cache_if_requested(clear_cache: bool) -> None:
    if not clear_cache:
        return

    cache_dir = ".cache"
    if not os.path.exists(cache_dir):
        print("  [CACHE] Kein .cache-Ordner vorhanden, nichts zu loeschen.")
        return

    try:
        shutil.rmtree(cache_dir)
        print("  [CACHE] .cache wurde erfolgreich geloescht.")
    except Exception as e:
        print(f"  [CACHE] Konnte .cache nicht loeschen: {e}")
        sys.exit(1)


def _ask_video_paths() -> list:
    """
    Fragt den Benutzer nach einem oder mehreren Video-Pfaden.
    Eingabe beenden mit leerem Enter nach mindestens einem gültigen Pfad.
    """
    print()
    print("Bitte gib deine Video-Quellen ein.")
    print("  → Mehrere Videos: Drücke nach jedem Pfad Enter.")
    print("  → Fertig:         Einfach Enter ohne Eingabe (nach mind. 1 Video).")
    print()

    paths: List[str] = []
    while True:
        idx = len(paths) + 1
        prompt = f"  Video {idx} (oder Enter zum Beenden): " if paths else f"  Video {idx}: "
        raw = input(prompt).strip().strip('"\'')

        if raw == "":
            if paths:
                break
            print("  Bitte gib mindestens ein Video ein.")
            continue

        if not os.path.exists(raw):
            print(f"  ✗ Datei nicht gefunden: '{raw}' – bitte erneut versuchen.")
            continue

        paths.append(raw)
        label = f"Kamera {idx}" if len(paths) > 1 else "Hauptvideo"
        print(f"  ✓ {label} hinzugefügt: {os.path.basename(raw)}")

    return paths


def run_pipeline(config: PipelineConfig, preview: bool = False, no_cache: bool = False):
    if no_cache:
        os.environ["KI_AUTO_DISABLE_CACHE"] = "1"

    print("=" * 60)
    print("   Intelligenter Simracing TikTok Editor")
    print("=" * 60)

    os.makedirs(".cache", exist_ok=True)
    with open(os.path.join(".cache", "last_run_config.json"), "w", encoding="utf-8") as cfg_f:
        import json
        json.dump(config.to_dict(), cfg_f, ensure_ascii=True, indent=2)

    # ── Schritt 1: Audio analysieren ─────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Schritt 1: Audioanalyse (mit Cache) ...")
    beats, hard_beats, main_drop_time, song_sections = get_audio_analysis_cached(config.audio_path)

    # ── Schritt 1b: Auto-Pilot Preset Wahl ───────────────────────────────────
    trend_preset = config.trend_preset
    if trend_preset is None:
        from audio_analyzer import suggest_trend_preset
        auto_preset = suggest_trend_preset(beats, song_sections)
        print(f"\n  [AUTO-PILOT] Musik-Analyse empfiehlt Preset: '{auto_preset}'")
        trend_preset = auto_preset

    # ── Schritt 1c: Musik-adaptiven Schnitt-Plan erstellen ───────────────────
    print("\n" + "=" * 60)
    print("Schritt 1c: Erstelle adaptiven Schnitt-Plan (Phase-aware, nicht jeder Beat)...")
    import librosa
    _y, _sr = librosa.load(config.audio_path)
    audio_duration = librosa.get_duration(y=_y, sr=_sr)
    del _y  # RAM freigeben
    cut_schedule = build_cut_schedule(
        beat_times=beats,
        sections=song_sections,
        hard_beat_times=hard_beats,
        audio_duration=audio_duration,
    )

    num_highlights = min(60, max(40, len(beats)))

    # ── Schritt 2: Video(s) analysieren ──────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"Schritt 2: Analysiere Videos (YOLO + Optischer Fluss + Audio)")
    print(f"  Suche {num_highlights} Highlights – bitte etwas Geduld...")

    highlights = get_highlights_cached(
        config.video_paths,
        num_clips_total=num_highlights,
        clip_duration=2.0,
    )

    # ── Schritt 3: Edit erstellen ─────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Schritt 3: Erstelle den intelligenten TikTok-Edit...")

    template_overrides = None
    if config.template_path and os.path.exists(config.template_path):
        template_overrides = load_edit_template(config.template_path)

    stats = create_tiktok_edit(
        video_paths=config.video_paths,
        audio_path=config.audio_path,
        beat_times=beats,
        hard_beat_times=hard_beats,
        main_drop_time=main_drop_time,
        highlight_times=highlights,
        output_path=config.output_path,
        grade_preset=config.grade_preset,
        grade_randomize=config.grade_randomize,
        vignette_strength=config.vignette_strength,
        gain_staging=config.gain_staging,
        volume_dip=config.volume_dip,
        visualizer=config.visualizer,
        visualizer_bars=config.visualizer_bars,
        use_jump_cut_burst=config.use_jump_cut_burst,
        use_speed_ramp=config.use_speed_ramp,
        use_reverse_clip=config.use_reverse_clip,
        use_white_flash=config.use_white_flash,
        use_freeze_frame=config.use_freeze_frame,
        use_overlap_transition=config.use_overlap_transition,
        use_text_mask=config.use_text_mask,
        use_pip=config.use_pip,
        use_zoom_punch=config.use_zoom_punch,
        use_glitch=config.use_glitch,
        use_letterbox=config.use_letterbox,
        text_mask_word=config.text_mask_word,
        text_mask_use_lyrics=config.text_mask_use_lyrics,
        lyrics_strict_mode=config.lyrics_strict_mode,
        use_blend_text=config.use_blend_text,
        use_intro_text_sequence=config.use_intro_text_sequence,
        use_split_screen_glitch=config.use_split_screen_glitch,
        use_bw_intro=config.use_bw_intro,
        watermark_text=config.watermark_text,
        watermark_opacity=config.watermark_opacity,
        editing_mode=config.editing_mode,
        trend_preset=trend_preset,
        template_overrides=template_overrides,
        sections=song_sections,
        cut_schedule=cut_schedule,
        preview=preview,
    )

    print("\n" + "=" * 60)
    print(f"✓ Abgeschlossen! Video gespeichert: {os.path.abspath(config.output_path)}")
    if stats and "tag_stats" in stats:
        print(f"  Clip-Verteilung: {stats['tag_stats']}")
    print("=" * 60)


def main():
    args = _parse_args()

    if args.gui:
        try:
            from gui import run_gui
            run_gui()
            return
        except ImportError:
            print("Fehler: gui.py nicht gefunden oder tkinter fehlt.")
            sys.exit(1)

    if args.no_cache:
        os.environ["KI_AUTO_DISABLE_CACHE"] = "1"

    print("=" * 60)
    print("   Intelligenter Simracing TikTok Editor  v3")
    print("   (Optischer Fluss + Clip-Klassifizierung + Smart-Cut)")
    print("=" * 60)
    if args.no_cache:
        print("  [CACHE] --no-cache aktiv: Cache-Lesen/Schreiben ist fuer diesen Run deaktiviert.")
    print(f"  [MODE] {args.mode}")
    if args.preset:
        print(f"  [PRESET] {args.preset}")
    _clear_cache_if_requested(args.clear_cache)

    # ── Video-Eingabe ────────────────────────────────────────────────────────
    video_paths = _ask_video_paths()
    multi = len(video_paths) > 1
    if multi:
        print(f"\n  Multi-Kamera Modus: {len(video_paths)} Quellen werden intelligent kombiniert.")

    # ── Audio-Eingabe ────────────────────────────────────────────────────────
    print()
    audio_path = input("Pfad zur Musik (MP3/WAV): ").strip().strip('"\'')
    if not os.path.exists(audio_path):
        print(f"Fehler: Audiodatei '{audio_path}' nicht gefunden.")
        sys.exit(1)

    # ── Output ───────────────────────────────────────────────────────────────
    output_path = input("Name der fertigen Datei (z.B. output.mp4): ").strip().strip('"\'')
    if not output_path.endswith(".mp4"):
        output_path += ".mp4"

    # ── Farbkorrektur / Cinema-Grade ──────────────────────────────────────
    print()
    print("Cinema Color Grading:")
    print("  [1] Teal & Orange  (beliebtester Film-Look, Standard)")
    print("  [2] Cinematic      (kühle Schatten, warme Highlights, gehoben)")
    print("  [3] Neutral        (minimale Korrektur, kein Look)")
    grade_choice = input("  Auswahl [1]: ").strip()
    grade_preset = {"1": "teal_orange", "2": "cinematic", "3": "neutral"}.get(grade_choice, "teal_orange")
    print(f"  ✓ Grade-Preset: '{grade_preset}'")

    grade_randomize_raw = input("  Leichte Variation pro Clip? [J/n]: ").strip().upper()
    grade_randomize = grade_randomize_raw not in ("N", "NEIN", "NO", "0")

    def _ask_float(label, default):
        raw = input(f"  {label} [{default}]: ").strip()
        try:
            return float(raw) if raw else default
        except ValueError:
            print(f"  Ungültige Eingabe, nutze Standard ({default}).")
            return default

    vignette = _ask_float("  Vignette (0.0 = aus, 0.5 = Standard, 1.0 = max)", 0.50)

    # ── Audio-reaktive Effekte ───────────────────────────────────────────────
    print()
    print("Audio-reaktive Effekte (Enter = Standard übernehmen):")

    def _ask_bool(label, default=True):
        default_str = "J" if default else "N"
        raw = input(f"  {label} [{'J' if default else 'N'}]: ").strip().upper()
        if raw in ("J", "Y", "JA", "YES", "1"):
            return True
        if raw in ("N", "NO", "NEIN", "0"):
            return False
        return default

    use_gain_staging = _ask_bool("Gain-Staging an?  (Lautstärke automatisch normalisieren)", True)
    use_volume_dip   = _ask_bool("Volume-Dip an?    (Stille vor dem Drop → BOOM)",          True)
    use_visualizer   = _ask_bool("Audio-Visualizer an?  (Beat-Balken unten im Video)",        True)

    viz_bars = 24
    if use_visualizer:
        raw = input("  Anzahl Visualizer-Bars [24] (16 / 24 / 32): ").strip()
        if raw in ("16", "32"):
            viz_bars = int(raw)

    # ── Schnitt-Techniken ────────────────────────────────────────────────────
    print()
    print("Schnitt-Techniken (Enter = Standard übernehmen):")
    use_jump_cut_burst     = _ask_bool("Jump-Cut-Burst beim Drop?", True)
    use_speed_ramp         = _ask_bool("Speed Ramp (Slowmo/Zeitraffer)?", True)
    use_reverse_clip       = _ask_bool("Reverse Clip?", True)
    use_white_flash        = _ask_bool("White-Flash auf harten Beats?", True)
    use_freeze_frame       = _ask_bool("Freeze Frame?", True)
    use_overlap_transition = _ask_bool("Overlap Transition?", True)

    # ── Visuelle Effekte ─────────────────────────────────────────────────────
    print()
    print("🎬 Visuelle Effekte (Enter = Standard übernehmen):")
    use_text_mask  = _ask_bool("Text-Mask-Clip vor dem Drop? (Video durch Buchstaben)", True)
    text_mask_word = None
    text_mask_use_lyrics = False
    if use_text_mask:
        raw_word = input("  Text-Mask Wort (leer = aus Musik-Metadaten): ").strip().upper()
        text_mask_word = raw_word if raw_word else None
        if text_mask_word:
            print(f"  ✓ Text-Mask-Wort: '{text_mask_word}'")
        else:
            text_mask_use_lyrics = _ask_bool("  Soll das Text-Mask Wort automatisch mit KI (Whisper) aus dem Gesang erkannt werden?", True)
            print("  ✓ Wort wird automatisch aus dem Gesang oder Metadaten ermittelt.")
    use_pip        = _ask_bool("Picture-in-Picture? (2. Kamera als kleines Fenster)", True)
    use_zoom_punch = _ask_bool("Zoom-Punch auf harten Beats?", True)
    use_glitch     = _ask_bool("Glitch-Effekt auf harten Beats?", True)
    use_camera_shake = _ask_bool("Camera-Shake auf harten Beats?", True)
    use_mirror_x   = _ask_bool("Video horizontal spiegeln (Zufall)?", True)
    use_letterbox  = _ask_bool("Cinematic Letterbox? (schwarze Balken oben/unten)", True)

    print()
    print("🎨 Neue TikTok-Stil Effekte:")
    watermark_text = input("  Wasserzeichen Text (leer = aus): ").strip()
    watermark_text = watermark_text if watermark_text else None
    watermark_opacity = 0.4
    if watermark_text:
        watermark_opacity = _ask_float("  Wasserzeichen Deckkraft (0.0 - 1.0)", 0.4)

    use_blend_text         = _ask_bool("Blend-Text? (Screen-Modus, Footage scheint durch Text – @editdd032-Stil)", True)
    use_intro_text_sequence = _ask_bool("Intro Text-Sequenz? (3-5 Wörter schnell auf Schwarz vor erstem Beat – @azmiedtz03-Stil)", True)
    lyrics_strict_mode = _ask_bool("Lyrics Strict Mode? (nur exakte Wörter; sonst lockerer + Fallback)", True)
    use_split_screen_glitch = _ask_bool("Split-Screen-Glitch am Ende? (3 Streifen zeitversetzt)", True)
    use_bw_intro           = _ask_bool("Schwarz-Weiß in Intro-Phase? (klinisch-cleaner Look)", False)

    config = PipelineConfig(
        video_paths=video_paths,
        audio_path=audio_path,
        output_path=output_path,
        grade_preset=grade_preset,
        grade_randomize=grade_randomize,
        vignette_strength=vignette,
        gain_staging=use_gain_staging,
        volume_dip=use_volume_dip,
        visualizer=use_visualizer,
        visualizer_bars=viz_bars,
        use_jump_cut_burst=use_jump_cut_burst,
        use_speed_ramp=use_speed_ramp,
        use_reverse_clip=use_reverse_clip,
        use_white_flash=use_white_flash,
        use_freeze_frame=use_freeze_frame,
        use_overlap_transition=use_overlap_transition,
        use_text_mask=use_text_mask,
        text_mask_word=text_mask_word,
        text_mask_use_lyrics=text_mask_use_lyrics,
        use_pip=use_pip,
        use_zoom_punch=use_zoom_punch,
        use_glitch=use_glitch,
        use_camera_shake=use_camera_shake,
        use_mirror_x=use_mirror_x,
        use_letterbox=use_letterbox,
        use_blend_text=use_blend_text,
        use_intro_text_sequence=use_intro_text_sequence,
        lyrics_strict_mode=lyrics_strict_mode,
        use_split_screen_glitch=use_split_screen_glitch,
        use_bw_intro=use_bw_intro,
        watermark_text=watermark_text if watermark_text else args.watermark,
        watermark_opacity=watermark_opacity if watermark_text else args.watermark_opacity,
        editing_mode=args.mode,
        trend_preset=args.preset,
        template_path=args.template,
    )

    template_overrides = None
    if args.template:
        if not os.path.exists(args.template):
            print(f"Fehler: Template-Datei '{args.template}' nicht gefunden.")
            sys.exit(1)
        try:
            template_overrides = load_edit_template(args.template)
            print(f"  ✓ Template geladen: {args.template}")
        except Exception as e:
            print(f"Fehler: Template konnte nicht geladen werden: {e}")
            sys.exit(1)

    if args.save_template:
        try:
            save_payload = {
                "grade_preset": config.grade_preset,
                "grade_randomize": config.grade_randomize,
                "visualizer": config.visualizer,
                "visualizer_bars": config.visualizer_bars,
                "use_jump_cut_burst": config.use_jump_cut_burst,
                "use_speed_ramp": config.use_speed_ramp,
                "use_reverse_clip": config.use_reverse_clip,
                "use_white_flash": config.use_white_flash,
                "use_freeze_frame": config.use_freeze_frame,
                "use_overlap_transition": config.use_overlap_transition,
                "use_text_mask": config.use_text_mask,
                "use_pip": config.use_pip,
                "use_zoom_punch": config.use_zoom_punch,
                "use_glitch": config.use_glitch,
                "use_camera_shake": config.use_camera_shake,
                "use_mirror_x": config.use_mirror_x,
                "use_letterbox": config.use_letterbox,
                "use_blend_text": config.use_blend_text,
                "use_intro_text_sequence": config.use_intro_text_sequence,
                "use_split_screen_glitch": config.use_split_screen_glitch,
                "use_bw_intro": config.use_bw_intro,
                "visualizer_height": 0.13,
            }
            save_edit_template(args.save_template, save_payload)
            print(f"  ✓ Template gespeichert: {args.save_template}")
        except Exception as e:
            print(f"Fehler: Template konnte nicht gespeichert werden: {e}")
            sys.exit(1)

    run_pipeline(config, preview=args.preview, no_cache=args.no_cache)


if __name__ == "__main__":
    main()
