import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pipeline_config import PipelineConfig
from main import run_pipeline

class SimRacingEditorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("AI Sim-Racing TikTok Editor")
        self.root.geometry("800x900")

        self.main_container = ttk.Frame(self.root, padding="10")
        self.main_container.pack(fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(self.main_container)
        self.scrollbar = ttk.Scrollbar(self.main_container, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = ttk.Frame(self.canvas)

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(
                scrollregion=self.canvas.bbox("all")
            )
        )

        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")

        self._build_ui()

    def _build_ui(self):
        # --- File Selection ---
        file_frame = ttk.LabelFrame(self.scrollable_frame, text="Dateiauswahl", padding="10")
        file_frame.pack(fill="x", pady=5)

        self.video_paths = []
        self.video_listbox = tk.Listbox(file_frame, height=4)
        self.video_listbox.pack(side="left", fill="x", expand=True)

        btn_frame = ttk.Frame(file_frame)
        btn_frame.pack(side="right", padx=5)
        ttk.Button(btn_frame, text="Videos hinzufügen", command=self._add_videos).pack(fill="x")
        ttk.Button(btn_frame, text="Entfernen", command=self._remove_video).pack(fill="x")

        self.audio_path_var = tk.StringVar()
        ttk.Label(file_frame, text="Musik (MP3/WAV):").pack(anchor="w")
        ttk.Entry(file_frame, textvariable=self.audio_path_var).pack(fill="x", side="left", expand=True)
        ttk.Button(file_frame, text="Durchsuchen", command=self._browse_audio).pack(side="right")

        self.output_path_var = tk.StringVar(value="output.mp4")
        ttk.Label(file_frame, text="Ausgabedatei:").pack(anchor="w", pady=(10, 0))
        ttk.Entry(file_frame, textvariable=self.output_path_var).pack(fill="x")

        # --- Presets & Mode ---
        preset_frame = ttk.LabelFrame(self.scrollable_frame, text="Modus & Presets", padding="10")
        preset_frame.pack(fill="x", pady=5)

        self.editing_mode_var = tk.StringVar(value="pro")
        ttk.Label(preset_frame, text="Modus:").grid(row=0, column=0, sticky="w")
        ttk.OptionMenu(preset_frame, self.editing_mode_var, "pro", "quick", "pro").grid(row=0, column=1, sticky="w")

        self.trend_preset_var = tk.StringVar(value="None")
        ttk.Label(preset_frame, text="Trend Preset:").grid(row=1, column=0, sticky="w")
        ttk.OptionMenu(preset_frame, self.trend_preset_var, "None", "None", "storytime", "motivation", "fast_meme_cut").grid(row=1, column=1, sticky="w")

        # --- Visual Effects ---
        fx_frame = ttk.LabelFrame(self.scrollable_frame, text="Visuelle Effekte", padding="10")
        fx_frame.pack(fill="x", pady=5)

        self.vars = {}
        effects = [
            ("visualizer", "Audio-Visualizer", True),
            ("use_jump_cut_burst", "Jump-Cut-Burst", True),
            ("use_speed_ramp", "Speed Ramp", True),
            ("use_reverse_clip", "Reverse Clip", True),
            ("use_white_flash", "White-Flash", True),
            ("use_freeze_frame", "Freeze Frame", True),
            ("use_overlap_transition", "Overlap Transition", True),
            ("use_text_mask", "Text-Mask", True),
            ("use_pip", "Picture-in-Picture", True),
            ("use_zoom_punch", "Zoom-Punch", True),
            ("use_glitch", "Glitch-Effekt", True),
            ("use_letterbox", "Cinematic Letterbox", True),
            ("use_blend_text", "Blend-Text", True),
            ("use_intro_text_sequence", "Intro Text-Sequenz", True),
            ("use_split_screen_glitch", "Split-Screen-Glitch", True),
            ("use_bw_intro", "Schwarz-Weiß Intro", False),
            ("text_mask_use_lyrics", "Text-Mask via Lyrics", False),
            ("lyrics_strict_mode", "Lyrics Strict Mode", True),
            ("grade_randomize", "Farbe randomisieren", True),
            ("gain_staging", "Gain-Staging", True),
            ("volume_dip", "Volume-Dip", True),
        ]

        row = 0
        col = 0
        for key, label, default in effects:
            var = tk.BooleanVar(value=default)
            self.vars[key] = var
            ttk.Checkbutton(fx_frame, text=label, variable=var).grid(row=row, column=col, sticky="w")
            col += 1
            if col > 2:
                col = 0
                row += 1

        # --- Advanced Settings ---
        adv_frame = ttk.LabelFrame(self.scrollable_frame, text="Erweiterte Einstellungen", padding="10")
        adv_frame.pack(fill="x", pady=5)

        self.grade_preset_var = tk.StringVar(value="teal_orange")
        ttk.Label(adv_frame, text="Grade Preset:").grid(row=0, column=0, sticky="w")
        ttk.OptionMenu(adv_frame, self.grade_preset_var, "teal_orange", "teal_orange", "cinematic", "neutral").grid(row=0, column=1, sticky="w")

        self.vignette_var = tk.DoubleVar(value=0.5)
        ttk.Label(adv_frame, text="Vignette:").grid(row=1, column=0, sticky="w")
        ttk.Scale(adv_frame, from_=0.0, to=1.0, variable=self.vignette_var, orient="horizontal").grid(row=1, column=1, sticky="ew")

        self.viz_bars_var = tk.IntVar(value=24)
        ttk.Label(adv_frame, text="Visualizer Bars:").grid(row=2, column=0, sticky="w")
        ttk.Entry(adv_frame, textvariable=self.viz_bars_var).grid(row=2, column=1, sticky="w")

        self.tm_word_var = tk.StringVar()
        ttk.Label(adv_frame, text="Text-Mask Wort:").grid(row=3, column=0, sticky="w")
        ttk.Entry(adv_frame, textvariable=self.tm_word_var).grid(row=3, column=1, sticky="ew")

        # --- Watermark ---
        wm_frame = ttk.LabelFrame(self.scrollable_frame, text="Wasserzeichen", padding="10")
        wm_frame.pack(fill="x", pady=5)

        self.wm_text_var = tk.StringVar()
        ttk.Label(wm_frame, text="Text:").grid(row=0, column=0, sticky="w")
        ttk.Entry(wm_frame, textvariable=self.wm_text_var).grid(row=0, column=1, sticky="ew")

        self.wm_opacity_var = tk.DoubleVar(value=0.4)
        ttk.Label(wm_frame, text="Deckkraft:").grid(row=1, column=0, sticky="w")
        ttk.Scale(wm_frame, from_=0.0, to=1.0, variable=self.wm_opacity_var, orient="horizontal").grid(row=1, column=1, sticky="ew")

        # --- Run options ---
        run_opt_frame = ttk.Frame(self.scrollable_frame, padding="10")
        run_opt_frame.pack(fill="x")

        self.preview_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(run_opt_frame, text="Vorschau-Modus (Schneller Export)", variable=self.preview_var).pack(side="left")

        self.no_cache_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(run_opt_frame, text="Cache ignorieren", variable=self.no_cache_var).pack(side="left", padx=20)

        # --- Run Button ---
        ttk.Button(self.scrollable_frame, text="EDITOR STARTEN", command=self._run).pack(pady=20, fill="x")

    def _add_videos(self):
        files = filedialog.askopenfilenames(filetypes=[("Video files", "*.mp4 *.mov *.avi *.mkv")])
        for f in files:
            if f not in self.video_paths:
                self.video_paths.append(f)
                self.video_listbox.insert(tk.END, os.path.basename(f))

    def _remove_video(self):
        selection = self.video_listbox.curselection()
        if selection:
            idx = selection[0]
            self.video_listbox.delete(idx)
            self.video_paths.pop(idx)

    def _browse_audio(self):
        f = filedialog.askopenfilename(filetypes=[("Audio files", "*.mp3 *.wav")])
        if f:
            self.audio_path_var.set(f)

    def _run(self):
        if not self.video_paths:
            messagebox.showerror("Fehler", "Bitte mindestens ein Video auswählen.")
            return
        if not self.audio_path_var.get():
            messagebox.showerror("Fehler", "Bitte eine Musikdatei auswählen.")
            return

        config = PipelineConfig(
            video_paths=self.video_paths,
            audio_path=self.audio_path_var.get(),
            output_path=self.output_path_var.get(),
            grade_preset=self.grade_preset_var.get(),
            grade_randomize=self.vars["grade_randomize"].get(),
            vignette_strength=self.vignette_var.get(),
            gain_staging=self.vars["gain_staging"].get(),
            volume_dip=self.vars["volume_dip"].get(),
            visualizer=self.vars["visualizer"].get(),
            visualizer_bars=self.viz_bars_var.get(),
            use_jump_cut_burst=self.vars["use_jump_cut_burst"].get(),
            use_speed_ramp=self.vars["use_speed_ramp"].get(),
            use_reverse_clip=self.vars["use_reverse_clip"].get(),
            use_white_flash=self.vars["use_white_flash"].get(),
            use_freeze_frame=self.vars["use_freeze_frame"].get(),
            use_overlap_transition=self.vars["use_overlap_transition"].get(),
            use_text_mask=self.vars["use_text_mask"].get(),
            text_mask_word=self.tm_word_var.get() if self.tm_word_var.get() else None,
            text_mask_use_lyrics=self.vars["text_mask_use_lyrics"].get(),
            use_pip=self.vars["use_pip"].get(),
            use_zoom_punch=self.vars["use_zoom_punch"].get(),
            use_glitch=self.vars["use_glitch"].get(),
            use_letterbox=self.vars["use_letterbox"].get(),
            use_blend_text=self.vars["use_blend_text"].get(),
            use_intro_text_sequence=self.vars["use_intro_text_sequence"].get(),
            lyrics_strict_mode=self.vars["lyrics_strict_mode"].get(),
            use_split_screen_glitch=self.vars["use_split_screen_glitch"].get(),
            use_bw_intro=self.vars["use_bw_intro"].get(),
            editing_mode=self.editing_mode_var.get(),
            trend_preset=self.trend_preset_var.get() if self.trend_preset_var.get() != "None" else None,
            watermark_text=self.wm_text_var.get() if self.wm_text_var.get() else None,
            watermark_opacity=self.wm_opacity_var.get(),
        )

        try:
            self.root.withdraw() # Hide GUI
            run_pipeline(config, preview=self.preview_var.get(), no_cache=self.no_cache_var.get())
            messagebox.showinfo("Erfolg", f"Video erfolgreich erstellt: {config.output_path}")
        except Exception as e:
            messagebox.showerror("Fehler", f"Ein Fehler ist aufgetreten: {e}")
        finally:
            self.root.deiconify() # Show GUI again

def run_gui():
    root = tk.Tk()
    app = SimRacingEditorGUI(root)
    root.mainloop()

if __name__ == "__main__":
    run_gui()
