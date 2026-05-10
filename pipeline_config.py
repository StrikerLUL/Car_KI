from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional


@dataclass
class PipelineConfig:
    video_paths: List[str]
    audio_path: str
    output_path: str
    grade_preset: str = "teal_orange"
    grade_randomize: bool = True
    vignette_strength: float = 0.50
    gain_staging: bool = True
    volume_dip: bool = True
    visualizer: bool = True
    visualizer_bars: int = 24
    use_jump_cut_burst: bool = True
    use_speed_ramp: bool = True
    use_reverse_clip: bool = True
    use_white_flash: bool = True
    use_freeze_frame: bool = True
    use_overlap_transition: bool = True
    use_text_mask: bool = True
    text_mask_word: Optional[str] = None
    text_mask_use_lyrics: bool = False
    use_pip: bool = True
    use_zoom_punch: bool = True
    use_glitch: bool = True
    use_camera_shake: bool = True
    use_mirror_x: bool = True
    use_letterbox: bool = True
    use_blend_text: bool = True
    use_intro_text_sequence: bool = True
    lyrics_strict_mode: bool = True
    use_split_screen_glitch: bool = True
    use_bw_intro: bool = False
    editing_mode: str = "pro"
    trend_preset: Optional[str] = None
    template_path: Optional[str] = None
    watermark_text: Optional[str] = None
    watermark_opacity: float = 0.4

    def validate(self) -> None:
        import os
        if not self.video_paths:
            raise ValueError("video_paths darf nicht leer sein.")
        for p in self.video_paths:
            if not os.path.exists(p):
                raise FileNotFoundError(f"Video-Datei nicht gefunden: {p}")
        if not os.path.exists(self.audio_path):
            raise FileNotFoundError(f"Audio-Datei nicht gefunden: {self.audio_path}")
        if not self.output_path.lower().endswith(".mp4"):
            raise ValueError("output_path muss auf '.mp4' enden.")
        if not (0.0 <= self.vignette_strength <= 1.0):
            raise ValueError(f"vignette_strength muss zwischen 0.0 und 1.0 liegen, ist aber {self.vignette_strength}.")
        if not (0.0 <= self.watermark_opacity <= 1.0):
            raise ValueError(f"watermark_opacity muss zwischen 0.0 und 1.0 liegen, ist aber {self.watermark_opacity}.")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
