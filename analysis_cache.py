import hashlib
import json
import os
from typing import Dict, List, Optional, Tuple

from audio_analyzer import SongSection, detect_song_sections, extract_beats
from video_analyzer import ClipInfo, find_highlights

_CACHE_VERSION = "v1"


def _cache_disabled() -> bool:
    return os.environ.get("KI_AUTO_DISABLE_CACHE", "").strip().lower() in ("1", "true", "yes", "on")


def _cache_dir() -> str:
    root = os.path.join(os.getcwd(), ".cache")
    os.makedirs(root, exist_ok=True)
    return root


def _file_fingerprint(path: str) -> Dict[str, object]:
    st = os.stat(path)
    return {
        "path": os.path.abspath(path),
        "size": int(st.st_size),
        "mtime_ns": int(st.st_mtime_ns),
    }


def _cache_path(namespace: str, payload: Dict[str, object]) -> str:
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=True)
    digest = hashlib.sha1(blob.encode("utf-8")).hexdigest()
    return os.path.join(_cache_dir(), f"{namespace}_{digest}.json")


def _load_json(path: str) -> Optional[Dict[str, object]]:
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _save_json(path: str, data: Dict[str, object]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=True, indent=2)


def _sections_to_json(sections: List[SongSection]) -> List[Dict[str, object]]:
    return [
        {
            "start": float(s.start),
            "end": float(s.end),
            "phase": s.phase,
            "energy": float(s.energy),
        }
        for s in sections
    ]


def _sections_from_json(payload: List[Dict[str, object]]) -> List[SongSection]:
    return [
        SongSection(
            start=float(item["start"]),
            end=float(item["end"]),
            phase=str(item["phase"]),
            energy=float(item["energy"]),
        )
        for item in payload
    ]


def _clips_to_json(clips: List[ClipInfo]) -> List[Dict[str, object]]:
    return [
        {
            "timestamp": float(c.timestamp),
            "score": float(c.score),
            "motion_score": float(c.motion_score),
            "drift_score": float(c.drift_score),
            "audio_score": float(c.audio_score),
            "telemetry_score": float(c.telemetry_score),
            "vehicle_count": float(c.vehicle_count),
            "cam_type": c.cam_type,
            "tag": c.tag,
            "source": c.source,
        }
        for c in clips
    ]


def _clips_from_json(payload: List[Dict[str, object]]) -> List[ClipInfo]:
    return [
        ClipInfo(
            timestamp=float(item["timestamp"]),
            score=float(item["score"]),
            motion_score=float(item["motion_score"]),
            drift_score=float(item["drift_score"]),
            audio_score=float(item["audio_score"]),
            telemetry_score=float(item["telemetry_score"]),
            vehicle_count=float(item["vehicle_count"]),
            cam_type=str(item["cam_type"]),
            tag=str(item["tag"]),
            source=str(item["source"]),
        )
        for item in payload
    ]


def get_audio_analysis_cached(audio_path: str) -> Tuple[List[float], List[float], Optional[float], List[SongSection]]:
    key = {
        "version": _CACHE_VERSION,
        "audio": _file_fingerprint(audio_path),
        "kind": "audio_analysis",
    }
    cache_off = _cache_disabled()
    path = _cache_path("audio_analysis", key)
    if not cache_off:
        cached = _load_json(path)
        if cached:
            print("  [CACHE] Audio-Analyse aus Cache geladen.")
            beats = [float(x) for x in cached.get("beats", [])]
            hard = [float(x) for x in cached.get("hard_beats", [])]
            main_drop_raw = cached.get("main_drop_time")
            main_drop = float(main_drop_raw) if main_drop_raw is not None else None
            sections = _sections_from_json(cached.get("sections", []))
            return beats, hard, main_drop, sections

    beats, hard_beats, main_drop_time = extract_beats(audio_path)
    sections = detect_song_sections(
        audio_path,
        main_drop_time=main_drop_time,
        beat_times=beats,
    )
    if not cache_off:
        _save_json(
            path,
            {
                "beats": beats,
                "hard_beats": hard_beats,
                "main_drop_time": main_drop_time,
                "sections": _sections_to_json(sections),
            },
        )
    return beats, hard_beats, main_drop_time, sections


def get_highlights_cached(
    video_paths: List[str],
    num_clips_total: int,
    clip_duration: float = 2.0,
) -> Dict[str, List[ClipInfo]]:
    n = len(video_paths)
    if n == 0:
        return {}

    clips_per_video = max(1, num_clips_total // n)
    remainder = num_clips_total - clips_per_video * n
    result: Dict[str, List[ClipInfo]] = {}

    for i, vp in enumerate(video_paths):
        extra = 1 if i < remainder else 0
        num = clips_per_video + extra
        key = {
            "version": _CACHE_VERSION,
            "video": _file_fingerprint(vp),
            "kind": "video_highlights",
            "num_clips": num,
            "clip_duration": clip_duration,
        }
        cache_off = _cache_disabled()
        path = _cache_path("video_highlights", key)
        if not cache_off:
            cached = _load_json(path)
            if cached:
                print(f"  [CACHE] Highlights geladen: {os.path.basename(vp)}")
                result[vp] = _clips_from_json(cached.get("clips", []))
                continue

        print(f"\n── Analysiere Quelle {i+1}/{n}: {os.path.basename(vp)} ({num} Highlights) ──")
        clips = find_highlights(vp, num_clips=num, clip_duration=clip_duration)
        result[vp] = clips
        if not cache_off:
            _save_json(path, {"clips": _clips_to_json(clips)})

    return result

