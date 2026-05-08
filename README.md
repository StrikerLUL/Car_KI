# AI Sim-Racing TikTok Editor

An intelligent, AI-powered automated video editor designed to create high-energy Sim-Racing highlights for TikTok, Reels, and Shorts.

> **⚠️ DISCLAIMER:** This software is intended for **strictly private use only**. It is not permitted to be used for commercial purposes, public distribution, or on social media platforms without explicit permission.

## 🚀 Key Features

- **🤖 AI Video Analysis:** Uses **YOLOv8** for vehicle detection and **Optical Flow** (with CUDA GPU acceleration) to find the most exciting moments (drifts, overtakes, high-speed action).
- **🎶 Beat-Synchronized Editing:** Automatically analyzes audio to detect beats, drops, and song phases (Intro, Verse, Buildup, Drop) for perfect musical timing.
- **⚡ Parallel Processing:** Analyzes multiple video sources simultaneously to drastically reduce processing time.
- **🏎️ Auto-Pilot Mode:** Automatically suggests and applies the best editing style based on the music's BPM and energy.
- **✨ Advanced Visual Effects:**
  - **Text Masking:** Video playing through large typography.
  - **Glitch & Zoom Punches:** Dynamic transitions on hard beats.
  - **Split-Screen Glitch:** Stylish multi-panel endings.
  - **Audio-Reactive Visualizer:** Real-time beat bars.
- **🎨 Professional Color Grading:** Built-in cinema looks (Teal & Orange, Cinematic) with GPU-accelerated processing.
- **🎥 Multi-Camera Support:** Intelligently switches between multiple camera angles to keep the edit dynamic.
- **⏩ Ultra-Fast Preview:** New `--preview` mode for lightning-fast exports at reduced resolution to iterate quickly.

## 🛠 Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd car-ki
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   *Note: For GPU acceleration, ensure you have NVIDIA drivers and CUDA installed.*

3. **(Optional) Download YOLO weights:**
   The script will automatically download `yolov8n.pt` on the first run if not present.

## 📖 Usage

Run the main script to start the interactive editor:

```bash
python main.py
```

### Command Line Arguments

- `--preview`: Export a low-res (540p), 30fps preview quickly.
- `--mode [quick|pro]`: Choose between faster "quick" mode or more detailed "pro" analysis.
- `--preset [storytime|motivation|fast_meme_cut]`: Manually select a style (or let Auto-Pilot decide).
- `--clear-cache`: Deletes the `.cache` folder before starting.
- `--no-cache`: Disables reading/writing to cache for the current run.

## 🎨 Presets

- **Fast Meme Cut:** High-energy, rapid-fire cuts for high BPM tracks.
- **Motivation:** Dramatic effects and strong color grading for epic moments.
- **Storytime:** Longer cuts and cleaner transitions for a more cinematic feel.

---
*Developed for the ultimate Sim-Racing experience.*
