# AI Sim-Racing TikTok Editor

An intelligent, AI-powered automated video editor designed to create high-energy Sim-Racing highlights for TikTok, Reels, and Shorts.

> ⚠️DISCLAIMER: This software is still under development and not yet finished. Therefore, it is restricted to strictly private use only. Commercial use, public distribution, or sharing on social media is prohibited unless you have explicit permission until the official full release.

## 🚀 Key Features

- **🤖 AI Video Analysis:** Uses **YOLOv8** for vehicle detection and **Optical Flow** (with CUDA GPU acceleration) to find the most exciting moments (drifts, overtakes, high-speed action).
- **🎶 Beat-Synchronized Editing:** Automatically analyzes audio to detect beats, drops, and song phases (Intro, Verse, Buildup, Drop) for perfect musical timing.
- **⚡ Parallel Processing:** Analyzes multiple video sources simultaneously to drastically reduce processing time.
- **🏎️ Auto-Pilot Mode:** Automatically suggests and applies the best editing style based on the music's BPM and energy.
- **🖥️ Graphical User Interface:** New Tkinter-based GUI for easy configuration and execution.
- **✨ Advanced Visual Effects:**
  - **Watermark:** Add a semi-transparent branding text to your videos.
  - **Text Masking:** Video playing through large typography.
  - **Glitch, Zoom Punches & Camera Shake:** Dynamic and impactful transitions on hard beats.
  - **Mirror X:** Randomly flip the video horizontally to keep the view fresh and engaging.
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

- `--gui`: Start the graphical user interface.
- `--preview`: Export a low-res (540p), 30fps preview quickly.
- `--mode [quick|pro]`: Choose between faster "quick" mode or more detailed "pro" analysis.
- `--preset [storytime|motivation|fast_meme_cut]`: Manually select a style (or let Auto-Pilot decide).
- `--watermark "TEXT"`: Add a watermark text to the final video.
- `--watermark-opacity 0.4`: Set the opacity of the watermark (0.0 to 1.0).
- `--clear-cache`: Deletes the `.cache` folder before starting.
- `--no-cache`: Disables reading/writing to cache for the current run.

## 🎨 Presets

- **Fast Meme Cut:** High-energy, rapid-fire cuts for high BPM tracks.
- **Motivation:** Dramatic effects and strong color grading for epic moments.
- **Storytime:** Longer cuts and cleaner transitions for a more cinematic feel.

---
*Developed for the ultimate Sim-Racing experience.*
