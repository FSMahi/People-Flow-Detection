# 🧠 People Flow Detection using Object Tracking & Heatmap Visualization

This project implements a complete pipeline to detect, track, and count people entering or exiting a defined area in a video using YOLOv8 and ByteTrack. Additionally, it generates a heatmap showing areas of the most movement or presence intensity.

📎 **Sample Video Used**:  
[People Walking - Video Link](https://media.roboflow.com/supervision/video-examples/people-walking.mp4)

---

## 🎯 Objectives

- **Count people entering/exiting** based on their movement across two pre-defined lines.
- **Draw two horizontal lines** (IN and OUT) in the video frame.
- **Track each individual** using a unique ID across frames.
- **Visualize motion patterns** through a dynamic heatmap.

---

## ✅ Features

### 1. **People Detection**
- Uses the [YOLOv8](https://github.com/ultralytics/ultralytics) model to detect people (COCO class ID: 0).
- Bounding boxes are drawn for each detected person.

### 2. **Object Tracking**
- Uses the ByteTrack algorithm (`botsort.yaml`) to maintain consistent IDs for people across frames for people tracking.
- Uses the ByteTrack algorithm (`bytetrack.yaml`) to maintain consistent IDs for people across frames for heatmap generation.
- Each person is assigned a unique ID shown on their bounding box.

### 3. **People Counting Logic**
- Two horizontal lines are defined manually:
  - Green = IN (top line)
  - Red = OUT (bottom line)
- Tracks each person’s movement across frames.
- If a person crosses from top to bottom across the IN line → count as `IN`.
- If a person crosses from bottom to top across the OUT line → count as `OUT`.

### 4. **Heatmap Visualization**
- For every tracked person, their center position is recorded over time.
- These centers accumulate intensity on a heatmap.
- Heatmap is colorized using OpenCV's `COLORMAP_JET`.
- Two outputs:
  - `heatmap_only.jpg` - visual heatmap.
  - `heatmap_overlay.jpg` - heatmap blended with the final frame.

---

## 🗂️ File Structure

```bash
.
├── main_tracking_counting.py      # Tracking + IN/OUT counting code
├── heatmap_generator.py          # Heatmap generation code
├── yolov8n.pt                    # YOLOv8 model weights
├── people-walking.mp4            # Sample input video
├── output_people_flow.mp4        # Processed output video with tracking
├── heatmap_only.jpg              # Output heatmap
└── heatmap_overlay.jpg           # Overlay of heatmap on final frame
```

---

## 🚀 How to Run

### 1. Install Dependencies

```bash
pip install ultralytics opencv-python
```

### 2. Run Tracking + Counting

```bash
python main_tracking_counting.py
```

### 3. Run Heatmap Generation

```bash
python heatmap_generator.py
```

---

## 📊 Output Preview

- ✅ Bounding boxes with unique IDs
- ✅ Live `IN` and `OUT` counters
- ✅ Two horizontal lines (IN = green, OUT = red)
- ✅ Heatmap showing areas of highest presence/movement

---

## 🧪 Notes

- Line coordinates are customizable using tools like [PolygonZone](https://polygonzone.roboflow.com/).
- You can change the YOLO model by replacing `yolov8n.pt` with another variant (e.g., `yolov8s.pt`).

---

## 📌 License

This project is open-source and free to use under the MIT License.
