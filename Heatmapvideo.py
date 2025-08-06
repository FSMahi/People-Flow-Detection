import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict
from google.colab.patches import cv2_imshow

VIDEO_PATH = '/content/people-walking.mp4'
MODEL_PATH = '/content/yolov8n.pt'
OUTPUT_VIDEO_PATH = '/content/heatmap_video.mp4'


def main():
    # Load model with tracking
    model = YOLO(MODEL_PATH)

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print('Failed to open video file.')
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Create empty heatmap
    heatmap = np.zeros((height, width), dtype=np.float32)

    # Setup VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (width, height))

    frame_idx = 0
    # Process video frame by frame
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Run tracking on the current frame
        results = model.track(frame, persist=True, tracker="bytetrack.yaml", verbose=False)[0]

        if results.boxes is not None:
            boxes = results.boxes.xyxy.cpu().numpy()       # x1, y1, x2, y2
            ids = results.boxes.id.cpu().numpy().astype(int) if results.boxes.id is not None else None

            for box, track_id in zip(boxes, ids):
                x1, y1, x2, y2 = box.astype(int)
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                # Accumulate center on heatmap
                cv2.circle(heatmap, (cx, cy), radius=5, color=1, thickness=-1)

                # Draw bounding box with ID on the frame
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f'ID: {track_id}', (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Normalize and colorize heatmap for the current frame
        heatmap_img_frame = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
        heatmap_img_frame = np.uint8(heatmap_img_frame)
        heatmap_color_frame = cv2.applyColorMap(heatmap_img_frame, cv2.COLORMAP_JET)

        # Blend heatmap over the current frame
        overlay_frame = cv2.addWeighted(frame, 0.6, heatmap_color_frame, 0.4, 0)

        # Write the overlaid frame to the output video
        out.write(overlay_frame)

        frame_idx += 1
        if frame_idx % 50 == 0:
            print(f'Processed {frame_idx} frames')

    cap.release()
    out.release()
    print(f"✅ Saved video: {OUTPUT_VIDEO_PATH}")


if __name__ == '__main__':
    main()
