import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict
from google.colab.patches import cv2_imshow

VIDEO_PATH = '/content/people-walking.mp4'
MODEL_PATH = '/content/yolov8n.pt'

def main():
    # Load model with tracking
    model = YOLO(MODEL_PATH)

    # Enable tracker
    model.predict(source=VIDEO_PATH, stream=True, tracker="bytetrack.yaml")

    cap = cv2.VideoCapture(VIDEO_PATH)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Create empty heatmap
    heatmap = np.zeros((height, width), dtype=np.float32)

    frame_idx = 0
    for result in model.track(source=VIDEO_PATH, stream=True, tracker="bytetrack.yaml"):
        frame = result.orig_img.copy()

        if result.boxes is None:
            continue

        boxes = result.boxes.xyxy.cpu().numpy()       # x1, y1, x2, y2
        ids = result.boxes.id.cpu().numpy().astype(int) if result.boxes.id is not None else None

        for box, track_id in zip(boxes, ids):
            x1, y1, x2, y2 = box.astype(int)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            # Accumulate center on heatmap
            cv2.circle(heatmap, (cx, cy), radius=5, color=1, thickness=-1)

            # Draw bounding box with ID
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f'ID: {track_id}', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        frame_idx += 1
        if frame_idx % 50 == 0:
            print(f'Processed {frame_idx} frames')

    cap.release()

    # Normalize and colorize heatmap
    heatmap_img = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
    heatmap_img = np.uint8(heatmap_img)
    heatmap_color = cv2.applyColorMap(heatmap_img, cv2.COLORMAP_JET)

    # Blend heatmap over last frame (optional)
    overlay = cv2.addWeighted(frame, 0.6, heatmap_color, 0.4, 0)

    # Save results
    cv2.imwrite('/content/heatmap_only.jpg', heatmap_color)
    cv2.imwrite('/content/heatmap_overlay.jpg', overlay)
    print("✅ Saved: heatmap_only.jpg")
    print("✅ Saved: heatmap_overlay.jpg")

    # Display in Colab
    print("📌 Heatmap:")
    cv2_imshow(heatmap_color)
    print("📌 Overlay with last frame:")
    cv2_imshow(overlay)

if __name__ == '__main__':
    main()
