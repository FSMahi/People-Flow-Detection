import cv2
import numpy as np
from ultralytics import YOLO
import sys

# ==== CONFIG ====
SLOWDOWN_MS = 0
VIDEO_PATH = '/content/people-walking.mp4'
MODEL_PATH = '/content/yolov8n.pt'
OUTPUT_PATH = '/content/output_people_flow.mp4'

# Person class ID in COCO
PERSON_CLASS_ID = 0

# Line coordinates: [x1, y1], [x2, y2]
LINE_IN = np.array([[0, 218], [1967, 202]])      # Green line = IN
LINE_OUT = np.array([[0, 516], [1967, 504]])     # Red line = OUT


def ccw(A, B, C):
    """Check if points are in counter-clockwise order"""
    return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

def intersect(A, B, C, D):
    """Return True if line AB and CD intersect"""
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)


def main():
    try:
        model = YOLO(MODEL_PATH)
    except Exception as e:
        print("[ERROR] Could not load YOLO model:", e)
        sys.exit(1)

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print("Failed to open video.")
        return

    ret, frame = cap.read()
    if not ret:
        print("Failed to read first frame.")
        return

    height, width = frame.shape[:2]
    RESIZE_DIMS = (960, 540)

    # Setup VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)
    out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, RESIZE_DIMS)

    in_count = 0
    out_count = 0
    already_counted_in = set()
    already_counted_out = set()
    track_history = {}

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, RESIZE_DIMS)

        results = model.track(frame, persist=True, tracker="botsort.yaml", verbose=False)[0]
        boxes = results.boxes

        if boxes is None or boxes.xyxy is None:
            out.write(frame)
            continue

        class_ids = boxes.cls.cpu().numpy().astype(int)
        xyxy = boxes.xyxy.cpu().numpy()
        track_ids = boxes.id.cpu().numpy().astype(int) if boxes.id is not None else [None] * len(xyxy)

        person_indices = [i for i, cid in enumerate(class_ids) if cid == PERSON_CLASS_ID]

        # Draw IN and OUT lines
        cv2.line(frame, tuple(LINE_IN[0]), tuple(LINE_IN[1]), (0, 255, 0), 3)   # Green = IN
        cv2.line(frame, tuple(LINE_OUT[0]), tuple(LINE_OUT[1]), (0, 0, 255), 3) # Red = OUT

        for idx in person_indices:
            x1, y1, x2, y2 = map(int, xyxy[idx])
            track_id = track_ids[idx]

            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            center = (center_x, center_y)

            if track_id not in track_history:
                track_history[track_id] = []

            track_history[track_id].append(center)

            # Only consider last 2 positions
            if len(track_history[track_id]) > 2:
                track_history[track_id] = track_history[track_id][-2:]

            if len(track_history[track_id]) == 2:
                prev, curr = track_history[track_id]

                # IN line crossed (top to bottom)
                if (
                    intersect(prev, curr, LINE_IN[0], LINE_IN[1])
                    and (curr[1] > prev[1])  # downward movement
                    and track_id not in already_counted_in
                ):
                    in_count += 1
                    already_counted_in.add(track_id)

                # OUT line crossed (bottom to top)
                if (
                    intersect(prev, curr, LINE_OUT[0], LINE_OUT[1])
                    and (curr[1] < prev[1])  # upward movement
                    and track_id not in already_counted_out
                ):
                    out_count += 1
                    already_counted_out.add(track_id)

            # Draw bounding box and ID
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
            cv2.circle(frame, center, 5, (0, 255, 0), -1)
            cv2.putText(frame, f'ID {track_id}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Display counters
        cv2.putText(frame, f'IN: {in_count}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
        cv2.putText(frame, f'OUT: {out_count}', (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

        out.write(frame)

    cap.release()
    out.release()
    print(f"[✅ DONE] Output saved to: {OUTPUT_PATH}")


if __name__ == '__main__':
    main()
