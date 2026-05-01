from __future__ import annotations
import cv2
import torch
from deep_sort_realtime.deepsort_tracker import DeepSort
from pathlib import Path

VIDEO_PATH  = "15sec_input_720p.mp4"        # <<<<<<  PUT YOUR VIDEO FILE HERE  <<<<<<
OUTPUT_PATH = "football_clip_tracked.mp4"
MODEL_NAME  = "yolov5s"                  # yolov5s / yolov5m / yolov8n etc.
DEVICE      = "cuda" if __import__('torch').cuda.is_available() else "cpu"  # auto-detect
MAX_AGE     = 45                         # frames to keep lost tracks (higher = more stable IDs)
DISPLAY     = True                       # set False for headless servers


print(" Loading YOLO model …")
model = torch.hub.load("ultralytics/yolov5", MODEL_NAME, pretrained=True)
model = model.to(DEVICE).eval()

print(" Initialising Deep SORT …")
tracker = DeepSort(max_age=MAX_AGE, embedder="mobilenet")

cap = cv2.VideoCapture(str(VIDEO_PATH))
if not cap.isOpened():
    raise FileNotFoundError(f"Cannot open video: {VIDEO_PATH}")

w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps   = cap.get(cv2.CAP_PROP_FPS) or 30
writer = cv2.VideoWriter(str(OUTPUT_PATH), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))


frame_idx = 0
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        #Detection
        results = model(frame, size=640)
        detections = results.xyxy[0].cpu().numpy()  # x1,y1,x2,y2,conf,cls

        #Prepare detections for Deep SORT (only class 0 = person)
        det_list = []
        for x1, y1, x2, y2, conf, cls in detections:
            if int(cls) != 0:
                continue
            l, t, w_box, h_box = x1, y1, x2 - x1, y2 - y1
            det_list.append(([l, t, w_box, h_box], float(conf), "person"))

        #Tracker step
        tracks = tracker.update_tracks(det_list, frame=frame)

        #Draw
        for track in tracks:
            if not track.is_confirmed():
                continue
            tid = track.track_id
            l, t, r, b = map(int, track.to_ltrb())
            cv2.rectangle(frame, (l, t), (r, b), (0, 255, 0), 2)
            cv2.putText(frame, f"ID {tid}", (l, max(15, t - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        writer.write(frame)
        if DISPLAY:
            cv2.imshow("Player Tracking", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("🛑 Stopped by user.")
                break

        frame_idx += 1
        if frame_idx % 100 == 0:
            print(f"Processed {frame_idx} frames …")
finally:
    cap.release()
    writer.release()
    cv2.destroyAllWindows()
    print(f"Done! Annotated video saved to {Path(OUTPUT_PATH).resolve()}")
