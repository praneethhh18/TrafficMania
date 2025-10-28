import argparse
import time
from datetime import datetime
import csv
import os
from typing import Optional

import cv2
import numpy as np

try:
    from ultralytics import YOLO
except Exception as e:
    YOLO = None

from data_pipeline.privacy import anonymize_frame
from data_pipeline.tracker import CentroidTracker, estimate_speed


DEFAULT_MODEL = 'yolov8n.pt'


"""
Collector uses YOLO to detect vehicles, a simple tracker to assign IDs,
and estimates speeds. Output is a compact CSV for downstream RL/state builders.
"""


def ensure_dir(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)


def collect(source: str, save_csv: str, model_path: Optional[str], display: bool, anonymize: bool, camera_id: str):
    if YOLO is None:
        raise RuntimeError('ultralytics not available; install requirements first')
    model = YOLO(model_path or DEFAULT_MODEL)
    cap = cv2.VideoCapture(0 if source == '0' else source)
    if not cap.isOpened():
        raise RuntimeError(f'Cannot open source {source}')

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    tracker = CentroidTracker()

    ensure_dir(save_csv)
    with open(save_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['timestamp','camera_id','cls','conf','track_id','speed_kmph'])

        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if anonymize:
                frame = anonymize_frame(frame)
            ts = datetime.utcnow().isoformat()
            # YOLO prediction
            results = model.predict(frame, verbose=False)[0]
            detections = []
            for b in results.boxes:
                x1,y1,x2,y2 = b.xyxy[0].tolist()
                conf = float(b.conf[0])
                cls = int(b.cls[0])
                # Only common vehicle classes (indices follow COCO for YOLOv8n)
                # car=2, motorcycle=3, bus=5, train=6, truck=7 (if trained on COCO)
                if conf < 0.35:
                    continue
                detections.append((x1,y1,x2,y2,cls,conf))
            assignments = tracker.update(detections)
            # rudimentary speed: compare current center with previous stored in tracker.objects
            for tid, j in assignments.items():
                x1,y1,x2,y2,cls,conf = detections[j]
                center = (int((x1+x2)/2), int((y1+y2)/2))
                prev = tracker.objects.get(tid, center)
                spd = estimate_speed(prev, center, fps)
                writer.writerow([ts, camera_id, cls, f"{conf:.3f}", tid, f"{spd:.2f}"])

            if display:
                for x1,y1,x2,y2,cls,conf in detections:
                    cv2.rectangle(frame, (int(x1),int(y1)), (int(x2),int(y2)), (0,255,0), 2)
                cv2.imshow('collector', frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    break
    cap.release()
    cv2.destroyAllWindows()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', type=str, default='0', help='0 for webcam or path to video')
    p.add_argument('--save-csv', type=str, default='data/events.csv')
    p.add_argument('--model', type=str, default=None, help='YOLO model path (default yolov8n.pt)')
    p.add_argument('--display', action='store_true')
    p.add_argument('--anonymize', action='store_true')
    p.add_argument('--camera-id', type=str, default='cam01')
    args = p.parse_args()
    collect(args.source, args.save_csv, args.model, args.display, args.anonymize, args.camera_id)


if __name__ == '__main__':
    main()
