import argparse
import json
from typing import List, Tuple, Optional

import cv2
import numpy as np
from ultralytics import YOLO

# Simple polygon contains test
def point_in_poly(x: float, y: float, poly: List[Tuple[float, float]]) -> bool:
    return cv2.pointPolygonTest(np.array(poly, dtype=np.int32), (int(x), int(y)), False) >= 0


def parse_args():
    p = argparse.ArgumentParser(description="YOLO-based ambulance lane priority observer (demo)")
    p.add_argument("--source", type=str, default="0", help="Video source (file path or '0' for webcam)")
    p.add_argument("--weights", type=str, default="yolov8n.pt", help="YOLOv8 weights path")
    p.add_argument("--classes", type=str, default="car,truck,bus", help="Comma-separated class names to consider")
    p.add_argument("--conf", type=float, default=0.35, help="Confidence threshold")
    p.add_argument("--imgsz", type=int, default=640, help="Inference image size")
    p.add_argument("--device", type=str, default="auto", choices=["auto","cpu","cuda"], help="Device")
    p.add_argument("--config", type=str, default="camera_rois.json", help="JSON with lane ROIs and lane_ids")
    p.add_argument("--out-file", type=str, default="ambulance_requests.txt", help="File to write lane_ids for priority")
    p.add_argument("--show", action="store_true", help="Show annotated window")
    p.add_argument("--treat-all-as-ambulance", action="store_true", help="Demo mode: treat any vehicle in ROI as ambulance")
    return p.parse_args()


def load_config(path: str):
    with open(path, "r") as f:
        cfg = json.load(f)
    # expect {"lanes": [{"lane_id": "<id>", "polygon": [[x,y], ...]}, ...]}
    lanes = cfg.get("lanes", [])
    rois = []
    for ln in lanes:
        lane_id = ln.get("lane_id")
        poly = ln.get("polygon")
        if lane_id and isinstance(poly, list) and len(poly) >= 3:
            rois.append({"lane_id": lane_id, "polygon": [(float(x), float(y)) for x,y in poly]})
    return rois


def open_source(src: str) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(0) if src == "0" else cv2.VideoCapture(src)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open source: {src}")
    return cap


def main():
    args = parse_args()
    model = YOLO(args.weights)
    rois = load_config(args.config)
    target_classes = [c.strip() for c in args.classes.split(',') if c.strip()]

    cap = open_source(args.source)
    win = "Ambulance Observer"
    if args.show:
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        H, W = frame.shape[:2]

        results = model.predict(frame, conf=args.conf, imgsz=args.imgsz, device=None if args.device=="auto" else args.device, verbose=False)
        res = results[0]

        # collect candidate points per ROI
        lane_hits = set()
        for box in res.boxes:
            cls_id = int(box.cls[0])
            cls_name = model.names.get(cls_id, str(cls_id)) if isinstance(model.names, dict) else model.names[cls_id]
            if cls_name not in target_classes:
                continue
            xyxy = box.xyxy[0].tolist()  # [x1,y1,x2,y2]
            cx = 0.5 * (xyxy[0] + xyxy[2])
            cy = 0.5 * (xyxy[1] + xyxy[3])
            for roi in rois:
                if point_in_poly(cx, cy, roi["polygon"]):
                    lane_hits.add(roi["lane_id"])

        # Demo mode: treat any vehicle in ROI as ambulance
        if lane_hits and args.treat_all_as_ambulance:
            try:
                with open(args.out_file, 'a') as f:
                    for ln in lane_hits:
                        f.write(f"{ln}\n")
            except Exception:
                pass

        # annotate
        if args.show:
            for roi in rois:
                poly = np.array(roi["polygon"], dtype=np.int32)
                color = (0,255,0) if roi["lane_id"] in lane_hits else (0,0,255)
                cv2.polylines(frame, [poly], isClosed=True, color=color, thickness=2)
                # draw label at polygon centroid
                M = cv2.moments(poly)
                if M["m00"] != 0:
                    cx = int(M["m10"]/M["m00"]); cy = int(M["m01"]/M["m00"])
                    cv2.putText(frame, roi["lane_id"], (cx-20, cy-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            cv2.imshow(win, frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    if args.show:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
