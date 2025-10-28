import argparse
from typing import List, Optional

import cv2
import numpy as np
from ultralytics import YOLO

# PyTorch 2.6+ defaults to weights_only=True for torch.load which can break
# loading some third-party checkpoints. We trust official Ultralytics weights,
# so we patch torch.load to set weights_only=False by default here.
try:
    import torch  # noqa: E402
    _orig_torch_load = torch.load

    def _patched_torch_load(*args, **kwargs):
        kwargs.setdefault('weights_only', False)
        return _orig_torch_load(*args, **kwargs)

    torch.load = _patched_torch_load  # type: ignore[assignment]
except Exception:
    pass


def parse_args():
    ap = argparse.ArgumentParser(description="YOLOv8 vehicle detection and counting")
    ap.add_argument("--source", type=str, default="traffic.mp4", help="Video path, directory, image path, or '0' for webcam")
    ap.add_argument("--weights", type=str, default="yolov8n.pt", help="Path to YOLOv8 weights")
    ap.add_argument("--conf", type=float, default=0.35, help="Confidence threshold")
    ap.add_argument("--imgsz", type=int, default=640, help="Inference image size")
    ap.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"], help="Inference device")
    ap.add_argument("--classes", type=str, default="car,truck,bus,motorbike", help="Comma-separated class names to count")
    ap.add_argument("--show", action="store_true", help="Show a live window")
    ap.add_argument("--save", type=str, default=None, help="Optional output video file path to save annotated result")
    ap.add_argument("--max-frames", type=int, default=None, help="Process only N frames then exit (for quick tests)")
    return ap.parse_args()


def open_source(src: str):
    if src == "0":
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise RuntimeError(f"Error: Could not open source: {src}")
    return cap


def build_writer(save_path: str, cap: cv2.VideoCapture) -> Optional[cv2.VideoWriter]:
    if not save_path:
        return None
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    return cv2.VideoWriter(save_path, fourcc, fps, (w, h))


def main():
    args = parse_args()
    model = YOLO(args.weights)

    # Prepare class filter
    target_classes: List[str] = [c.strip() for c in args.classes.split(",") if c.strip()]

    # Open source and optional writer
    cap = open_source(args.source)
    writer = build_writer(args.save, cap)

    window_name = "YOLOv8 Vehicle Detection & Counting"
    if args.show:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    frame_i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # Run inference
        results = model.predict(frame, conf=args.conf, imgsz=args.imgsz, device=None if args.device == "auto" else args.device, verbose=False)
        res = results[0]

        # Count vehicles and annotate
        vehicle_count = 0
        per_class = {}
        for box in res.boxes:
            cls_id = int(box.cls[0])
            cls_name = model.names.get(cls_id, str(cls_id)) if isinstance(model.names, dict) else model.names[cls_id]
            if cls_name in target_classes:
                vehicle_count += 1
                per_class[cls_name] = per_class.get(cls_name, 0) + 1

        annotated = res.plot()  # ndarray BGR

        # Overlay text
        text = f"Vehicles: {vehicle_count}  " + ", ".join([f"{k}:{v}" for k, v in per_class.items()])
        cv2.putText(annotated, text, (18, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        if args.show:
            cv2.imshow(window_name, annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        if writer is not None:
            writer.write(annotated)

        frame_i += 1
        if args.max_frames is not None and frame_i >= args.max_frames:
            break

    cap.release()
    if writer is not None:
        writer.release()
    if args.show:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
