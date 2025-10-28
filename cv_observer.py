import argparse
import time
import yaml
import cv2
import numpy as np
import torch

from ultralytics import YOLO
from data_pipeline.tracker import CentroidTracker
from train_dqn_advanced import DuelingDQN, Device


def load_rois(yaml_path):
    with open(yaml_path, 'r') as f:
        cfg = yaml.safe_load(f)
    # cfg: { lanes: [ {name: str, polygon: [[x,y],...]} , ... ] }
    lanes = []
    for l in cfg.get('lanes', []):
        name = l['name']
        poly = np.array(l['polygon'], dtype=np.int32)
        lanes.append((name, poly))
    return lanes


def count_in_rois(detections, lanes):
    counts = [0]*len(lanes)
    for x1,y1,x2,y2,cls,conf in detections:
        cx, cy = int((x1+x2)/2), int((y1+y2)/2)
        for i, (_, poly) in enumerate(lanes):
            if cv2.pointPolygonTest(poly, (cx,cy), False) >= 0:
                counts[i] += 1
                break
    return counts


def run(source, rois_yaml, model_path, policy_path=None, ppm=8.0):
    cap = cv2.VideoCapture(0 if source == '0' else source)
    if not cap.isOpened():
        raise RuntimeError(f'Cannot open source {source}')
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    lanes = load_rois(rois_yaml)
    names = [n for n,_ in lanes]
    model = YOLO(model_path)
    tracker = CentroidTracker()

    policy = None
    obs_size = len(lanes) + 4  # [per-lane q] + total_q,total_w,phase_idx,time_since_switch
    if policy_path:
        policy = DuelingDQN(obs_size, 2).to(Device)
        policy.load_state_dict(torch.load(policy_path, map_location=Device))
        policy.eval()

    phase_idx = 0
    time_since_switch = 0
    min_green = 8

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        res = model.predict(frame, verbose=False)[0]
        dets = []
        for b in res.boxes:
            x1,y1,x2,y2 = b.xyxy[0].tolist()
            conf = float(b.conf[0])
            cls = int(b.cls[0])
            if conf < 0.35: continue
            dets.append((x1,y1,x2,y2,cls,conf))
        tracker.update(dets)
        # Build observation-like vector
        qs = count_in_rois(dets, lanes)
        total_q = float(sum(qs))
        total_w = 0.0  # unknown from camera; keep 0 or approximate by queue length
        obs = np.array(qs + [total_q, total_w, float(phase_idx), float(time_since_switch)], dtype=np.float32)

        action = 0
        if policy is not None:
            with torch.no_grad():
                q = policy(torch.tensor(obs, dtype=torch.float32, device=Device).unsqueeze(0))
                action = int(torch.argmax(q, dim=1).item())
        # Simple local timing without actuation: emulate decision interval and min green
        if action == 1 and time_since_switch >= min_green:
            phase_idx = (phase_idx + 1) % 2
            time_since_switch = 0
        else:
            time_since_switch += 1

        # Draw
        for name, poly in lanes:
            cv2.polylines(frame, [poly], isClosed=True, color=(255,0,0), thickness=2)
        cv2.putText(frame, f"phase={phase_idx} total_q={int(total_q)} action={action}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
        for x1,y1,x2,y2,cls,conf in dets:
            cv2.rectangle(frame, (int(x1),int(y1)), (int(x2),int(y2)), (0,255,0), 2)
        cv2.imshow('CV Observer', frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break
    cap.release()
    cv2.destroyAllWindows()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', type=str, default='0')
    ap.add_argument('--rois', type=str, required=True, help='YAML defining lane ROIs')
    ap.add_argument('--yolo', type=str, default='yolov8n.pt')
    ap.add_argument('--policy', type=str, default=None, help='Optional trained policy to suggest actions')
    args = ap.parse_args()
    run(args.source, args.rois, args.yolo, args.policy)


if __name__ == '__main__':
    main()
