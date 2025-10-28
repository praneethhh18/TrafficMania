from typing import List, Tuple


class CentroidTracker:
    def __init__(self, max_distance: float = 50.0):
        self.next_id = 1
        self.objects = {}  # id -> (x, y)
        self.max_distance = max_distance

    def update(self, detections: List[Tuple[float,float,float,float,int,float]]):
        # detections: list of (x1,y1,x2,y2,cls,conf)
        centers = [(int((x1+x2)/2), int((y1+y2)/2)) for x1,y1,x2,y2,_,_ in detections]
        assigned = {}
        # Greedy nearest assignment
        for cid, (cx, cy) in list(self.objects.items()):
            best_j = -1
            best_d = self.max_distance
            for j, (ux, uy) in enumerate(centers):
                if j in assigned.values():
                    continue
                d = ((cx-ux)**2 + (cy-uy)**2) ** 0.5
                if d < best_d:
                    best_d = d
                    best_j = j
            if best_j >= 0:
                self.objects[cid] = centers[best_j]
                assigned[cid] = best_j
        # New tracks
        for j, c in enumerate(centers):
            if j not in assigned.values():
                self.objects[self.next_id] = c
                assigned[self.next_id] = j
                self.next_id += 1
        return assigned  # id -> index


def estimate_speed(prev_center, center, fps: float, ppm: float = 8.0) -> float:
    dx = center[0] - prev_center[0]
    dy = center[1] - prev_center[1]
    dist_pixels = (dx*dx + dy*dy) ** 0.5
    dist_meters = dist_pixels / ppm
    speed_mps = dist_meters * fps
    return speed_mps * 3.6
