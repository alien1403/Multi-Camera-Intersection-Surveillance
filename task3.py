import os
import sys
import argparse
import cv2
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import torch
from ultralytics import YOLO
from scipy.optimize import linear_sum_assignment
from filterpy.kalman import KalmanFilter


class Track:
    _id_counter = 0

    def __init__(self, bbox: np.ndarray):
        Track._id_counter += 1
        self.id = Track._id_counter
        self.kf = self._make_kf(bbox)
        self.hits = 1
        self.age = 0
        self.history = [bbox.copy()]
        self.entry_zone: int | None = None
        self.exit_zone: int | None = None

    @staticmethod
    def _make_kf(bbox):
        kf = KalmanFilter(dim_x=8, dim_z=4)
        kf.F = np.eye(8) + np.diag([1, 1, 1, 1], k=4)
        kf.H = np.eye(4, 8)
        kf.P *= 10.0
        kf.R = np.eye(4) * 5.0
        kf.Q = np.eye(8) * 0.1
        x1, y1, x2, y2 = bbox
        w, h = x2 - x1, y2 - y1
        kf.x = np.array([x1, y1, w, h, 0, 0, 0, 0], dtype=np.float64)
        return kf

    def predict(self) -> np.ndarray:
        self.kf.predict()
        self.age += 1
        x, y, w, h = self.kf.x[:4]
        return np.array([x, y, x + w, y + h])

    def update(self, bbox: np.ndarray):
        x1, y1, x2, y2 = bbox
        self.kf.update(np.array([x1, y1, x2 - x1, y2 - y1], dtype=np.float64))
        self.hits += 1
        self.age = 0
        self.history.append(bbox.copy())

    @property
    def bbox(self) -> np.ndarray:
        x, y, w, h = self.kf.x[:4]
        return np.array([x, y, x + w, y + h])

    @property
    def center(self) -> np.ndarray:
        b = self.bbox
        return np.array([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2])


def iou_matrix(bboxes_a: np.ndarray, bboxes_b: np.ndarray) -> np.ndarray:
    N, M = len(bboxes_a), len(bboxes_b)
    mat = np.zeros((N, M))
    for i, a in enumerate(bboxes_a):
        for j, b in enumerate(bboxes_b):
            x1 = max(a[0], b[0])
            y1 = max(a[1], b[1])
            x2 = min(a[2], b[2])
            y2 = min(a[3], b[3])
            if x2 <= x1 or y2 <= y1:
                continue
            inter = (x2 - x1) * (y2 - y1)
            aa = (a[2] - a[0]) * (a[3] - a[1])
            ab = (b[2] - b[0]) * (b[3] - b[1])
            denom = aa + ab - inter
            mat[i, j] = inter / denom if denom > 0 else 0
    return mat


ZONE_MARGIN = 80


def get_zone(center: np.ndarray, frame_w: int, frame_h: int) -> int | None:
    cx, cy = center
    near_top = cy < ZONE_MARGIN
    near_bottom = cy > frame_h - ZONE_MARGIN
    near_left = cx < ZONE_MARGIN
    near_right = cx > frame_w - ZONE_MARGIN

    if near_top:
        return 1
    if near_right:
        return 2
    if near_bottom:
        return 3
    if near_left:
        return 4
    return None


class MOT:
    def __init__(
        self, max_age: int = 10, min_hits: int = 2, iou_threshold: float = 0.25
    ):
        self.tracks: list[Track] = []
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold

    def update(self, detections: np.ndarray, frame_w: int, frame_h: int) -> list[Track]:
        predicted = (
            np.array([t.predict() for t in self.tracks])
            if self.tracks
            else np.empty((0, 4))
        )

        matched, unmatched_dets, unmatched_trks = self._match(detections, predicted)

        for d_idx, t_idx in matched:
            self.tracks[t_idx].update(detections[d_idx])

        for d_idx in unmatched_dets:
            self.tracks.append(Track(detections[d_idx]))

        alive = []
        for i, trk in enumerate(self.tracks):
            zone = get_zone(trk.center, frame_w, frame_h)
            if zone is not None:
                if trk.entry_zone is None:
                    trk.entry_zone = zone
                trk.exit_zone = zone

            if trk.age <= self.max_age:
                alive.append(trk)

        self.tracks = alive

        return [t for t in self.tracks if t.hits >= self.min_hits]

    def _match(self, dets, preds):
        if len(preds) == 0 or len(dets) == 0:
            return [], list(range(len(dets))), list(range(len(preds)))

        iou_mat = iou_matrix(dets, preds)
        cost = 1 - iou_mat
        row_ind, col_ind = linear_sum_assignment(cost)

        matched, unmatched_d, unmatched_t = [], [], []
        matched_d_set, matched_t_set = set(), set()

        for r, c in zip(row_ind, col_ind):
            if iou_mat[r, c] >= self.iou_threshold:
                matched.append((r, c))
                matched_d_set.add(r)
                matched_t_set.add(c)

        unmatched_d = [i for i in range(len(dets)) if i not in matched_d_set]
        unmatched_t = [i for i in range(len(preds)) if i not in matched_t_set]
        return matched, unmatched_d, unmatched_t


VEHICLE_CLASSES = {2, 7}


def process_video(
    video_path: str, yolo: YOLO, src_zone: int, dst_zone: int
) -> set[int]:

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  Cannot open {video_path}")
        return set()

    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    mot = MOT(max_age=12, min_hits=2, iou_threshold=0.25)
    completions: set[int] = set()

    with tqdm(
        total=n_frames, desc=f"  {os.path.basename(video_path)}", unit="fr"
    ) as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            res = yolo(frame, conf=0.35, verbose=False)
            boxes = res[0].boxes.xyxy.cpu().numpy()
            classes = res[0].boxes.cls.cpu().numpy()

            dets = (
                np.array(
                    [
                        boxes[i]
                        for i in range(len(boxes))
                        if int(classes[i]) in VEHICLE_CLASSES
                    ]
                )
                if len(boxes) > 0
                else np.empty((0, 4))
            )

            active = mot.update(dets, frame_w, frame_h)

            for trk in active:
                if (
                    trk.id not in completions
                    and trk.entry_zone == src_zone
                    and trk.exit_zone == dst_zone
                    and trk.hits >= 5
                ):
                    completions.add(trk.id)

            pbar.update(1)

    for trk in mot.tracks:
        if (
            trk.id not in completions
            and trk.entry_zone == src_zone
            and trk.exit_zone == dst_zone
            and trk.hits >= 5
        ):
            completions.add(trk.id)

    cap.release()
    return completions


def parse_od(od_str: str):
    od_str = od_str.strip()
    for arrow in ["→", "->", ">>"]:
        od_str = od_str.replace(arrow, "|")
    od_str = od_str.replace(">", "|")
    parts = od_str.split("|")
    parts = [p.strip().lstrip("-") for p in parts if p.strip()]
    return int(parts[0]), int(parts[1])


def main():
    parser = argparse.ArgumentParser(
        description="Task 3 – Directional Traffic Counting"
    )
    parser.add_argument(
        "--input", default="./data/train/task3/", help="Directory with training pairs"
    )
    parser.add_argument("--od", default="1->2", help="Oriented direction, e.g. '1->2'")
    args = parser.parse_args()

    src_zone, dst_zone = parse_od(args.od)
    print(f"Counting vehicles: zone {src_zone} → zone {dst_zone}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    yolo = YOLO("yolo11m.pt").to(device)

    os.makedirs("submission/task3", exist_ok=True)

    input_dir = args.input
    video_pairs = sorted(
        set(f.split("_1.mp4")[0] for f in os.listdir(input_dir) if f.endswith("_1.mp4"))
    )

    if not video_pairs:
        print(f"No *_1.mp4 files found in {input_dir}")
        sys.exit(1)

    for pair_id in video_pairs:
        print(f"\n{'='*50}")
        print(f"Pair: {pair_id}")
        path_a = os.path.join(input_dir, f"{pair_id}_1.mp4")
        path_b = os.path.join(input_dir, f"{pair_id}_2.mp4")

        completions_a = process_video(path_a, yolo, src_zone, dst_zone)
        completions_b = set()
        if os.path.exists(path_b):
            completions_b = process_video(path_b, yolo, src_zone, dst_zone)

        total = len(completions_a) + len(completions_b)

        od_str = f"{src_zone}->{dst_zone}"
        out_path = f"submission/task3/{pair_id}_predicted.txt"
        with open(out_path, "w") as f:
            f.write(f"{od_str}\n{total}\n")

        print(
            f"Camera A: {len(completions_a)} | Camera B: {len(completions_b)} | Total: {total}"
        )


if __name__ == "__main__":
    main()
