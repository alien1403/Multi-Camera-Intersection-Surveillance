import os
import cv2
import numpy as np
from tqdm import tqdm
import torch
from ultralytics import YOLO
from filterpy.kalman import KalmanFilter


def make_kalman() -> KalmanFilter:
    kf = KalmanFilter(dim_x=8, dim_z=4)
    kf.F = np.eye(8) + np.diag([1, 1, 1, 1], k=4)
    kf.H = np.eye(4, 8)
    kf.P *= 1000.0
    kf.R = np.eye(4) * 10.0
    kf.Q = np.eye(8) * 0.1
    return kf


def kalman_init(kf: KalmanFilter, bbox_xyxy: np.ndarray):
    x, y, x2, y2 = bbox_xyxy
    w, h = x2 - x, y2 - y
    kf.x = np.array([x, y, w, h, 0, 0, 0, 0], dtype=np.float64)


def kalman_update(kf: KalmanFilter, bbox_xyxy: np.ndarray):
    x, y, x2, y2 = bbox_xyxy
    w, h = x2 - x, y2 - y
    kf.update(np.array([x, y, w, h], dtype=np.float64))


def kalman_predict_xyxy(kf: KalmanFilter) -> np.ndarray:
    kf.predict()
    x, y, w, h = kf.x[:4]
    return np.array([x, y, x + w, y + h])


def compute_iou(b1: np.ndarray, b2: np.ndarray) -> float:
    x1, y1 = max(b1[0], b2[0]), max(b1[1], b2[1])
    x2, y2 = min(b1[2], b2[2]), min(b1[3], b2[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    denom = a1 + a2 - inter
    return inter / denom if denom > 0 else 0.0


class SingleCameraTracker:
    VEHICLE_CLASSES = {2, 7}

    def __init__(self, yolo_model: YOLO, device):
        self.yolo = yolo_model
        self.device = device

    def track(
        self, video_path: str, initial_bbox: np.ndarray, frame_count: int
    ) -> list:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Cannot open {video_path}")
            return []

        kf = make_kalman()
        tracker = cv2.TrackerKCF_create()

        ret, first_frame = cap.read()
        if not ret:
            cap.release()
            return []

        x1, y1, x2, y2 = map(int, initial_bbox)
        tracker.init(first_frame, (x1, y1, x2 - x1, y2 - y1))
        kalman_init(kf, initial_bbox)

        results = [[0] + initial_bbox.tolist()]
        prev_bbox = initial_bbox.copy()
        missed = 0
        MAX_MISSED = 30
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        box_area = (x2 - x1) * (y2 - y1)
        iou_thresh = 0.25 if box_area < 5000 else 0.30

        with tqdm(
            total=frame_count, desc=f"  {os.path.basename(video_path)}", unit="fr"
        ) as pbar:

            for frame_idx in range(frame_count):
                ret, frame = cap.read()
                if not ret:
                    break

                current_bbox = None

                ok, kcf_box = tracker.update(frame)
                if ok:
                    fx, fy, fw, fh = map(int, kcf_box)
                    cand = np.array([fx, fy, fx + fw, fy + fh], dtype=float)
                    if compute_iou(cand, prev_bbox) >= iou_thresh:
                        current_bbox = cand

                if current_bbox is None or frame_idx % 5 == 0:
                    res = self.yolo(frame, conf=0.35, verbose=False)
                    boxes = res[0].boxes.xyxy.cpu().numpy()
                    classes = res[0].boxes.cls.cpu().numpy()

                    best_iou, best_box = -1, None
                    for box, cls in zip(boxes, classes):
                        if int(cls) not in self.VEHICLE_CLASSES:
                            continue
                        iou = compute_iou(box, prev_bbox)
                        if iou > best_iou and iou >= iou_thresh:
                            best_iou, best_box = iou, box

                    if best_box is not None:
                        current_bbox = best_box
                        missed = 0
                        fx1, fy1, fx2, fy2 = map(int, current_bbox)
                        tracker = cv2.TrackerKCF_create()
                        tracker.init(frame, (fx1, fy1, fx2 - fx1, fy2 - fy1))

                if current_bbox is None:
                    missed += 1
                    current_bbox = kalman_predict_xyxy(kf)
                else:
                    kalman_update(kf, current_bbox)

                if missed > MAX_MISSED:
                    pbar.update(frame_count - frame_idx)
                    break

                results.append([frame_idx] + current_bbox.tolist())
                prev_bbox = current_bbox
                pbar.update(1)

        cap.release()
        return results[1:]


def estimate_initial_bbox_camB(
    frame_a: np.ndarray,
    frame_b: np.ndarray,
    bbox_a: np.ndarray,
) -> np.ndarray | None:
    gray_a = cv2.cvtColor(frame_a, cv2.COLOR_BGR2GRAY)
    gray_b = cv2.cvtColor(frame_b, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create(2000)
    kp_a, des_a = orb.detectAndCompute(gray_a, None)
    kp_b, des_b = orb.detectAndCompute(gray_b, None)

    if des_a is None or des_b is None or len(kp_a) < 10 or len(kp_b) < 10:
        return None

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des_a, des_b)
    matches = sorted(matches, key=lambda m: m.distance)[:200]

    if len(matches) < 10:
        return None

    pts_a = np.float32([kp_a[m.queryIdx].pt for m in matches])
    pts_b = np.float32([kp_b[m.trainIdx].pt for m in matches])

    H, mask = cv2.findHomography(pts_a, pts_b, cv2.RANSAC, 5.0)
    if H is None or mask.sum() < 8:
        return None

    x1, y1, x2, y2 = bbox_a
    corners_a = np.float32([[x1, y1], [x2, y1], [x2, y2], [x1, y2]]).reshape(-1, 1, 2)
    corners_b = cv2.perspectiveTransform(corners_a, H).reshape(-1, 2)

    bx1 = max(0, int(corners_b[:, 0].min()))
    by1 = max(0, int(corners_b[:, 1].min()))
    bx2 = int(corners_b[:, 0].max())
    by2 = int(corners_b[:, 1].max())

    h, w = frame_b.shape[:2]
    bx2 = min(bx2, w - 1)
    by2 = min(by2, h - 1)

    if bx2 <= bx1 or by2 <= by1:
        return None

    return np.array([bx1, by1, bx2, by2], dtype=float)


class VehicleTracker:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.yolo_model = YOLO("yolo11m.pt").to(self.device)
        self.cam_tracker = SingleCameraTracker(self.yolo_model, self.device)
        os.makedirs("submission/task2", exist_ok=True)

    def _read_initial_bbox(self, txt_path: str) -> np.ndarray | None:
        try:
            with open(txt_path) as f:
                lines = f.readlines()
            row = lines[1].strip().replace("[", "").replace("]", "").split()
            return np.array([float(v) for v in row[1:]])
        except Exception as e:
            print(f"  Error reading {txt_path}: {e}")
            return None

    def _frame_count(self, path: str) -> int:
        cap = cv2.VideoCapture(path)
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.isOpened() else 0
        cap.release()
        return n

    def _first_frame(self, path: str) -> np.ndarray | None:
        cap = cv2.VideoCapture(path)
        ret, frame = cap.read()
        cap.release()
        return frame if ret else None

    def _save_results(self, path: str, frame_count: int, results: list):
        with open(path, "w") as f:
            f.write(f"{frame_count} -1 -1 -1 -1\n")
            for r in results:
                f.write(f"{int(r[0])} {r[1]:.0f} {r[2]:.0f} {r[3]:.0f} {r[4]:.0f}\n")

    def process_videos(self, input_dir: str):
        video_files = sorted(f for f in os.listdir(input_dir) if f.endswith("_1.mp4"))
        if not video_files:
            print(f"No *_1.mp4 files found in {input_dir}")
            return

        for vf in video_files:
            video_id = vf.replace("_1.mp4", "")
            path_a = os.path.join(input_dir, f"{video_id}_1.mp4")
            path_b = os.path.join(input_dir, f"{video_id}_2.mp4")
            annot_a = os.path.join(input_dir, f"{video_id}_1.txt")

            print(f"\n{'='*50}")
            print(f"Pair: {video_id}")

            bbox_a = self._read_initial_bbox(annot_a)
            if bbox_a is None:
                continue

            n_a = self._frame_count(path_a)
            print(f"Camera A: {n_a} frames")
            res_a = self.cam_tracker.track(path_a, bbox_a, n_a)
            res_a = [[0] + bbox_a.tolist()] + res_a
            self._save_results(
                f"submission/task2/{video_id}_1_predicted.txt", n_a, res_a
            )
            print(f"Saved Camera A ({len(res_a)} detections)")

            if not os.path.exists(path_b):
                print(f"Camera B not found, skipping.")
                continue

            frame_a0 = self._first_frame(path_a)
            frame_b0 = self._first_frame(path_b)
            bbox_b = None

            if frame_a0 is not None and frame_b0 is not None:
                bbox_b = estimate_initial_bbox_camB(frame_a0, frame_b0, bbox_a)

            if bbox_b is None:
                print("Homography failed – using Camera A bbox as Camera B init.")
                bbox_b = bbox_a.copy()

            n_b = self._frame_count(path_b)
            print(f"Camera B: {n_b} frames, init bbox: {bbox_b.astype(int).tolist()}")
            res_b = self.cam_tracker.track(path_b, bbox_b, n_b)
            res_b = [[0] + bbox_b.tolist()] + res_b
            self._save_results(
                f"submission/task2/{video_id}_2_predicted.txt", n_b, res_b
            )
            print(f"Saved Camera B ({len(res_b)} detections)")


def main():
    input_directory = "./data/train/task2"
    tracker = VehicleTracker()
    tracker.process_videos(input_directory)


if __name__ == "__main__":
    main()
