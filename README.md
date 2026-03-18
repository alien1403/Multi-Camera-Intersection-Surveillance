# Multi-Camera Intersection Surveillance

A fully automatic computer vision pipeline for multi-camera analysis of urban intersections. Built for the *Beyond One View* project at the University of Bucharest, Faculty of Mathematics and Computer Science.

---

## Tasks

| # | Task | Method | Output |
|---|------|--------|--------|
| 1 | **Temporal Localisation** | DINOv2 embeddings + multi-scale sliding window | Frame offset (integer) |
| 2 | **Cross-View Vehicle Tracking** | YOLOv11m + KCF + per-track Kalman filter + homography-based Camera B init | Bounding box track per frame |
| 3 | **Directional Traffic Counting** | YOLOv11m + SORT-style MOT + zone-crossing logic | Vehicle count per oriented direction |

---

## Architecture

### Task 1 – Temporal Localisation
Determines the frame index in a 30-second reference video (Camera B) at which a 3-second query video (Camera A) begins.

- Extracts keyframes at every 2nd frame from both videos
- Computes CLS-token embeddings using **DINOv2-small** (`facebook/dinov2-small`)
- Slides the query embedding sequence over the reference sequence and scores each offset using **cosine similarity**
- Applies a Hanning-weighted mean to emphasise central frames and smooths scores with a 3-point moving average
- Refines the top-5 candidate offsets at single-keyframe resolution

### Task 2 – Cross-View Single Vehicle Tracking
Tracks a target vehicle through both Camera A and Camera B for the entire duration of each video.

- **Camera A**: Initialised from the provided bounding box; uses KCF for frame-to-frame tracking, YOLO for periodic re-detection (every 5 frames), and a per-track Kalman filter as fallback when both fail
- **Camera B**: Estimates the initial bounding box by computing an ORB-feature homography between the first frames of Camera A and Camera B, then projects the Camera A box through it; falls back to the raw Camera A box if homography fails
- Adaptive IoU threshold based on box area; stops emitting detections after 30 consecutive missed frames

### Task 3 – Directional Traffic Counting
Counts vehicles completing a specified directional trajectory `od ∈ {1→2, 1→3, 1→4, 2→1, 2→4, 3→1, 3→2, 4→2, 4→3}` across both video feeds.

- Detects all vehicles per frame with YOLOv11m (classes: car, truck)
- Associates detections across frames using a SORT-style multi-object tracker (IoU-based Hungarian matching + per-track Kalman filter)
- Defines 4 entry/exit zones (80 px margin) at the frame edges corresponding to the 4 canonical approach directions
- A vehicle counts when it enters through the origin zone and later exits through the destination zone with ≥5 confirmed detections
- Results from Camera A and Camera B are summed

---

## How to Run

### Task 1
```bash
python task1.py
```

### Task 2
```bash
python task2.py
```

### Task 3
```bash
python task3.py --input ../../train/task3/ --od "1->2"
```
Replace `1->2` with any valid oriented direction (e.g. `2->4`, `3->1`).

---

## Output Format

All results are saved in `submission/`.

**Task 1** — `submission/task1/{id}_predicted.txt`
```
427
```

**Task 2** — `submission/task2/{id}_1_predicted.txt` and `{id}_2_predicted.txt`
```
900 -1 -1 -1 -1
0 312 201 389 274
1 314 203 391 276
...
```

**Task 3** — `submission/task3/{id}_predicted.txt`
```
1->2
7
```