import cv2
import numpy as np
import os
import torch
from transformers import AutoModel, AutoImageProcessor
from PIL import Image
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

BASE_PATH = "./data/train/task1/"
KEYFRAME_STEP = 2
SIMILARITY_THRESHOLD = 0.25
OUTPUT_DIR = "submission/task1"


def load_frames(path, step=KEYFRAME_STEP):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError(f"Failed to open video: {path}")
    frames, frame_indices = [], []
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % step == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame_rgb))
            frame_indices.append(frame_count)
        frame_count += 1
    cap.release()
    if not frames:
        raise ValueError(f"No frames read from {path}")
    return frames, frame_indices


def get_embeddings(frames, model, processor, device, batch_size=4):
    embeddings = []
    for i in range(0, len(frames), batch_size):
        batch = frames[i : i + batch_size]
        inputs = processor(images=batch, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            batch_emb = outputs.last_hidden_state[:, 0, :]
        embeddings.append(batch_emb.cpu())
    embeddings = torch.cat(embeddings, dim=0)
    embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
    return embeddings.numpy()


def align_keyframes(
    query_frames, ref_frames, q_indices, ref_indices, model, processor, device
):
    query_emb = get_embeddings(query_frames, model, processor, device)
    ref_emb = get_embeddings(ref_frames, model, processor, device)

    q_len = len(query_frames)
    r_len = len(ref_frames)

    best_offset_idx = 0
    best_score = -np.inf

    scores = np.zeros(r_len - q_len + 1)
    for offset in range(r_len - q_len + 1):
        sims = np.dot(query_emb, ref_emb[offset : offset + q_len].T).diagonal()
        scores[offset] = np.mean(sims)

    kernel = np.ones(3) / 3
    smoothed = np.convolve(scores, kernel, mode="same")

    top_candidates = np.argsort(smoothed)[::-1][:5]

    for cand in top_candidates:
        lo = max(0, cand - 2)
        hi = min(r_len - q_len, cand + 2)
        for offset in range(lo, hi + 1):
            sims = np.dot(query_emb, ref_emb[offset : offset + q_len].T).diagonal()
            weights = np.hanning(q_len) + 1e-6
            weights /= weights.sum()
            avg_sim = np.dot(weights, sims)
            if avg_sim >= SIMILARITY_THRESHOLD and avg_sim > best_score:
                best_score = avg_sim
                best_offset_idx = offset

    mid_q_frame = q_indices[q_len // 2]
    best_ref_frame = ref_indices[best_offset_idx]
    predicted_frame = best_ref_frame - mid_q_frame
    return predicted_frame


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model_name = "facebook/dinov2-small"
    print(f"Loading model: {model_name}")
    model = AutoModel.from_pretrained(model_name).to(device)
    processor = AutoImageProcessor.from_pretrained(model_name)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for pair_id in [f"{i:02d}" for i in range(1, 16)]:
        try:
            print(f"\nProcessing pair {pair_id}")
            query_path = os.path.join(BASE_PATH, f"{pair_id}_query.mp4")
            ref_path = os.path.join(BASE_PATH, f"{pair_id}_reference.mp4")

            query_frames, query_indices = load_frames(query_path)
            ref_frames, ref_indices = load_frames(ref_path)

            pred_frame = align_keyframes(
                query_frames,
                ref_frames,
                query_indices,
                ref_indices,
                model,
                processor,
                device,
            )

            pred_frame = max(0, pred_frame)
            output_file = os.path.join(OUTPUT_DIR, f"{pair_id}_predicted.txt")
            with open(output_file, "w") as f:
                f.write(str(pred_frame))
            print(f"Predicted start frame: {pred_frame}")

        except Exception as e:
            print(f"Error processing pair {pair_id}: {e}")
