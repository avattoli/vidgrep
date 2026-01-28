import argparse
import contextlib
import hashlib
import io
import json
import sys
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from search import search_videos, save_results_videos, save_results_to_folder


def create_stub_clip(frame_path: Path, dest_path: Path, duration: float = 10.0, fps: int = 5) -> bool:
    if not frame_path.exists():
        return False
    frame = cv2.imread(str(frame_path))
    if frame is None:
        return False
    height, width = frame.shape[:2]
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    codecs = [('avc1', '.mp4'), ('H264', '.mp4'), ('X264', '.mp4'), ('mp4v', '.mp4')]

    for fourcc, ext in codecs:
        out_path = dest_path.with_suffix(ext)
        writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*fourcc), fps, (width, height))
        if not writer.isOpened():
            continue
        total_frames = max(1, int(duration * fps))
        for _ in range(total_frames):
            writer.write(frame)
        writer.release()
        if out_path.exists() and out_path.stat().st_size > 0:
            # Ensure final path is dest_path (.mp4)
            if out_path != dest_path:
                try:
                    dest_path.unlink(missing_ok=True)
                except Exception:
                    pass
                out_path.rename(dest_path)
            return True
    return False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("query")
    parser.add_argument("--top-k", type=int, default=None)
    args = parser.parse_args()

    def filter_non_overlapping(items, window: float = 10.0, limit: int = 10):
        kept = []
        for item in items:
            vid = item.get("video_id")
            ts = float(item.get("timestamp", 0.0))
            conflict = any(
                k.get("video_id") == vid and abs(float(k.get("timestamp", 0.0)) - ts) < window
                for k in kept
            )
            if conflict:
                continue
            kept.append(item)
            if len(kept) >= limit:
                break
        return kept

    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(buffer):
        raw_results = search_videos(args.query, top_k=args.top_k)
        results = filter_non_overlapping(raw_results, window=10.0, limit=10)
        if results:
            # Hash filenames before saving to avoid long names and clashes
            hashed_results = []
            for idx, r in enumerate(results, 1):
                timestamp = float(r.get("timestamp", 0.0))
                score = float(r.get("score", 0.0))
                video_id = r.get("video_id", "video")
                hash_input = f"{video_id}-{timestamp:.3f}-{score:.3f}-{idx}"
                hash_name = hashlib.md5(hash_input.encode("utf-8")).hexdigest()
                hashed_results.append({**r, "hash_name": hash_name})

            # Save videos and images using hashes
            save_results_videos(
                [{**r, "video_id": r.get("video_id"), "timestamp": r.get("timestamp"), "hash_name": r["hash_name"]} for r in hashed_results],
                args.query,
                use_hash_names=True,
            )
            save_results_to_folder(
                [{**r, "video_id": r.get("video_id"), "timestamp": r.get("timestamp"), "hash_name": r["hash_name"]} for r in hashed_results],
                args.query,
                use_hash_names=True,
            )
            results = hashed_results

    results_dir = PROJECT_ROOT / "results_video"
    images_dir = PROJECT_ROOT / "results"
    enriched = []
    for i, result in enumerate(results, 1):
        timestamp = float(result.get("timestamp", 0.0))
        score = float(result.get("score", 0.0))
        video_id = result.get("video_id", "video")
        hash_input = f"{video_id}-{timestamp:.3f}-{score:.3f}-{i}"
        hash_name = hashlib.md5(hash_input.encode("utf-8")).hexdigest()
        filename = f"{hash_name}.mp4"
        clip_path = results_dir / filename
        clip_url = None
        if clip_path.exists():
            clip_url = f"/results_video/{filename}"
        else:
            # Fallback: synthesize a short clip from the stored frame if original video is missing
            frame_path = Path(result.get("frame_path", ""))
            if frame_path.exists() and create_stub_clip(frame_path, clip_path):
                clip_url = f"/results_video/{filename}"

        image_filename = f"{hash_name}.jpg"
        image_path = images_dir / image_filename
        image_url = f"/results/{image_filename}" if image_path.exists() else None
        enriched.append({**result, "clip_url": clip_url, "image_url": image_url})

    clip_count = sum(1 for item in enriched if item.get("clip_url"))
    debug_output = buffer.getvalue()

    payload = {
        "results": enriched,
        "clip_count": clip_count,
        "debug": debug_output if clip_count == 0 else ""
    }
    print(json.dumps(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
