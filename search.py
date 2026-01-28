"""Search module: query videos using text prompts."""
import os
# Fix OpenMP conflicts on macOS
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '1'

import cv2
import shutil
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
import config
from models import EmbeddingModel
from vector_store import VectorStore
from PIL import Image

VIDEO_EXTENSIONS = ('.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv')


def resolve_video_path(video_id: str, video_path_hint: Optional[str] = None) -> Optional[Path]:
    """
    Resolve a video file path from a video_id and optional hint.
    """
    if video_path_hint:
        hinted_path = Path(video_path_hint)
        if hinted_path.exists():
            return hinted_path
    
    for base_dir in [config.VIDEOS_DIR, config.BASE_DIR]:
        for ext in VIDEO_EXTENSIONS:
            candidate = base_dir / f"{video_id}{ext}"
            if candidate.exists():
                return candidate
    
    return None


def extract_video_clip(video_path: Path, start_time: float, end_time: float, output_path: Path) -> bool:
    """
    Extract a clip from video_path between start_time and end_time (seconds).
    
    Returns True if a clip was written.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return False
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = (total_frames / fps) if total_frames > 0 else None
    
    start_time = max(0.0, start_time)
    end_time = max(start_time, end_time)
    if duration is not None:
        end_time = min(end_time, duration)
    
    start_frame = int(start_time * fps)
    end_frame = int(end_time * fps)
    
    if total_frames > 0:
        start_frame = max(0, min(start_frame, total_frames - 1))
        end_frame = max(start_frame + 1, min(end_frame, total_frames))
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width <= 0 or height <= 0:
        cap.release()
        return False
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        cap.release()
        return False
    
    frames_to_write = (end_frame - start_frame) if total_frames > 0 else None
    frames_written = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        writer.write(frame)
        frames_written += 1
        if frames_to_write is not None and frames_written >= frames_to_write:
            break
    
    cap.release()
    writer.release()
    return frames_written > 0


def search_videos(query_text: str, top_k: int = None, vector_store: VectorStore = None) -> List[Dict]:
    """
    Search for frames matching a text query.
    
    Args:
        query_text: Text description to search for (e.g., "a dog on screen")
        top_k: Number of results to return (defaults to config.DEFAULT_TOP_K)
        vector_store: Optional VectorStore instance (loads existing if None)
        
    Returns:
        List of result dicts with keys: video_id, timestamp, frame_path, score, index
    """
    if vector_store is None:
        vector_store = VectorStore()
    
    stats = vector_store.get_stats()
    if stats['total_embeddings'] == 0:
        print("⚠️  WARNING: Index is empty! No videos have been ingested yet.")
        print("   Please run: python ingest.py video1.mp4")
        return []
    
    if top_k is None:
        top_k = config.DEFAULT_TOP_K
    
    embedding_model = EmbeddingModel()
    print(f"Encoding query: '{query_text}'")
    query_embedding = embedding_model.encode_text(query_text)
    
    print(f"Query embedding - shape: {query_embedding.shape}, norm: {np.linalg.norm(query_embedding):.6f}, sample: {query_embedding[:3]}")
    
    print(f"Searching for top {top_k} results...")
    results = vector_store.search(query_embedding, top_k=top_k)
    
    if results:
        scores = [r['score'] for r in results]
        print(f"Result scores - min: {min(scores):.4f}, max: {max(scores):.4f}, mean: {np.mean(scores):.4f}")
    
    return results


def format_results(results: List[Dict], show_preview: bool = False) -> str:
    """
    Format search results for display.
    
    Args:
        results: List of result dicts from search_videos
        show_preview: Whether to include preview frame paths
        
    Returns:
        Formatted string
    """
    if not results:
        return "No results found."
    
    output = []
    output.append(f"\nFound {len(results)} results:\n")
    output.append("-" * 80)
    
    for i, result in enumerate(results, 1):
        video_id = result['video_id']
        timestamp = result['timestamp']
        score = result.get('score', 0.0)
        frame_path = result.get('frame_path', '')
        
        minutes = int(timestamp // 60)
        seconds = int(timestamp % 60)
        time_str = f"{minutes:02d}:{seconds:02d}"
        
        output.append(f"\n{i}. Video: {video_id}")
        output.append(f"   Timestamp: {time_str} ({timestamp:.2f}s)")
        output.append(f"   Similarity Score: {score:.4f}")
        
        if show_preview and frame_path:
            output.append(f"   Frame: {frame_path}")
    
    output.append("\n" + "-" * 80)
    return "\n".join(output)


def get_preview_frame(result: Dict) -> Image.Image:
    """
    Load preview frame image for a search result.
    
    Args:
        result: Result dict from search_videos
        
    Returns:
        PIL Image object
    """
    frame_path = result.get('frame_path')
    if not frame_path:
        return None
    
    frame_path = Path(frame_path)
    if not frame_path.exists():
        return None
    
    return Image.open(frame_path)


def save_results_to_folder(results: List[Dict], query_text: str):
    """
    Save matching frame images to results folder.
    Clears the results folder before adding new images.
    
    Args:
        results: List of result dicts from search_videos
        query_text: The search query (used for naming)
    """
    results_dir = config.RESULTS_DIR
    
    if results_dir.exists():
        print(f"Clearing results folder: {results_dir}")
        for item in results_dir.iterdir():
            if item.is_file():
                item.unlink()
            elif item.is_dir():
                shutil.rmtree(item)
    
    if not results:
        print("No results to save.")
        return
    
    safe_query = "".join(c for c in query_text if c.isalnum() or c in (' ', '-', '_')).strip()
    safe_query = safe_query.replace(' ', '_')[:50]  # Limit length
    
    print(f"\nSaving {len(results)} results to {results_dir}")
    
    for i, result in enumerate(results, 1):
        frame_path = Path(result.get('frame_path', ''))
        if not frame_path.exists():
            print(f"  Warning: Frame not found: {frame_path}")
            continue
        
        video_id = result['video_id']
        timestamp = result['timestamp']
        score = result.get('score', 0.0)
        
        minutes = int(timestamp // 60)
        seconds = int(timestamp % 60)
        time_str = f"{minutes:02d}_{seconds:02d}"
        
        filename = f"{i:02d}_{video_id}_{time_str}_score{score:.3f}.jpg"
        dest_path = results_dir / filename
        
        try:
            shutil.copy2(frame_path, dest_path)
            print(f"  Saved: {filename}")
        except Exception as e:
            print(f"  Error copying {frame_path}: {e}")
    
    print(f"\nResults saved to: {results_dir}")


def save_results_videos(results: List[Dict], query_text: str, clip_seconds: float = 10.0):
    """
    Save matching video clips to results_video folder.
    Clips are centered around the result timestamp.
    """
    results_dir = config.RESULTS_VIDEO_DIR
    
    if results_dir.exists():
        print(f"Clearing results_video folder: {results_dir}")
        for item in results_dir.iterdir():
            if item.is_file():
                item.unlink()
            elif item.is_dir():
                shutil.rmtree(item)
    
    if not results:
        print("No results to save.")
        return
    
    half_window = clip_seconds / 2.0
    print(f"\nSaving {len(results)} video clips to {results_dir}")
    
    for i, result in enumerate(results, 1):
        video_id = result.get('video_id')
        timestamp = float(result.get('timestamp', 0.0))
        score = result.get('score', 0.0)
        
        video_path = resolve_video_path(video_id, result.get('video_path'))
        if not video_path:
            print(f"  Warning: Video file not found for {video_id}")
            continue
        
        start_time = timestamp - half_window
        end_time = timestamp + half_window
        
        minutes = int(timestamp // 60)
        seconds = int(timestamp % 60)
        time_str = f"{minutes:02d}_{seconds:02d}"
        
        filename = f"{i:02d}_{video_id}_{time_str}_score{score:.3f}.mp4"
        dest_path = results_dir / filename
        
        ok = extract_video_clip(video_path, start_time, end_time, dest_path)
        if ok:
            print(f"  Saved: {filename}")
        else:
            print(f"  Error: Could not create clip for {video_id} at {timestamp:.2f}s")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python search.py '<query text>' [--top-k N] [--preview]")
        print("Example: python search.py 'a dog on screen' --top-k 5")
        sys.exit(1)
    
    query_text = sys.argv[1]
    top_k = config.DEFAULT_TOP_K
    show_preview = False
    
    i = 2
    while i < len(sys.argv):
        if sys.argv[i] == "--top-k" and i + 1 < len(sys.argv):
            top_k = int(sys.argv[i + 1])
            i += 2
        elif sys.argv[i] == "--preview":
            show_preview = True
            i += 1
        else:
            i += 1
    
    results = search_videos(query_text, top_k=top_k)
    
    print(format_results(results, show_preview=show_preview))
    
    save_results_to_folder(results, query_text)
    
    save_results_videos(results, query_text)
    
    vector_store = VectorStore()
    stats = vector_store.get_stats()
    print(f"\nIndex stats: {stats['total_embeddings']} embeddings from {stats['unique_videos']} videos")
