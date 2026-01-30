"""Video ingestion: extract frames and create embeddings."""
import os
# Fix OpenMP conflicts on macOS
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '1'

import cv2
import numpy as np
from pathlib import Path
from PIL import Image
from typing import List, Dict
from tqdm import tqdm
import config
from models import EmbeddingModel
from vector_store import VectorStore


def extract_frames(video_path: Path, output_dir: Path, interval_seconds: float = 1.0) -> List[Dict]:
    """
    Extract frames from video at regular intervals.
    
    Args:
        video_path: Path to video file
        output_dir: Directory to save extracted frames
        interval_seconds: Extract frame every N seconds
        
    Returns:
        List of dicts with keys: timestamp, frame_path
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0
        print(f"Warning: Invalid FPS, using default {fps}")
    
    frame_interval = max(1, int(fps * interval_seconds))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0
    
    video_id = video_path.stem
    frame_data = []
    frame_count = 0
    extracted_count = 0
    
    print(f"Extracting frames from {video_path.name} (duration: {duration:.2f}s, fps: {fps:.2f})")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % frame_interval == 0:
            timestamp = frame_count / fps if fps > 0 else 0
            
            # Convert BGR to RGB for PIL
            frame_rgb_full = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image_full = Image.fromarray(frame_rgb_full)
            
            # Resize frame for CLIP model (smaller, faster processing)
            frame_resized = cv2.resize(frame, (config.FRAME_WIDTH, config.FRAME_HEIGHT))
            frame_rgb_small = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            pil_image_small = Image.fromarray(frame_rgb_small)
            
            # Save full-resolution frame for display
            frame_filename = f"{video_id}_frame_{extracted_count:06d}_t{timestamp:.2f}.jpg"
            frame_path = output_dir / frame_filename
            pil_image_full.save(frame_path, quality=config.FRAME_JPEG_QUALITY, optimize=False)
            
            frame_data.append({
                'timestamp': timestamp,
                'frame_path': str(frame_path),
                'frame_image': pil_image_small  # Use small image for CLIP encoding (faster)
            })
            
            extracted_count += 1
        
        frame_count += 1
    
    cap.release()
    print(f"Extracted {extracted_count} frames from {video_path.name}")
    return frame_data


def process_video(video_path: Path, embedding_model: EmbeddingModel, vector_store: VectorStore) -> int:
    """
    Process a single video: extract frames, encode, and add to index.
    
    Args:
        video_path: Path to video file
        embedding_model: EmbeddingModel instance
        vector_store: VectorStore instance
        
    Returns:
        Number of frames processed
    """
    video_id = video_path.stem
    
    frames_dir = config.FRAMES_DIR / video_id
    frame_data = extract_frames(video_path, frames_dir, config.FRAME_SAMPLING_INTERVAL)
    
    if not frame_data:
        print(f"No frames extracted from {video_path.name}")
        return 0
    
    batch_size = int(os.getenv("BATCH_SIZE", "32"))
    all_embeddings = []
    metadata_list = []
    
    print(f"Encoding {len(frame_data)} frames...")
    for i in tqdm(range(0, len(frame_data), batch_size), desc="Encoding frames"):
        batch = frame_data[i:i + batch_size]
        images = [item['frame_image'] for item in batch]
        
        embeddings = embedding_model.encode_images_batch(images)
        all_embeddings.append(embeddings)
        
        for item in batch:
            metadata_list.append({
                'video_id': video_id,
                'timestamp': item['timestamp'],
                'frame_path': item['frame_path'],
                'video_path': str(video_path)
            })
    
    all_embeddings = np.vstack(all_embeddings)
    
    vector_store.add_embeddings(all_embeddings, metadata_list)
    
    return len(frame_data)


def ingest_videos(video_paths: List[Path], vector_store: VectorStore = None):
    """
    Ingest multiple videos into the system.
    
    Args:
        video_paths: List of paths to video files
        vector_store: Optional VectorStore instance (creates new if None)
    """
    if vector_store is None:
        vector_store = VectorStore()
    
    embedding_model = EmbeddingModel()
    
    total_frames = 0
    for video_path in video_paths:
        if not video_path.exists():
            print(f"Warning: Video not found: {video_path}")
            continue
        
        try:
            frames_count = process_video(video_path, embedding_model, vector_store)
            total_frames += frames_count
        except Exception as e:
            print(f"Error processing {video_path}: {e}")
            continue
    
    vector_store.save()
    
    stats = vector_store.get_stats()
    print(f"\nIngestion complete!")
    print(f"Total frames processed: {total_frames}")
    print(f"Total embeddings in index: {stats['total_embeddings']}")
    print(f"Unique videos: {stats['unique_videos']}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python ingest.py <video1> [video2] [video3] ...")
        print("Or: python ingest.py --dir <directory_with_videos>")
        sys.exit(1)
    
    video_paths = []
    
    if sys.argv[1] == "--dir":
        video_dir = Path(sys.argv[2])
        video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv'}
        video_paths = [p for p in video_dir.iterdir() 
                      if p.suffix.lower() in video_extensions]
        print(f"Found {len(video_paths)} videos in {video_dir}")
    else:
        video_paths = [Path(p) for p in sys.argv[1:]]
    
    if not video_paths:
        print("No videos found to ingest")
        sys.exit(1)
    
    ingest_videos(video_paths)
