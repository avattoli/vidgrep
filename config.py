"""Configuration settings for VidGrep system."""
import os
from pathlib import Path

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
VIDEOS_DIR = DATA_DIR / "videos"
FRAMES_DIR = DATA_DIR / "frames"
INDEX_DIR = DATA_DIR / "index"
METADATA_DIR = DATA_DIR / "metadata"
RESULTS_DIR = BASE_DIR / "results"
RESULTS_VIDEO_DIR = BASE_DIR / "results_video"

for dir_path in [DATA_DIR, VIDEOS_DIR, FRAMES_DIR, INDEX_DIR, METADATA_DIR, RESULTS_DIR, RESULTS_VIDEO_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

FRAME_SAMPLING_INTERVAL = 1.0
FRAME_WIDTH = 224
FRAME_HEIGHT = 224
SAVE_FULL_RESOLUTION = True
FRAME_JPEG_QUALITY = 95

CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
DEVICE = "cuda" if os.getenv("CUDA_AVAILABLE") == "true" else "cpu"

FAISS_INDEX_TYPE = "L2"
EMBEDDING_DIM = 512

DEFAULT_TOP_K = 10
