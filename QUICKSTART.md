# VidGrep Quick Start Guide

## Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

This will install:
- PyTorch (for CLIP model)
- OpenCV (for video processing)
- FAISS (for vector search)
- Transformers (for CLIP)
- Other dependencies

**Note:** On first run, the system will automatically download the CLIP model (~150MB).

## Step 2: Prepare Your Videos

Place your video files anywhere on your system. Supported formats:
- `.mp4`, `.avi`, `.mov`, `.mkv`, `.flv`, `.wmv`

## Step 3: Ingest Videos

### Option A: Ingest specific video files

```bash
python ingest.py path/to/video1.mp4 path/to/video2.mp4
```

### Option B: Ingest all videos from a directory

```bash
python ingest.py --dir /path/to/video/directory
```

**What happens during ingestion:**
- Extracts frames every 1 second (configurable in `config.py`)
- Encodes frames using CLIP model
- Stores embeddings in FAISS index
- Saves metadata (video_id, timestamp, frame_path)

**Example output:**
```
Loading CLIP model: openai/clip-vit-base-patch32
Model loaded on device: cpu
Extracting frames from video1.mp4 (duration: 120.50s, fps: 30.00)
Extracted 120 frames from video1.mp4
Encoding 120 frames...
Encoding frames: 100%|████████| 4/4 [00:15<00:00,  3.75s/it]
Added 120 embeddings. Total: 120

Ingestion complete!
Total frames processed: 120
Total embeddings in index: 120
Unique videos: 1
```

## Step 4: Search Videos

### Basic search

```bash
python search.py "a dog on screen"
```

### Search with more results

```bash
python search.py "a dog on screen" --top-k 20
```

### Search with preview frame paths

```bash
python search.py "a dog on screen" --preview
```

**Example output:**
```
Encoding query: 'a dog on screen'
Searching for top 10 results...

Found 10 results:

--------------------------------------------------------------------------------

1. Video: video1
   Timestamp: 00:45 (45.00s)
   Similarity Score: 0.8234

2. Video: video1
   Timestamp: 01:23 (83.00s)
   Similarity Score: 0.7891

...
```

## Step 5: Use Programmatically (Python)

```python
from pathlib import Path
from ingest import ingest_videos
from search import search_videos, format_results

# Ingest videos
video_paths = [Path("video1.mp4"), Path("video2.mp4")]
ingest_videos(video_paths)

# Search
results = search_videos("a dog on screen", top_k=10)

# Print formatted results
print(format_results(results))

# Access individual results
for result in results:
    print(f"Video: {result['video_id']}")
    print(f"Timestamp: {result['timestamp']}s")
    print(f"Score: {result['score']}")
    print(f"Frame: {result['frame_path']}")
```

## Configuration

Edit `config.py` to customize:

```python
FRAME_SAMPLING_INTERVAL = 1.0  # Extract frame every N seconds
FRAME_WIDTH = 224              # Frame resize width
FRAME_HEIGHT = 224             # Frame resize height
DEFAULT_TOP_K = 10             # Default search results
```

## File Structure After Ingestion

```
VidGrep/
├── data/
│   ├── videos/          # (optional: original videos)
│   ├── frames/          # Extracted frame images
│   │   └── video1/
│   │       ├── video1_frame_000000_t0.00.jpg
│   │       ├── video1_frame_000001_t1.00.jpg
│   │       └── ...
│   ├── index/
│   │   └── faiss.index  # Vector index
│   └── metadata/
│       └── metadata.json # Video metadata
```

## Tips

1. **First ingestion takes longer**: The CLIP model downloads on first use (~150MB)
2. **Batch processing**: Videos are processed sequentially, but frames are encoded in batches
3. **Incremental ingestion**: You can add more videos anytime - they'll be added to the existing index
4. **Search quality**: More descriptive queries work better (e.g., "a brown dog running" vs "dog")
5. **Storage**: Frame images are saved for previews - delete `data/frames/` if you don't need them

## Troubleshooting

**"Could not open video"**
- Check video file path is correct
- Verify video format is supported
- Try with absolute paths

**"No results found"**
- Make sure you've ingested videos first
- Check that `data/index/faiss.index` exists
- Try a different search query

**Out of memory**
- Reduce batch size in `ingest.py` (line ~140: `batch_size = 32`)
- Process videos one at a time
- Use smaller frame dimensions in `config.py`

**Slow performance**
- Use GPU if available: Set `CUDA_AVAILABLE=true` environment variable
- Reduce `FRAME_SAMPLING_INTERVAL` to extract fewer frames
- Use a smaller CLIP model variant
