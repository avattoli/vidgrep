# VidGrep

Search videos with text prompts to get matching timestamps.

## Features
- Ingest videos into frame embeddings
- CLIP text/image search with FAISS
- Timestamped results and optional previews/clips

## Installation

```bash
pip install -r requirements.txt
```

The CLIP model downloads on first use.

## Usage

### Ingest

```bash
python ingest.py video1.mp4 video2.mp4
python ingest.py --dir /path/to/videos
```

### Search

```bash
python search.py "a dog on screen"
python search.py "a dog on screen" --top-k 20
python search.py "a dog on screen" --preview
```

### Programmatic

```python
from ingest import ingest_videos
from search import search_videos, format_results
from pathlib import Path

ingest_videos([Path("video1.mp4"), Path("video2.mp4")])
results = search_videos("a dog on screen", top_k=10)
print(format_results(results))
```

## Configuration

Edit `config.py`:
- `FRAME_SAMPLING_INTERVAL`
- `FRAME_WIDTH` / `FRAME_HEIGHT`
- `CLIP_MODEL_NAME`
- `DEFAULT_TOP_K`

## Data layout

```
data/
├── videos/
├── frames/{video_id}/
├── index/faiss.index
└── metadata/metadata.json
```

## Requirements

Python 3.8+ and the packages in `requirements.txt`.
