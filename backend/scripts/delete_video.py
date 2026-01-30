import json
import sys
from pathlib import Path

import faiss
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import config  # noqa: E402

METADATA_PATH = PROJECT_ROOT / "data" / "metadata" / "metadata.json"
INDEX_PATH = PROJECT_ROOT / "data" / "index" / "faiss.index"


def main(video_id: str) -> int:
    if not METADATA_PATH.exists() or not INDEX_PATH.exists():
        print(json.dumps({"ok": False, "error": "index or metadata missing"}))
        return 1

    with open(METADATA_PATH, "r") as f:
        metadata = json.load(f)

    keep_indices = [i for i, m in enumerate(metadata) if m.get("video_id") != video_id]
    remove_count = len(metadata) - len(keep_indices)
    if remove_count == 0:
        print(json.dumps({"ok": True, "removed": 0}))
        return 0

    index = faiss.read_index(str(INDEX_PATH))
    dim = index.d

    # Reconstruct kept vectors
    kept_vectors = []
    for idx in keep_indices:
        vec = index.reconstruct(idx)  # returns a 1-D numpy array
        kept_vectors.append(np.asarray(vec, dtype="float32"))

    if kept_vectors:
        new_index = faiss.IndexFlatIP(dim)
        new_index.add(np.vstack(kept_vectors))
    else:
        new_index = faiss.IndexFlatIP(dim)
    faiss.write_index(new_index, str(INDEX_PATH))

    new_metadata = [metadata[i] for i in keep_indices]
    with open(METADATA_PATH, "w") as f:
        json.dump(new_metadata, f, indent=2)

    print(json.dumps({"ok": True, "removed": remove_count, "remaining": len(new_metadata)}))
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(json.dumps({"ok": False, "error": "video_id required"}))
        raise SystemExit(1)
    raise SystemExit(main(sys.argv[1]))
