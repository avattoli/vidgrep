"""Vector storage and retrieval using FAISS."""
import os
# Fix OpenMP conflicts on macOS
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
# Set FAISS to use single thread to avoid OpenMP issues
os.environ['OMP_NUM_THREADS'] = '1'

import faiss
import numpy as np
import json
from pathlib import Path
from typing import List, Dict, Tuple
import config


class VectorStore:
    """Manages vector embeddings and metadata for video frames."""
    
    def __init__(self, index_path: Path = None, metadata_path: Path = None):
        """
        Initialize vector store.
        
        Args:
            index_path: Path to save/load FAISS index
            metadata_path: Path to save/load metadata
        """
        self.index_path = index_path or config.INDEX_DIR / "faiss.index"
        self.metadata_path = metadata_path or config.METADATA_DIR / "metadata.json"
        
        self.index = None
        self.metadata = []  # List of dicts: {video_id, timestamp, frame_path}
        self.embedding_dim = config.EMBEDDING_DIM
        self._initialize_index()
    
    def _initialize_index(self):
        """Initialize or load FAISS index."""
        if self.index_path.exists() and self.metadata_path.exists():
            print(f"Loading existing index from {self.index_path}")
            self.index = faiss.read_index(str(self.index_path))
            with open(self.metadata_path, 'r') as f:
                self.metadata = json.load(f)
            print(f"Loaded {len(self.metadata)} existing embeddings")
        else:
            # Create new index - using inner product for cosine similarity (since vectors are normalized)
            self.index = faiss.IndexFlatIP(self.embedding_dim)
            print(f"Created new FAISS index with dimension {self.embedding_dim}")
    
    def add_embeddings(self, embeddings: np.ndarray, metadata_list: List[Dict]):
        """
        Add embeddings and metadata to the index.
        
        Args:
            embeddings: numpy array of shape (n, embedding_dim)
            metadata_list: List of metadata dicts, one per embedding
        """
        if len(embeddings) != len(metadata_list):
            raise ValueError(f"Mismatch: {len(embeddings)} embeddings but {len(metadata_list)} metadata entries")
        
        if embeddings.shape[1] != self.embedding_dim:
            raise ValueError(f"Embedding dimension mismatch: expected {self.embedding_dim}, got {embeddings.shape[1]}")
        
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / (norms + 1e-8)
        
        self.index.add(embeddings.astype('float32'))
        
        self.metadata.extend(metadata_list)
        
        print(f"Added {len(embeddings)} embeddings. Total: {len(self.metadata)}")
    
    def search(self, query_embedding: np.ndarray, top_k: int = 10) -> List[Dict]:
        """
        Search for similar frames.
        
        Args:
            query_embedding: numpy array of shape (embedding_dim,)
            top_k: Number of results to return
            
        Returns:
            List of result dicts with keys: video_id, timestamp, frame_path, score, index
        """
        if self.index.ntotal == 0:
            return []
        
        query_embedding = query_embedding.reshape(1, -1)
        norm = np.linalg.norm(query_embedding)
        if norm > 0:
            query_embedding = query_embedding / norm
        
        query_embedding = query_embedding.astype('float32')
        distances, indices = self.index.search(query_embedding, min(top_k, self.index.ntotal))
        
        results = []
        for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
            if idx < len(self.metadata):
                result = self.metadata[idx].copy()
                result['score'] = float(distance)  # Cosine similarity score
                result['index'] = int(idx)
                results.append(result)
        
        return results
    
    def save(self):
        """Save index and metadata to disk."""
        print(f"Saving index to {self.index_path}")
        faiss.write_index(self.index, str(self.index_path))
        
        print(f"Saving metadata to {self.metadata_path}")
        with open(self.metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)
        
        print("Save complete")
    
    def get_stats(self) -> Dict:
        """Get statistics about the index."""
        return {
            'total_embeddings': self.index.ntotal if self.index else 0,
            'embedding_dim': self.embedding_dim,
            'unique_videos': len(set(m['video_id'] for m in self.metadata)) if self.metadata else 0
        }
