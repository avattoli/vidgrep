"""Model utilities for CLIP embeddings."""
import os
import warnings
import threading

# Disable Hugging Face auto-conversion background thread to avoid network timeouts
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = '1'
os.environ['HF_HUB_DISABLE_SAFETENSORS_WARNING'] = '1'
os.environ['HF_HUB_DISABLE_EXPERIMENTAL_WARNING'] = '1'

# Suppress warnings from background threads
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# Suppress httpx timeout errors in background threads
def suppress_thread_exceptions():
    """Suppress exceptions in background threads to avoid noise."""
    original_excepthook = threading.excepthook
    
    def custom_excepthook(args):
        if args.exc_value is not None:
            exc_type_name = type(args.exc_value).__name__
            exc_msg = str(args.exc_value) if args.exc_value else ""
            
            if any(keyword in exc_type_name or keyword in exc_msg for keyword in 
                   ['ReadTimeout', 'Timeout', 'timeout', 'Thread-auto_conversion']):
                return
        
        if original_excepthook:
            original_excepthook(args)
    
    threading.excepthook = custom_excepthook

suppress_thread_exceptions()

import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import numpy as np
from typing import List, Union
import config


class EmbeddingModel:
    """Wrapper for CLIP model to encode images and text."""
    
    def __init__(self):
        """Initialize CLIP model and processor."""
        print(f"Loading CLIP model: {config.CLIP_MODEL_NAME}")
        self.device = config.DEVICE
        
        self.model = CLIPModel.from_pretrained(config.CLIP_MODEL_NAME).to(self.device)
        self.model.eval()
        
        try:
            self.processor = CLIPProcessor.from_pretrained(config.CLIP_MODEL_NAME)
        except Exception as e:
            print(f"Warning: Could not load processor from hub ({e})")
            print("Attempting to load from local cache...")
            try:
                self.processor = CLIPProcessor.from_pretrained(
                    config.CLIP_MODEL_NAME, 
                    local_files_only=True
                )
            except Exception as e2:
                print(f"Error: Could not load processor: {e2}")
                raise
        
        print(f"Model loaded on device: {self.device}")
    
    def encode_image(self, image: Image.Image) -> np.ndarray:
        """
        Encode a single image into an embedding vector.
        
        Args:
            image: PIL Image object
            
        Returns:
            numpy array of shape (embedding_dim,)
        """
        with torch.no_grad():
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)
            vision_outputs = self.model.vision_model(**inputs)
            pooled_output = vision_outputs.pooler_output if hasattr(vision_outputs, 'pooler_output') and vision_outputs.pooler_output is not None else vision_outputs.last_hidden_state[:, 0, :]
            image_features = self.model.visual_projection(pooled_output)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            return image_features.cpu().numpy().flatten()
    
    def encode_images_batch(self, images: List[Image.Image]) -> np.ndarray:
        """
        Encode a batch of images into embedding vectors.
        
        Args:
            images: List of PIL Image objects
            
        Returns:
            numpy array of shape (batch_size, embedding_dim)
        """
        with torch.no_grad():
            inputs = self.processor(images=images, return_tensors="pt").to(self.device)
            vision_outputs = self.model.vision_model(**inputs)
            pooled_output = vision_outputs.pooler_output if hasattr(vision_outputs, 'pooler_output') and vision_outputs.pooler_output is not None else vision_outputs.last_hidden_state[:, 0, :]
            image_features = self.model.visual_projection(pooled_output)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            return image_features.cpu().numpy()
    
    def encode_text(self, text: str) -> np.ndarray:
        """
        Encode a text query into an embedding vector.
        
        Args:
            text: Text string to encode
            
        Returns:
            numpy array of shape (embedding_dim,)
        """
        with torch.no_grad():
            inputs = self.processor(text=text, return_tensors="pt", padding=True, truncation=True).to(self.device)
            text_outputs = self.model.text_model(**inputs)
            
            attention_mask = inputs.get('attention_mask', None)
            if attention_mask is not None:
                seq_lengths = attention_mask.sum(dim=1) - 1
                batch_size = text_outputs.last_hidden_state.shape[0]
                pooled_output = text_outputs.last_hidden_state[torch.arange(batch_size), seq_lengths]
            else:
                pooled_output = text_outputs.last_hidden_state[:, -1, :]
            
            text_features = self.model.text_projection(pooled_output)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            return text_features.cpu().numpy().flatten()
    
    def encode_texts_batch(self, texts: List[str]) -> np.ndarray:
        """
        Encode a batch of text queries into embedding vectors.
        
        Args:
            texts: List of text strings to encode
            
        Returns:
            numpy array of shape (batch_size, embedding_dim)
        """
        with torch.no_grad():
            inputs = self.processor(text=texts, return_tensors="pt", padding=True, truncation=True).to(self.device)
            text_outputs = self.model.text_model(**inputs)
            
            attention_mask = inputs.get('attention_mask', None)
            if attention_mask is not None:
                seq_lengths = attention_mask.sum(dim=1) - 1
                batch_size = text_outputs.last_hidden_state.shape[0]
                pooled_output = text_outputs.last_hidden_state[torch.arange(batch_size), seq_lengths]
            else:
                pooled_output = text_outputs.last_hidden_state[:, -1, :]
            
            text_features = self.model.text_projection(pooled_output)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            return text_features.cpu().numpy()
