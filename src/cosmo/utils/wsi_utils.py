#!/usr/bin/env python3
"""
WSI Utilities

Utilities for processing whole slide image (WSI) bags and features.

Author: Philip Chikontwe
"""

import torch
import numpy as np
from typing import Tuple, List
import os

def pad_bag(bag: np.ndarray, max_sz: int = 1024) -> np.ndarray:
    """
    Pad WSI bag to specified maximum size using mean imputation.
    
    Args:
        bag: Input bag of shape (n_patches, feature_dim)
        max_sz: Maximum bag size for padding
        
    Returns:
        Padded bag of shape (max_sz, feature_dim)
    """
    bag_f = np.zeros((max_sz, bag.shape[1])).astype(dtype=np.float32)
    bag_m = np.mean(bag, 0)
    bag_m = bag_m[np.newaxis, :]
    
    # Fill with mean values
    bag_f[:, :] = bag_m
    
    # Replace first n patches with actual data
    bag_f[:bag.shape[0], :] = bag[:max_sz, :]
    
    return bag_f

def knn_bag(bag: torch.Tensor, n_tokens: int = 1024, k: int = 16) -> torch.Tensor:
    """
    Process bag using k-nearest neighbor aggregation.
    
    Args:
        bag: Input bag tensor of shape (n_patches, feature_dim)
        n_tokens: Number of tokens for random sampling if k < 16
        k: Number of nearest neighbors
        
    Returns:
        Processed bag tensor of shape (min(n_patches, k), feature_dim)
    """
    k = 16 if bag.shape[0] > 16 else bag.shape[0] - 1
    
    # If insufficient patches, use random sampling
    if k < 16:
        patch_indices = torch.randint(0, bag.size(0), (n_tokens,)).tolist()
        bag = bag[patch_indices]
        return bag

    # Normalize features
    all_feats = torch.nn.functional.normalize(bag, dim=-1)
    
    # Compute similarity matrix
    classwise_sim = torch.einsum('b d, n d -> b n', all_feats, all_feats)
    
    # Get top k+1 similar patches (excluding self)
    _, indices = classwise_sim.topk(k=k+1, dim=-1, largest=True, sorted=True)
    indices = indices[:, 1:]  # Remove self-similarity
    
    # Average k nearest neighbors
    bag = torch.mean(bag[indices, :].view(-1, k, all_feats.shape[-1]), dim=0)
    
    return bag

def get_bag_sizes(wsi_paths: List[str]) -> Tuple[int, float, int]:
    """
    Compute bag size statistics from WSI feature files.
    
    Args:
        wsi_paths: List of paths to WSI feature files (.pt or .pth)
        
    Returns:
        Tuple of (min_size, mean_size, max_size)
    """
    bags = []
    
    for wsi_path in wsi_paths:
        if "virtual.svs" in wsi_path:
            continue
            
        try:
            # Load WSI features
            wsi_features = torch.load(wsi_path, map_location=torch.device('cpu'))
            num_patches = np.asarray(wsi_features).shape[0]
            bags.append(num_patches)
        except Exception as e:
            print(f"Error loading {wsi_path}: {e}")
            continue
    
    if not bags:
        return 0, 0.0, 0
    
    return np.min(bags), np.mean(bags), np.max(bags)

def load_wsi_features(wsi_path: str) -> torch.Tensor:
    """
    Load WSI features from file with proper error handling.
    
    Args:
        wsi_path: Path to WSI feature file (.pt or .pth)
        
    Returns:
        WSI features as float tensor
        
    Raises:
        FileNotFoundError: If WSI file doesn't exist
        RuntimeError: If file loading fails
    """
    if not os.path.isfile(wsi_path):
        raise FileNotFoundError(f"WSI file not found: {wsi_path}")
    
    try:
        wsi_features = torch.load(wsi_path, map_location=torch.device('cpu')).detach()
        wsi_features = np.asarray(wsi_features).astype(dtype=np.float32)
        wsi_features = torch.from_numpy(wsi_features).float()
        return wsi_features
    except Exception as e:
        raise RuntimeError(f"Failed to load WSI features from {wsi_path}: {e}")

def process_wsi_paths(slide_id: str, root_dir: str) -> str:
    """
    Convert slide ID to WSI feature file path.
    
    Args:
        slide_id: Slide identifier (e.g., "12345.svs")
        root_dir: Root directory containing WSI features
        
    Returns:
        Full path to WSI feature file
    """
    # Extract extension and replace with .pt
    ext = os.path.split(slide_id)[-1].split(".")[-1]
    wsi_path = os.path.join(root_dir, slide_id.replace(f".{ext}", ".pt"))
    
    # Try .pth if .pt doesn't exist
    if not os.path.isfile(wsi_path):
        wsi_path = os.path.join(root_dir, slide_id.replace(f".{ext}", ".pth"))
    
    return wsi_path