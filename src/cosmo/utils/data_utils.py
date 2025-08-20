#!/usr/bin/env python3
"""
Data Utilities

Utilities for data loading, sampling, and class weighting.
Includes balanced sampling, class weights, and dataloader builders.

Author: Philip Chikontwe
"""

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from typing import List, Tuple, Dict, Any
from collections import Counter

def make_weights_for_balanced_classes_split(labels: List[int]) -> torch.Tensor:
    """
    Create sample weights for balanced class sampling.
    
    Args:
        labels: List of class labels
        
    Returns:
        Tensor of sample weights for WeightedRandomSampler
    """
    uniques, counts = np.unique(labels, return_counts=True)
    weights = {uniques[i]: 1. / counts[i] for i in range(len(uniques))}
    samples_weight = np.array([weights[t] for t in labels])
    samples_weight = torch.from_numpy(samples_weight)  
    return torch.DoubleTensor(samples_weight).type('torch.DoubleTensor')

def sk_class_weights(labels: List[int]) -> np.ndarray:
    """
    Compute sklearn-style class weights for loss weighting.
    
    Args:
        labels: List of class labels
        
    Returns:
        Array of class weights
    """
    u, c = np.unique(labels, return_counts=True)
    n_samples = sum(c)
    weights = n_samples / (len(c) * c)
    return weights

def create_balanced_dataloader(dataset: Any, batch_size: int, num_workers: int = 0, 
                             pin_memory: bool = False) -> DataLoader:
    """
    Create a DataLoader with balanced class sampling.
    
    Args:
        dataset: Dataset instance with .targets attribute
        batch_size: Batch size for DataLoader
        num_workers: Number of worker processes
        pin_memory: Whether to pin memory
        
    Returns:
        DataLoader with WeightedRandomSampler
    """
    weights = make_weights_for_balanced_classes_split(dataset.targets)
    sampler = WeightedRandomSampler(weights, len(weights))
    
    return DataLoader(
        dataset=dataset, 
        sampler=sampler, 
        batch_size=batch_size,
        num_workers=num_workers, 
        pin_memory=pin_memory
    )

def create_standard_dataloader(dataset: Any, batch_size: int, shuffle: bool = False,
                             num_workers: int = 0, pin_memory: bool = False) -> DataLoader:
    """
    Create a standard DataLoader without special sampling.
    
    Args:
        dataset: Dataset instance
        batch_size: Batch size for DataLoader
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes
        pin_memory: Whether to pin memory
        
    Returns:
        Standard DataLoader
    """
    return DataLoader(
        dataset=dataset, 
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers, 
        pin_memory=pin_memory
    )

def get_dataset_statistics(targets: List[int]) -> Dict[str, Any]:
    """
    Compute dataset statistics including class distribution.
    
    Args:
        targets: List of target labels
        
    Returns:
        Dictionary with statistics
    """
    counter = Counter(targets)
    unique_classes = np.unique(targets)
    
    stats = {
        'total_samples': len(targets),
        'num_classes': len(unique_classes),
        'class_distribution': dict(counter),
        'unique_classes': unique_classes.tolist(),
        'class_weights': sk_class_weights(targets).tolist()
    }
    
    return stats

def print_dataset_info(dataset_name: str, targets: List[int]) -> None:
    """
    Print formatted dataset information.
    
    Args:
        dataset_name: Name of the dataset
        targets: List of target labels
    """
    stats = get_dataset_statistics(targets)
    
    print('**' * 20)
    print(f"Dataset: {dataset_name}")
    print(f"Total samples: {stats['total_samples']}")
    print(f"Number of classes: {stats['num_classes']}")
    print(f"Class distribution: {stats['class_distribution']}")
    print(f"Unique classes: {stats['unique_classes']}")
    print(f"Class weights: {[f'{w:.3f}' for w in stats['class_weights']]}")
    print('**' * 20)

def validate_dataset_split(train_targets: List[int], val_targets: List[int], 
                          test_targets: List[int] = None) -> bool:
    """
    Validate that dataset splits have consistent classes.
    
    Args:
        train_targets: Training set labels
        val_targets: Validation set labels  
        test_targets: Test set labels (optional)
        
    Returns:
        True if splits are valid, False otherwise
    """
    train_classes = set(train_targets)
    val_classes = set(val_targets)
    
    # Check train/val overlap
    if not train_classes.intersection(val_classes):
        print("Warning: No class overlap between train and validation sets")
        return False
    
    # Check test set if provided
    if test_targets is not None:
        test_classes = set(test_targets)
        if not train_classes.intersection(test_classes):
            print("Warning: No class overlap between train and test sets")
            return False
    
    return True