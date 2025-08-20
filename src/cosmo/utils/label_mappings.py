#!/usr/bin/env python3
"""
Label Mappings for WSI Dataset

Contains label mappings and class definitions for brain cancer types.
Supports prevalence-based seen-unseen splits.

Author: Philip Chikontwe
"""

from typing import Dict, List

# Brain cancer label mappings based on prevalence
LABEL_MAP: Dict[str, Dict[str, int]] = {
    "brainNIH": {
        # Common (seen classes)
        'Adult-type diffuse gliomas': 0, 
        'Circumscribed astrocytic gliomas': 1,
        'Meningiomas': 2,
        
        # Rare (unseen classes)
        'Ependymal Tumours': 3, 
        'Paediatric-type diffuse low-grade gliomas': 4,
        'Glioneuronal and neuronal tumours': 5, 
        'Embryonal Tumours': 6,
    }, 
}

# Classes to ignore during processing
IGNORE_CLASSES: List[str] = [
    "Metastatic tumours", 
    "Germ Cell Tumours", 
    "Chroid Plexus Tumours", 
    "Other", 
    "Pineal Tumor", 
    "Haematolymphoid tumours involving the CNS", 
    "Lipoma", 
    "Germ Cell Tumor", 
    "Mesenchymal non-meningothelial tumours involving the CNS", 
    "Tumours of the sellar region",  
    "Cranial and paraspinal nerve tumours"
]

def get_label_mapping(dataset: str) -> Dict[str, int]:
    """
    Get label mapping for specified dataset.
    
    Args:
        dataset: Dataset name (e.g., 'brainNIH', 'tvgh')
        
    Returns:
        Dictionary mapping class names to integer labels
        
    Raises:
        KeyError: If dataset not found in LABEL_MAP
    """
    if dataset not in LABEL_MAP:
        raise KeyError(f"Dataset '{dataset}' not found. Available: {list(LABEL_MAP.keys())}")
    
    return LABEL_MAP[dataset].copy()

def get_class_names(dataset: str) -> List[str]:
    """
    Get list of class names for specified dataset.
    
    Args:
        dataset: Dataset name
        
    Returns:
        List of class names in label order
    """
    label_map = get_label_mapping(dataset)
    # Sort by label value to get names in correct order
    sorted_items = sorted(label_map.items(), key=lambda x: x[1])
    return [name for name, _ in sorted_items]

def get_zero_shot_split(dataset: str, zero_shot_precision: float) -> tuple:
    """
    Get seen/unseen class splits.
    
    Args:
        dataset: Dataset name
        zero_shot_precision: Fraction of classes to use as "seen"
        
    Returns:
        Tuple of (seen_classes, unseen_classes) as lists of class indices
    """
    label_map = get_label_mapping(dataset)
    num_classes = len(label_map)
    seen_cls = int(num_classes * zero_shot_precision)
    
    all_classes = list(range(num_classes))
    seen_classes = all_classes[:seen_cls]
    unseen_classes = all_classes[seen_cls:]
    
    return seen_classes, unseen_classes