#!/usr/bin/env python3
"""
WSI Bag Dataset

Whole Slide Image (WSI) bag dataset for brain cancer classification.
Supports zero-shot learning with seen/unseen splits and spatial feature processing.

Author: Philip Chikontwe
"""

import warnings
warnings.filterwarnings("ignore")

import os
import json
import math
import copy
import random
from typing import Optional, Tuple, List, Dict, Any
from collections import Counter

import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from ...utils.label_mappings import get_label_mapping, IGNORE_CLASSES
from ...utils.wsi_utils import load_wsi_features, process_wsi_paths, knn_bag


class WSIBagDataset(Dataset):
    """
    WSI Bag Dataset .
    """
    
    def __init__(self, 
                 root: str,
                 patient_info: str,
                 split_file: str,
                 template_paths: Optional[Dict[str, str]] = None,
                 train: str = 'train',
                 zs_state: str = 'seen',
                 zsmode: bool = True,
                 proto: bool = False,
                 ratio: float = 1.0,
                 prev_type: str = "brainNIH",
                 use_spatial: bool = False,
                 wsi_tokens: int = 1024,
                 text_model: str = "brainkt",
                 oversample: bool = False):
        """
        Initialize WSI Bag Dataset.
        
        Args:
            root: Root directory containing WSI feature files
            patient_info: Path to CSV file with patient information
            split_file: Path to CSV file with train/val/test splits
            template_paths: Optional dict of template file paths
            train: Split mode ('train', 'val', 'test', 'test_all')
            zs_state: Zero-shot state ('seen' or 'unseen')
            zsmode: Whether to use zero-shot mode
            proto: Whether in prototype mode
            ratio: Data sampling ratio (0.1 to 1.0)
            prev_type: Prevalence type for label mapping
            use_spatial: Whether to use spatial feature processing
            wsi_tokens: Number of WSI tokens for processing
            text_model: Text model type
            oversample: Whether to oversample minority classes
        """
        self.root = root
        self.csv_pat = patient_info
        self.split = split_file
        self.template_paths = template_paths or {}
        self.train = train
        self.zstate = zs_state
        self.bag_max = 1024
        self.max_slides = -1
        self.prevalence = prev_type
        self.usespatial = use_spatial
        self.proto = proto
        self.zsmode = zsmode
        self.zeroshotprec = self._get_zero_shot_precision()
        self.n_tokens = wsi_tokens
        self.text_model = text_model
        self.oversample = oversample
        
        # Ratio mapping for data subsampling
        self.ratio_dict = {
            1.0: -1,
            0.1: 1,
            0.2: 2,
            0.4: 4,
            0.8: 8,
            0.16: 16,
        }
        
        # Validate inputs
        self._validate_inputs()
        
        # Load and process data
        print('**' * 20)
        print(f"Prevalence type: {self.prevalence} [{self.zeroshotprec}]")
        
        self.wsis, self.targets_all, self.wsi_names = self.process()
        print("Initial data statistics:")
        print(Counter([str(i) for i in self.targets_all]))
        
        # Map string labels to integers
        LBL_MAP = get_label_mapping(self.prevalence)
        self.targets = [LBL_MAP[k] for k in self.targets_all]
        print(Counter(self.targets), np.unique(self.targets))
        self.all_classes = len(np.unique(self.targets))
        
        # Apply train/val split if needed
        if self.train in ['train', 'val']:
            print("Splitting data...")
            self.wsis, self.targets, self.wsi_names = self.split_val(self.train)
        
        # Apply data subsampling for training
        if self.train == 'train' or self.proto:
            print(f"\nData subsampling ---- [{ratio:.1%}]")
            
            if self.usespatial:
                if zsmode:
                    _, targets, _ = self.zs_split()
                else:
                    _, targets, _ = self.data_ratio(ratio, oversample=False)
                
                cls_dict = dict(Counter(targets))
                self.minimum_slides = max(list(cls_dict.values()))
                
                self.wsis, self.targets, self.wsi_names = self.data_ratio(
                    ratio, use_all=True, max_i=self.minimum_slides, oversample=oversample
                )
            else:
                self.wsis, self.targets, self.wsi_names = self.data_ratio(ratio, oversample=False)
            
            print(Counter(self.targets))
            print(np.unique(self.targets))
            print()
        
        # Apply zero-shot splits
        if not self.proto:
            if zsmode:
                print(f"[Zero-Shot Set][{zsmode}]")
                self.wsis, self.targets, self.wsi_names = self.zs_split()
            else:
                print(f"[Zero-Shot Set][{zsmode}]")
                self.wsis, self.targets, self.wsi_names = self.low_split(max_slides=-1)
            print(Counter([int(i) for i in self.targets]))
            print(np.unique(self.targets))
            print()
        
        # Get class information
        self.names = [k for k, v in get_label_mapping(self.prevalence).items()]
        
        # Determine feature dimensions from first WSI
        wsi_ex = load_wsi_features(self.wsis[0])
        self.ndims = wsi_ex.shape[-1]
        del wsi_ex
        
        print(f"WSIs: {len(self.wsis)} | {self.split}")
        self.label_ids = np.unique(self.targets)
        self.num_classes = len(np.unique(self.targets))
        print(f"{self.num_classes} classes: {np.unique(self.targets)}")
        print(f"[spatial-feat]: {self.usespatial}")
        print('**' * 20)
        
        # Create class-wise slide indices
        self.slide_cls_ids = [[] for i in range(self.num_classes)]
        for i in range(self.num_classes):
            self.slide_cls_ids[i] = np.where(np.array(self.targets) == i)[0]
        
    def _get_zero_shot_precision(self) -> float:
        """Get zero-shot precision based on prevalence type."""
        if self.prevalence in ["brainNIH"]:
            return 0.50
        return 0.75  # Default
    
    def _validate_inputs(self) -> None:
        """Validate input paths and parameters."""
        assert os.path.isdir(self.root), f'{self.root} is not a valid directory.'
        assert os.path.isfile(self.csv_pat), f'{self.csv_pat} is not a valid file.'
        assert os.path.isfile(self.split), f'{self.split} is not a valid file.'
    
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int, str]:
        """
        Get a single WSI bag sample.
        
        Args:
            index: Sample index
            
        Returns:
            Tuple of (bag_features, target_label, wsi_name)
        """
        target   = self.targets[index]
        wsi_name = self.wsi_names[index]
        bag_path = self.wsis[index]
        
        # Load WSI bag features
        bag = load_wsi_features(bag_path)
        
        # Apply spatial processing if enabled during training
        if self.usespatial and self.train == 'train':
            return bag, target, wsi_name
        else:
            # Convert to numpy for compatibility
            bag = np.asarray(bag).astype(dtype=np.float32)
            bag = torch.from_numpy(bag).float()
            
        return bag, target, wsi_name
    
    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.targets)
    
    def split_val(self, train: str) -> Tuple[List[str], np.ndarray, List[str]]:
        """
        Split data into train/validation sets (50/50 per class).
        
        Args:
            train: 'train' or 'val'
            
        Returns:
            Tuple of (wsi_paths, targets, wsi_names)
        """
        wsi_ = []
        wsi_y = []
        wsi_name_ = []
        self.targets = np.array(self.targets)
        
        classes = np.unique(self.targets).tolist()
        
        for i in classes:
            i = int(i)
            wsi_paths = np.array(self.wsis)[self.targets == i]
            wsi_names = np.array(self.wsi_names)[self.targets == i]
            max_s = len(wsi_paths) // 2
            
            if train == 'train':
                wsi_paths = wsi_paths[:max_s]
                wsi_name_.extend(wsi_names[:max_s])
            else:
                wsi_paths = wsi_paths[-max_s:]
                wsi_name_.extend(wsi_names[-max_s:])
            
            wsi_.extend(wsi_paths)
            wsi_y.extend([i] * max_s)
        
        return wsi_, np.array(wsi_y), wsi_name_
    
    def data_ratio(self, ratio: float = 1.0, use_all: bool = False, 
                   oversample: bool = True, max_i: int = 32) -> Tuple[List[str], List[int], List[str]]:
        """
        Sample a fixed number of samples for each class.
        
        Args:
            ratio: Sampling ratio
            use_all: Whether to use all available data
            oversample: Whether to oversample minority classes
            max_i: Maximum samples per class
            
        Returns:
            Tuple of (wsi_paths, targets, wsi_names)
        """
        num_samples = self.ratio_dict[ratio]
        
        if num_samples == -1 and not use_all:
            return self._data_ratio_percentage(1.0)
        
        if num_samples == -1:
            num_samples = max_i
        
        print("NUM SAMPLES ::", num_samples)
        
        num_classes = len(np.unique(self.targets))
        self.targets = np.array(self.targets)
        wsi_ = []
        wsi_y = []
        wsi_name_ = []
        
        for i in range(num_classes):
            wsi_paths = np.array(self.wsis)[self.targets == i]
            wsi_names = np.array(self.wsi_names)[self.targets == i]
            
            # Handle insufficient samples
            if len(wsi_paths) < num_samples:
                if not oversample:
                    num_samples_ = len(wsi_paths)
                    multiplier = 1
                    wsi_paths = np.tile(wsi_paths, multiplier)[:num_samples_]
                    wsi_names = np.tile(wsi_names, multiplier)[:num_samples_]
                else:
                    # Oversample by repeating existing samples
                    multiplier = int(np.ceil(num_samples / len(wsi_paths)))
                    wsi_paths = np.tile(wsi_paths, multiplier)[:num_samples]
                    wsi_names = np.tile(wsi_names, multiplier)[:num_samples]
            
            elif len(wsi_paths) > num_samples:
                # Random sampling if too many samples
                indices = np.random.choice(len(wsi_paths), num_samples, replace=False)
                wsi_paths = wsi_paths[indices]
                wsi_names = wsi_names[indices]
            
            wsi_.extend(wsi_paths)
            wsi_y.extend([i] * len(wsi_paths))
            wsi_name_.extend(wsi_names)
        
        return wsi_, wsi_y, wsi_name_
    
    def _data_ratio_percentage(self, ratio: float = 1.0) -> Tuple[List[str], List[int], List[str]]:
        """Subsample X% of each label."""
        num_classes = len(np.unique(self.targets))
        self.targets = np.array(self.targets)
        wsi_ = []
        wsi_y = []
        wsi_name_ = []
        
        for i in range(num_classes):
            wsi_paths = np.array(self.wsis)[self.targets == i]
            num_samples = wsi_paths.shape[0]
            max_slides = int(math.ceil(num_samples * ratio))
            
            if max_slides == 0:
                max_slides = num_samples
            
            wsi_paths = wsi_paths[:max_slides]
            wsi_.extend(wsi_paths)
            wsi_y.extend([i] * len(wsi_paths))
            wsi_names = np.array(self.wsi_names)[self.targets == i]
            wsi_name_.extend(wsi_names[:max_slides])
        
        return wsi_, wsi_y, wsi_name_
    
    def low_split(self, max_slides: int = 1) -> Tuple[List[str], List[int], List[str]]:
        """Create low-shot split with limited unseen class samples."""
        num_classes = len(np.unique(self.targets))
        seen_cls = int(num_classes * self.zeroshotprec)
        
        print("SEEN CLASSES   : [", seen_cls, "|", num_classes, "]")
        
        classes = list(range(num_classes))
        unseen_classes = classes[seen_cls:]
        print("UNSEEN CLASSES : [", unseen_classes, "|", num_classes, "]")
        
        wsi_ = []
        wsi_y = []
        wsi_name_ = []
        self.targets = np.array(self.targets)
        
        for i in classes:
            wsi_paths = np.array(self.wsis)[self.targets == i]
            if i in unseen_classes:
                if max_slides == -1:
                    wsi_.extend(wsi_paths)
                    wsi_y.extend([i] * len(wsi_paths))
                    wsi_names = np.array(self.wsi_names)[self.targets == i]
                    wsi_name_.extend(wsi_names)
                else:
                    max_s = min(max_slides, len(wsi_paths))
                    wsi_paths = wsi_paths[:max_s]
                    wsi_.extend(wsi_paths)
                    wsi_y.extend([i] * len(wsi_paths))
                    wsi_names = np.array(self.wsi_names)[self.targets == i]
                    wsi_name_.extend(wsi_names[:max_s])
            else:
                wsi_.extend(wsi_paths)
                wsi_y.extend([i] * len(wsi_paths))
                wsi_names = np.array(self.wsi_names)[self.targets == i]
                wsi_name_.extend(wsi_names)
        
        return wsi_, wsi_y, wsi_name_
    
    def zs_split(self, return_splits: bool = False) -> Tuple[List[str], List[int], List[str]]:
        """
        Create zero-shot splits (seen/unseen classes).
        
        Args:
            return_splits: If True, return class indices instead of data
            
        Returns:
            Tuple of (wsi_paths, targets, wsi_names) or class indices if return_splits=True
        """
        num_classes = len(np.unique(self.targets))
        seen_cls = int(num_classes * self.zeroshotprec)
        
        print("SEEN CLASSES : ", seen_cls, " | ", num_classes, " [TOTAL] ")
        
        if self.zstate == 'seen':
            classes = list(np.unique(self.targets))[:seen_cls]
        else:
            classes = list(np.unique(self.targets))[seen_cls:]
        
        if return_splits:
            return classes
        
        wsi_ = []
        wsi_y = []
        wsi_name_ = []
        self.targets = np.array(self.targets)
        
        for i in classes:
            wsi_paths = np.array(self.wsis)[self.targets == i]
            wsi_.extend(wsi_paths)
            wsi_y.extend([i] * len(wsi_paths))
            wsi_names = np.array(self.wsi_names)[self.targets == i]
            wsi_name_.extend(wsi_names)
        
        return wsi_, wsi_y, wsi_name_
    
    def process(self) -> Tuple[List[str], List[str], List[str]]:
        """
        Process patient CSV and split files to create dataset.
        
        Returns:
            Tuple of (wsi_paths, class_names, wsi_names)
        """
        print(f"Processing patient info | [set_split:{self.train}] [{self.csv_pat}] [{self.split}]")
        
        # Load patient data
        patient_data = pd.read_csv(self.csv_pat, low_memory=False)
        
        # Remove ignored classes
        patient_data = patient_data[~patient_data["class_name"].isin(IGNORE_CLASSES)]
        patient_data = patient_data.reset_index()
        
        # Get unique patients
        patient_unq = patient_data.drop_duplicates(['case_id']).copy()
        cases_list = list(patient_data['case_id'].values)
        patient_unq = patient_unq.set_index('slide_id').copy()
        patient_data = patient_data.set_index('case_id').copy()
        
        # Load split information
        split_state = self.train
        split_csv = pd.read_csv(self.split)
        
        # Handle different dataset naming conventions
        if "dfcibrain" in self.csv_pat and split_state == 'test_all':
            split_state = 'test'
        
        # Get patient list from split
        if split_state == 'test_all':
            split_tr = split_csv['train'].dropna().drop_duplicates()
            split_tr = list(split_tr.values)
            split_ts = split_csv['val'].dropna().drop_duplicates()
            split_ts = list(split_ts.values)
            
            patient_slds = []
            if len(split_tr) > 1:
                patient_slds.extend(split_tr)
            if len(split_ts) > 1:
                patient_slds.extend(split_ts)
        else:
            split_csv = split_csv[split_state].dropna().drop_duplicates()
            patient_slds = list(split_csv.values)
        
        # Convert to appropriate type
        try:
            patient_slds = [int(i) for i in patient_slds]
        except ValueError:
            patient_slds = [str(i) for i in patient_slds]
        
        # Process WSI features and labels
        libs = []
        lib_y = []
        wsi_names = []
        
        print(f"Loading wsi features and labels || {len(patient_slds)}")
        
        for idx, patient_id in enumerate(patient_slds):
            if patient_id not in cases_list:
                continue
            
            slide_ids = patient_data.loc[patient_id, 'slide_id']
            slide_ys = patient_data.loc[patient_id, 'class_name']
            
            if isinstance(slide_ids, str):
                slide_ids = np.array(slide_ids).reshape(-1)
                slide_ys = np.array(slide_ys).reshape(-1)
            else:
                slide_ids = slide_ids.values
                slide_ys = slide_ys.values
            
            for slide_idx, wsi_id in enumerate(slide_ids):
                wsi_path = process_wsi_paths(wsi_id, self.root)
                
                if os.path.isfile(wsi_path):
                    libs.append(wsi_path)
                    lib_y.append(slide_ys[slide_idx])
                    wsi_names.append(wsi_id)
                else:
                    print(os.path.split(wsi_path)[-1], " missing!!!")
            
            if int(idx) % int(len(patient_slds) * 0.25) == 0:
                print(f"Loading : [{idx+1}/{len(patient_slds)}] :::: ")
        
        print(len(libs), len(lib_y))
        return libs, lib_y, wsi_names