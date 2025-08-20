#!/usr/bin/env python3
"""
PathKT Data Loaders

Data loading utilities for COSMO pathology knowledge training.
Includes dataset classes and samplers for multi-tissue training.

Author: Philip Chikontwe
"""

import os
import json
import pandas as pd
import torch
import numpy as np
import random
import copy
from collections import defaultdict, deque
from typing import List, Tuple, Dict, Optional

from torch.utils.data import Dataset, Sampler

from collections import defaultdict
from collections import Counter

# LabelMAp 
LABEL_MAP = {

    "brainNIH": {
        # Common
        'Adult-type diffuse gliomas': 0, 
        'Circumscribed astrocytic gliomas': 1,
        'Meningiomas'      : 2,
        
        # Rare
        'Ependymal Tumours': 3, 
        'Paediatric-type diffuse low-grade gliomas': 4,
        'Glioneuronal and neuronal tumours': 5, 
        'Embryonal Tumours': 6,
    },  

    # Lung 
    "lungNIH" : {
        "Lung Adenocarcinoma" : 0,
        "Lung Squamous Cell Carcinoma": 1,
        

        # Rare
        "Small Cell Lung Cancer" : 2,
        "Lung Carcinoid": 3,
        "Lung Neuroendocrine Tumor": 4,
        "Lung Adenosquamous Carcinoma": 5
    },

    "kidneyNIH" : {
        "Chromophobe Renal Cell Carcinoma": 0,
        "Papillary Renal Cell Carcinoma"  : 1,
        "Renal Clear Cell Carcinoma"      : 2,

        # Rare
        "Collecting Duct Renal Cell Carcinoma" : 3,
        "Wilms Tumor": 4
    },

    # Example of custom data missing other types (common/uncommon)
    "tcgakidney" : {
        "Chromophobe Renal Cell Carcinoma": 0,
        "Papillary Renal Cell Carcinoma"  : 1,
        "Renal Clear Cell Carcinoma"      : 2,
    },
    
}

class BalancedIdentitySampler(Sampler):
    """
    Enhanced identity sampler that ensures better coverage of the dataset.
    Balances sampling across all classes and repeats data as needed.
    """
    def __init__(self, data_source, batch_size, num_instances, max_iter=None, multi=False):
        self.data_source = data_source
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.num_dids_per_batch = self.batch_size // self.num_instances
        self.max_iter = max_iter  # Optional maximum iterations per epoch
        
        # Index dictionary for each class ID
        self.index_dic = defaultdict(list)
        
        # Count instances per class for statistics
        self.class_counts = defaultdict(int)
        
        if multi:
            for index, (_, _, did, _, attr, *_) in enumerate(self.data_source):
                self.index_dic[did].append([index, attr])
                self.class_counts[did] += 1
        else:
            for index, (_, did, _, attr) in enumerate(self.data_source):
                self.index_dic[did].append([index, attr])
                self.class_counts[did] += 1
        
        # # Update for the new format with slide_path
        # for index, (_, _, did, _, attr, *_) in enumerate(self.data_source):
        #     self.index_dic[did].append([index, attr])
            
        
        self.dids = list(self.index_dic.keys())
        
        # Log class distribution
        print(f"Dataset class distribution:")
        for did in self.dids:
            print(f"Class {did}: {self.class_counts[did]} instances")
        
        # Calculate the expected length with better approximation
        if self.max_iter is not None:
            self.length = self.max_iter * self.batch_size
        else:
            # Estimate more accurately based on all available combinations
            total_instances = sum(self.class_counts.values())
            self.length = (total_instances // self.num_instances) * self.num_instances
            
        print(f"Sampler will generate approximately {self.length // self.batch_size} batches per epoch")
    
    def __iter__(self):
        # Create batches with class IDs ensuring all classes are used
        all_batch_idxs = []
        all_dids = self.dids.copy()
        
        # Shuffle class IDs
        random.shuffle(all_dids)
        
        # Create a queue of class IDs to ensure balanced sampling
        class_queue = deque(all_dids)
        
        # Keep track of remaining instances per class
        instance_tracker = {}
        for did in all_dids:
            # Create a copy of the original indices
            instance_tracker[did] = copy.deepcopy(self.index_dic[did])
            # Shuffle to ensure randomness
            random.shuffle(instance_tracker[did])
        
        # Function to get instances for a class, regenerating if needed
        def get_instances_for_class(did, count):
            instances = []
            
            # If we need more instances than available
            if len(instance_tracker[did]) < count:
                # Use all remaining instances
                instances.extend([idx[0] for idx in instance_tracker[did]])
                
                # Regenerate the instances for this class
                instance_tracker[did] = copy.deepcopy(self.index_dic[did])
                random.shuffle(instance_tracker[did])
                
                # Get additional instances needed
                additional_needed = count - len(instances)
                instances.extend([idx[0] for idx in instance_tracker[did][:additional_needed]])
                
                # Remove the used instances
                instance_tracker[did] = instance_tracker[did][additional_needed:]
            else:
                # Get the requested number of instances
                instances = [idx[0] for idx in instance_tracker[did][:count]]
                instance_tracker[did] = instance_tracker[did][count:]
            
            return instances
        
        # Generate batches until we reach the desired epoch size
        batch_count = 0
        max_batches = self.length // self.batch_size if self.max_iter is None else self.max_iter
        
        while batch_count < max_batches:
            # Get class IDs for this batch
            batch_dids = []
            for _ in range(self.num_dids_per_batch):
                # If queue is empty, refill it
                if not class_queue:
                    class_queue = deque(all_dids.copy())
                    random.shuffle(class_queue)
                
                # Get next class ID
                batch_dids.append(class_queue.popleft())
            
            # Get instances for each class
            batch_indices = []
            for did in batch_dids:
                batch_indices.extend(get_instances_for_class(did, self.num_instances))
            
            # Add to final batches
            all_batch_idxs.extend(batch_indices)
            batch_count += 1
        
        return iter(all_batch_idxs)

    def __len__(self):
        return self.length
    

class RandomIdentitySampler(Sampler):
    """
    Randomly sample N identities, then for each identity,
    randomly sample K instances, therefore batch size is N*K.
    Args:
    - data_source (list): list of (text, did, tid).
    - num_instances (int): number of instances per identity in a batch.
    - batch_size (int): number of examples in a batch.
    """

    def __init__(self, data_source, batch_size, num_instances, multi=False):
        self.data_source = data_source
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.num_dids_per_batch = self.batch_size // self.num_instances
        self.index_dic = defaultdict(list) #dict with list value
        #{783: [0, 5, 116, 876, 1554, 2041],...,}
        if multi:
            # for index, (_, _, did, _, attr) in enumerate(self.data_source):
            #     self.index_dic[did].append([index, attr])
            
            for index, (_, _, did, _, attr, *_) in enumerate(self.data_source):
                self.index_dic[did].append([index, attr])
        else:
            for index, (_, did, _, attr) in enumerate(self.data_source):
                self.index_dic[did].append([index, attr])

        self.dids = list(self.index_dic.keys())

        # estimate number of examples in an epoch
        self.length = 0
        for did in self.dids:
            idxs = self.index_dic[did]
            num = len(idxs)
            if num < self.num_instances:
                num = self.num_instances
            self.length += num - num % self.num_instances

    def __iter__(self):
        batch_idxs_dict = defaultdict(list)

        for did in self.dids:
            idxs = copy.deepcopy(self.index_dic[did])
            if len(idxs) < self.num_instances:
                sample_index = np.random.choice(list(range(len(idxs))), size=self.num_instances-len(idxs), replace=True)
                for i in list(sample_index):
                    idxs.append(idxs[i])
                random.shuffle(idxs)
                batch_idxs_dict[did].append([idx[0] for idx in idxs])

            ## each id contains at least main
            else:
                random.shuffle(idxs)
                for idx in idxs:
                    if idx[1] == 'main':
                        main_idx = idx[0]
                        break
                batch_idxs = []
                for idx in idxs:
                    if idx[1] == 'main':
                        continue
                    batch_idxs.append(idx[0])
                    if len(batch_idxs) == self.num_instances-1:
                        batch_idxs.append(main_idx)
                        batch_idxs_dict[did].append(batch_idxs)
                        batch_idxs = []

                ## don't discard the rest
                if len(batch_idxs) > 0:
                    batch_idxs.append(main_idx)
                    sample_index = np.random.choice(batch_idxs, size=self.num_instances-len(batch_idxs), replace=True)
                    batch_idxs += list(sample_index)
                    random.shuffle(batch_idxs)
                    batch_idxs_dict[did].append(batch_idxs)


        avai_dids  = copy.deepcopy(self.dids)
        final_idxs = []

        while len(avai_dids) >= self.num_dids_per_batch:
            selected_dids = random.sample(avai_dids, self.num_dids_per_batch)
            for did in selected_dids:
                batch_idxs = batch_idxs_dict[did].pop(0)
                final_idxs.extend(batch_idxs)
                if len(batch_idxs_dict[did]) == 0:
                    avai_dids.remove(did)

        return iter(final_idxs)

    def __len__(self):
        return self.length
    

class PKDataset(Dataset):
    def __init__(self, dataset, transform=None, istrain = True):
        self.dataset   = dataset
        self.transform = transform
        self.istrain   = istrain
        self.ndims     = 0

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        text, did, tid, attr = self.dataset[index]
        
        if self.transform is not None:
            text = self.transform(text)
         
        return text, did, tid, attr
    
    
class PathKnowledge(Dataset):
    """
    """

    def __init__(self, dataset_root, tissue="brain", wsi_json=None, k_val=0):

        self.k_val  = k_val
        self.tissue = tissue
        # cancers with high level categorizations
        self.coarse_types = ['brainNIH','lungNIH']

        print("Cancer in Coarse Types [", self.tissue in self.coarse_types,"]")
        # prevelance dict.
        prev_map = copy.deepcopy(LABEL_MAP)[tissue]
        print(prev_map)
        # Load wsi_json 
        with open(wsi_json, 'r') as f:
            cancer_names = json.load(f)

        self.label2name = {int(v):k for k,v in prev_map.items()}
        # {0: [all names for class 0],  1 : [], etc}
        self.text2name  = {int(k):cancer_names[v] for k,v in self.label2name.items()}

        print()
        print(self.text2name.items())
    
        print(f"Loading from {dataset_root} | [{tissue}] ")
        train_dir = os.path.join(dataset_root, f'{tissue}_knowledge_train.csv')
        self.train_list = pd.read_csv(train_dir,sep='\t',).values.tolist()

        # If not working with coarse types. 
        # if not self.tissue in self.coarse_types:
        #     self.text2name = self.get_names(self.train_list)
        # print(self.text2name.items())

        self.train = self.process_list(self.train_list)

        dids = []
        for i in self.train:
            dids.append(i[1])
            
        self.uniq_cls = len(np.unique(dids))
        print(Counter(dids))
        print(np.unique(dids))
        print(self.uniq_cls," classes.")
        print(f" Train | {len(self.train_list):^5}")
        

        # assert len(np.unique(dids)) == len(self.text2name.keys()), "len(dids) != text2name.keys()"

    def get_names(self, data_list):
        dataset = []
        for item in data_list:
            entity_name, entity_text = item[0],item[1]
            tid  = int(entity_name.split('_')[1].split('t')[1])
            attr = entity_name.split('_')[2]

            if attr.startswith('main'):
                dataset.append(entity_text)

        if self.k_val:
            dataset = {k+self.k_val:[v] for k,v in enumerate(dataset)}
        else:
            dataset = {k:[v] for k,v in enumerate(dataset)}

        return dataset

    def is_related(self, text, keywords):
        if self.tissue in self.coarse_types:
            # Okay for brain not lung
            return any(keyword.lower() in text.lower() for keyword in keywords)
        else:
            return True if text in keywords else False

    def format_data(self, data_list):

        dataset  = []
        curr_cls = None
    
        for item in data_list:
            entity_name, entity_text = item[0],item[1]
            tid  = int(entity_name.split('_')[1].split('t')[1])
            attr = entity_name.split('_')[2]

            if attr.startswith('main'):
                for cls_idx, text_entities in self.text2name.items():
                    
                    if self.is_related(entity_text, text_entities):
                        curr_cls = cls_idx
                        break
                    
                if curr_cls is not None:
                    dataset.append((entity_text, curr_cls+self.k_val, tid, attr))
            else:
                if curr_cls is not None:
                    dataset.append((entity_text, curr_cls+self.k_val, tid, attr))

        return dataset

    def process_list(self, data_list):
        if isinstance(data_list,dict):
            dataset = {}
            for k,v in data_list.items():
                dataset[k] = self.format_data(v)
        elif isinstance(data_list,list):
            dataset = self.format_data(data_list)
        return dataset
    

class WSIPathKnowledge(Dataset):
   
    def __init__(self, dataset_root, wsi_info, tissue="brain", wsi_json=None, 
                 slide_max=16, max_texts_per_slide=5):

        self.max_text  = max_texts_per_slide
        self.slide_max = slide_max
        print("MAX SLIDES :: ", self.slide_max)
        print("MAX TEXT   :: ", self.max_text)
        
        self.tissue    = tissue
        # cancers with high level categorizations
        # used in relation mapping
        self.coarse_types = ['brainNIH','lungNIH']

        print("Cancer in Coarse Types [", self.tissue in self.coarse_types,"]")
        # prevelance dict.
        prev_map = copy.deepcopy(LABEL_MAP)[tissue]
        print(prev_map)
        
        # Load wsi_json 
        with open(wsi_json, 'r') as f:
            cancer_names = json.load(f)

        self.label2name = {int(v):k for k,v in prev_map.items()}
        self.text2name  = {int(k):cancer_names[v] for k,v in self.label2name.items()}

        #print(self.text2name.items())
        
        # wsi2text
        wsi_paths, wsi_targets = wsi_info
        self.wsipaths = {}
        for cls_idx in self.label2name.keys():
            slide_idxes = np.where(np.array(wsi_targets) == cls_idx)[0]
            self.wsipaths[cls_idx] = np.array(wsi_paths)[slide_idxes]
    
        print(f"Loading from {dataset_root} | [{tissue}] ")
        train_dir = os.path.join(dataset_root, f'{tissue}_knowledge_train.csv')
        
        self.train_list = pd.read_csv(train_dir,sep='\t',).values.tolist()       
        self.train      = self.process_list(self.train_list)
       
        dids = []
        for i in self.train:
            dids.append(i[2])

        print(Counter(dids), np.unique(dids), len(np.unique(dids))," classes.")
        print(f" Train | {len(self.train):^5}")
        assert len(np.unique(dids)) == len(self.text2name.keys())

    def is_related(self, text, keywords):
        return any(keyword.lower() in text.lower() for keyword in keywords)

    def format_data_(self, data_list,max_texts_per_slide=None):

        dataset  = []
        curr_cls = None
    
        for item in data_list:
            entity_name, entity_text = item[0],item[1]
            tid  = int(entity_name.split('_')[1].split('t')[1])
            attr = entity_name.split('_')[2]

            if attr.startswith('main'):
                for cls_idx, text_entities in self.text2name.items():
    
                    if self.is_related(entity_text, text_entities):
                        curr_cls = cls_idx
                        break
                
                #dataset.append((entity_text, curr_cls, tid, attr))
                if curr_cls is not None:
                    slides   = self.wsipaths[curr_cls][:self.slide_max]
                    for slide_path in slides:
                        dataset.append((slide_path, entity_text, curr_cls, tid, attr))
            else:
                #dataset.append((entity_text, curr_cls, tid, attr))
                if curr_cls is not None:
                    slides   = self.wsipaths[curr_cls][:self.slide_max]
                    for slide_path in slides:
                        dataset.append((slide_path, entity_text, curr_cls, tid, attr))

        return dataset

    def format_data(self, data_list, max_texts_per_slide=5, text_type_balance=True):
        """
        Process the data list with balanced text sampling
        
        Args:
            data_list: Raw data list to process
            max_texts_per_slide: Maximum number of text descriptions per slide
            text_type_balance: Whether to balance different text types (main, syn, def, etc.)
        
        Returns:
            Processed dataset with balanced text representations
        """
        dataset = []
        
        # Group data by class and text type
        class_text_data = {}
        
        # First pass: Group all texts by class and attribute type
        for item in data_list:
            entity_name, entity_text = item[0], item[1]
            tid  = int(entity_name.split('_')[1].split('t')[1])
            attr = entity_name.split('_')[2]
            
            # Determine class for main entries
            if attr.startswith('main'):
                for cls_idx, text_entities in self.text2name.items():
                    if self.is_related(entity_text, text_entities):
                        curr_cls = cls_idx
                        
                        # Initialize class entry if not exists
                        if curr_cls not in class_text_data:
                            class_text_data[curr_cls] = {'main': [], 'syn': [], 'def': [], 'histology': [], 'cytology': [], 'other': []}
                        
                        # Add main text
                        class_text_data[curr_cls]['main'].append((entity_text, tid, attr))
                        break
            else:
                # For non-main entries, determine the text type
                text_type = 'other'
                if attr.startswith('syn'):
                    text_type = 'syn'
                elif attr.startswith('def'):
                    text_type = 'def'
                elif attr.startswith('histology'):
                    text_type = 'histology'
                elif attr.startswith('cytology'):
                    text_type = 'cytology'
                
                # Add to the appropriate class and text type if we've seen the class before
                for cls_idx, cls_data in class_text_data.items():
                    if self.is_related(entity_text, self.text2name[cls_idx]):
                        class_text_data[cls_idx][text_type].append((entity_text, tid, attr))
                        break
        
        # Second pass: Create balanced dataset
        for cls_idx, cls_data in class_text_data.items():
            # Get slides for this class
            slides = self.wsipaths[cls_idx][:self.slide_max]
            
            if len(slides) == 0:
                continue  # Skip if no slides for this class
            
            # For each slide, select a balanced set of text descriptions
            for slide_path in slides:
                slide_texts = []
                
                # Always include main text if available
                if cls_data['main']:
                    slide_texts.append(random.choice(cls_data['main']))
                
                # Add balanced text types if requested
                if text_type_balance:
                    # Calculate how many text entries to include from each type
                    # Prioritize different text types
                    remaining_slots = max_texts_per_slide - len(slide_texts)
                    text_types = ['syn', 'def', 'histology', 'cytology', 'other']
                    
                    # Remove empty text types
                    text_types = [t for t in text_types if cls_data[t]]
                    
                    if text_types:
                        # Calculate slots per text type
                        slots_per_type = max(1, remaining_slots // len(text_types))
                        
                        # Add texts from each type
                        for text_type in text_types:
                            type_texts = cls_data[text_type]
                            if type_texts:
                                # Sample without replacement if possible
                                sample_size = min(slots_per_type, len(type_texts))
                                selected_texts = random.sample(type_texts, sample_size)
                                slide_texts.extend(selected_texts)
                                
                                # Stop if we've reached the maximum
                                if len(slide_texts) >= max_texts_per_slide:
                                    break
                else:
                    # Simply add texts until we reach the maximum
                    remaining_slots = max_texts_per_slide - len(slide_texts)
                    all_texts = []
                    for text_type in ['syn', 'def', 'histology', 'cytology', 'other']:
                        all_texts.extend(cls_data[text_type])
                    
                    if all_texts and remaining_slots > 0:
                        # Sample without replacement if possible
                        sample_size = min(remaining_slots, len(all_texts))
                        selected_texts = random.sample(all_texts, sample_size)
                        slide_texts.extend(selected_texts)
                
                # Add entries to dataset
                for entity_text, tid, attr in slide_texts:
                    dataset.append((slide_path, entity_text, cls_idx, tid, attr))
        
        return dataset
    
    def process_list(self, data_list):
        if isinstance(data_list,dict):
            dataset = {}
            for k,v in data_list.items():
                dataset[k] = self.format_data(v,max_texts_per_slide=self.max_text)
        elif isinstance(data_list,list):
            dataset = self.format_data(data_list,max_texts_per_slide=self.max_text)
        return dataset


