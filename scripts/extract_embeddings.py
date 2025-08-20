#!/usr/bin/env python3
"""
Text Embedding Extraction 

Extract concept and class embeddings from pretrained models (COSMO, CONCH, etc.)
for use in multimodal training and inference.

Author: Philip Chikontwe

Usage:
    # Extract embeddings using COSMO text model
    python scripts/extract_embeddings.py --dataset brain --text-model cosmo \
                                        --prev-type brainNIH --output-dir ./pretrained/concepts

    # Extract embeddings using CONCH model
    python scripts/extract_embeddings.py --dataset brain --text-model conch \
                                        --prev-type brainNIH --output-dir ./pretrained/concepts
"""

import warnings
warnings.filterwarnings("ignore")

import os
import sys
import json
import torch
import argparse
import numpy as np
from typing import List, Dict, Optional
from pathlib import Path

import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from cosmo.models.llm import load_peft_model_checkpoint
from cosmo.utils.label_mappings import get_label_mapping

# Configure logging
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class EmbeddingExtractor:
    """
    Extract text embeddings for concepts and classes using various pretrained models.
    """
    
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = None
        self.model = None
        
        # Template prompts for different contexts
        self.templates = [
            "CLASSNAME.",
            "a photomicrograph showing CLASSNAME.",
            "a photomicrograph of CLASSNAME.", 
            "an image of CLASSNAME.",
            "an image showing CLASSNAME.",
            "an example of CLASSNAME.",
            "CLASSNAME is shown.",
            "this is CLASSNAME.",
            "there is CLASSNAME.",
            "a histopathological image showing CLASSNAME.",
            "a histopathological image of CLASSNAME.",
            "a histopathological photograph of CLASSNAME.",
            "a histopathological photograph showing CLASSNAME.",
            "shows CLASSNAME.",
            "presence of CLASSNAME.",
            "CLASSNAME is present.",
            "an H&E stained image of CLASSNAME.",
            "an H&E stained image showing CLASSNAME.",
            "an H&E image showing CLASSNAME.",
            "an H&E image of CLASSNAME.",
            "CLASSNAME, H&E stain.",
            "CLASSNAME, H&E."
        ]
        
        logger.info(f"Initializing embedding extractor on {self.device}")
    
    def get_template_prompts(self, class_names: List[List[str]], template_type: str = "full") -> List[List[str]]:
        """
        Generate template-based prompts for class names.
        
        Args:
            class_names: List of lists containing class name variants
            template_type: Type of templates to use ("full", "simple", "hne")
            
        Returns:
            List of lists containing templated prompts
        """
        if template_type == "simple":
            templates = ["CLASSNAME."]
        elif template_type == "hne":
            templates = ["an H&E stained image of CLASSNAME."]
        elif template_type == "minimal":
            templates = ["CLASSNAME.", "an H&E stained image of CLASSNAME.", "this is CLASSNAME."]
        else:
            templates = self.templates
            
        cls_templates = []
        for i, names in enumerate(class_names):
            cls_template = []
            for name in names:
                cls_template.extend([template.replace('CLASSNAME', name) for template in templates])
            cls_templates.append(cls_template)
            
        return cls_templates
    
    def load_pretrained_model(self) -> tuple:
        """
        Load pretrained text model based on specified type.
        
        Returns:
            Tuple of (model, tokenizer)
        """
        text_model = self.args.text_model
        logger.info(f"Loading pretrained model: {text_model}")
        
        if text_model == "cosmo":
            return self._load_cosmo_model()
        elif text_model == "conch":
            return self._load_conch_model()
        elif text_model == "biobert":
            return self._load_biobert_model()
        elif text_model == "clinicalbert":
            return self._load_clinicalbert_model()
        else:
            raise ValueError(f"Unsupported text model: {text_model}")
    
    def _load_cosmo_model(self) -> tuple:
        """Load COSMO PEFT model."""
        checkpoint_path = self.args.checkpoint or "./checkpoints/cosmollm"
        
        if not os.path.exists(checkpoint_path):
            raise ValueError(f"COSMO checkpoint not found: {checkpoint_path}")
        
        model, tokenizer = load_peft_model_checkpoint(checkpoint_path, self.device)
        return model, tokenizer
    
    def _load_conch_model(self) -> tuple:
        """Load CONCH VLM model."""
        try:
            from models.conch.open_clip_custom import create_model_from_pretrained, get_tokenizer
        except ImportError:
            raise ImportError("CONCH model dependencies not available")
        
        checkpoint_path = self.args.checkpoint or "./checkpoints/conch/pytorch_model.bin"
        
        if not os.path.exists(checkpoint_path):
            raise ValueError(f"CONCH checkpoint not found: {checkpoint_path}")
        
        with torch.no_grad():
            model, _ = create_model_from_pretrained("conch_ViT-B-16", checkpoint_path)
            model.eval()
            tokenizer = get_tokenizer()
            
        return model, tokenizer
    
    def _load_biobert_model(self) -> tuple:
        """Load BioBERT PEFT model."""
        checkpoint_path = self.args.checkpoint or "./checkpoints/biobertllm"
        
        if not os.path.exists(checkpoint_path):
            raise ValueError(f"BioBERT checkpoint not found: {checkpoint_path}")
        
        model, tokenizer = load_peft_model_checkpoint(checkpoint_path, self.device)
        return model, tokenizer
    
    def _load_clinicalbert_model(self) -> tuple:
        """Load ClinicalBERT PEFT model.""" 
        checkpoint_path = self.args.checkpoint or "./checkpoints/clinicalbertllm"
        
        if not os.path.exists(checkpoint_path):
            raise ValueError(f"ClinicalBERT checkpoint not found: {checkpoint_path}")
        
        model, tokenizer = load_peft_model_checkpoint(checkpoint_path, self.device)
        return model, tokenizer
    
    def encode_text(self, texts: List[str], average: bool = True) -> torch.Tensor:
        """
        Encode text using the loaded model.
        
        Args:
            texts: List of text strings to encode
            average: Whether to average embeddings across texts
            
        Returns:
            Text embeddings tensor
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_pretrained_model() first.")
        
        text_model = self.args.text_model
        
        with torch.no_grad():
            if text_model == "conch":
                # INSTALL CONCH
                from models.conch.open_clip_custom import tokenize
                tokenized = tokenize(texts=texts, tokenizer=self.tokenizer).to(self.device)
                outputs = self.model.encode_text(tokenized, normalize=False)
                
            elif text_model in ["cosmo", "biobert", "clinicalbert"]:
                # Tokenize with proper settings for BERT models
                tokenized = self.tokenizer(
                    texts,
                    add_special_tokens=True,
                    max_length=256,
                    truncation=True,
                    padding='max_length',
                    return_tensors='pt'
                )
                
                # Move to device
                tokenized['input_ids'] = tokenized['input_ids'].to(self.device)
                tokenized['attention_mask'] = tokenized['attention_mask'].to(self.device)
                
                outputs = self.model.encode_text(tokenized)
                
            else:
                raise ValueError(f"Encoding not implemented for {text_model}")
        
        if average and len(outputs) > 1:
            return outputs.mean(0, keepdim=True).cpu()
        else:
            return outputs.cpu()
    
    def load_knowledge_data(self) -> tuple:
        """
        Load concept and class name data from knowledge templates.
        
        Returns:
            Tuple of (concepts_dict, class_names_dict)
        """
        dataset       = self.args.dataset
        templates_dir = self.args.templates_dir
        concepts_dir  = self.args.concepts_dir
        
        # Template file paths
        concepts_file    = os.path.join(concepts_dir, f"{dataset}_closed_concepts.json")
        class_names_file = os.path.join(templates_dir, f"{dataset}KT_prompt_names_vlm.json")
        
        if not os.path.exists(concepts_file):
            raise ValueError(f"Concepts file not found: {concepts_file}")
        if not os.path.exists(class_names_file):
            raise ValueError(f"Class names file not found: {class_names_file}")
        
        # Load concepts
        with open(concepts_file, 'r') as f:
            concepts_dict = json.load(f)
        logger.info(f"Loaded concepts from {concepts_file}")
        
        # Load class names
        with open(class_names_file, 'r') as f:
            class_names_dict = json.load(f)
        logger.info(f"Loaded class names from {class_names_file}")
        
        return concepts_dict, class_names_dict
    
    def extract_concept_embeddings(self, concepts_dict: Dict) -> np.ndarray:
        """
        Extract embeddings for all concepts.
        
        Args:
            concepts_dict: Dictionary mapping class names to concept lists
            
        Returns:
            Concept embeddings array
        """
        logger.info("Extracting concept embeddings...")
        
        # Get class mapping
        label_mapping = get_label_mapping(self.args.prev_type)
        classes = list(label_mapping.keys())
        
        all_embeddings = []
        
        for idx, class_name in enumerate(classes):
            if class_name not in concepts_dict:
                logger.warning(f"Class {class_name} not found in concepts dict")
                continue
            
            concepts = concepts_dict[class_name]
            logger.info(f"Processing {len(concepts)} concepts for class {idx}: {class_name}")
            
            class_embeddings = []
            for concept_idx, concept in enumerate(concepts):
                # Create templated prompts for this concept
                if self.args.text_model == "conch":
                    prompts = [f"an H&E stained image of {concept}."]
                else:
                    prompts = [f"{concept}."]
                
                # Extract embeddings
                embedding = self.encode_text(prompts, average=True)
                class_embeddings.append(embedding)
                
                if (concept_idx + 1) % 10 == 0:
                    logger.info(f"  Processed {concept_idx + 1}/{len(concepts)} concepts")
            
            # Concatenate all concept embeddings for this class
            class_embeddings = torch.cat(class_embeddings, 0)
            all_embeddings.append(class_embeddings)
            logger.info(f"Class {idx} embeddings shape: {class_embeddings.shape}")
        
        # Concatenate all embeddings
        all_embeddings = torch.cat(all_embeddings, 0)
        logger.info(f"Final concept embeddings shape: {all_embeddings.shape}")
        
        return all_embeddings.numpy()
    
    def extract_class_embeddings(self, class_names_dict: Dict) -> np.ndarray:
        """
        Extract embeddings for class names.
        
        Args:
            class_names_dict: Dictionary mapping class names to name variants
            
        Returns:
            Class embeddings array
        """
        logger.info("Extracting class name embeddings...")
        
        # Get class mapping
        label_mapping = get_label_mapping(self.args.prev_type)
        classes = list(label_mapping.keys())
        
        # Prepare class names
        class_name_lists = []
        for class_name in classes:
            if class_name in class_names_dict:
                class_name_lists.append(class_names_dict[class_name])
            else:
                logger.warning(f"Class {class_name} not found in class names dict, using class name")
                class_name_lists.append([class_name])
        
        # Generate templated prompts
        template_type = "hne" if self.args.text_model == "conch" else "simple"
        templated_prompts = self.get_template_prompts(class_name_lists, template_type)
        
        all_embeddings = []
        for idx, prompts in enumerate(templated_prompts):
            logger.info(f"Processing {len(prompts)} prompts for class {idx}: {classes[idx]}")
            
            # Extract embeddings
            embedding = self.encode_text(prompts, average=True)
            all_embeddings.append(embedding)
            logger.info(f"Class {idx} embedding shape: {embedding.shape}")
        
        # Concatenate all embeddings
        all_embeddings = torch.cat(all_embeddings, 0)
        logger.info(f"Final class embeddings shape: {all_embeddings.shape}")
        
        return all_embeddings.numpy()
    
    def save_embeddings(self, embeddings: np.ndarray, filename: str):
        """
        Save embeddings to file.
        
        Args:
            embeddings: Embeddings array to save
            filename: Output filename
        """
        os.makedirs(self.args.output_dir, exist_ok=True)
        filepath = os.path.join(self.args.output_dir, filename)
        
        np.save(filepath, embeddings)
        logger.info(f"Saved embeddings to {filepath}")
        logger.info(f"Embeddings shape: {embeddings.shape}")
    
    def run_extraction(self):
        """Run the complete embedding extraction pipeline."""
        logger.info("Starting embedding extraction...")
        
        # Load model
        self.model, self.tokenizer = self.load_pretrained_model()
        self.model.eval()
        
        # Load knowledge data
        concepts_dict, class_names_dict = self.load_knowledge_data()
        
        # Extract concept embeddings
        concept_embeddings = self.extract_concept_embeddings(concepts_dict)
        concept_filename = f"{self.args.prev_type}_{self.args.text_model}_concepts.npy"
        self.save_embeddings(concept_embeddings, concept_filename)
        
        # Extract class embeddings
        class_embeddings = self.extract_class_embeddings(class_names_dict)
        class_filename = f"{self.args.prev_type}_{self.args.text_model}_class.npy"
        self.save_embeddings(class_embeddings, class_filename)
        
        
        logger.info(f"Output files:")
        logger.info(f"  - {os.path.join(self.args.output_dir, concept_filename)}")
        logger.info(f"  - {os.path.join(self.args.output_dir, class_filename)}")


def main():
    parser = argparse.ArgumentParser(description='Extract text embeddings for COSMO')
    
    # Data arguments
    parser.add_argument('--dataset', type=str, default='brain',
                       choices=['brain', 'lung', 'kidney'],
                       help='Dataset/organ type')
    parser.add_argument('--prev-type', type=str, default='brainNIH',
                       choices=['brainNIH', 'lungNIH', 'kidneyNIH'],
                       help='Prevalence type for label mapping')
    parser.add_argument('--templates-dir', type=str, 
                       default='./src/cosmo/data/knowledge/templates',
                       help='Directory containing knowledge templates')
    parser.add_argument('--concepts-dir', type=str, 
                       default='./pretrained/concepts',
                       help='Directory containing closed concepts')
    
    # Model arguments
    parser.add_argument('--text-model', type=str, default='cosmo',
                       choices=['cosmo', 'conch', 'biobert', 'clinicalbert'],
                       help='Text model type')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to model checkpoint (optional)')
    
    # Output arguments
    parser.add_argument('--output-dir', type=str, default='./pretrained/concepts_example',
                       help='Output directory for embeddings')
    
    args = parser.parse_args()
    
    # Validate arguments
    if not os.path.exists(args.templates_dir):
        raise ValueError(f"Templates directory not found: {args.templates_dir}")
    
    if not os.path.exists(args.concepts_dir):
        raise ValueError(f"Concepts directory not found: {args.concepts_dir}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Run extraction
    extractor = EmbeddingExtractor(args)
    extractor.run_extraction()
    
    logger.info("Done!")


if __name__ == '__main__':
    main()