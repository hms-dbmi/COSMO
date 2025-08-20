#!/usr/bin/env python3
"""
Inference Script

Author: Philip Chikontwe

Usage:
    # inference with pretrained model
    python test_cosmo.py --wsi-root ./data/features --patient-csv ./data/patient.csv \
                        --split-csv ./data/splits_0.csv --checkpoint ./checkpoints/cosmo
"""

import warnings
warnings.filterwarnings("ignore")

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset

from sklearn.metrics import (
    accuracy_score, 
    balanced_accuracy_score, 
    cohen_kappa_score, 
    classification_report, 
    roc_auc_score
)

# COSMO imports
from src.cosmo.models.llm import MultiModalCLP
from src.cosmo.data.dataloaders.wsi_data import WSIBagDataset
from src.cosmo.utils.label_mappings import get_class_names

# Configure logging
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class COSMOInferenceEngine:
    """
    Inference engine for COSMO multimodal model.
    """
    
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Model configuration
        self.feature_embed_dim = 512
        self.prev_type         = args.prev_type
        
        # Results storage
        self.results = {}
        
    def load_model(self, checkpoint_path: str, visual_dim: int) -> nn.Module:
        """
        Load COSMO model with pretrained weights.
        
        Args:
            checkpoint_path: Path to model checkpoint
            visual_dim: Dimension of visual features
            
        Returns:
            Loaded COSMO model
        """
        logger.info(f"Loading COSMO model from {checkpoint_path}")
        
        # Initialize COSMO model
        model = MultiModalCLP(
            feature_embed_dim=self.feature_embed_dim,
            visual_dim=visual_dim,
            device=self.device,
            logit_data=None,
            vlm=False,
            prev_type=self.prev_type,
            no_concept=self.args.no_concept,
            text_model=self.args.text_model,
            concept_root=self.args.concept_root
        )
        
        # Load pretrained weights if provided
        if checkpoint_path and os.path.exists(checkpoint_path):
            if os.path.isdir(checkpoint_path):
                # Load from directory structure (visual_model_0.pt format)
                model_file = os.path.join(checkpoint_path, f"visual_model_{self.args.run_id}.pt")
                if not os.path.exists(model_file):
                    model_file = os.path.join(checkpoint_path, "visual_model_0.pt")
            else:
                # Direct file path
                model_file = checkpoint_path
                
            if os.path.exists(model_file):
                logger.info(f"Loading weights from {model_file}")
                checkpoint = torch.load(model_file, map_location=self.device)
                model.load_state_dict(checkpoint, strict=True)
                logger.info("Successfully loaded pretrained weights")
            else:
                logger.warning(f"Checkpoint file not found: {model_file}")
                logger.warning("Proceeding with randomly initialized weights")
        else:
            logger.warning("No checkpoint provided. Using randomly initialized weights")
            sys.exit(1)
        
        model = nn.DataParallel(model.to(self.device))
        model.eval()
        return model
        
    def create_dataloaders(self) -> Tuple[DataLoader, DataLoader]:
        """
        Create data loaders for seen and unseen classes.
        
        Returns:
            Tuple of (seen_loader, unseen_loader)
        """
        logger.info("Creating data loaders...")
        
        # Create seen dataset
        seen_dataset = WSIBagDataset(
            root=self.args.wsi_root,
            patient_info=self.args.patient_csv,
            split_file=os.path.join(self.args.split_csv,f"splits_{self.args.run_id}.csv"),
            train='test',
            zs_state='seen',
            zsmode=True,
            prev_type=self.prev_type,
            use_spatial=False
        )
        
        # Create unseen dataset
        unseen_dataset = WSIBagDataset(
            root=self.args.wsi_root,
            patient_info=self.args.patient_csv,
            split_file=os.path.join(self.args.split_csv,f"splits_{self.args.run_id}.csv"),
            train='test',
            zs_state='unseen',
            zsmode=True,
            prev_type=self.prev_type,
            use_spatial=False
        )
        
        # Create data loaders
        seen_loader = DataLoader(
            seen_dataset, 
            batch_size=1, 
            shuffle=False, 
            num_workers=self.args.num_workers,
            pin_memory=True
        )
        
        unseen_loader = DataLoader(
            unseen_dataset, 
            batch_size=1, 
            shuffle=False, 
            num_workers=self.args.num_workers,
            pin_memory=True
        )
        
        logger.info(f"Created data loaders - Seen: {len(seen_dataset)}, Unseen: {len(unseen_dataset)}")
        return seen_loader, unseen_loader
    
    def evaluate_model(self, model: nn.Module, dataloader: DataLoader, 
                      class_ids: List[int]) -> Dict:
        """
        Evaluate model on given dataloader.
        
        Args:
            model: COSMO model
            dataloader: Data loader
            class_ids: List of class IDs for this split
            
        Returns:
            Dictionary with predictions and metrics
        """
        model.eval()
        
        # Label remapping for evaluation
        label_remap = {v: i for i, v in enumerate(class_ids)}
        
        scores = []
        labels = []
        predictions = []
        slide_names = []
        
        logger.info(f"Evaluating on {len(dataloader)} samples...")
        
        with torch.no_grad():
            for sample in dataloader:
                bag = sample[0].float().to(self.device, non_blocking=True)
                label = sample[1].item()
                slide_name = sample[2][0]
                
                # Model inference
                logits = model.module.inference(bag)
                
                # Store results
                scores.append(logits)
                remapped_label = label_remap[label]
                labels.append(remapped_label)
                slide_names.append(slide_name)
                
                # Get predictions for this split's classes only
                split_logits = logits[:, class_ids]
                prob_y = torch.softmax(split_logits, dim=1).cpu().numpy()[0]
                predictions.append((slide_name, remapped_label, prob_y.tolist()))
        
        # Concatenate scores and compute final predictions
        scores = torch.cat(scores, dim=0)
        logits = scores.cpu().numpy()[:, class_ids]
        labels = np.array(labels)
        preds = np.argmax(logits, axis=1)
        
        # Compute metrics
        results = self.compute_metrics(labels, preds, logits)
        results['predictions'] = predictions
        results['logits'] = logits
        results['labels'] = labels
        results['preds'] = preds
        
        return results
    
    def compute_metrics(self, labels: np.ndarray, preds: np.ndarray, 
                       logits: np.ndarray) -> Dict:
        """
        Compute evaluation metrics.
        
        Args:
            labels: Ground truth labels
            preds: Predicted labels  
            logits: Model logits
            
        Returns:
            Dictionary with computed metrics
        """
        # Compute probabilities
        probs = F.softmax(torch.from_numpy(logits), dim=1).numpy()
        
        # Basic metrics
        acc = accuracy_score(labels, preds)
        bacc = balanced_accuracy_score(labels, preds)
        weighted_kappa = cohen_kappa_score(labels, preds, weights='quadratic')
        kappa = cohen_kappa_score(labels, preds)
        
        # Classification report
        cls_rep = classification_report(labels, preds, output_dict=True, zero_division=0)
        f1_score = cls_rep['weighted avg']['f1-score']
        
        # ROC AUC
        n_classes = probs.shape[1]
        try:
            if n_classes == 2:
                roc_auc = roc_auc_score(labels, probs[:, 1])
            else:
                roc_auc = roc_auc_score(labels, probs, multi_class='ovo', average='macro')
        except ValueError:
            roc_auc = np.nan
        
        return {
            'accuracy': acc,
            'balanced_accuracy': bacc,
            'weighted_kappa': weighted_kappa,
            'cohen_kappa': kappa,
            'f1_score': f1_score,
            'roc_auc': roc_auc
        }
    
    def save_predictions(self, predictions: List, split_name: str, class_names: List[str]):
        """
        Save predictions to CSV file.
        
        Args:
            predictions: List of (slide_name, label, probabilities)
            split_name: Name of split ('seen', 'unseen', 'all')
            class_names: List of class names for this split
        """
        output_dir = self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        csv_file = os.path.join(output_dir, f"predictions_cosmo_{self.args.run_id}_{split_name}.csv")
        
        # Create header
        class_header = ','.join(class_names)
        
        with open(csv_file, 'w') as f:
            f.write(f'slide,label,{class_header}\n')
            
            for slide_name, label, probs in predictions:
                prob_str = ','.join([f'{p:.6f}' for p in probs])
                f.write(f'{slide_name},{label},{prob_str}\n')
        
        logger.info(f"Saved predictions to {csv_file}")
    
    def save_metrics(self, results: Dict, run_id: int = 0):
        """
        Save evaluation metrics to CSV file.
        
        Args:
            results: Dictionary with seen/unseen results
            run_id: Run identifier
        """
        output_dir = self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        csv_file = os.path.join(output_dir, 'results_summary.csv')
        
        # Prepare results
        seen_res = results['seen']
        unseen_res = results['unseen']
        
        # Compute harmonic means
        h_acc = 2 * seen_res['balanced_accuracy'] * unseen_res['balanced_accuracy'] / \
                (seen_res['balanced_accuracy'] + unseen_res['balanced_accuracy'])
        h_f1 = 2 * seen_res['f1_score'] * unseen_res['f1_score'] / \
               (seen_res['f1_score'] + unseen_res['f1_score'])
        
        # Create summary row
        summary_data = {
            'Method': 'COSMO',
            'Run': run_id,
            'Seen_Accuracy': seen_res['accuracy'],
            'Seen_BalancedAccuracy': seen_res['balanced_accuracy'],
            'Seen_F1Score': seen_res['f1_score'],
            'Seen_ROC_AUC': seen_res['roc_auc'],
            'Unseen_Accuracy': unseen_res['accuracy'],
            'Unseen_BalancedAccuracy': unseen_res['balanced_accuracy'],
            'Unseen_F1Score': unseen_res['f1_score'],
            'Unseen_ROC_AUC': unseen_res['roc_auc'],
            'Harmonic_BalancedAccuracy': h_acc,
            'Harmonic_F1Score': h_f1,
            'Dataset': self.args.prev_type
        }
        
        # Write to CSV
        df = pd.DataFrame([summary_data])
        if os.path.exists(csv_file):
            df.to_csv(csv_file, mode='a', header=False, index=False)
        else:
            df.to_csv(csv_file, index=False)
        
        logger.info(f"Saved metrics to {csv_file}")
        
        # Print results
        logger.info("="*60)
        logger.info("EVALUATION RESULTS")
        logger.info("="*60)
        logger.info(f"Seen Classes - Accuracy: {seen_res['accuracy']:.3f}, "
                   f"Balanced Accuracy: {seen_res['balanced_accuracy']:.3f}")
        logger.info(f"Unseen Classes - Accuracy: {unseen_res['accuracy']:.3f}, "
                   f"Balanced Accuracy: {unseen_res['balanced_accuracy']:.3f}")
        logger.info(f"Harmonic Mean - Balanced Accuracy: {h_acc:.3f}")
        logger.info("="*60)
    
    def run_inference(self):
        """Run complete inference pipeline."""
        logger.info("Starting COSMO inference...")
        
        # Create data loaders
        seen_loader, unseen_loader = self.create_dataloaders()
        
        # Get feature dimensions from first loader
        visual_dim = seen_loader.dataset.ndims
        logger.info(f"Visual feature dimension: {visual_dim}")
        
        # Load model
        model = self.load_model(self.args.checkpoint, visual_dim)
        
        # Get class information
        seen_classes    = seen_loader.dataset.label_ids
        unseen_classes  = unseen_loader.dataset.label_ids
        all_class_names = get_class_names(self.prev_type)
        
        logger.info(f"Seen classes  : {seen_classes}")
        logger.info(f"Unseen classes: {unseen_classes}")
        
        # Evaluate on seen classes
        logger.info("Evaluating on seen classes...")
        seen_results = self.evaluate_model(model, seen_loader, seen_classes)
        
        # Evaluate on unseen classes  
        logger.info("Evaluating on unseen classes...")
        unseen_results = self.evaluate_model(model, unseen_loader, unseen_classes)
        
        # Save predictions
        seen_class_names   = [all_class_names[i] for i in seen_classes]
        unseen_class_names = [all_class_names[i] for i in unseen_classes]
        
        self.save_predictions(seen_results['predictions'], 'seen', seen_class_names)
        self.save_predictions(unseen_results['predictions'], 'unseen', unseen_class_names)
        
        # Combined evaluation for completeness
        if self.args.eval_all:
            logger.info("Evaluating on all classes...")
            concat_dataset = ConcatDataset([seen_loader.dataset, unseen_loader.dataset])
            all_loader = DataLoader(concat_dataset, batch_size=1, shuffle=False, 
                                  num_workers=self.args.num_workers)
            
            
            all_classes = [i for i in seen_classes.tolist()]
            all_classes.extend(unseen_classes.tolist())
            seen_class_names.extend(unseen_class_names)
            
            all_results = self.evaluate_model(model, all_loader, all_classes)
            self.save_predictions(all_results['predictions'], 'all', seen_class_names)
        
        # Save metrics
        results = {'seen': seen_results, 'unseen': unseen_results}
        self.save_metrics(results, self.args.run_id)
        
        return seen_results['balanced_accuracy'], unseen_results['balanced_accuracy']


def main():
    parser = argparse.ArgumentParser(description='Model Inference')
    
    # Data arguments
    parser.add_argument('--wsi-root', type=str, required=True,
                       help='Root directory containing WSI feature files')
    parser.add_argument('--patient-csv', type=str, required=True,
                       help='CSV file with patient information')
    parser.add_argument('--split-csv', type=str, required=True,
                       help='CSV file with train/val/test splits')
    
    # Model arguments
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to model checkpoint (file or directory)')
    parser.add_argument('--prev-type', type=str, default='brainNIH',
                       choices=['brainNIH'],
                       help='Prevalence type for label mapping')
    parser.add_argument('--text-model', type=str, default='cosmo',
                       help='Text model type')
    parser.add_argument('--concept-root', type=str, default='./pretrained/concepts',
                       help='Root directory for concept embeddings')
    parser.add_argument('--no-concept', action='store_true',
                       help='Disable concept-based deconfounding')
    
    # Inference arguments
    parser.add_argument('--runs', type=int, default=1,
                       help='Number of inference runs')
    parser.add_argument('--run-id', type=int, default=0,
                       help='Run ID for checkpoint loading')
    parser.add_argument('--num-workers', type=int, default=12,
                       help='Number of data loader workers')
    parser.add_argument('--eval-all', action='store_true',
                       help='evaluate on combined seen+unseen classes')
    
    # Output arguments
    parser.add_argument('--output-dir', type=str, default='./results',
                       help='Output directory for results')
    
    # Cross-dataset evaluation
    parser.add_argument('--cross-dataset', type=str, default=None,
                       help='Cross-dataset evaluation target')
    
    args = parser.parse_args()
    
    # Validate arguments
    if not os.path.exists(args.wsi_root):
        raise ValueError(f"WSI root directory not found: {args.wsi_root}")
    if not os.path.exists(args.patient_csv):
        raise ValueError(f"Patient CSV not found: {args.patient_csv}")
    if not os.path.exists(args.split_csv):
        raise ValueError(f"Split CSV not found: {args.split_csv}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Run inference
    if args.runs > 1:
        logger.info(f"Running inference for {args.runs} runs...")
        seen_accs = []
        unseen_accs = []
        
        for run in range(args.runs):
            args.run_id = run
            logger.info(f"Run {run + 1}/{args.runs}")
            
            engine = COSMOInferenceEngine(args)
            seen_acc, unseen_acc = engine.run_inference()
            
            seen_accs.append(seen_acc)
            unseen_accs.append(unseen_acc)
        
        # Summary statistics
        seen_mean, seen_std = np.mean(seen_accs), np.std(seen_accs)
        unseen_mean, unseen_std = np.mean(unseen_accs), np.std(unseen_accs)
        harmonic_mean = 2 * seen_mean * unseen_mean / (seen_mean + unseen_mean)
        
        logger.info("="*60)
        logger.info("FINAL RESULTS ACROSS ALL RUNS")
        logger.info("="*60)
        logger.info(f"Seen Classes: {seen_mean:.3f} ± {seen_std:.3f}")
        logger.info(f"Unseen Classes: {unseen_mean:.3f} ± {unseen_std:.3f}")
        logger.info(f"Harmonic Mean: {harmonic_mean:.3f}")
        logger.info("="*60)
        
    else:
        engine = COSMOInferenceEngine(args)
        engine.run_inference()
    
    logger.info("Inference completed!")


if __name__ == '__main__':
    main()