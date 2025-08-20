#!/usr/bin/env python3
"""
Language Model Training

Multi-tissue language model training using PEFT and adaptive sampling.
Supports training on brain, lung, and kidney pathology knowledge simultaneously.

Author: Philip Chikontwe

Usage:
    # Train on all tissues with default settings
    python train_llm.py --data-dir ./examples/data/ --output-dir ./checkpoints/
    
    # Train with custom parameters
    python train_llm.py --data-dir ./examples/data/ --output-dir ./checkpoints/ \
                       --batch-size 64 --learning-rate 2e-5 --epochs 10
    
    # Train on specific tissues only
    python train_llm.py --data-dir ./examples/data/ --tissues brain lung \
                       --output-dir ./checkpoints/
"""

import warnings
warnings.filterwarnings("ignore")

import os
import json
import torch
import numpy as np
import argparse
import logging
from typing import List, Dict, Optional
from collections import Counter, defaultdict

from torch.utils.data import DataLoader

from src.cosmo.models.llm import CLP_clinical_PEFT
from src.cosmo.data.dataloaders.pathkt_data import RandomIdentitySampler, PKDataset, PathKnowledge # type: ignore
from src.cosmo.utils.training_utils import AdaSPLoss, set_seed

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MultiTissueDataCollator:
    """
    Data collator for multi-tissue cancer dataset with identity preservation
    """
    
    def __init__(self, tokenizer, max_length=256):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __call__(self, features):
        # Extract data from features
        texts = [f[0] for f in features]
        dids = torch.tensor([f[1] for f in features])
        tids = [f[2] for f in features]
        attrs = [f[3] for f in features]
        
        # Tokenize texts
        tokenized = self.tokenizer(
            texts,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        return {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "dids": dids,
            "tids": tids,
            "attrs": attrs
        }


class COSMOTrainer:
    """
    COSMO trainer with multi-tissue support and adaptive sampling
    """
    
    def __init__(
        self, 
        model,
        train_dataset,
        args,
        tokenizer=None,
        batch_size=32,
        num_instances=4,
        learning_rate=2e-5,
        epochs=3,
        temp=0.04,
        loss_type='adasp',
        save_path='./checkpoints',
    ):
        self.model = model
        self.train_dataset = train_dataset
        self.tokenizer = tokenizer if tokenizer else model.tokenizer
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.device = args.device
        self.temp = temp
        self.loss_type = loss_type
        self.save_path = save_path
        
        # Create sampler and data loader
        self.sampler = RandomIdentitySampler(
            train_dataset,
            batch_size=batch_size,
            num_instances=num_instances
        )
        
        self.data_collator = MultiTissueDataCollator(
            tokenizer=self.tokenizer,
            max_length=args.max_length
        )
        
        self.data_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            sampler=self.sampler,
            collate_fn=self.data_collator,
            num_workers=args.num_workers,
            pin_memory=True
        )
        
        # Create optimizer and scheduler
        self.optimizer = self._create_optimizer()
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=epochs * len(self.data_loader),
            eta_min=1e-6
        )
        
        # Create loss function
        self.loss_fn = AdaSPLoss(
            device=self.device,
            temp=self.temp,
            loss_type=self.loss_type
        )
        
        os.makedirs(self.save_path, exist_ok=True)
        
    def _create_optimizer(self):
        """Create optimizer with proper parameter handling"""
        params = []
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                params.append(param)
                
        return torch.optim.AdamW(
            params,
            lr=self.learning_rate,
            weight_decay=0.01,
            betas=(0.9, 0.999),
            eps=1e-8
        )
    
    def train(self):
        """Train the model with adaptive sampling loss"""
        self.model.to(self.device)
        self.model.train()
        
        best_loss = float('inf')
        
        logger.info(f"Starting training for {self.epochs} epochs...")
        
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            batch_count = 0
            
            for batch in self.data_loader:
                # Move batch to device
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                dids = batch["dids"].to(self.device)
                
                # Zero gradients
                self.optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model({
                    "input_ids": input_ids,
                    "attention_mask": attention_mask
                })
                
                # Compute loss
                loss = self.loss_fn(outputs, dids)
                
                # Backward pass
                loss.backward()
                
                # Update weights
                self.optimizer.step()
                self.scheduler.step()
                
                epoch_loss += loss.item()
                batch_count += 1
                
                if batch_count % 10 == 0:
                    logger.info(
                        f"Epoch {epoch+1}/{self.epochs}, Batch {batch_count}/{len(self.data_loader)}, "
                        f"Loss: {loss.item():.6f}, LR: {self.scheduler.get_last_lr()[0]:.6f}"
                    )
                    
            # Epoch complete
            avg_epoch_loss = epoch_loss / batch_count
            logger.info(f"Epoch {epoch+1} complete. Average loss: {avg_epoch_loss:.4f}")
            
            # Save checkpoint if best so far
            if avg_epoch_loss < best_loss:
                best_loss = avg_epoch_loss
                self.save_checkpoint()
                logger.info(f"New best model saved with loss: {best_loss:.4f}")
        
        logger.info(f"Training complete. Best loss: {best_loss:.4f}")
        return best_loss
    
    def save_checkpoint(self):
        """Save model checkpoint"""
        path = self.save_path
        
        # Save BERT model (PEFT adapters)
        if hasattr(self.model.bert_model, 'save_pretrained'):
            self.model.bert_model.save_pretrained(path)
        else:
            torch.save(self.model.bert_model.state_dict(), os.path.join(path, 'pytorch_model.bin'))
        
        # Save MLP embedding layer
        torch.save(self.model.mlp_embed.state_dict(), os.path.join(path, 'mlp_embed.pt'))
        
        # Save tokenizer
        self.tokenizer.save_pretrained(path)
        
        # Save training configuration
        with open(os.path.join(path, 'training_config.json'), 'w') as f:
            json.dump({
                'batch_size': self.batch_size,
                'num_instances': self.num_instances,
                'learning_rate': self.learning_rate,
                'epochs': self.epochs,
                'temp': self.temp,
                'loss_type': self.loss_type,
                'bert_embed_dim': self.model.bert_model.config.hidden_size 
                    if hasattr(self.model.bert_model, 'config') else 768,
                'feature_embed_dim': self.model.embed_dim
            }, f, indent=2)
        
        logger.info(f"Checkpoint saved to {path}")


class MultiTissueDataLoader:
    """
    Data loader for multiple tissue types with balanced sampling
    """
    
    # Matches the saved names and LABEL MAP in data/dataloaders/pathkt_data
    CODE_MAPPING = {
        'brain'   : 'brainNIH',
        'kidney'  : 'kidneyNIH',
        'lung'    : 'lungNIH'
    }
    
    DEFAULT_PATHS = {
        'brain': {
            'data_path'  : "./examples/data/brainwsikt_all",
            'json_path'  : "./src/cosmo/data/knowledge/templates/brainKT_prompt_names.json",
            'tissue_code': "brainNIH",
            'k_val'      : 0
        },
        'lung': {
            'data_path'  : "./examples/data", 
            'json_path'  : "./src/cosmo/data/knowledge/templates/lungKT_prompt_names.json",
            'tissue_code': "lungNIH",
            'k_val': 10
        },
        'kidney': {
            'data_path'  : "./examples/data",
            'json_path'  : "./src/cosmo/data/knowledge/templates/kidneyKT_prompt_names.json", 
            'tissue_code': "kidneyNIH",
            'k_val'      : 16
        }
    }
    
    @classmethod
    def load_multi_tissue_data(cls, data_dir: str = None, 
                               tissues: List[str] = None, 
                               json_dir: str = None) -> List:
        """
        Load training data for multiple tissues
        
        Args:
            data_dir: Base directory containing tissue data
            tissues: List of tissues to load (default: all)
            
        Returns:
            Combined training data from all tissues
        """
        
        
        if tissues is None:
            tissues = ['brain', 'lung', 'kidney']
            
    
        for tissue in tissues:
            tissue_code = cls.CODE_MAPPING[tissue]
            cls.DEFAULT_PATHS[tissue]['data_path'] = data_dir
            cls.DEFAULT_PATHS[tissue]['json_path'] = os.path.join(json_dir,f"{tissue}KT_prompt_names.json")
            cls.DEFAULT_PATHS[tissue]['tissue_code'] = tissue_code
            cls.DEFAULT_PATHS[tissue]['k_val'] = 0
        
        print("Configuration ..")
        print(cls.DEFAULT_PATHS.keys())
        print()
        
        data_train = []
        k_val      = 0
        for tissue in tissues:
            if tissue not in cls.DEFAULT_PATHS:
                logger.warning(f"Unknown tissue type: {tissue}, skipping...")
                continue
                
            config = cls.DEFAULT_PATHS[tissue].copy()
             
            # Check if files exist
            if not os.path.exists(config['data_path']):
                logger.warning(f"Training data not found for {tissue}: {config['json_path']}")
                continue
            
            logger.info(f"Loading {tissue} data from {config['data_path']} k_val : [class index starting at] {k_val}")
            
            
            try:
                # Load tissue data using existing PathKnowledge class
                data = PathKnowledge(
                    config['data_path'], 
                    config['tissue_code'], 
                    config['json_path'],
                    k_val=k_val
                )
                k_val = k_val + data.uniq_cls
                
                data_train.extend(data.train)
                logger.info(f"Loaded {len(data.train)} samples from {tissue}")
                
            except Exception as e:
                logger.error(f"Error loading {tissue} data: {e}")
                continue
        
        return data_train


def create_model(args):
    """Create and configure the language model"""
    model = CLP_clinical_PEFT(
        bert_model_name=args.model_name,
        bert_embed_dim=args.bert_embed_dim,
        feature_embed_dim=args.feature_embed_dim,
        device=args.device
    )
    
    # Log model statistics
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger.info(f"Model created:")
    logger.info(f"  - Base model: {args.model_name}")
    logger.info(f"  - Total parameters: {total_params:,}")
    logger.info(f"  - Trainable parameters: {trainable_params:,}")
    logger.info(f"  - Trainable percentage: {100 * trainable_params / total_params:.2f}%")
    
    return model


def main(args):
    """Main training function"""
    logger.info("Starting COSMO LLM training...")
    logger.info(f"Configuration: {vars(args)}")
    
    # Set random seed
    set_seed(args.seed)

    # Load multi-tissue data
    logger.info(f"Loading data for tissues: {args.tissues}")
    data_train = MultiTissueDataLoader.load_multi_tissue_data(
        data_dir=args.data_dir,
        json_dir=args.json_dir,
        tissues=args.tissues
    )
    
    if not data_train:
        logger.error("No training data loaded. Please check your data paths.")
        return
    
    # Analyze data distribution
    dids = [item[1] for item in data_train]
    tissue_counts = Counter(dids)
    
    logger.info(f"Total training instances: {len(data_train)}")
    logger.info(f"Tissue distribution: {dict(tissue_counts)}")
    logger.info(f"Unique tissue IDs: {sorted(set(dids))}")
    
    # Create dataset
    train_dataset = PKDataset(data_train, istrain=True)
    
    # Create model
    model = create_model(args)
    
    # Create trainer
    save_path = os.path.join(args.output_dir, f"cosmollm")
    os.makedirs(save_path, exist_ok=True)
    
    trainer = COSMOTrainer(
        model=model,
        train_dataset=train_dataset,
        args=args,
        batch_size=args.batch_size,
        num_instances=args.num_instances,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        temp=args.temp,
        loss_type=args.loss_type,
        save_path=save_path
    )
    
    # Train model
    logger.info(f"Training will save to: {save_path}")
    best_loss = trainer.train()
    
    logger.info(f"Training completed!")
    logger.info(f"Best loss achieved: {best_loss:.4f}")
    logger.info(f"Model saved to: {save_path}")


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Train language model on multi-tissue pathology knowledge"
    )
    
    # Data arguments
    parser.add_argument('--data-dir', type=str, required=True,
                       help="Directory containing tissue text training data (csvs)")
    parser.add_argument('--json-dir', type=str, required=True,
                       help="Directory containing tissue json data")
    parser.add_argument('--tissues', nargs='+', 
                       choices=['brain', 'lung', 'kidney'],
                       default=['brain', 'lung', 'kidney'],
                       help="Tissues to include in training (default: all)")
    parser.add_argument('--output-dir', type=str, required=True,
                       help="Directory to save trained model")
    
    # Model arguments
    parser.add_argument('--model-name', type=str, 
                       default="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
                       help="Pre-trained BERT model name")
    parser.add_argument('--bert-embed-dim', type=int, default=768,
                       help="BERT embedding dimension")
    parser.add_argument('--feature-embed-dim', type=int, default=512,
                       help="Final feature embedding dimension")
    parser.add_argument('--max-length', type=int, default=256,
                       help="Maximum sequence length for tokenization")
    
    # Training arguments
    parser.add_argument('--batch-size', type=int, default=64,
                       help="Training batch size")
    parser.add_argument('--num-instances', type=int, default=8,
                       help="Number of instances per class in batch")
    parser.add_argument('--learning-rate', type=float, default=2e-5,
                       help="Learning rate for optimizer")
    parser.add_argument('--epochs', type=int, default=400,
                       help="Number of training epochs")
    parser.add_argument('--temp', type=float, default=0.04,
                       help="Temperature for contrastive loss")
    parser.add_argument('--loss-type', type=str, default='adasp',
                       choices=['adasp', 'triplet', 'contrastive'],
                       help="Loss function type")
    
    # System arguments  
    parser.add_argument('--device', type=str, 
                       default='cuda' if torch.cuda.is_available() else 'cpu',
                       help="Device for training (cuda/cpu)")
    parser.add_argument('--num-workers', type=int, default=12,
                       help="Number of data loader workers")
    parser.add_argument('--seed', type=int, default=1,
                       help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    args.device = torch.device(args.device)
    
    return args


if __name__ == "__main__":
    args = parse_arguments()
    main(args)