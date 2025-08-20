import warnings
warnings.filterwarnings("ignore")

import os
import json
import torch
import argparse
import logging
from torch.utils.data import Dataset, DataLoader
from datetime import datetime

from transformers import get_cosine_schedule_with_warmup
import torch.nn.functional as F 

from src.cosmo.utils.training_utils import AdaSPLossRobust, set_seed, pprint
from src.cosmo.data.dataloaders.wsi_data import WSIBagDataset
from src.cosmo.models.llm import MultiModalCLP_PEFT
from src.cosmo.data.dataloaders.pathkt_data import WSIPathKnowledge, BalancedIdentitySampler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)
logger = logging.getLogger(__name__)


class MultiModalPKDataset(Dataset):
    """Dataset wrapper for cancer data that includes visual features"""
    def __init__(self, dataset, transform=None, istrain=True, n_tokens=256):
        self.dataset = dataset
        self.transform = transform
        self.istrain = istrain
        self.n_tokens = n_tokens  # Max number of visual feature regions to use

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        slide_path, text, did, tid, attr = self.dataset[index]
        
        bag = torch.load(slide_path, map_location=torch.device('cpu'))
        patch_indices = torch.randint(0, bag.size(0), (self.n_tokens,)).tolist() if bag.shape[0] < self.n_tokens else torch.randperm(bag.size(0))[:self.n_tokens].tolist()
        bag = bag[patch_indices]
             
        return slide_path, text, did, tid, attr, bag


class MultiModalDataCollator:
    """
    Data collator that handles both text and visual features
    """
    def __init__(self, tokenizer, max_length=256, visual_dim=1024, vlm=False):
        self.tokenizer  = tokenizer
        self.max_length = max_length
        self.visual_dim = visual_dim
        self.vlm        = vlm
        
    def __call__(self, features):
        # Extract data from features
        slide_paths = [f[0] for f in features]
        texts = [f[1] for f in features]
        dids  = torch.tensor([f[2] for f in features])
        tids  = [f[3] for f in features]
        attrs = [f[4] for f in features]
        
        # Handle visual features - pad to same number of regions
        visual_features = [f[5] for f in features]
        
        # Stack into batches
        visual_features_batch = torch.stack(visual_features)
        
        # Tokenize texts
        if self.vlm:
            tokenized = self.tokenizer(
                texts, 
                max_length = 127,
                add_special_tokens=True, 
                return_token_type_ids=False,
                truncation = True,
                padding = 'max_length',
                return_tensors = 'pt')
            
            tokenized["input_ids"]  = F.pad(tokenized['input_ids'], (0, 1), value=self.tokenizer.pad_token_id)
        else:
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
            "attrs": attrs,
            "visual_features": visual_features_batch,
            "slide_paths": slide_paths
        }


class MultiModalTrainer(torch.nn.Module):
    """
    Trainer for multi-modal AdaSP loss with identity-based sampling
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
        temp_text=0.04,
        temp_cross=0.07,
        loss_type='adasp',
        text_weight=1.0,
        cross_weight=1.0,
        save_path='./model_checkpoints',
        i_run=0,
    ):
        super().__init__()
        self.model = model
        self.train_dataset = train_dataset
        self.tokenizer = tokenizer if tokenizer else model.tokenizer
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.device = args.device
        self.save_path = save_path
        self.run  = i_run
        self.args = args
            
        self.sampler = BalancedIdentitySampler(
            train_dataset,
            batch_size=batch_size,
            num_instances=num_instances,
            multi=True,
        )
        
        self.data_collator = MultiModalDataCollator(
            tokenizer=self.tokenizer,
            max_length=args.max_length,
            vlm=self.model.vlm,
        )
        
        self.data_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            sampler=self.sampler,
            collate_fn=self.data_collator,
            num_workers=12,
            pin_memory=True
        )
        
        self.loss_fn = AdaSPLossRobust(
            device=self.device,
            temp=temp_text,
            loss_type=loss_type
        )
        
        # Learning rate scheduler
        parameters    = [{'params': filter(lambda p: p.requires_grad, self.model.parameters()), 
                          "weight_decay": 1e-5}]
        self.optimizer = torch.optim.AdamW(parameters, lr=self.learning_rate)
        num_steps      = self.epochs * (len(self.data_loader)  // self.batch_size)
        
        self.scheduler = get_cosine_schedule_with_warmup(
            optimizer=self.optimizer,
            num_warmup_steps=-1,
            num_training_steps=num_steps,
        )
        
        # Create directory for saving checkpoints
        os.makedirs(self.save_path, exist_ok=True)
        
        # Training config to save
        self.training_config = {
            'batch_size': self.batch_size,
            'num_instances': self.num_instances,
            'learning_rate': self.learning_rate,
            'epochs': self.epochs,
            'temp_text': temp_text,
            'temp_cross': temp_cross,
            'loss_type': loss_type,
            'text_weight': text_weight,
            'cross_weight': cross_weight,
        }
        
    def calculate_loss(self, emb_loss, emb_v, logits, dids, verbose=False):
        # Define the three component losses
        embedding_loss = emb_loss
        triplet_loss   = self.loss_fn(emb_v, dids)
        ce_loss        = F.cross_entropy(logits, dids)
        
        active_losses   = 3  
        loss_components = [embedding_loss, triplet_loss, ce_loss]
        loss_weights    = [1.0, 1.0, 1.0]  # Default weights
        loss_name       = "default"
        
        # Configure loss components based on arguments
        if self.args.no_embloss:
            loss_weights[0] = 0.0 
            active_losses  -= 1
            loss_name       = "no emb"
        
        if self.args.no_trploss:
            loss_weights[1] = 0.0 
            active_losses   -= 1
            loss_name       = "no trp"
        
        if self.args.no_embloss and self.args.no_trploss:
            loss_name = "ce only"
        
        # Calculate weighted loss - avoid division by zero if all losses are disabled
        if active_losses > 0:
            total_loss = sum(w * l for w, l in zip(loss_weights, loss_components)) / active_losses
        else:
            total_loss = ce_loss  # Fallback to CE loss if everything else is disabled
        
        if verbose:
            print(f"Using [{loss_name}] loss configuration")
        
        return total_loss
    
    def train(self):
        """Train the model with multi-modal loss"""
        self.model.to(self.device)
        self.model.visual_model.train()
        
        best_epoch = 0
        best_loss  = float('inf')
        patience   = 10 #10  # Stop training after 15 epochs without improvement
        counter    = 0       # Counter to track epochs without improvement
        
        for epoch in range(self.epochs):
            epoch_loss  = 0.0
            batch_count = 0
            
            log_percent   = 0.25
            log_interval  = int(len(self.data_loader) * log_percent)
            start_time    = datetime.now() 
            
            for batch in self.data_loader:
                
                input_ids       = batch["input_ids"].to(self.device)
                attention_mask  = batch["attention_mask"].to(self.device)
                dids            = batch["dids"].to(self.device)
                visual_features = batch["visual_features"].to(self.device)
                
                # print(input_ids.shape)
                # print(visual_features.shape)
                # print(dids)
                
                # Zero gradients
                self.optimizer.zero_grad()
                
                logits, emb_v, emb_loss =  self.model({
                    "input_ids"      : input_ids,
                    "attention_mask" : attention_mask,
                    "visual_features": visual_features,
                })
                
                loss = self.calculate_loss(emb_loss, emb_v, logits, dids)
            
                # Backward pass
                loss.backward()
                
                # Update weights
                self.optimizer.step()
                self.scheduler.step()
                
                # Track losses
                epoch_loss  += loss.item()
                batch_count += 1
                
                if batch_count % log_interval == 0 or batch_count == 1 or batch_count == len(self.data_loader):
                    logger.info(
                        f"[train] epo : [{epoch+1}/{self.epochs}] [{batch_count}/{len(self.data_loader)}] "
                        f"loss: {loss.item():.5f}, "
                        f"lr: {self.scheduler.get_last_lr()[0]:.6f}"
                    )
            
            avg_epoch_loss = epoch_loss / batch_count
            if avg_epoch_loss < best_loss:
                counter    = 0
                best_epoch = epoch
                best_loss  = avg_epoch_loss
                self.save_checkpoint()
            else:
                counter += 1
                
            epoch_time = (datetime.now()  - start_time).total_seconds() / 60
            time_left  = f'{(self.epochs - (epoch+1)) / 60. * epoch_time:.2f} h left | avg_loss: {avg_epoch_loss:.8f} | best_loss: {best_loss:.6f}'
            logger.info(f'[ log ] roughly {time_left}\n')
            
            if counter >= patience:
                logger.info(f"[ log ] early stop after {epoch} epochs. Best at epoch {best_epoch}")
                break
        
        logger.info(f"Training complete. Best loss: {best_loss:.4f}")
        return best_loss
    
    def save_checkpoint(self):
        """Save model checkpoint"""
        path = self.save_path
        os.makedirs(path, exist_ok=True)
        
        # Save the MLP embedding layer separately
        torch.save(self.model.visual_model.state_dict(), os.path.join(path, f'visual_model_{self.run}.pt'))
        
        if self.model.vlm: # Parameter object : use '.data'
            torch.save(self.model.bert_model.text.text_projection.data,os.path.join(path, f'mlp_embed_{self.run}.pt'))
        else:
            torch.save(self.model.bert_model.mlp_embed.state_dict(), os.path.join(path, f'mlp_embed_{self.run}.pt'))
        # Save training configuration
        with open(os.path.join(path, 'training_args.json'), 'w') as f:
            json.dump(self.training_config, f, indent=2)
            
        logger.info(f"[ log ] saving @ {path}")


def train_main(args, i_run=0):
    """
    Main training function
    """
    # Set random seed for reproducibility
    set_seed()
    
    pprint(vars(args))
    
    # Process data
    logger.info(f"Loading data from [wsifeatures] {args.wsi_root}")
    logger.info(f"Loading data from [knowledge  ] {args.data_path}")
    
    data_train   = []
    oversample   = True 
    print(f"Oversample : {oversample}")
    
    dset   = WSIBagDataset(root=args.wsi_root, 
                           patient_info=args.patient_csv, 
                           split_file=f"{args.split_csv}/splits_{i_run}.csv", 
                           train='train', 
                           zs_state="seen", 
                           zsmode=False, 
                           ratio=args.ratio, 
                           proto=False,
                           use_spatial=True,
                           prev_type=args.prev,
                           text_model="",
                           oversample=oversample) # True
     
    data = WSIPathKnowledge(args.data_path, (dset.wsis, dset.targets), 
                            args.prev, args.json, 
                            slide_max=dset.minimum_slides,
                            max_texts_per_slide=1 if args.no_txtaug else 16)
    data_train.extend(data.train)
    
    
    print("All instances : ", len(data_train), " | tokens :: ", args.tokens)
    train_dataset = MultiModalPKDataset(data_train, istrain=True, n_tokens=args.tokens) 
 
    # Create model
    if args.feat == "conch":
        args.model_name = "conch"
    
    model = MultiModalCLP_PEFT(
        bert_model_name=args.model_name,
        visual_dim=dset.ndims,# uni
        device=args.device,
        prev_type=args.prev,
        no_concept=args.no_concpt,
        concept_root=args.concept_root
    )
    
    logger.info("\n")
    logger.info(model.visual_model)
        
    # Create trainer
    if args.no_embloss and args.no_trploss:
        save_path = os.path.join(args.output_dir,f"{args.dataset}_{args.ratio}_{args.feat}_ceonly")
    elif args.no_embloss:
        save_path = os.path.join(args.output_dir,f"{args.dataset}_{args.ratio}_{args.feat}_noembloss")
    elif args.no_trploss:
        save_path = os.path.join(args.output_dir,f"{args.dataset}_{args.ratio}_{args.feat}_notrploss")
    elif args.no_txtaug:
        save_path = os.path.join(args.output_dir,f"{args.dataset}_{args.ratio}_{args.feat}_notxtaug")
    elif args.no_concpt:
        save_path = os.path.join(args.output_dir,f"{args.dataset}_{args.ratio}_{args.feat}_noconcpt")
    elif args.biobert:
        save_path = os.path.join(args.output_dir,f"{args.dataset}_{args.ratio}_{args.feat}_biobert")
    elif args.clinicalbert:
        save_path = os.path.join(args.output_dir,f"{args.dataset}_{args.ratio}_{args.feat}_clinicalbert")
    else:
        save_path = os.path.join(args.output_dir,f"{args.dataset}_{args.ratio}_{args.feat}")
    
    os.makedirs(save_path,exist_ok=True)
    
    
    # Log model info
    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    trainable_names  = [n for n,p in model.named_parameters() if p.requires_grad]
    
    logger.info(f"- Base model: {args.model_name}")
    logger.info(f"- Total parameters: {total_params:,}")
    logger.info(f"- Trainable parameters: {trainable_params:,}")
    logger.info(f"- Percentage of trainable parameters: {100 * trainable_params / total_params:.2f}%")
    logger.info(f"- Trainable parameters (names): {trainable_names}")
    logger.info(f"save_dir: {save_path}")
    logger.info("\n")  

    trainer = MultiModalTrainer(
        model=model,
        train_dataset=train_dataset,
        args=args,
        batch_size=args.batch_size,
        num_instances=args.num_instances,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        temp_text=args.temp,
        loss_type=args.loss_type,
        save_path=save_path,
        i_run=i_run
    )
    
    # Train model
    logger.info("Starting training...\n\n")
    best_loss = trainer.train()
    
    logger.info(f"Training completed with best loss: {best_loss:.4f}")
    logger.info(f"Model saved to {args.output_dir}")

def main(args):
    
    train_main(args, args.runs)
        
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MultiModal Model")
    
    # Data arguments
    parser.add_argument('--device',  type=str, default='cpu')
    parser.add_argument("--data_path", type=str, 
                      default="", 
                      help="Path to brain cancer knowledge files")
    parser.add_argument("--prev", type=str, default="brainNIH", 
                      help="prev type")
    parser.add_argument("--json", type=str, 
                      default="/pathto/brainKT_prompt_names.json", 
                      help="json file")
    parser.add_argument("--split_csv", type=str, 
                      default="", 
                      help="Root containing split csvs.")
    parser.add_argument("--patient_csv", type=str, 
                      default="", 
                      help="Patient csv file")
    parser.add_argument("--output_dir", type=str, 
                      default="./checkpoints/cosmo",
                      help="Directory to save model and results")
    
    # Model arguments
    parser.add_argument("--concept_root", type=str, 
                      default="./pretrained/concepts",
                      help="Concept root dir.")
    parser.add_argument("--model_name", type=str, 
                      default="./checkpoints/cosmollm",
                      help="Pretrained model name")
    parser.add_argument("--bert_embed_dim", type=int, default=768,
                      help="BERT embedding dimension")
    parser.add_argument("--feature_embed_dim", type=int, default=512,
                      help="Final embedding dimension")
    parser.add_argument("--max_length", type=int, default=256,
                      help="Maximum sequence length")
    
    # Training arguments
    parser.add_argument("--batch_size", type=int, default=16,
                      help="Batch size for training ( 4 * 4)")
    parser.add_argument("--num_instances", type=int, default=4,
                      help="Number of instances per class in a batch")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                      help="Learning rate")
    parser.add_argument("--epochs", type=int, default=100,
                      help="Number of training epochs")
    parser.add_argument("--temp", type=float, default=0.04,
                      help="temperature scaling for Adasploss ")
    parser.add_argument("--loss_type", type=str, default="adasp",
                      help="loss type")
    
    ''' about dataset '''
    parser.add_argument("--dataset", type=str, default="dfcibrain_coarse",
                      help="dataset name")
    parser.add_argument("--wsi_root", type=str, 
                      default="", 
                      help="Root directory of the WSI features")
    parser.add_argument("--feat", type=str, default="uni",
                      help="feature directory name")
    parser.add_argument('--ratio', type=float, default=0.16, 
                        help='Proportion of dataset to subsample per class')
    parser.add_argument('--tokens', type=int, default=1024, 
                        help='number of wsi instances to sample')
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    parser.add_argument("--runs", type=int, default=5, help="number of runs")
    
    ''' about ablations '''
    parser.add_argument('--no_embloss',  action='store_true', help='not using emb_loss')
    parser.add_argument('--no_trploss',  action='store_true', help='not using triplet_loss')
    parser.add_argument('--no_txtaug',   action='store_true', help='not using text sampling (set to max 1 description)')
    parser.add_argument('--no_concpt',   action='store_true', help='not using concept fusion')
    parser.add_argument('--biobert',     action='store_true', help='using biobert LLM')
    parser.add_argument('--clinicalbert',action='store_true', help='using clinicalbert LLM')
    
    args = parser.parse_args()
    main(args)
    
    print("Done!")