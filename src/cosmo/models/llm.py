#!/usr/bin/env python3
"""
All Model (Multimodal/Language) Definitions
Includes PEFT-based BERT models and multimodal components.

Author: Philip Chikontwe
"""
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import json
import sys
import random
from transformers import AutoTokenizer, AutoModel

from ..utils.label_mappings import LABEL_MAP

# pip install accelerate
# pip install -i https://pypi.org/simple/ bitsandbytes>=0.40.0
# pip install peft>=0.4.0
# pip install transformers>=4.36.0


def get_lora_config(r=16, alpha=32):
    """
    Get LoRA configuration for BERT model
    
    Args:
        r: rank of the update matrices
        alpha: scaling factor for the updated weights
    """
    from peft import get_peft_config
    
    config = {
        "peft_type": "LORA",
        "task_type": "FEATURE_EXTRACTION",
        "inference_mode": False,
        "r": r,
        "target_modules": ["query", "key", "value", "output.dense"],
        "lora_alpha": alpha,
        "lora_dropout": 0.05,
        "fan_in_fan_out": False,
        "bias": "none",
    }
    return get_peft_config(config)


def initialize_weights(module):
    """Initialize model weights following COSMO conventions"""
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


class AttentionFC(nn.Module):
    """Attention-based feature aggregation layer"""
    
    def __init__(self, L=512, D=128, dropout=False, n_classes=1):
        super(AttentionFC, self).__init__()
        self.attention_a = [
            nn.Linear(L, D),
            nn.Tanh()
        ]
        
        self.attention_b = [
            nn.Linear(L, D),
            nn.Sigmoid()
        ]
        
        self.dropout = dropout 
        if self.dropout:
            self.attention_a.append(nn.Dropout(0.25))
            self.attention_b.append(nn.Dropout(0.25))

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)
        self.attention_c = nn.Sequential(nn.Linear(D, n_classes))

    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A)
        return A


class TaskAdapter(nn.Module):
    """Task-specific adaptation layer for text embeddings"""
    
    def __init__(self, embedding, alpha=0.5):
        super(TaskAdapter, self).__init__()
        self.alpha = alpha
        self.register_buffer("base_text_features", embedding)
        self.ctx = nn.Parameter(torch.zeros_like(embedding))

    def forward(self):
        return self.base_text_features + self.alpha * self.ctx


class MultiModalCLP(nn.Module):
    """
    Multimodal Contrastive Language-Pathology model
    
    Combines visual features with concept-based text representations for pathology analysis.
    """
    
    def __init__(
        self,
        bert_embed_dim=512,
        feature_embed_dim=512,
        visual_dim=1024,
        device=torch.device('cpu'),
        logit_data=None,
        vlm=False,
        prev_type='brainNIH',
        no_concept=False,
        text_model='cosmo',
        concept_root="./pretrained/concepts",
        
    ):
        super().__init__()
        alpha = 1.0
        self.no_concept = no_concept
        self.device = device
        self.embed_dim    = feature_embed_dim
        self.text_model   = 'conch' if vlm else text_model
        self.concept_root = concept_root
        
        # Visual projection to match text dimension
        self.visual_projection = nn.Sequential(
            nn.Linear(visual_dim, bert_embed_dim),
            nn.LayerNorm(bert_embed_dim),
            nn.ReLU(),
            nn.Dropout(0.25)
        )
        self.attention = AttentionFC(L=bert_embed_dim, D=bert_embed_dim//2, dropout=True)
        
        # Load concept embeddings and class prototypes
        c_list, concepts, cls_emb = self.get_concepts(prev_type, txt_model=self.text_model)
        self.concept_list = c_list
        self.concepts = concepts.to(self.device)
        self.class_embeds = TaskAdapter(cls_emb.detach().clone(), alpha=alpha)
        
        # Deconfounding layers
        if self.no_concept:
            self.w_q = nn.Identity()
            self.w_k = nn.Identity()
        else:
            self.w_q = nn.Linear(bert_embed_dim, 128)
            self.w_k = nn.Linear(bert_embed_dim, 128)
        
        # Logit scale parameter
        if logit_data:
            print("Loading pre-trained logit_scale")
            self.logit_scale = nn.Parameter(logit_data)
        else:
            self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.04))
            
        initialize_weights(self)
        
        self.mode   = 0
        self.x_path = None
    
    def get_concepts(self, prev_type, txt_model='cosmo'):
        """Load concept embeddings and class prototypes"""
        organ         = prev_type.split("NIH")[0]
        FILE_JSON     = f"{self.concept_root}/{organ}_closed_concepts.json"
        all_concepts_ = json.load(open(FILE_JSON, "r"))
        all_concepts  = {k: v for k, v in all_concepts_.items()}
        
        root_folder   = self.concept_root 
        concept_list  = []
        path_c = os.path.join(root_folder, f"{prev_type}_{txt_model}_concepts.npy")
        concept_embeddings = np.load(path_c)
        class_embs = np.load(os.path.join(root_folder, f"{prev_type}_{txt_model}_class.npy"))
        print(f"Loaded - [{path_c}]")
        
        # IMPORTANT.
        label_names = [i for i in LABEL_MAP[prev_type]]
        
        for idx, class_name in enumerate(label_names):
            concept_descrip = all_concepts.get(class_name, [])
            concept_list.extend(concept_descrip)
        
        # Ensure unique concepts while maintaining order
        unique_concepts = {}
        for idx, concept in enumerate(concept_list):
            if concept not in unique_concepts:
                unique_concepts[concept] = concept_embeddings[idx]

        # Extract unique concepts and corresponding embeddings
        concept_list = list(unique_concepts.keys())
        concept_embeddings = np.stack(list(unique_concepts.values()), axis=0)
        print(f"[{txt_model.upper()}] Concepts : ", len(concept_list), " | Embeddings : ", concept_embeddings.shape)
        
        return concept_list, torch.from_numpy(concept_embeddings).float(), torch.from_numpy(class_embs).float()
    
    def deconfound(self, slide_emb, inst_proto):
        """Apply deconfounding using concept prototypes"""
        if self.no_concept:
            return slide_emb
        
        bag_feat = self.w_q(slide_emb)
        conf_k = self.w_k(inst_proto)
        
        deconf_A = torch.mm(conf_k, bag_feat.transpose(0, 1))
        deconf_A = F.softmax(deconf_A / torch.sqrt(torch.tensor(conf_k.shape[1], 
                    dtype=torch.float32, device=bag_feat.device)), 0)  
        
        conf_feats = torch.mm(deconf_A.transpose(0, 1), inst_proto) + slide_emb
        return conf_feats
    
    def visual_encode(self, features):
        """Encode visual features with attention and deconfounding"""
        emb_v = self.visual_projection(features)
        emb_A = self.attention(emb_v)
        emb_A = torch.transpose(emb_A, 2, 1)
        emb_A = F.softmax(emb_A, -1)  
        emb_v = torch.bmm(emb_A, emb_v)
        emb_v = self.deconfound(emb_v[:, 0], self.concepts)
        return emb_v
    
    def sim_score(self, a, b):
        """Compute similarity score between features"""
        text_features  = F.normalize(b, dim=-1)
        image_features = F.normalize(a, dim=-1)
        score = image_features @ text_features.t() * self.logit_scale.exp()
        return score
    
    def forward(self, visual_features=None, norm=True, use_att=False):
        """Forward pass for multimodal model"""
        if use_att:
            x = self.visual_projection(visual_features[0])
            A = self.attention(x)
            
            A     = torch.transpose(A, 1, 0)
            A     = F.softmax(A, 1) 
            emb_v = torch.mm(A, x)
            
            text_features  = F.normalize(self.concepts.to(emb_v.device), dim=-1)
            image_features = F.normalize(emb_v, dim=-1)
            logits = (image_features @ text_features.t()) * self.logit_scale.exp()
        else:
            emb_v  = self.visual_encode(visual_features)
            logits = self.sim_score(emb_v, self.class_embeds())
            
        return logits, emb_v

    def inference(self, bag, use_att=False):
        if self.mode == 1: # all instances
            # bag is already [K,D]
            x  = self.visual_projection(bag)
            if use_att:
                text_features  = F.normalize(self.concepts.to(bag.device),dim=-1)
                image_features = F.normalize(x,dim=-1)
                logits         = (image_features @ text_features.t()) * self.logit_scale.exp()
            else:
                conf_ft  = self.deconfound(x, self.concepts.to(bag.device))
                logits   = self.sim_score(conf_ft, self.class_embeds())
            
        else:
            logits, _ = self.forward(bag, use_att=use_att)
            
        return logits


class MultiModalCLP_PEFT(nn.Module):
    """
    Integrates CLP_clinical_PEFT with MultiModalCLP.
    """
    
    def __init__(
        self,
        bert_model_name: str = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
        visual_dim: int = 1024,
        device=torch.device('cpu'),
        prev_type='brainNIH',
        no_concept=False,
        concept_root: str = None
    ):
        super().__init__()
        self.device = device
        self.prev_type = prev_type
        self.vlm = False
        text_model = 'cosmo'
        
        # Initialize BERT model or VLM
        if bert_model_name == "conch":
            self.vlm = True
            self.bert_model, self.tokenizer, logit_data = self._get_vlm_basemodel()
            self.bert_model.eval()
            self.bert_model.text.eval()
            self.bert_model.text.requires_grad_(False)
            self.bert_model.logit_scale.requires_grad_(False)
            self.bert_model.text.text_projection.requires_grad_(False)
        else:
            with torch.no_grad():
                self.bert_model, self.tokenizer = self._get_bert_basemodel(bert_model_name)
                logit_data = None 
                self.bert_model.requires_grad_ = False
                self.bert_model.mlp_embed.requires_grad_ = True
                
            # Set text model type based on BERT variant
            if 'biobert' in os.path.split(bert_model_name)[-1]:
                text_model = 'biobert'
            elif 'clinicalbert' in os.path.split(bert_model_name)[-1]:
                text_model = 'clinicalbert'
            else:
                text_model = 'cosmo'
            
        feature_embed_dim = self.bert_model.embed_dim
        
        # Initialize visual model
        self.visual_model = MultiModalCLP(
            feature_embed_dim=feature_embed_dim,
            visual_dim=visual_dim,
            device=self.device,
            logit_data=logit_data,
            vlm=self.vlm,
            prev_type=prev_type,
            no_concept=no_concept,
            text_model=text_model,
            concept_root=concept_root
        )
        
        # Compatibility attributes
        self.embed_dim = feature_embed_dim
        self.class_embeds = None
    
    def _get_vlm_basemodel(self, 
        checkpoint_path="/path/to/CONCH_WEIGHTS/pytorch_model.bin"):
        """Load CONCH VLM model"""
        from models.conch.open_clip_custom import create_model_from_pretrained, get_tokenizer
        
        with torch.no_grad():
            model, _ = create_model_from_pretrained("conch_ViT-B-16", checkpoint_path)
            tokenizer = get_tokenizer()
        
            print(f"Loading VLM model from {checkpoint_path} to {self.device}")
            model.text_decoder = None
            model.visual = None
            model.eval()
            model.text.eval()
        
        print(f"Successfully loaded VLM model [embedding dim : {model.embed_dim}]")
        return model, tokenizer, model.logit_scale.data
        
    def _get_bert_basemodel(self, checkpoint_path):
        """Load PEFT BERT model from checkpoint"""
        from peft import PeftModel, PeftConfig
        from transformers import AutoModel, AutoTokenizer
        
        print(f"Loading PEFT model from {checkpoint_path} to {self.device}")
        
        # Check for PEFT configuration
        adapter_config_path = os.path.join(checkpoint_path, "adapter_config.json")
        if not os.path.exists(adapter_config_path):
            raise ValueError(f"No adapter_config.json found in {checkpoint_path}")
        
        # Load PEFT configuration and model
        peft_config = PeftConfig.from_pretrained(checkpoint_path)
        print(f"Base model: {peft_config.base_model_name_or_path}")
        
        base_model = AutoModel.from_pretrained(
            peft_config.base_model_name_or_path,
            output_hidden_states=True
        )
        
        peft_model = PeftModel.from_pretrained(base_model, checkpoint_path)
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
        peft_model = peft_model.to(self.device)
        peft_model.eval()
        
        # Load MLP embedding layer
        model = CLP_clinical_PEFT(device=self.device)
        chck = os.path.join(checkpoint_path, "mlp_embed.pt")
        params = torch.load(chck, map_location=self.device)
        model.mlp_embed.load_state_dict(params)
        model.mlp_embed.requires_grad_(False)
        
        model.bert_model = peft_model
        model = model.to(self.device)
        print(f"Successfully loaded PEFT model")
        return model, tokenizer
    
    def forward(self, inputs):
        """Forward pass for multimodal PEFT model"""
        logits, emb_v = self.visual_model(inputs['visual_features'])
        
        with torch.no_grad():
            text_inputs = {
                'input_ids': inputs['input_ids'],
                'attention_mask': inputs['attention_mask']
            }
            
            if self.vlm:
                emb_t = self.bert_model.encode_text(text_inputs['input_ids'], normalize=False)
            else:
                emb_t = self.bert_model.encode_text(text_inputs)
                
        emb_loss = F.l1_loss(emb_v, emb_t)
        emb_v = F.normalize(emb_v, dim=-1)
        return logits, emb_v, emb_loss
    
    def inference(self, bag):
        """Inference using visual model"""
        return self.visual_model.inference(bag)
    
    def sim_score(self, a, b):
        """Compute similarity score"""
        text_features = F.normalize(b, dim=-1)
        image_features = F.normalize(a, dim=-1)
        score = image_features @ text_features.t()
        return score


class CLP_clinical_PEFT(nn.Module):
    """
    Clinical Language Processing model with PEFT (Parameter-Efficient Fine-Tuning)
    
    Uses LoRA for efficient adaptation of pre-trained BERT models for cancer pathology text.
    """
    
    def __init__(self,
                 bert_model_name: str = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
                 bert_embed_dim: int = 768,
                 feature_embed_dim: int = 512,
                 device=torch.device('cpu')):
        
        super().__init__()
        self.device = device
        self.bert_model = self._get_bert_basemodel(bert_model_name=bert_model_name)
        
        # Projection layer remains fully trainable
        self.mlp_embed = nn.Sequential(
            nn.Linear(bert_embed_dim, feature_embed_dim),
            nn.GELU(),
            nn.Linear(feature_embed_dim, feature_embed_dim)
        )
        self.embed_dim = feature_embed_dim

        # Tokenizer setup
        self.max_length = 256
        self.tokenizer  = AutoTokenizer.from_pretrained(bert_model_name)

        self.init_parameters()
    
    def init_parameters(self):
        """Initialize projection layer parameters"""
        for m in self.mlp_embed:
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=self.embed_dim ** -0.5)
        
    def _get_bert_basemodel(self, bert_model_name):
        """Create PEFT BERT model with LoRA adapters"""
        from peft import get_peft_config, PeftModelForFeatureExtraction
        
        # Configure LoRA for different BERT variants
        target_modules = ["query", "key", "value", "output.dense"]
        if bert_model_name in ['medicalai/ClinicalBERT']:
            target_modules = ["q_lin", "k_lin", "v_lin"]

        config = {
            "peft_type": "LORA",
            "task_type": "FEATURE_EXTRACTION",
            "inference_mode": False,
            "r": 16,
            "target_modules": target_modules,
            "lora_alpha": 32,
            "lora_dropout": 0.05,
            "fan_in_fan_out": False,
            "bias": "none",
        }
        peft_config = get_peft_config(config)
        model       = AutoModel.from_pretrained(bert_model_name)
        peft_model  = PeftModelForFeatureExtraction(model, peft_config)
        peft_model.print_trainable_parameters()
            
        return peft_model

    def encode_text(self, text):
        """Encode text using PEFT BERT model"""
        output = self.bert_model(
            input_ids=text['input_ids'],
            attention_mask=text['attention_mask']
        )
        
        # Access last hidden state and pooler output
        last_hidden_state = output.last_hidden_state
        pooler_output = output.pooler_output if hasattr(output, 'pooler_output') else last_hidden_state[:, 0]
        
        encode_out = self.mlp_embed(pooler_output)
        return encode_out
    
    def forward(self, input_text):
        """Forward pass with L2 normalization"""
        text_features = self.encode_text(input_text)
        text_features = F.normalize(text_features, dim=-1)
        return text_features
    
    def loss_function(self, text, target, loss_fn):
        """Compute loss for given text and target embeddings"""
        input_text = get_tokenizer(
            text,
            self.tokenizer,
            max_length=self.max_length, 
            ismask=False
        )
        text_features = self.forward(input_text.to(self.device))
        loss = loss_fn(text_features, target)
        return loss 

    def inference(self, text):
        """Generate embeddings for inference"""
        input_text = get_tokenizer(
            text,
            self.tokenizer,
            max_length=self.max_length, 
            ismask=False
        )
        text_features = self.forward(input_text.to(self.device))
        return text_features


def get_tokenizer(text, tokenizer, max_length, ismask=False):
    """Tokenize text with error handling"""
    try:
        token_list = tokenizer(
            list(text) if isinstance(text, list) else [text],
            add_special_tokens=True,
            max_length=max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
    except Exception as e:
        print(f"Error in tokenization: {e}")
        token_list = tokenizer(list(text) if isinstance(text, list) else [text])

    return token_list


def load_peft_model_checkpoint(checkpoint_path, device=None):
    """
    Load a saved PEFT model checkpoint
    
    Args:
        checkpoint_path: Path to checkpoint directory
        device: Target device for model
        
    Returns:
        Loaded model and tokenizer
    """
    from peft import PeftModel, PeftConfig
    from transformers import AutoModel, AutoTokenizer
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    print(f"Loading PEFT model from {checkpoint_path} to {device}")
    
    # Load PEFT configuration
    peft_config = PeftConfig.from_pretrained(checkpoint_path)
    
    # Load base model and PEFT adapters
    base_model = AutoModel.from_pretrained(
        peft_config.base_model_name_or_path,
        output_hidden_states=True
    )
    peft_model = PeftModel.from_pretrained(base_model, checkpoint_path)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    
    # Load complete CLP model
    model = CLP_clinical_PEFT(device=device)
    mlp_path = os.path.join(checkpoint_path, "mlp_embed.pt")
    if os.path.exists(mlp_path):
        params = torch.load(mlp_path, map_location=device)
        model.mlp_embed.load_state_dict(params)
    
    model.bert_model = peft_model.to(device)
    model = model.to(device)
    model.eval()
    
    print(f"Successfully loaded PEFT model")
    return model, tokenizer


if __name__ == '__main__':
    # Test model initialization
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Test CLP_clinical_PEFT Language Model
    # ************
    # Model options:
    # 1. default: 
    model_name = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
    
    # 2,3 : others (uncomment)
    #model_name = "dmis-lab/biobert-base-cased-v1.1"
    #model_name = "medicalai/ClinicalBERT"
    
    print("Initializing Model ...")
    model = CLP_clinical_PEFT(bert_model_name=model_name, device=device)
    
    print(model)
    print()
    
    # Test text encoding
    texts      = ["papillary strands with fibrovascular cores"]
    input_text = get_tokenizer(texts, model.tokenizer, max_length=256)
    
    with torch.no_grad():
        encoded_text = model.encode_text(input_text)
        norm_vec = model(input_text)
    
    print(f"Encoded text shape    : {encoded_text.shape}")
    print(f"Normalized text shape : {norm_vec.shape}")
    # ************
    # ************
    
    
    