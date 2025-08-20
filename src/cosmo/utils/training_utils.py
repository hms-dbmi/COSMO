#!/usr/bin/env python3
"""
Utilities

Author: Philip Chikontwe
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import os
import pprint
from typing import Optional

_utils_pp = pprint.PrettyPrinter()

def set_seed(seed: int = 1):
    """Set random seed for reproducibility"""
    print(f"Setting seed [{seed}]")
    if seed == 0:
        torch.backends.cudnn.benchmark = True
    else:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def pprint(x):
    _utils_pp.pprint(x)

def print_network(net, show_net=False):
    """Print network parameter statistics"""
    num_params = 0
    num_params_train = 0
    
    for param in net.parameters():
        n = param.numel()
        num_params += n
        if param.requires_grad:
            num_params_train += n
            
    if show_net:
        print(net)
    else:
        print("")
    
    num_params_m = num_params / 1000000.0
    num_params_train_m = num_params_train / 1000000.0
    
    print("----------------------------")
    print("MODEL     : {:.5f}M".format(num_params_m))
    print("TRAINABLE : {:.5f}M".format(num_params_train_m))
    try:
        print(f"RATIO     : {(num_params_train / num_params) * 100:.3f}%")
    except ZeroDivisionError:
        print(f"RATIO     : 0%")
    print("----------------------------")


class AdaSPLoss(object):
    """
    SP loss using HARD example mining,
    modified based on original triplet loss using hard example mining
    """

    def __init__(self, device, temp=0.04, loss_type = 'adasp'):
        self.device = device
        self.temp = temp
        self.loss_type = loss_type

    def __call__(self, feat_q, targets):
        
        #feat_q = nn.functional.normalize(feats, dim=1)
        
        bs_size = feat_q.size(0)
        N_id    = len(torch.unique(targets))
        N_ins   = bs_size // N_id

        scale = 1./self.temp

        sim_qq    = torch.matmul(feat_q, feat_q.T)
        sf_sim_qq = sim_qq*scale

        right_factor = torch.from_numpy(np.kron(np.eye(N_id),np.ones((N_ins,1)))).to(self.device)
        pos_mask = torch.from_numpy(np.kron(np.eye(N_id),np.ones((N_ins,1)))).to(self.device)
        left_factor = torch.from_numpy(np.kron(np.eye(N_id), np.ones((1,N_ins)))).to(self.device)
        
        ## hard-hard mining for pos
        mask_HH = torch.from_numpy(np.kron(np.eye(N_id),-1.*np.ones((N_ins,N_ins)))).to(self.device)
        mask_HH[mask_HH==0]=1.

        ID_sim_HH = torch.exp(sf_sim_qq.mul(mask_HH))
        ID_sim_HH = ID_sim_HH.mm(right_factor)
        ID_sim_HH = left_factor.mm(ID_sim_HH)

        pos_mask_id = torch.eye(N_id).to(self.device)
        pos_sim_HH = ID_sim_HH.mul(pos_mask_id)
        pos_sim_HH[pos_sim_HH==0]=1.
        pos_sim_HH = 1./pos_sim_HH
        ID_sim_HH = ID_sim_HH.mul(1-pos_mask_id) + pos_sim_HH.mul(pos_mask_id)
        
        ID_sim_HH_L1 = nn.functional.normalize(ID_sim_HH,p = 1, dim = 1)   
        
        ## hard-easy mining for pos
        mask_HE = torch.from_numpy(np.kron(np.eye(N_id),-1.*np.ones((N_ins,N_ins)))).to(self.device)
        mask_HE[mask_HE==0]=1.

        ID_sim_HE = torch.exp(sf_sim_qq.mul(mask_HE))
        ID_sim_HE = ID_sim_HE.mm(right_factor)

        pos_sim_HE = ID_sim_HE.mul(pos_mask)
        pos_sim_HE[pos_sim_HE==0]=1.
        pos_sim_HE = 1./pos_sim_HE
        ID_sim_HE = ID_sim_HE.mul(1-pos_mask) + pos_sim_HE.mul(pos_mask)

        # hard-hard for neg
        ID_sim_HE = left_factor.mm(ID_sim_HE)

        ID_sim_HE_L1 = nn.functional.normalize(ID_sim_HE,p = 1, dim = 1)
        
    
        l_sim = torch.log(torch.diag(ID_sim_HH))
        s_sim = torch.log(torch.diag(ID_sim_HE))

        weight_sim_HH = torch.log(torch.diag(ID_sim_HH)).detach()/scale
        weight_sim_HE = torch.log(torch.diag(ID_sim_HE)).detach()/scale
        wt_l = 2*weight_sim_HE.mul(weight_sim_HH)/(weight_sim_HH + weight_sim_HE)
        wt_l[weight_sim_HH < 0] = 0
        both_sim = l_sim.mul(wt_l) + s_sim.mul(1-wt_l) 
    
        adaptive_pos = torch.diag(torch.exp(both_sim))

        pos_mask_id = torch.eye(N_id).to(self.device)
        adaptive_sim_mat = adaptive_pos.mul(pos_mask_id) + ID_sim_HE.mul(1-pos_mask_id)

        adaptive_sim_mat_L1 = nn.functional.normalize(adaptive_sim_mat,p = 1, dim = 1)

        loss_HH = -1*torch.log(torch.diag(ID_sim_HH_L1)).mean()
        loss_HE = -1*torch.log(torch.diag(ID_sim_HE_L1)).mean()
        loss_adaptive = -1*torch.log(torch.diag(adaptive_sim_mat_L1)).mean()
        
        if self.loss_type == 'sp-h':
            loss = loss_HH.mean()
        elif self.loss_type == 'sp-lh':
            loss = loss_HE.mean()
        elif self.loss_type == 'adasp':
            loss = loss_adaptive
            
        return loss


class AdaSPLossRobust(object):
    """
    SP loss using HARD example mining,
    modified based on original triplet loss using hard example mining.
    Made more robust against irregular batch sizes.
    """

    def __init__(self, device, temp=0.04, loss_type='adasp'):
        self.device = device
        self.temp = temp
        self.loss_type = loss_type

    def __call__(self, feat_q, targets):
        # Ensure inputs are on the correct device
        feat_q  = feat_q.to(self.device)
        targets = targets.to(self.device)
        
        # Get unique classes and count instances per class
        unique_targets = torch.unique(targets)
        N_id = len(unique_targets)
        
        # Count instances per ID
        id_counts = {}
        for t in unique_targets:
            id_counts[t.item()] = (targets == t).sum().item()
        
        # Get the minimum instance count across classes
        min_instances = min(id_counts.values())
        N_ins = min_instances
        
        # For simplicity and robustness, if we have irregular counts, 
        # default to a simple contrastive loss
        if len(unique_targets) < 2 or any(count != N_ins for count in id_counts.values()):
            return self._compute_fallback_loss(feat_q, targets)
        
        # Proceed with the original AdaSP implementation
        scale = 1./self.temp
        
        sim_qq = torch.matmul(feat_q, feat_q.T)
        sf_sim_qq = sim_qq * scale
        
        # Create masks for computation
        try:
            right_factor = torch.from_numpy(np.kron(np.eye(N_id), np.ones((N_ins, 1)))).to(self.device).float()
            pos_mask = torch.from_numpy(np.kron(np.eye(N_id), np.ones((N_ins, 1)))).to(self.device).float()
            left_factor = torch.from_numpy(np.kron(np.eye(N_id), np.ones((1, N_ins)))).to(self.device).float()
            
            # Hard-hard mining for pos
            mask_HH = torch.from_numpy(np.kron(np.eye(N_id), -1.*np.ones((N_ins, N_ins)))).to(self.device).float()
            mask_HH[mask_HH==0] = 1.
            
            ID_sim_HH = torch.exp(sf_sim_qq.mul(mask_HH))
            ID_sim_HH = ID_sim_HH.mm(right_factor)
            ID_sim_HH = left_factor.mm(ID_sim_HH)
            
            pos_mask_id = torch.eye(N_id).to(self.device)
            pos_sim_HH = ID_sim_HH.mul(pos_mask_id)
            pos_sim_HH[pos_sim_HH==0] = 1.
            pos_sim_HH = 1./pos_sim_HH
            ID_sim_HH = ID_sim_HH.mul(1-pos_mask_id) + pos_sim_HH.mul(pos_mask_id)
            
            ID_sim_HH_L1 = nn.functional.normalize(ID_sim_HH, p=1, dim=1)
            
            # Hard-easy mining for pos
            mask_HE = torch.from_numpy(np.kron(np.eye(N_id), -1.*np.ones((N_ins, N_ins)))).to(self.device).float()
            mask_HE[mask_HE==0] = 1.
            
            ID_sim_HE = torch.exp(sf_sim_qq.mul(mask_HE))
            ID_sim_HE = ID_sim_HE.mm(right_factor)
            
            pos_sim_HE = ID_sim_HE.mul(pos_mask)
            pos_sim_HE[pos_sim_HE==0] = 1.
            pos_sim_HE = 1./pos_sim_HE
            ID_sim_HE = ID_sim_HE.mul(1-pos_mask) + pos_sim_HE.mul(pos_mask)
            
            # Hard-hard for neg
            ID_sim_HE = left_factor.mm(ID_sim_HE)
            
            ID_sim_HE_L1 = nn.functional.normalize(ID_sim_HE, p=1, dim=1)
            
            l_sim = torch.log(torch.diag(ID_sim_HH))
            s_sim = torch.log(torch.diag(ID_sim_HE))
            
            weight_sim_HH = torch.log(torch.diag(ID_sim_HH)).detach()/scale
            weight_sim_HE = torch.log(torch.diag(ID_sim_HE)).detach()/scale
            wt_l = 2*weight_sim_HE.mul(weight_sim_HH)/(weight_sim_HH + weight_sim_HE)
            wt_l[weight_sim_HH < 0] = 0
            both_sim = l_sim.mul(wt_l) + s_sim.mul(1-wt_l)
            
            adaptive_pos = torch.diag(torch.exp(both_sim))
            
            pos_mask_id = torch.eye(N_id).to(self.device)
            adaptive_sim_mat = adaptive_pos.mul(pos_mask_id) + ID_sim_HE.mul(1-pos_mask_id)
            
            adaptive_sim_mat_L1 = nn.functional.normalize(adaptive_sim_mat, p=1, dim=1)
            
            loss_HH = -1*torch.log(torch.diag(ID_sim_HH_L1)).mean()
            loss_HE = -1*torch.log(torch.diag(ID_sim_HE_L1)).mean()
            loss_adaptive = -1*torch.log(torch.diag(adaptive_sim_mat_L1)).mean()
            
            if self.loss_type == 'sp-h':
                loss = loss_HH
            elif self.loss_type == 'sp-lh':
                loss = loss_HE
            elif self.loss_type == 'adasp':
                loss = loss_adaptive
                
            return loss
        
        except (ValueError, RuntimeError) as e:
            print(f"Warning: AdaSPLoss encountered an error: {e}")
            print(f"Falling back to simpler contrastive loss.")
            return self._compute_fallback_loss(feat_q, targets)
    
    def _compute_fallback_loss(self, features, targets):
        """
        Compute a fallback InfoNCE-style contrastive loss when the original
        AdaSP loss cannot be computed due to batch composition issues.
        """
        # Compute pairwise similarities
        sim_matrix = torch.matmul(features, features.T) / self.temp
        
        # Create a mask for positive pairs (same class)
        pos_mask = (targets.unsqueeze(1) == targets.unsqueeze(0)).float()
        # Remove self-similarity
        pos_mask.fill_diagonal_(0)
        
        # InfoNCE loss
        exp_sim = torch.exp(sim_matrix)
        
        # For each sample, sum similarities with positives
        pos_sim = (pos_mask * exp_sim).sum(dim=1)
        
        # Sum of all similarities (excluding self)
        all_sim = exp_sim.sum(dim=1) - exp_sim.diagonal()
        
        # Compute loss only for samples that have positives
        valid_samples = pos_mask.sum(dim=1) > 0
        
        if valid_samples.sum() > 0:
            # Log probability of selecting the correct positives
            loss = -torch.log(pos_sim[valid_samples] / all_sim[valid_samples]).mean()
        else:
            # If no valid samples (no positives), use a dummy loss
            loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        
        return loss


if __name__ == '__main__':
    # Test loss functions
    print("Testing COSMO training utilities...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Test AdaSP loss
    batch_size  = 32
    embed_dim   = 512
    num_classes = 4
    
    features = torch.randn(batch_size, embed_dim).to(device)
    features = F.normalize(features, dim=-1)
    
    # Create balanced labels for AdaSP
    labels = torch.tensor([i // (batch_size // num_classes) for i in range(batch_size)]).to(device)
    
    adasp_loss = AdaSPLoss(device=device, temp=0.04, loss_type='adasp')
    loss_value = adasp_loss(features, labels)
    
    print(f"AdaSP Loss: {loss_value.item():.4f}")
    
    # Test fallback with irregular batch
    irregular_labels = torch.randint(0, num_classes, (batch_size,)).to(device)
    fallback_loss = adasp_loss(features, irregular_labels)
    
    print(f"Fallback Loss: {fallback_loss.item():.4f}")
    print("Training utilities ready!")