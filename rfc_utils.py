import json
import os
import numpy as np
try:
    from scipy.io import loadmat
except ImportError:
    loadmat = None
import mat73

import pytorch_lightning as pl
import torch
from torchmetrics.audio.stoi import ShortTimeObjectiveIntelligibility
# from torchmetrics.audio import ShortTimeObjectiveIntelligibility
# If stoi computation is not available, you should install the package:
# pip install pystoi
import fast_bss_eval # https://github.com/fakufaku/fast_bss_eval

class CustomLRLogger(pl.Callback):
    def on_validation_end(self, trainer, pl_module):
        lr = trainer.optimizers[0].param_groups[0]['lr']
        global_step = trainer.global_step
        # For TensorBoardLogger
        if hasattr(pl_module.logger, "experiment"):
            pl_module.logger.experiment.add_scalar("lr-Adam", lr, global_step)
            
def load_json_config(path):
    """Load a JSON config file from the given path."""
    with open(path, 'r') as f:
        return json.load(f)

def load_file_list(rellist_path):
    """Load a list of file paths from a text file (one path per line)."""
    with open(rellist_path) as file:
        return [line.rstrip('\n') for line in file]

def load_mat_file(file_path):
    """Load a .mat file using scipy.io.loadmat if possible, otherwise mat73.loadmat."""
    if loadmat is not None:
        try:
            return loadmat(file_path)
        except (NotImplementedError, ValueError):
            return mat73.loadmat(file_path)
    else:
        return mat73.loadmat(file_path)

def robust_complex_divide(A: torch.Tensor, B: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Helper for numerically stable complex division A / B."""
    B_conj = B.conj()
    B_mag_sq = (B * B_conj).real + eps
    return (A * B_conj) / B_mag_sq

def configure_optimizer_and_scheduler(module, lr, optimizer_name='adam', weight_decay=0.0, sch_factor=0.25, sch_patience=5, monitor='valid_loss'):
    """
    Shared utility to create optimizer and scheduler for LightningModule classes.
    Args:
        module: The LightningModule (self)
        lr: Learning rate
        optimizer_name: 'adam' or 'adamw'
        weight_decay: Weight decay for optimizer
        sch_factor: LR scheduler factor
        sch_patience: LR scheduler patience
        monitor: Metric to monitor for scheduler
    Returns:
        dict for Lightning configure_optimizers
    """
    import torch
    if optimizer_name.lower() == 'adam':
        mod_opt = torch.optim.Adam(module.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name.lower() == 'adamw':
        mod_opt = torch.optim.AdamW(module.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")
    mod_sch = torch.optim.lr_scheduler.ReduceLROnPlateau(
        mod_opt, mode='min', factor=sch_factor, patience=sch_patience,
        min_lr=lr*1e-2, threshold=1e-3
    )
    return {
        "optimizer": mod_opt,
        "lr_scheduler": {
            "scheduler": mod_sch,
            "monitor": monitor,
            "interval": "epoch",
            "frequency": 1
        }
    }

# === Evaluation utilities (migrated from common/common_utils.py) ===
def hse_eval(soi_est, srcs_true, soi_comp, fs=16000):
    """
    Calculates various metrics for extraction evaluation
    Single-channel evaluation (on reference channel)
    All metrics are calculated in the time domain
    
    Migrated from common/common_utils.py for consolidation
    
    Inputs:
        soi_est [1,N]: estimated soi
        srcs_true [D,N]: true source signals soi = srcs_true[0,:]
        soi_comp [1,N]: the soi component in the soi_est = soi_comp + noi_comps
        fs: sampling frequency
    
    Outputs:
        sdr_bss, sir_bss, sar_bss: BSS-EVAL metrics
        stoi: Short-Time Objective Intelligibility (STOI) score
        sdr_td: sdr computed as 10*log10(E_soi-est/E_dist), where E_dist = min_a (E_{soi-est - a*soi_true})
    """
    sdr_bss, sir_bss, sar_bss, _ = fast_bss_eval.bss_eval_sources(srcs_true, torch.cat([soi_est,soi_est],dim=0), compute_permutation=True)
    
    stoi_class = ShortTimeObjectiveIntelligibility(16000, 'wb') # here 'wb' is interpreted as true
    stoi = stoi_class(soi_est, srcs_true[0:1,:])
    
    # Calculate SDR in time domain
    alpha = torch.sum(torch.pinverse(srcs_true[0:1,:]).permute(1,0)* soi_comp)
    sdr_td = 10 * torch.log10(torch.mean(soi_comp**2, dim=1) / torch.mean((soi_comp - alpha * srcs_true[0:1,:])**2, dim=1))
    
    return sdr_bss[0], sir_bss[0], sar_bss[0], stoi, sdr_td

def check_cuda_for_bss_eval():
    """
    Helper to check CUDA availability for bss_eval
    Returns device if CUDA available, raises error otherwise
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        raise RuntimeError("fast_bss_eval requires a CUDA-enabled GPU (due to a bug in lu_solve)")

