"""
Training module for Discrete Diffusion Model
Uses Cross-Entropy loss instead of MSE
"""

import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from typing import Dict
import logging

from src.models.discrete_diffusion_3d import DiscreteDiscreteDiffusionModel3D

logger = logging.getLogger(__name__)


class DiscreteDiffusionLightningModule(pl.LightningModule):
    """PyTorch Lightning Module for Discrete Diffusion Training"""
    
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        self.save_hyperparameters()
        
        # Create model
        self.model = DiscreteDiscreteDiffusionModel3D(config)
        
        logger.info("Initialized Discrete Diffusion Model")
        logger.info(f"Number of block categories: {self.model.num_classes}")
        logger.info(f"Timesteps: {self.model.num_timesteps}")

        # WEIGHTED LOSS IMPLEMENTATION
        # Air (index 0) is ~95% of data. If we don't weight it down, model predicts only air.
        # We assign a small weight to air (0.1) and 1.0 to everything else.
        loss_weights = torch.ones(self.model.num_classes)
        loss_weights[0] = 0.05  # Drastically reduce importance of air (1/20)
        self.register_buffer('loss_weights', loss_weights)
    
    def forward(self, batch):
        """Forward pass"""
        # Extract batch data
        voxels_onehot = batch['voxels']  # (B, C, D, H, W) - already one-hot
        text_embedding = batch['text_embedding']  # (B, 384)
        size_name = batch['size_name'][0]  # String (same for whole batch)
        
        # Forward through model
        predicted_logits, target_onehot, t = self.model(
            x=voxels_onehot,
            text_embed=text_embedding,
            size=size_name
        )
        
        return predicted_logits, target_onehot
    
    def training_step(self, batch, batch_idx):
        """Training step"""
        predicted_logits, target_onehot = self(batch)
        
        # Cross-Entropy Loss
        # predicted_logits: (B, C, D, H, W)
        # target_onehot: (B, C, D, H, W)
        
        # Convert target one-hot to class indices
        target_classes = torch.argmax(target_onehot, dim=1)  # (B, D, H, W)
        
        # Cross-entropy expects (B, C, D, H, W) logits and (B, D, H, W) targets
        # Using WEIGHTED loss to handle class imbalance (Air vs Blocks)
        loss = F.cross_entropy(
            predicted_logits, 
            target_classes, 
            weight=self.loss_weights,
            reduction='mean'
        )
        
        # Log
        self.log('train/loss_step', loss, on_step=True, on_epoch=False, prog_bar=True)
        self.log('train/loss_epoch', loss, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        """Validation step"""
        predicted_logits, target_onehot = self(batch)
        
        # Cross-Entropy Loss with weights
        target_classes = torch.argmax(target_onehot, dim=1)
        loss = F.cross_entropy(
            predicted_logits, 
            target_classes, 
            weight=self.loss_weights,
            reduction='mean'
        )
        
        # Additional metrics
        predicted_classes = torch.argmax(predicted_logits, dim=1)
        accuracy = (predicted_classes == target_classes).float().mean()
        
        # Log
        self.log('val/loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val/accuracy', accuracy, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss
    
    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler"""
        optimizer = AdamW(
            self.parameters(),
            lr=self.config['training']['learning_rate'],
            weight_decay=self.config['training'].get('weight_decay', 0.01)
        )
        
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=self.config['training']['num_epochs'],
            eta_min=self.config['training']['learning_rate'] * 0.01
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch'
            }
        }


def create_trainer(config: Dict, logger_name: str = "minecraft_discrete_diffusion") -> pl.Trainer:
    """Create PyTorch Lightning trainer with callbacks"""
    from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
    from pytorch_lightning.loggers import TensorBoardLogger
    
    # Callbacks
    callbacks = []
    
    # Model checkpoint
    checkpoint_callback = ModelCheckpoint(
        dirpath=config['training']['checkpoint_dir'],
        filename='minecraft-discrete-diffusion-{epoch:02d}-{val/loss:.4f}',
        monitor='val/loss',
        mode='min',
        save_top_k=1,
        save_last=True,
        every_n_epochs=config['training'].get('save_every_n_epochs', 10),
        verbose=True
    )
    callbacks.append(checkpoint_callback)
    
    # Early stopping
    if config['training'].get('early_stopping', {}).get('enabled', True):
        early_stop_callback = EarlyStopping(
            monitor=config['training']['early_stopping'].get('monitor', 'val/loss'),
            patience=config['training']['early_stopping'].get('patience', 20),
            mode='min',
            verbose=True
        )
        callbacks.append(early_stop_callback)
    
    # Learning rate monitor
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    callbacks.append(lr_monitor)
    
    # Create trainer
    # Multi-GPU support
    num_devices = config['hardware'].get('num_gpus', 1)
    strategy = 'ddp' if num_devices > 1 or num_devices == -1 else 'auto'
    
    trainer = pl.Trainer(
        max_epochs=config['training']['num_epochs'],
        accelerator='gpu' if config['hardware']['device'] == 'cuda' else 'cpu',
        devices=num_devices,
        strategy=strategy,
        precision=config['hardware']['precision'],
        callbacks=callbacks,
        logger=TensorBoardLogger('lightning_logs', name=logger_name),
        gradient_clip_val=config['training'].get('gradient_clip_val', 1.0),
        log_every_n_steps=10,
        deterministic=False  # For speed
    )
    
    return trainer
