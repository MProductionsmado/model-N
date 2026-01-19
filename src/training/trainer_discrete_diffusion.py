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
    
    def __init__(self, config: Dict, target_size: str):
        super().__init__()
        self.config = config
        self.target_size = target_size
        self.save_hyperparameters()
        
        # Create model for specific size
        self.model = DiscreteDiscreteDiffusionModel3D(config, target_size=target_size)
        
        logger.info("Initialized Discrete Diffusion Model")
        logger.info(f"Target Size: {target_size}")
        logger.info(f"Number of block categories: {self.model.num_classes}")
        logger.info(f"Timesteps: {self.model.num_timesteps}")

        # WEIGHTED LOSS IMPLEMENTATION
        # Air (index 0) is ~95% of data. If we don't weight it down, model predicts only air.
        # We assign a small weight to air (0.1) and 1.0 to everything else.
        loss_weights = torch.ones(self.model.num_classes)
        loss_weights[0] = 0.20  # Increased from 0.05 to prevent "solid noise" artifacts
        self.register_buffer('loss_weights', loss_weights)
    
    def forward(self, batch):
        """Forward pass"""
        # Extract batch data
        voxels = batch['voxels']  # (B, D, H, W) - indices
        text_embedding = batch['text_embedding']  # (B, 384)
        # size_name is in batch but we ignore it as we only train one size
        
        # One-hot encode ON GPU (saves RAM in dataloader)
        voxels_onehot = F.one_hot(voxels, num_classes=self.model.num_classes)
        voxels_onehot = voxels_onehot.permute(0, 4, 1, 2, 3).float()  # (B, C, D, H, W)
        
        # Forward through model
        predicted_logits, target_onehot, t = self.model(
            x=voxels_onehot,
            text_embed=text_embedding,
            size=None # Ignored by new single-size model
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
        batch_size = predicted_logits.shape[0]
        self.log('train/loss_step', loss, on_step=True, on_epoch=False, prog_bar=True, batch_size=batch_size)
        self.log('train/loss_epoch', loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size)
        
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
        batch_size = predicted_logits.shape[0]
        self.log('val/loss', loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size)
        self.log('val/accuracy', accuracy, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size)
        
        return loss

    def on_validation_epoch_end(self):
        """Print metrics at end of epoch to console"""
        if self.trainer.sanity_checking:
             return
             
        metrics = self.trainer.callback_metrics
        train_loss = metrics.get('train/loss_epoch', 0.0)
        val_loss = metrics.get('val/loss', 0.0)
        val_acc = metrics.get('val/accuracy', 0.0)
        
        print(f"\n[Epoch {self.current_epoch}] Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
    
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
