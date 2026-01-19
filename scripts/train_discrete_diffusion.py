"""
Train Discrete Diffusion Model
Supports size-specific configurations with global defaults
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import yaml
import logging
import argparse
from datetime import datetime
from copy import deepcopy

import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from src.data.dataset import MinecraftSchematicDataset
from src.training.trainer_discrete_diffusion import DiscreteDiffusionLightningModule, create_trainer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

torch.set_float32_matmul_precision('medium')


def get_size_config(config: dict, size: str) -> dict:
    """
    Merge global defaults with size-specific overrides.
    Size-specific settings take priority over globals.
    
    Args:
        config: Full config dict
        size: 'normal', 'big', or 'huge'
    
    Returns:
        Merged config with size-specific values
    """
    merged = deepcopy(config)
    size_cfg = config['model']['sizes'].get(size, {})
    
    # Override encoder settings
    if 'encoder' in size_cfg:
        for key, value in size_cfg['encoder'].items():
            merged['model']['encoder'][key] = value
    
    # Override diffusion settings
    if 'diffusion' in size_cfg:
        for key, value in size_cfg['diffusion'].items():
            merged['model']['diffusion'][key] = value
    
    # Override training settings
    if 'training' in size_cfg:
        for key, value in size_cfg['training'].items():
            merged['training'][key] = value
    
    return merged


def main():
    parser = argparse.ArgumentParser(description='Train Discrete Diffusion Model')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='Path to config file')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume training')
    parser.add_argument('--debug', action='store_true',
                       help='Run in debug mode (fast_dev_run)')
    parser.add_argument('--size', type=str, required=True,
                       choices=['normal', 'big', 'huge'],
                       help='Target size to train (normal, big, huge)')
    args = parser.parse_args()
    
    # Load config
    logger.info(f"Loading config from {args.config}")
    with open(args.config, 'r') as f:
        raw_config = yaml.safe_load(f)
    
    # Merge global defaults with size-specific overrides
    config = get_size_config(raw_config, args.size)
    
    logger.info(f"=" * 60)
    logger.info(f"TARGET SIZE: {args.size.upper()}")
    logger.info(f"Dimensions: {config['model']['sizes'][args.size]['dims']}")
    logger.info(f"Channels: {config['model']['encoder']['channels']}")
    logger.info(f"Batch Size: {config['training']['batch_size']}")
    logger.info(f"Learning Rate: {config['training']['learning_rate']}")
    logger.info(f"Attention Levels: {config['model']['diffusion']['attention_levels']}")
    logger.info(f"=" * 60)
    
    # Create datasets
    logger.info("Creating datasets...")
    data_dir = Path(config['data']['processed_dir'])
    splits_dir = data_dir.parent / 'splits'
    
    train_dataset = MinecraftSchematicDataset(
        metadata_file=splits_dir / 'train_metadata.json',
        data_dir=data_dir,
        text_encoder_name=config['data'].get('text_encoder_name', 'all-MiniLM-L6-v2'),
        size_filter=[args.size],
        transform=True,
        num_classes=len(config['blocks'])
    )
    
    val_dataset = MinecraftSchematicDataset(
        metadata_file=splits_dir / 'val_metadata.json',
        data_dir=data_dir,
        text_encoder_name=config['data'].get('text_encoder_name', 'all-MiniLM-L6-v2'),
        size_filter=[args.size],
        transform=None,
        num_classes=len(config['blocks'])
    )
    
    # Check if datasets are empty
    if len(train_dataset) == 0:
        raise ValueError(f"No training data found for size '{args.size}'")
    
    logger.info(f"Train dataset size: {len(train_dataset)}")
    logger.info(f"Val dataset size: {len(val_dataset)}")
    
    # Get training params from merged config
    batch_size = config['training']['batch_size']
    num_workers = config['training'].get('num_workers', 4)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    # Create model
    logger.info("Creating model...")
    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        model = DiscreteDiffusionLightningModule.load_from_checkpoint(
            args.resume, config=config, target_size=args.size
        )
    else:
        model = DiscreteDiffusionLightningModule(config, target_size=args.size)
            
    # Create trainer with size-specific settings
    logger.info("Creating trainer...")
    
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=f'models/checkpoints/{args.size}',
        filename='{epoch}-{val_loss:.2f}',
        monitor='val/loss',
        mode='min',
        save_top_k=1,
        every_n_epochs=10  # Only save every 10 epochs to reduce file count
    )
    
    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='step')
    
    # Get accumulation from merged config (already size-specific!)
    acc_grad = config['training'].get('accumulate_grad_batches', 1)
    num_epochs = config['training']['num_epochs']
        
    log_every = config['training'].get('log_every_n_steps', 10)
    
    trainer = pl.Trainer(
        max_epochs=num_epochs,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        precision=16,
        callbacks=[checkpoint_callback, lr_monitor],
        logger=pl.loggers.TensorBoardLogger("logs", name=f"discrete_diffusion_{args.size}"),
        gradient_clip_val=1.0,
        accumulate_grad_batches=acc_grad,
        log_every_n_steps=log_every
    )
    
    # Train
    logger.info("Starting training...")
    logger.info(f"Model: Discrete Diffusion (Multinomial) - Size: {args.size}")
    
    if args.debug:
        logger.info("Running in debug mode (fast_dev_run)")
        trainer.fast_dev_run = True
    
    trainer.fit(
        model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader
    )
    
    logger.info("Training complete!")
    logger.info(f"Best checkpoint: {checkpoint_callback.best_model_path}")


if __name__ == "__main__":
    main()
