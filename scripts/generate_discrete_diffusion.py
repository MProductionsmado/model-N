"""
Generate Minecraft assets using trained Discrete Diffusion Model
"""

import argparse
import yaml
import torch
from pathlib import Path
import sys
import logging
from sentence_transformers import SentenceTransformer

sys.path.append(str(Path(__file__).parent.parent))

from src.training.trainer_discrete_diffusion import DiscreteDiffusionLightningModule
from src.data.schematic_parser import create_schematic

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

torch.set_float32_matmul_precision('medium')

def main():
    parser = argparse.ArgumentParser(description='Generate Minecraft assets with Discrete Diffusion')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='Path to config file')
    parser.add_argument('--prompt', type=str, required=True,
                       help='Text description (e.g., "oak tree", "birch tree big")')
    parser.add_argument('--size', type=str, default=None,
                       help='Size override: normal, big, or huge')
    parser.add_argument('--output', type=str, default='generated',
                       help='Output directory')
    parser.add_argument('--num-samples', type=int, default=1,
                       help='Number of samples to generate')
    parser.add_argument('--sampling-steps', type=int, default=50,
                       help='Number of denoising steps (50=fast, 1000=best quality)')
    parser.add_argument('--guidance-scale', type=float, default=7.5,
                       help='Classifier-Free Guidance scale (1.0=no guidance, 3.0-7.5=strong prompt following)')
    parser.add_argument('--temperature', type=float, default=0.9,
                       help='Temperature for sampling (0.7=conservative, 1.0=balanced, 1.2=diverse)')
    parser.add_argument('--top-k', type=int, default=0,
                       help='Top-k filtering (0=auto-adaptive, 10-20=recommended)')
    parser.add_argument('--top-p', type=float, default=1.0,
                       help='Nucleus sampling (1.0=auto-adaptive, 0.8-0.9=recommended)')
    parser.add_argument('--sample-mode', type=str, default='argmax', choices=['argmax','multinomial'],
                       help='argmax: pick most likely block; multinomial: sample per voxel')
    parser.add_argument('--seed', type=int, default=None,
                       help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Set random seed
    if args.seed is not None:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        logger.info(f"Random seed set to {args.seed}")
    
    # Load config
    logger.info(f"Loading config from {args.config}")
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Auto-detect size from prompt if not specified
    if args.size is None:
        prompt_lower = args.prompt.lower()
        if 'huge' in prompt_lower:
            size = 'huge'
        elif 'big' in prompt_lower or 'large' in prompt_lower:
            size = 'big'
        else:
            size = 'normal'
    else:
        size = args.size
    
    logger.info(f"Size: {size}")
    dims = config['model']['sizes'][size]['dims']
    logger.info(f"Dimensions: {dims}")
    
    # Load model
    logger.info(f"Loading model from {args.checkpoint}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load checkpoint to extract the config it was trained with
    checkpoint = torch.load(args.checkpoint, map_location=device)
    if 'hyper_parameters' in checkpoint and 'config' in checkpoint['hyper_parameters']:
        train_config = checkpoint['hyper_parameters']['config']
        logger.info("Using config from checkpoint (ensures architecture match)")
    else:
        train_config = config
        logger.warning("No config in checkpoint, using current config file")
    
    model = DiscreteDiffusionLightningModule.load_from_checkpoint(
        args.checkpoint,
        config=train_config,
        target_size=size,
        map_location=device,
        strict=False  # Allow loading even if some keys mismatch
    )
    model.eval()
    model.to(device)
    
    # Load text encoder
    logger.info("Loading text encoder...")
    text_encoder = SentenceTransformer(config['model']['text_encoder']['model_name'])
    text_encoder.to(device)
    
    # Encode prompt
    logger.info(f"Encoding prompt: '{args.prompt}'")
    text_embed = text_encoder.encode(
        [args.prompt] * args.num_samples,
        convert_to_tensor=True,
        device=device
    )
    
    # Generate
    logger.info(f"Generating {args.num_samples} sample(s)...")
    logger.info(f"Sampling steps: {args.sampling_steps}")
    logger.info(f"Guidance scale: {args.guidance_scale}")
    logger.info(f"Temperature: {args.temperature}, Top-k: {args.top_k}, Top-p: {args.top_p}")
    
    with torch.no_grad():
        # Generate using discrete diffusion with IMPROVED SAMPLING
        # Returns probabilities over classes (B, C, D, H, W)
        generated_probs = model.model.generate(
            text_embed=text_embed,
            size=size,
            num_samples=args.num_samples,
            sampling_steps=args.sampling_steps,
            guidance_scale=args.guidance_scale,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p
        )
        
        if args.temperature != 1.0:
            generated_probs = generated_probs ** (1.0 / max(args.temperature, 1e-6))
            generated_probs = generated_probs / (generated_probs.sum(dim=1, keepdim=True) + 1e-8)

        if args.sample_mode == 'argmax':
            generated_voxels = torch.argmax(generated_probs, dim=1)
        else:
            # Multinomial sampling per voxel
            B, C, D, H, W = generated_probs.shape
            probs_flat = generated_probs.permute(0,2,3,4,1).reshape(-1, C)
            samples = torch.multinomial(probs_flat, num_samples=1).view(B, D, H, W)
            generated_voxels = samples
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get block vocabulary
    block_id_to_name = {v: k for k, v in config['blocks'].items()}
    
    # Save generated assets
    for i in range(args.num_samples):
        voxels = generated_voxels[i].cpu().numpy()  # (D, H, W)
        
        # Count non-air blocks
        non_air_blocks = (voxels != 0).sum()
        logger.info(f"Sample {i+1}: {non_air_blocks} non-air blocks")
        
        # Create schematic file
        filename = f"{args.prompt.replace(' ', '_')}_{size}_{i+1}.schem"
        output_path = output_dir / filename
        
        create_schematic(
            voxel_data=voxels,
            block_id_to_name=block_id_to_name,
            output_path=str(output_path)
        )
        
        logger.info(f"✓ Saved: {output_path}")
    
    logger.info(f"\n✓ Generated {args.num_samples} sample(s) successfully!")
    logger.info(f"Output directory: {output_dir.absolute()}")


if __name__ == "__main__":
    main()
