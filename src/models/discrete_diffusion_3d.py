"""
Discrete Diffusion Model for 3D Voxel Generation
Uses Multinomial Diffusion instead of Gaussian Noise
Better for categorical data like Minecraft blocks

Based on:
- "Argmax Flows and Multinomial Diffusion" (Hoogeboom et al. 2021)
- "Structured Denoising Diffusion Models in Discrete State-Spaces" (Austin et al. 2021)

IMPROVEMENTS:
- Adaptive noise scheduling based on voxel resolution
- Top-k and nucleus (top-p) sampling for better quality
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
import math


def improved_sampling(
    probs: torch.Tensor, 
    temperature: float = 1.0, 
    top_k: int = 0, 
    top_p: float = 1.0
) -> torch.Tensor:
    """
    Enhanced sampling with top-k and nucleus (top-p) filtering.
    Prevents low-probability blocks from being sampled.
    
    Args:
        probs: Probability distribution (B*D*H*W, num_classes) - MUST BE PROBS!
        temperature: Sampling temperature (lower = more deterministic)
        top_k: Keep only top k classes (0 = disabled)
        top_p: Nucleus sampling threshold (1.0 = disabled)
    
    Returns:
        Sampled indices (B*D*H*W,)
    """
    # CRITICAL FIX: Work directly with probabilities, not logits
    # Temperature scaling on probabilities
    if temperature != 1.0:
        probs = probs ** (1.0 / max(temperature, 1e-6))
        probs = probs / (probs.sum(dim=-1, keepdim=True) + 1e-8)
    
    # Top-k filtering on probabilities
    if top_k > 0:
        top_k = min(top_k, probs.size(-1))
        values, indices = torch.topk(probs, top_k, dim=-1)
        
        # Create mask for top-k
        mask = torch.zeros_like(probs)
        mask.scatter_(-1, indices, 1.0)
        
        # Zero out non-top-k probabilities
        probs = probs * mask
        
        # Renormalize
        probs = probs / (probs.sum(dim=-1, keepdim=True) + 1e-8)
    
    # Nucleus (top-p) sampling on probabilities
    if top_p < 1.0:
        sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        
        # Find cutoff: first index where cumulative > top_p
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift right to keep first token above threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        
        # Set removed probabilities to zero in sorted order
        sorted_probs[sorted_indices_to_remove] = 0.0
        
        # Scatter back to original order
        probs = torch.zeros_like(probs).scatter_(-1, sorted_indices, sorted_probs)
        
        # Renormalize
        probs = probs / (probs.sum(dim=-1, keepdim=True) + 1e-8)
    
    # Handle numerical issues
    if torch.isnan(probs).any() or (probs.sum(dim=-1) == 0).any():
        probs = torch.ones_like(probs) / probs.size(-1)
    
    # Final sampling
    return torch.multinomial(probs, num_samples=1).squeeze(-1)


class SinusoidalPositionEmbeddings(nn.Module):
    """Sinusoidal time embeddings (same as before)"""
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class ConditionalGroupNorm(nn.Module):
    """FiLM conditioning (same as before)"""
    
    def __init__(self, num_groups: int, num_channels: int, cond_dim: int):
        super().__init__()
        self.norm = nn.GroupNorm(num_groups, num_channels, affine=False)
        self.scale = nn.Linear(cond_dim, num_channels)
        self.shift = nn.Linear(cond_dim, num_channels)
    
    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        # x: (B, C, D, H, W)
        # cond: (B, cond_dim)
        x = self.norm(x)
        scale = self.scale(cond)[:, :, None, None, None]
        shift = self.shift(cond)[:, :, None, None, None]
        return x * (1 + scale) + shift


class ResidualBlock3D(nn.Module):
    """3D Residual Block with FiLM conditioning (same as before)"""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        cond_dim: int,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = ConditionalGroupNorm(8, out_channels, cond_dim)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = ConditionalGroupNorm(8, out_channels, cond_dim)
        
        self.activation = nn.SiLU()
        self.dropout = nn.Dropout(dropout)
        
        # Shortcut
        if in_channels != out_channels:
            self.shortcut = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()
    
    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = self.conv1(x)
        h = self.norm1(h, cond)
        h = self.activation(h)
        h = self.dropout(h)
        
        h = self.conv2(h)
        h = self.norm2(h, cond)
        h = self.activation(h)
        
        return h + self.shortcut(x)


class AttentionBlock3D(nn.Module):
    """3D Self-Attention (same as before)"""
    
    def __init__(self, channels: int, num_heads: int = 8):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        
        assert channels % num_heads == 0, "channels must be divisible by num_heads"
        
        self.norm = nn.GroupNorm(8, channels)
        self.qkv = nn.Conv3d(channels, channels * 3, kernel_size=1)
        self.proj = nn.Conv3d(channels, channels, kernel_size=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, D, H, W = x.shape
        
        h = self.norm(x)
        qkv = self.qkv(h)
        
        # Reshape to (B, num_heads, head_dim, D*H*W)
        qkv = qkv.reshape(B, 3, self.num_heads, self.head_dim, D * H * W)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
        
        # Attention
        scale = self.head_dim ** -0.5
        attn = torch.einsum('bhdn,bhdm->bhnm', q, k) * scale
        attn = F.softmax(attn, dim=-1)
        
        # Apply attention
        out = torch.einsum('bhnm,bhdm->bhdn', attn, v)
        out = out.reshape(B, C, D, H, W)
        out = self.proj(out)
        
        return x + out


class UNet3D(nn.Module):
    """
    3D UNet for Discrete Diffusion
    REUSED from continuous diffusion - only output layer changes
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        model_channels: int = 64,
        channel_multipliers: Tuple[int, ...] = (1, 2, 4, 8),
        num_res_blocks: int = 2,
        cond_dim: int = 512,
        attention_levels: Tuple[int, ...] = (2, 3),
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.channel_multipliers = channel_multipliers
        self.num_levels = len(channel_multipliers)
        
        # Initial convolution
        self.input_conv = nn.Conv3d(in_channels, model_channels, kernel_size=3, padding=1)
        
        # Encoder
        self.encoder_blocks = nn.ModuleList()
        self.downsample_ops = nn.ModuleList()
        
        ch = model_channels
        for level, mult in enumerate(channel_multipliers):
            out_ch = model_channels * mult
            
            # Residual blocks for this level
            blocks = nn.ModuleList()
            for _ in range(num_res_blocks):
                blocks.append(ResidualBlock3D(ch, out_ch, cond_dim, dropout))
                ch = out_ch
            
            # Attention at deeper levels
            if level in attention_levels:
                blocks.append(AttentionBlock3D(out_ch))
            
            self.encoder_blocks.append(blocks)
            
            # Downsample (except last level)
            if level < self.num_levels - 1:
                self.downsample_ops.append(nn.Conv3d(out_ch, out_ch, kernel_size=3, stride=2, padding=1))
            else:
                self.downsample_ops.append(nn.Identity())
        
        # Bottleneck
        self.bottleneck = nn.ModuleList([
            ResidualBlock3D(ch, ch, cond_dim, dropout),
            AttentionBlock3D(ch),
            ResidualBlock3D(ch, ch, cond_dim, dropout)
        ])
        
        # Decoder
        self.decoder_blocks = nn.ModuleList()
        self.upsample_ops = nn.ModuleList()
        
        for level, mult in reversed(list(enumerate(channel_multipliers))):
            out_ch = model_channels * mult
            
            # Upsample first (except first decoder level)
            if level < self.num_levels - 1:
                # self.upsample_ops.append(nn.ConvTranspose3d(ch, ch, kernel_size=4, stride=2, padding=1))
                # IMPROVEMENT: Use Nearest Upsample + Conv to avoid checkerboard artifacts in 3D
                self.upsample_ops.append(nn.Sequential(
                    nn.Upsample(scale_factor=2, mode='nearest'),
                    nn.Conv3d(ch, ch, kernel_size=3, padding=1)
                ))
            else:
                self.upsample_ops.append(nn.Identity())
            
            # Residual blocks for this level (with skip connection)
            blocks = nn.ModuleList()
            for i in range(num_res_blocks + 1):
                # First block gets skip connection
                in_ch = ch + out_ch if i == 0 else out_ch
                blocks.append(ResidualBlock3D(in_ch, out_ch, cond_dim, dropout))
            
            # Attention at deeper levels
            if level in attention_levels:
                blocks.append(AttentionBlock3D(out_ch))
            
            self.decoder_blocks.append(blocks)
            ch = out_ch
        
        # Output - predicts LOGITS for each block category
        self.output_conv = nn.Sequential(
            nn.GroupNorm(8, ch),
            nn.SiLU(),
            nn.Conv3d(ch, out_channels, kernel_size=3, padding=1)
        )
    
    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        text_embed: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input (B, in_channels, D, H, W) - one-hot encoded
            t: Timesteps (B,)
            text_embed: Text embeddings (B, text_dim)
        
        Returns:
            Logits for each category (B, out_channels, D, H, W)
        """
        # Condition: concatenate time and text embeddings
        cond = torch.cat([t, text_embed], dim=1)  # (B, cond_dim)
        
        # Initial conv
        h = self.input_conv(x)
        
        # Encoder with skip connections
        skips = []
        for blocks, downsample in zip(self.encoder_blocks, self.downsample_ops):
            for block in blocks:
                if isinstance(block, AttentionBlock3D):
                    h = block(h)
                else:
                    h = block(h, cond)
            skips.append(h)
            h = downsample(h)
        
        # Bottleneck
        for block in self.bottleneck:
            if isinstance(block, AttentionBlock3D):
                h = block(h)
            else:
                h = block(h, cond)
        
        # Decoder with skip connections
        for blocks, upsample in zip(self.decoder_blocks, self.upsample_ops):
            h = upsample(h)
            skip = skips.pop()
            h = torch.cat([h, skip], dim=1)
            
            for block in blocks:
                if isinstance(block, AttentionBlock3D):
                    h = block(h)
                else:
                    h = block(h, cond)
        
        # Output logits
        return self.output_conv(h)


class DiscreteDiscreteDiffusionModel3D(nn.Module):
    """
    Discrete Diffusion Model using Multinomial Transitions
    Better for categorical data like Minecraft blocks
    Single-Size Version: Trains a dedicated model for one specific size category.
    """
    
    def __init__(self, config: Dict, target_size: str):
        super().__init__()
        self.config = config
        self.target_size = target_size
        self.num_classes = len(config['blocks'])
        self.num_timesteps = config['model']['diffusion']['num_timesteps']
        
        # Validate target size
        if target_size not in config['model']['sizes']:
            raise ValueError(f"Target size '{target_size}' not found in config. Available: {list(config['model']['sizes'].keys())}")
        
        size_config = config['model']['sizes'][target_size]
        
        # Time embeddings
        time_embed_dim = 256
        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbeddings(time_embed_dim),
            nn.Linear(time_embed_dim, time_embed_dim),
            nn.SiLU()
        )
        
        # Text embeddings projection
        text_embed_dim = config['model']['text_encoder']['embedding_dim']
        text_proj_dim = 256
        self.text_proj = nn.Sequential(
            nn.Linear(text_embed_dim, text_proj_dim),
            nn.SiLU()
        )
        
        cond_dim = time_embed_dim + text_proj_dim
        
        # Create SINGLE UNet for the target size
        print(f"Initializing 3D U-Net for size: {target_size} {size_config['dims']}")
        self.unet = UNet3D(
            in_channels=self.num_classes + 3,  # Added 3 channels for CoordConv (x,y,z)
            out_channels=self.num_classes,  # Predict logits for each class
            model_channels=config['model']['encoder']['channels'][0],
            channel_multipliers=tuple(
                c // config['model']['encoder']['channels'][0] 
                for c in config['model']['encoder']['channels']
            ),
            num_res_blocks=config['model']['diffusion']['num_res_blocks'],
            cond_dim=cond_dim,
            attention_levels=tuple(config['model']['diffusion']['attention_levels']),
            dropout=config['model']['diffusion']['dropout']
        )
        
        # Transition matrix schedule (probability of staying in same state)
        # ADAPTIVE: Schedule based on resolution of target size
        dims = size_config['dims']
        betas = self._adaptive_cosine_beta_schedule(
            self.num_timesteps, 
            voxel_size=dims
        )
        self.register_buffer('betas', betas)
        
        # Cumulative product of (1 - beta)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
    
    def _adaptive_cosine_beta_schedule(
        self, 
        timesteps: int, 
        voxel_size: tuple,
        s_base: float = 0.008
    ) -> torch.Tensor:
        """
        ADAPTIVE cosine schedule that adjusts based on voxel resolution.
        Larger structures get MORE GRADUAL noise schedules.
        
        Args:
            timesteps: Number of diffusion steps
            voxel_size: (D, H, W) dimensions
            s_base: Base smoothness parameter
            
        Returns:
            betas: Noise schedule tensor
        """
        max_dim = max(voxel_size)
        
        # Adjust smoothness and max noise based on resolution
        if max_dim <= 16:
            s = s_base  # 0.008
            max_noise = 0.95  # Aggressive for small structures
        elif max_dim <= 32:
            s = s_base * 1.5  # 0.012 - more gradual
            max_noise = 0.85  # Less aggressive
        else:  # 64+
            s = s_base * 2.5  # 0.020 - very gradual
            max_noise = 0.75  # Conservative
        
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        
        # Clip with adaptive max noise
        # REMOVED max_noise clipping to ensure full noise at t=T
        # This prevents distribution mismatch between training end and inference start
        return torch.clip(betas, 0.0001, 0.9999)
    
    def _cosine_beta_schedule(self, timesteps: int, s: float = 0.008) -> torch.Tensor:
        """
        Cosine schedule for transition probabilities
        From "Improved Denoising Diffusion Probabilistic Models"
        """
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)
    
    def q_sample(
        self,
        x_start: torch.Tensor,
        t: torch.Tensor,
        size: str = None
    ) -> torch.Tensor:
        """
        Forward diffusion: q(x_t | x_0)
        """
        alpha_cumprod_t = self.alphas_cumprod[t]
        
        # Reshape for broadcasting
        while len(alpha_cumprod_t.shape) < len(x_start.shape):
            alpha_cumprod_t = alpha_cumprod_t.unsqueeze(-1)
        
        # Probability of staying in same state
        stay_prob = alpha_cumprod_t
        
        # Probability of transitioning to uniform
        uniform_prob = (1.0 - alpha_cumprod_t) / self.num_classes
        
        # Create transition: stay in same state with prob stay_prob,
        # otherwise uniform over all states
        noised = x_start * stay_prob + uniform_prob
        
        return noised
    
    def _add_coordinate_channels(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add normalized coordinate channels (x, y, z) to the input.
        CoordConv implementation for 3D.
        
        Args:
            x: Input tensor (B, C, D, H, W)
            
        Returns:
            Tensor with added coordinates (B, C+3, D, H, W)
        """
        B, _, D, H, W = x.shape
        device = x.device
        
        # Create normalized coordinates (-1 to 1)
        # Z-axis (Depth)
        z_coords = torch.linspace(-1, 1, steps=D, device=device).view(1, 1, D, 1, 1)
        z_coords = z_coords.expand(B, 1, D, H, W)
        
        # Y-axis (Height)
        y_coords = torch.linspace(-1, 1, steps=H, device=device).view(1, 1, 1, H, 1)
        y_coords = y_coords.expand(B, 1, D, H, W)
        
        # X-axis (Width)
        x_coords = torch.linspace(-1, 1, steps=W, device=device).view(1, 1, 1, 1, W)
        x_coords = x_coords.expand(B, 1, D, H, W)
        
        # Concatenate
        return torch.cat([x, z_coords, y_coords, x_coords], dim=1)

    def forward(
        self,
        x: torch.Tensor,
        text_embed: torch.Tensor,
        size: str
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Training forward pass
        
        Args:
            x: Clean one-hot voxels (B, C, D, H, W)
            text_embed: Text embeddings (B, text_dim)
            size: Size category
        
        Returns:
            predicted_logits: (B, C, D, H, W)
            target_onehot: (B, C, D, H, W)
            t: (B,)
        """
        batch_size = x.shape[0]
        device = x.device
        
        # Sample random timesteps
        t = torch.randint(0, self.num_timesteps, (batch_size,), device=device, dtype=torch.long)
        
        # Forward diffusion WITH SIZE-SPECIFIC SCHEDULE
        # q_sample returns soft probabilities (mixture of signal + uniform)
        x_t_probs = self.q_sample(x, t, size=size)
        
        # CRITICAL FIX: Sample discrete states from the probabilities
        # This matches the discrete generation process (D3PM)
        # We reshape to (N, C) for multinomial sampling
        B, C, D, H, W = x_t_probs.shape
        x_t_flat = x_t_probs.permute(0, 2, 3, 4, 1).reshape(-1, C)
        
        # Sample indices
        x_t_indices = torch.multinomial(x_t_flat, num_samples=1)
        x_t_indices = x_t_indices.reshape(B, D, H, W)
        
        # Convert back to one-hot floats for UNet input
        x_t = F.one_hot(x_t_indices, num_classes=self.num_classes).permute(0, 4, 1, 2, 3).float()
        
        # Project embeddings
        time_embed = self.time_embed(t.float())
        text_proj = self.text_proj(text_embed)
        
        # Add coordinate channels (CoordConv)
        # This helps the model understand absolute position (floor vs roof)
        x_t_aug = self._add_coordinate_channels(x_t)
        
        # Predict original one-hot from noised version
        predicted_logits = self.unet(x_t_aug, time_embed, text_proj)
        # Return predicted logits and original clean target
        return predicted_logits, x, t

    @torch.no_grad()
    def p_sample(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        text_embed: torch.Tensor,
        text_proj: torch.Tensor,
        size: str = None,
        guidance_scale: float = 1.0
    ) -> torch.Tensor:
        """
        Reverse diffusion: p(x_{t-1} | x_t)
        """
        # Predict logits for x_0 (clean data estimate)
        time_embed = self.time_embed(t.float())
        
        # Add coordinate channels (CoordConv)
        x_aug = self._add_coordinate_channels(x)
        
        # Classifier-Free Guidance: conditional + unconditional prediction
        if guidance_scale != 1.0:
            # Conditional prediction (with text)
            predicted_logits_cond = self.unet(x_aug, time_embed, text_proj)
            
            # Unconditional prediction (without text - zero embedding)
            text_proj_uncond = torch.zeros_like(text_proj)
            predicted_logits_uncond = self.unet(x_aug, time_embed, text_proj_uncond)
            
            # CFG: interpolate logits before softmax
            predicted_logits = predicted_logits_uncond + guidance_scale * (predicted_logits_cond - predicted_logits_uncond)
        else:
            # No guidance: just conditional
            predicted_logits = self.unet(x_aug, time_embed, text_proj)

        # Estimate x_0 distribution
        x_0_pred = F.softmax(predicted_logits, dim=1)

        if t[0] > 0:
            alpha_cumprod_t_prev = self.alphas_cumprod[t - 1]
            
            while len(alpha_cumprod_t_prev.shape) < len(x.shape):
                alpha_cumprod_t_prev = alpha_cumprod_t_prev.unsqueeze(-1)

            # Prior over x_{t-1} constructed from predicted clean data
            stay_prob_prev = alpha_cumprod_t_prev
            uniform_prob_prev = (1.0 - alpha_cumprod_t_prev) / self.num_classes
            prior_prev = x_0_pred * stay_prob_prev + uniform_prob_prev

            # Incorporate current noisy state x (acts like likelihood term)
            # p(x_t | x_{t-1})
            # Retrieve beta_t
            betas_t = self.betas[t].view(-1, 1, 1, 1, 1)

            # Likelihood p(x_t | x_{t-1})

            # This is a vector over x_{t-1} states
            uniform_jump = betas_t / self.num_classes
            stay_p = 1.0 - betas_t
            
            # If x_{t-1} is same as observed x_t, prob is high.
            # likelihood[c] = P(x_t | x_{t-1}=c)
            # = (1-beta)*I(x_t==c) + beta/K
            likelihood = x * stay_p + uniform_jump

            # Element-wise product then renormalize
            posterior_unnorm = prior_prev * likelihood
            posterior = posterior_unnorm / (posterior_unnorm.sum(dim=1, keepdim=True) + 1e-8)
            return posterior
        else:
            # Final step: return categorical distribution for x_0
            return x_0_pred
    
    @torch.no_grad()
    def generate(
        self,
        text_embed: torch.Tensor,
        size: str = None, # kept but ignored, uses self.target_size
        num_samples: int = 1,
        sampling_steps: Optional[int] = None,
        guidance_scale: float = 1.0,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0
    ) -> torch.Tensor:
        """
        Generate samples using reverse diffusion with Classifier-Free Guidance
        
        FIXED: Keep probability distributions through the denoising process,
        only sample discretely at the final step. This prevents error accumulation
        and allows the model to smoothly refine its predictions.
        """
        device = text_embed.device
        size = self.target_size
        
        # Get dimensions
        dims = self.config['model']['sizes'][size]['dims']
        D, H, W = dims
        max_dim = max(dims)
        
        # ADAPTIVE sampling parameters based on resolution
        if top_k <= 0:
            if max_dim <= 16:
                top_k = 15
            elif max_dim <= 32:
                top_k = 12
            else:
                top_k = 10
        
        # RELAXED Top-P:
        # Prevents masking out rare but correct blocks (like a trunk base surrounded by air)
        if top_p >= 1.0:
            if max_dim <= 16:
                top_p = 0.99  # Was 0.9 - caused floating trees (censored sparse trunks)
            elif max_dim <= 32:
                top_p = 0.95
            else:
                top_p = 0.92
        
        print(f"Adaptive sampling for {size}: temp={temperature:.2f}, top_k={top_k}, top_p={top_p}")
        
        # Project text embeddings once
        text_proj = self.text_proj(text_embed)
        
        # Start from RANDOM NOISE (Categorical Prior)
        # Instead of uniform probabilities, we start with a sample from the prior
        # This matches the training assumption where x_t is always a discrete state (one-hot)
        x_indices = torch.randint(0, self.num_classes, (num_samples, D, H, W), device=device)
        x = F.one_hot(x_indices, num_classes=self.num_classes).permute(0, 4, 1, 2, 3).float()
        
        # Determine timesteps
        if sampling_steps is None:
            timesteps = list(range(self.num_timesteps - 1, -1, -1))
        else:
            # DDIM-style: subset of timesteps
            timesteps = torch.linspace(
                self.num_timesteps - 1, 0, sampling_steps, dtype=torch.long, device=device
            ).tolist()
        
        # Iterative denoising
        for i, t in enumerate(timesteps):
            t_batch = torch.full((num_samples,), int(t), device=device, dtype=torch.long)
            
            # Get posterior probabilities p(x_{t-1} | x_t)
            # x is always one-hot encoded here
            posterior = self.p_sample(x, t_batch, text_embed, text_proj, size, guidance_scale=guidance_scale)
            
            # SAMPLE from the posterior distribution
            # This is critical: Discrete diffusion requires sampling at each step
            # to maintain the "one-hot" manifold the UNet was trained on.
            
            # Apply temperature if needed (sharpen/flatten distribution)
            if temperature != 1.0:
                 posterior = posterior ** (1.0 / temperature)
                 posterior = posterior / (posterior.sum(dim=1, keepdim=True) + 1e-8)

            B, C, D, H, W = posterior.shape
            probs_flat = posterior.permute(0, 2, 3, 4, 1).reshape(-1, C)
            
            # Determine sampling strategy for this step
            # Use improved sampling (top-k/p) for ALL steps to maintain structure
            # This helps prevent "floating" structures by avoiding low-probability air tokens
            # in critical structural positions (like the base of a tree)
            sampled_indices = improved_sampling(
                probs_flat, 
                temperature=temperature, 
                top_k=top_k, 
                top_p=top_p
            )
            
            sampled_indices = sampled_indices.reshape(B, D, H, W)
            
            # Convert to one-hot for next step
            x = F.one_hot(sampled_indices.long(), num_classes=self.num_classes).permute(0, 4, 1, 2, 3).float()
        
        return x
