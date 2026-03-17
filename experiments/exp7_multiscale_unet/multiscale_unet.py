#!/usr/bin/env python3
# OpenSKIZZE
# Copyright (C) 2025 [Alexander Hagg]
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""
Multi-Scale U-Net for KLAM_21 Surrogate Modeling

This U-Net variant can handle variable parcel sizes by:
1. Using adaptive pooling to handle variable input dimensions
2. Encoding parcel size as an additional input channel/embedding
3. Dynamic unpooling to match input resolution
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Tuple


@dataclass
class MultiScaleUNetConfig:
    """Configuration for multi-scale U-Net."""
    input_channels: int = 3  # terrain, buildings, landuse
    output_channels: int = 6  # uq, vq, uz, vz, Ex, Hx
    base_channels: int = 64
    depth: int = 4
    dropout: float = 0.1
    use_size_embedding: bool = True
    size_embedding_dim: int = 16


class SizeEmbedding(nn.Module):
    """Embed parcel size as learnable features."""
    
    def __init__(self, embedding_dim: int):
        super().__init__()
        self.embedding_dim = embedding_dim
        # MLP to encode (height, width) -> embedding
        self.mlp = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Linear(64, embedding_dim),
        )
    
    def forward(self, h: int, w: int, batch_size: int) -> torch.Tensor:
        """
        Args:
            h, w: Input height and width
            batch_size: Batch size
        
        Returns:
            Embedding of shape (batch_size, embedding_dim, 1, 1)
        """
        # Create size tensor [h, w]
        size_tensor = torch.tensor([[h, w]], dtype=torch.float32).to(next(self.mlp.parameters()).device)
        size_tensor = size_tensor.repeat(batch_size, 1)
        
        # Embed
        embedding = self.mlp(size_tensor)  # (batch, embedding_dim)
        embedding = embedding.unsqueeze(-1).unsqueeze(-1)  # (batch, embedding_dim, 1, 1)
        
        return embedding


class DoubleConv(nn.Module):
    """(Conv2d -> BN -> ReLU) × 2"""
    
    def __init__(self, in_channels, out_channels, dropout=0.0):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x):
        return self.double_conv(x)


class MultiScaleUNet(nn.Module):
    """
    Multi-scale U-Net that can handle variable input dimensions.
    
    Key features:
    - Adaptive pooling at bottleneck to fixed size
    - Size embedding to inform network about parcel dimensions
    - Flexible encoder/decoder that adapts to input size
    """
    
    def __init__(self, config: MultiScaleUNetConfig):
        super().__init__()
        self.config = config
        
        # Size embedding
        if config.use_size_embedding:
            self.size_embedding = SizeEmbedding(config.size_embedding_dim)
            first_in_channels = config.input_channels + config.size_embedding_dim
        else:
            self.size_embedding = None
            first_in_channels = config.input_channels
        
        # Encoder (downsampling)
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        
        in_ch = first_in_channels
        for i in range(config.depth):
            out_ch = config.base_channels * (2 ** i)
            self.encoders.append(DoubleConv(in_ch, out_ch, config.dropout))
            if i < config.depth - 1:
                self.pools.append(nn.MaxPool2d(2))
            in_ch = out_ch
        
        # Bottleneck with adaptive pooling
        self.bottleneck_size = 8  # Fixed bottleneck spatial size
        self.adaptive_pool = nn.AdaptiveAvgPool2d(self.bottleneck_size)
        
        bottleneck_ch = config.base_channels * (2 ** (config.depth - 1))
        self.bottleneck = DoubleConv(bottleneck_ch, bottleneck_ch * 2, config.dropout)
        
        # Decoder (upsampling)
        self.decoders = nn.ModuleList()
        self.upconvs = nn.ModuleList()
        
        for i in range(config.depth - 1, 0, -1):
            in_ch = config.base_channels * (2 ** i) * 2  # *2 from bottleneck or previous decoder
            out_ch = config.base_channels * (2 ** (i - 1))
            
            self.upconvs.append(nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2))
            self.decoders.append(DoubleConv(out_ch * 2, out_ch, config.dropout))  # *2 from skip connection
        
        # Final output layer
        self.output = nn.Conv2d(config.base_channels, config.output_channels, kernel_size=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with variable input dimensions.
        
        Args:
            x: Input tensor of shape (B, C, H, W) where H and W can vary
        
        Returns:
            Output tensor of shape (B, output_channels, H, W)
        """
        batch_size, _, h, w = x.shape
        
        # Add size embedding
        if self.size_embedding is not None:
            size_emb = self.size_embedding(h, w, batch_size)
            # Broadcast to match input spatial dimensions
            size_emb = F.interpolate(size_emb, size=(h, w), mode='nearest')
            x = torch.cat([x, size_emb], dim=1)
        
        # Store original size for final resize
        original_size = (h, w)
        
        # Encoder
        skip_connections = []
        for i, (encoder, pool) in enumerate(zip(self.encoders[:-1], self.pools)):
            x = encoder(x)
            skip_connections.append(x)
            x = pool(x)
        
        # Last encoder without pooling
        x = self.encoders[-1](x)
        
        # Adaptive pooling to fixed bottleneck size
        pre_bottleneck_size = x.shape[2:]  # Store for upsampling
        x = self.adaptive_pool(x)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Upsample from bottleneck to last encoder size
        x = F.interpolate(x, size=pre_bottleneck_size, mode='bilinear', align_corners=False)
        
        # Decoder
        for i, (upconv, decoder) in enumerate(zip(self.upconvs, self.decoders)):
            skip = skip_connections[-(i+1)]
            
            # Upsample
            x = upconv(x)
            
            # Match skip connection size (handle odd dimensions)
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
            
            # Concatenate skip connection
            x = torch.cat([x, skip], dim=1)
            
            # Decoder conv
            x = decoder(x)
        
        # Final output
        x = self.output(x)
        
        # Resize to original input dimensions
        if x.shape[2:] != original_size:
            x = F.interpolate(x, size=original_size, mode='bilinear', align_corners=False)
        
        return x


def create_multiscale_unet(
    use_size_embedding: bool = True,
    base_channels: int = 64,
    depth: int = 4,
    dropout: float = 0.1
) -> MultiScaleUNet:
    """Factory function to create multi-scale U-Net."""
    config = MultiScaleUNetConfig(
        input_channels=3,
        output_channels=6,
        base_channels=base_channels,
        depth=depth,
        dropout=dropout,
        use_size_embedding=use_size_embedding,
    )
    return MultiScaleUNet(config)


if __name__ == '__main__':
    # Test multi-scale capability
    model = create_multiscale_unet()
    
    print("Testing multi-scale U-Net:")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test various input sizes
    test_sizes = [
        (66, 94),   # 27m parcel (training size)
        (110, 158), # 51m parcel
        (146, 214), # 69m parcel
    ]
    
    model.eval()
    with torch.no_grad():
        for h, w in test_sizes:
            x = torch.randn(2, 3, h, w)  # Batch of 2
            y = model(x)
            print(f"  Input: {x.shape} -> Output: {y.shape} ✓")
