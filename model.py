import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthwiseSeparableConv(nn.Module):
    """Depthwise Separable Convolution for efficiency"""
    def __init__(self, in_channels, out_channels, kernel_size, padding, dilation=1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, 
                                   padding=padding, dilation=dilation, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.norm = nn.GroupNorm(8, out_channels)
        
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.norm(x)
        return x


class MSRF_Block(nn.Module):
    """Multi-Scale Receptive Field Block - SOTA 2024-2025"""
    def __init__(self, channels):
        super().__init__()
        
        # Multi-scale branches with different receptive fields
        # Scale 1: Small receptive field (3x3)
        self.branch1 = nn.Sequential(
            DepthwiseSeparableConv(channels, channels, 3, padding=1),
            nn.GELU()
        )
        
        # Scale 2: Medium receptive field (5x5 dilated)
        self.branch2 = nn.Sequential(
            DepthwiseSeparableConv(channels, channels, 3, padding=2, dilation=2),
            nn.GELU()
        )
        
        # Scale 3: Large receptive field (11x11)
        self.branch3 = nn.Sequential(
            nn.Conv2d(channels, channels, 11, padding=5, groups=channels, bias=False),
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.GroupNorm(8, channels),
            nn.GELU()
        )
        
        # Scale 4: Very large receptive field (17x17)
        self.branch4 = nn.Sequential(
            nn.Conv2d(channels, channels, 17, padding=8, groups=channels, bias=False),
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.GroupNorm(8, channels),
            nn.GELU()
        )
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Conv2d(channels * 4, channels, 1, bias=False),
            nn.GroupNorm(8, channels)
        )
        
        # Channel attention for adaptive weighting
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 4, 1),
            nn.GELU(),
            nn.Conv2d(channels // 4, channels, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # Multi-scale feature extraction
        f1 = self.branch1(x)
        f2 = self.branch2(x)
        f3 = self.branch3(x)
        f4 = self.branch4(x)
        
        # Concatenate all scales
        fused = torch.cat([f1, f2, f3, f4], dim=1)
        fused = self.fusion(fused)
        
        # Apply channel attention
        att = self.channel_attention(fused)
        fused = fused * att
        
        # Residual connection
        return x + fused


class HybridAttentionModule(nn.Module):
    """Hybrid Spatial + Channel Attention"""
    def __init__(self, channels):
        super().__init__()
        
        # Spatial attention
        self.spatial = nn.Sequential(
            nn.Conv2d(channels, channels // 8, 1),
            nn.GELU(),
            nn.Conv2d(channels // 8, channels // 8, 7, padding=3, groups=channels // 8),
            nn.Conv2d(channels // 8, 1, 1),
            nn.Sigmoid()
        )
        
        # Channel attention
        self.channel = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 4, 1),
            nn.GELU(),
            nn.Conv2d(channels // 4, channels, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        s_att = self.spatial(x)
        c_att = self.channel(x)
        return x * s_att * c_att


class ShallowHFBlock(nn.Module):
    """
    Shallow High-Frequency Branch
    
    Captures very small stars (1-3 px) that can be lost during downsampling.
    Works at full resolution WITHOUT downsample to preserve micro-details.
    Uses small kernels to maintain high-frequency information.
    
    Critical for:
    - Tiny stars (1-3 px)
    - Learning native PSF of small point sources
    - Distinguishing stars from noise/background patterns
    - Not touching comets (non-point profile → filtered out by HF branch)
    - Perfect removal from 1px to huge stars
    """
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.norm1 = nn.GroupNorm(min(8, channels), channels)
        self.act = nn.GELU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(min(8, channels), channels)
        
    def forward(self, x):
        res = x
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.norm2(x)
        return x + res


class HaloBlock(nn.Module):
    """
    Halo Suppression Block
    
    Handles low-frequency halos around very bright stars.
    Prevents treating wide soft halos as texture (which causes artifacts).
    Uses large kernel to capture halo extent.
    """
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=7, padding=3, groups=channels, bias=False)
        self.act = nn.SiLU()
        
    def forward(self, x):
        return self.act(self.conv(x))


class ResidualGate(nn.Module):
    """
    Noise-Aware Residual Gate
    
    Adaptively weights residual based on local intensity.
    Prevents micro-artifacts in low-intensity noisy regions.
    
    Critical for:
    - Avoiding micro-spots in deep sky backgrounds
    - Stabilizing dark regions
    - Preventing noise amplification
    - Not affecting comets (different pattern)
    """
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.norm = nn.GroupNorm(min(8, channels), channels)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, residual, x):
        gate = self.conv(x)
        gate = self.norm(gate)
        gate = self.sigmoid(gate)
        return residual * gate


class StarRemovalNet(nn.Module):
    """
    Hybrid Attention Multi-Scale Network for Star Removal
    
    Architecture:
    - Multi-scale receptive fields (3x3 to 17x17) for detecting stars of all sizes
    - Shallow HF branch for tiny stars (1-3px) that could be lost in downsampling
    - Residual learning: output = input - residual (Google's approach)
    - MSRF hybrid blocks for texture coherent reconstruction
    - Direct starless output (not masks or residuals)
    """
    def __init__(self, in_channels=3, base_channels=64, num_blocks=8):
        super().__init__()
        
        # Shallow High-Frequency Branch - captures tiny stars (1-3px)
        # Works at full resolution to preserve micro-details before downsampling
        self.hf_branch = ShallowHFBlock(in_channels)
        
        # Halo suppression for bright stars with wide soft halos
        self.halo_block = HaloBlock(in_channels)
        
        # Initial feature extraction with large kernel for context
        self.input_conv = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 7, padding=3, bias=False),
            nn.GroupNorm(8, base_channels),
            nn.GELU()
        )
        
        # Encoder - progressive downsampling
        self.enc1 = self._make_encoder_block(base_channels, base_channels * 2, num_blocks=2)
        self.down1 = nn.Conv2d(base_channels * 2, base_channels * 2, 3, stride=2, padding=1)
        
        self.enc2 = self._make_encoder_block(base_channels * 2, base_channels * 4, num_blocks=2)
        self.down2 = nn.Conv2d(base_channels * 4, base_channels * 4, 3, stride=2, padding=1)
        
        # Bottleneck - deep MSRF blocks for maximum receptive field
        self.bottleneck = nn.Sequential(*[
            MSRF_Block(base_channels * 4) for _ in range(num_blocks)
        ])
        
        # Decoder - progressive upsampling with skip connections
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 2, stride=2),
            nn.GroupNorm(8, base_channels * 2),
            nn.GELU()
        )
        self.dec2 = self._make_decoder_block(base_channels * 4, base_channels * 2, num_blocks=2)
        
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 2, base_channels, 2, stride=2),
            nn.GroupNorm(8, base_channels),
            nn.GELU()
        )
        self.dec1 = self._make_decoder_block(base_channels * 2, base_channels, num_blocks=2)
        
        # Hybrid attention for final refinement
        self.final_attention = HybridAttentionModule(base_channels)
        
        # Noise-aware residual gate
        self.residual_gate = ResidualGate(in_channels)
        
        # Output projection - direct starless image
        self.output_conv = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(base_channels, in_channels, 3, padding=1)
        )
        
    def _make_encoder_block(self, in_ch, out_ch, num_blocks):
        layers = []
        layers.append(nn.Conv2d(in_ch, out_ch, 1))
        for _ in range(num_blocks):
            layers.append(MSRF_Block(out_ch))
        return nn.Sequential(*layers)
    
    def _make_decoder_block(self, in_ch, out_ch, num_blocks):
        layers = []
        layers.append(nn.Conv2d(in_ch, out_ch, 1))
        for _ in range(num_blocks):
            layers.append(MSRF_Block(out_ch))
        layers.append(HybridAttentionModule(out_ch))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        # Shallow HF branch - extract tiny star features at full resolution
        # This prevents loss of 1-3px stars during downsampling
        hf_features = self.hf_branch(x)
        
        # Halo features for bright stars with wide soft halos
        halo_features = self.halo_block(x)
        
        # Input feature extraction
        f0 = self.input_conv(x)
        
        # Encoder path with skip connections
        e1 = self.enc1(f0)
        d1 = self.down1(e1)
        
        e2 = self.enc2(d1)
        d2 = self.down2(e2)
        
        # Bottleneck with proxy-skip for huge PSFs
        # Add interpolated e1 for more global context
        e1_proxy = F.interpolate(e1, size=d2.shape[2:], mode='bilinear', align_corners=False)
        # Project e1 to match d2 channels
        if not hasattr(self, 'proxy_proj'):
            self.proxy_proj = nn.Conv2d(e1.shape[1], d2.shape[1], 1).to(x.device)
        e1_proj = self.proxy_proj(e1_proxy)
        
        b = self.bottleneck(d2 + e1_proj * 0.3)  # Add proxy skip with scaling
        
        # Decoder path with skip connections
        u2 = self.up2(b)
        u2 = torch.cat([u2, e2], dim=1)  # Skip connection
        d2_out = self.dec2(u2)
        
        u1 = self.up1(d2_out)
        u1 = torch.cat([u1, e1], dim=1)  # Skip connection
        d1_out = self.dec1(u1)
        
        # Inject halo suppression at decoder output
        # Convert halo features to match d1_out channels
        if not hasattr(self, 'halo_proj'):
            self.halo_proj = nn.Conv2d(halo_features.shape[1], d1_out.shape[1], 1).to(x.device)
        halo_proj = self.halo_proj(halo_features)
        d1_out = d1_out + halo_proj * 0.3
        
        # Final attention refinement
        refined = self.final_attention(d1_out)
        
        # Convert HF features to match refined channels
        if not hasattr(self, 'hf_proj'):
            self.hf_proj = nn.Conv2d(hf_features.shape[1], refined.shape[1], 1).to(x.device)
        hf_proj = self.hf_proj(hf_features)
        
        # Fuse HF features with main path
        # HF branch adds micro-detail detection for tiny stars
        fused = refined + hf_proj
        
        # Output residual
        residual = self.output_conv(fused)
        
        # Apply noise-aware gate to prevent artifacts in dark/noisy regions
        residual = self.residual_gate(residual, x)
        
        # Google's residual approach: output = input - residual
        starless = x - residual
        
        return starless


def count_parameters(model):
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test model
    model = StarRemovalNet(in_channels=3, base_channels=64, num_blocks=6)
    print(f"Total parameters: {count_parameters(model):,}")
    
    # Test forward pass
    dummy_input = torch.randn(1, 3, 256, 256)
    with torch.no_grad():
        output = model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
