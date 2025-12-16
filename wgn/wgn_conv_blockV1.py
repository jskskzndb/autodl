"""
wgn_conv_block.py

This file contains the complete implementation of the "Wavelet-Gated Recursive 
Convolution" block (Wg^nConv Block).

This novel block represents a deep fusion of two cutting-edge ideas:
1.  **SFFNet**: Feature decoupling into spatial-frequency domains using the
    Discrete Wavelet Transform (DWT).
2.  **HorNet**: High-order spatial interaction in the spatial domain using
    Recursive Gated Convolution (g^nConv).

Our Wg^nConv block performs the high-order spatial interaction *directly* on
the decoupled low-frequency and high-frequency sub-bands, enabling a more
specialized and potentially more powerful feature learning paradigm.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_wavelets import DWTForward, DWTInverse
from timm.layers import DropPath, trunc_normal_
from functools import partial

# =====================================================================================
# PART 1: CORE COMPONENTS - GnConv from HorNet
# We use the purified, standalone version of GnConv.
# =====================================================================================

def get_dwconv(dim, kernel, bias):
    """Helper function to create a depthwise convolution."""
    return nn.Conv2d(dim, dim, kernel_size=kernel, padding=(kernel - 1) // 2, bias=bias, groups=dim)

class GnConv(nn.Module):
    """
    The core of HorNet: g^nConv (Recursive Gated Convolution).
    """
    def __init__(self, dim, order=5, s=1.0):
        super().__init__()
        self.order = order
        self.dims = [dim // 2 ** i for i in range(order)]
        self.dims.reverse()
        self.proj_in = nn.Conv2d(dim, 2 * dim, 1)
        self.dwconv = get_dwconv(sum(self.dims), 7, True)
        self.proj_out = nn.Conv2d(dim, dim, 1)
        self.pws = nn.ModuleList(
            [nn.Conv2d(self.dims[i], self.dims[i + 1], 1) for i in range(order - 1)]
        )
        self.scale = s

    def forward(self, x):
        fused_x = self.proj_in(x)
        pwa, abc = torch.split(fused_x, (self.dims[0], sum(self.dims)), dim=1)
        dw_abc = self.dwconv(abc) * self.scale
        dw_list = torch.split(dw_abc, self.dims, dim=1)
        x = pwa * dw_list[0]
        for i in range(self.order - 1):
            x = self.pws[i](x) * dw_list[i + 1]
        x = self.proj_out(x)
        return x

# =====================================================================================
# PART 2: THE FINAL ASSEMBLY - Wg^nConv Block
# =====================================================================================
class Wg_nConv_Block(nn.Module):
    """
    Wavelet-Gated Recursive Convolution Block.
    
    This block performs g^nConv interactions in the wavelet domain.
    """
    def __init__(self, channels, order_low=4, order_high=3):
        super().__init__()
        
        # 1. --- Wavelet Transforms ---
        self.dwt = DWTForward(J=1, wave='haar', mode='zero')
        self.idwt = DWTInverse(wave='haar', mode='zero')
        
        # 2. --- Parallel High-Order Interaction Experts ---
        
        # Expert for Low-Frequency (LL) sub-band
        # This expert learns the propagation of global, structural information.
        self.gnconv_low_freq = GnConv(dim=channels, order=order_low)
        
        # Expert for High-Frequency (LH, HL, HH) sub-bands
        # This expert learns the propagation of local details and edges.
        # It first needs to project the 3*C channels down to C.
        self.high_freq_proj_in = nn.Conv2d(channels * 3, channels, 1)
        self.gnconv_high_freq = GnConv(dim=channels, order=order_high)
        # And project back to 3*C channels for reconstruction.
        self.high_freq_proj_out = nn.Conv2d(channels, channels * 3, 1)
        
    def forward(self, x):
        identity = x
        b, c, h, w = x.shape

        # 1. --- Decompose with DWT ---
        # ll shape: [B, C, H/2, W/2]
        # high_freq shape: [B, C, 3, H/2, W/2]
        ll, high_freq_list = self.dwt(x)
        high_freq = high_freq_list[0].view(b, c * 3, h // 2, w // 2)

        # 2. --- Parallel High-Order Interaction in Wavelet Domain ---
        
        # Process low-frequency (global context) path
        ll_enhanced = self.gnconv_low_freq(ll)
        
        # Process high-frequency (spatial details) path
        high_freq_projected = self.high_freq_proj_in(high_freq)
        high_freq_enhanced = self.gnconv_high_freq(high_freq_projected)
        high_freq_reconstructed = self.high_freq_proj_out(high_freq_enhanced)
        
        # 3. --- Reconstruct with IDWT ---
        
        # Reshape the high-frequency tensor back to the list format for IDWT
        high_freq_out_list = [high_freq_reconstructed.view(b, c, 3, h // 2, w // 2)]
        
        y = self.idwt((ll_enhanced, high_freq_out_list))
        
        # 4. --- Final Residual Connection ---
        return identity + y

# =====================================================================================
# Test block to verify the model
# =====================================================================================
if __name__ == '__main__':
    print("Testing the novel Wg^nConv Block...")
    
    # Create a dummy input tensor
    # Batch=4, Channels=64, Height=32, Width=32
    input_tensor = torch.randn(4, 64, 32, 32)

    # Instantiate our new block with different orders for low and high freqs
    wgn_block = Wg_nConv_Block(channels=64, order_low=5, order_high=3)
    
    # Perform a forward pass
    output_tensor = wgn_block(input_tensor)

    print(f"Input shape:  {input_tensor.shape}")
    print(f"Output shape: {output_tensor.shape}")
    assert input_tensor.shape == output_tensor.shape, "Shape mismatch!"
    
    num_params = sum(p.numel() for p in wgn_block.parameters() if p.requires_grad)
    print(f"Wg^nConv Block (64 channels) Parameters: {num_params / 1e6:.3f} M")

    print("\n✅ Wg^nConv Block is assembled and works correctly!")