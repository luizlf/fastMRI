"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import torch
from torch import nn
from torch.nn import functional as F


class Unet(nn.Module):
    """
    PyTorch implementation of a U-Net model.

    O. Ronneberger, P. Fischer, and Thomas Brox. U-net: Convolutional networks
    for biomedical image segmentation. In International Conference on Medical
    image computing and computer-assisted intervention, pages 234–241.
    Springer, 2015.
    """

    def __init__(
        self,
        in_chans: int,
        out_chans: int,
        chans: int = 32,
        num_pool_layers: int = 4,
        drop_prob: float = 0.0,
        roi_weight: float = 0.1,
        attn_layer: bool = False,
    ):
        """
        Args:
            in_chans: Number of channels in the input to the U-Net model.
            out_chans: Number of channels in the output to the U-Net model.
            chans: Number of output channels of the first convolution layer.
            num_pool_layers: Number of down-sampling and up-sampling layers.
            drop_prob: Dropout probability.
        """
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.chans = chans
        self.num_pool_layers = num_pool_layers
        self.drop_prob = drop_prob
        self.roi_weight = roi_weight
        self.attn_layer = attn_layer

        # Not sure it will work. Adding CBAM to the model
        self.down_sample_layers = nn.ModuleList([ConvBlock(in_chans, chans, drop_prob)])
        if self.attn_layer:
            self.cbam_down = nn.ModuleList([CBAM(chans)])

        ch = chans
        for _ in range(num_pool_layers - 1):
            self.down_sample_layers.append(ConvBlock(ch, ch * 2, drop_prob))
            if self.attn_layer:
                self.cbam_down.append(CBAM(ch * 2))
            ch *= 2
        
        # Bottleneck
        self.conv = ConvBlock(ch, ch * 2, drop_prob)
        if self.attn_layer:
            self.cbam_bottleneck = CBAM(ch * 2)

        self.up_conv = nn.ModuleList()
        self.up_transpose_conv = nn.ModuleList()
        self.attention_gates = nn.ModuleList()
        
        if self.attn_layer:
            self.cbam_up = nn.ModuleList()

        for i in range(num_pool_layers):
            
            # Add attention gate for each level
            # F_g is the number of channels in the gating signal (decoder)
            # F_l is the number of channels in the skip connection (encoder)
            # F_int is the intermediate channel dimension (reduced for efficiency)
            if i == 0:
                # First level upsampling from the bottleneck
                self.attention_gates.append(
                    AttentionGate(
                        F_g=ch * 2,  # Bottleneck features
                        F_l=ch,      # Last encoder level features
                        F_int=ch // 2 # Reduced dimension
                    )
                )
            else:
                # Subsequent levels
                self.attention_gates.append(
                    AttentionGate(
                        F_g=ch,        # Previous decoder level features
                        F_l=ch // 2,   # Corresponding encoder level features
                        F_int=ch // 4  # Reduced dimension
                    )
                )
            
            self.up_transpose_conv.append(TransposeConvBlock(ch * 2, ch))
            self.up_conv.append(ConvBlock(ch * 2, ch, drop_prob) if i < num_pool_layers - 1 
                                else nn.Sequential(
                                    ConvBlock(ch * 2, ch, drop_prob),
                                    nn.Conv2d(ch, self.out_chans, kernel_size=1, stride=1),
                                ))
            if self.attn_layer:
                self.cbam_up.append(CBAM(ch * 2))
            ch //= 2

        self.up_transpose_conv.append(TransposeConvBlock(ch * 2, ch))
        self.up_conv.append(
            nn.Sequential(
                ConvBlock(ch * 2, ch, drop_prob),
                nn.Conv2d(ch, self.out_chans, kernel_size=1, stride=1),
            )
        )
        if self.attn_layer:
            self.cbam_up.append(CBAM(ch * 2))

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: Input 4D tensor of shape `(N, in_chans, H, W)`.

        Returns:
            Output tensor of shape `(N, out_chans, H, W)`.
        """
        encoder_outputs = []
        output = image

        # apply down-sampling layers
        if self.attn_layer:
            for layer, cbam in zip(self.down_sample_layers, self.cbam_down):
                output = layer(output)
                output = cbam(output)
                encoder_outputs.append(output)
                output = F.avg_pool2d(output, kernel_size=2, stride=2, padding=0)
        else:
            for layer in self.down_sample_layers:
                output = layer(output)
                encoder_outputs.append(output)
                output = F.avg_pool2d(output, kernel_size=2, stride=2, padding=0)

        output = self.conv(output)
        if self.attn_layer:
            output = self.cbam_bottleneck(output)
            
        # Decoder path with attention gates
        for i, (transpose_conv, conv, attention_gate) in enumerate(
            zip(self.up_transpose_conv, self.up_conv, self.attention_gates)
        ):
            # Get corresponding encoder output for skip connection
            encoder_output = encoder_outputs[-(i + 1)]
            
            # Upsample decoder features
            output = transpose_conv(output)
            
            # Handle odd dimensions with padding if needed
            padding = [0, 0, 0, 0]
            if output.shape[-1] != encoder_output.shape[-1]:
                padding[1] = 1
            if output.shape[-2] != encoder_output.shape[-2]:
                padding[3] = 1
            if torch.sum(torch.tensor(padding)) != 0:
                output = F.pad(output, padding, "reflect")
            
            # Apply attention mechanism
            gated_encoder_output = attention_gate(output, encoder_output)
            
            # Concatenate with attention-gated skip connection
            output = torch.cat([output, gated_encoder_output], dim=1)
            
            # Apply CBAM if used
            if self.attn_layer and i < len(self.cbam_up):
                output = self.cbam_up[i](output)
                
            # Apply convolution
            output = conv(output)

        return output


class ConvBlock(nn.Module):
    """
    A Convolutional Block that consists of two convolution layers each followed by
    instance normalization, LeakyReLU activation and dropout.
    """

    def __init__(self, in_chans: int, out_chans: int, drop_prob: float):
        """
        Args:
            in_chans: Number of channels in the input.
            out_chans: Number of channels in the output.
            drop_prob: Dropout probability.
        """
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.drop_prob = drop_prob

        self.layers = nn.Sequential(
            nn.Conv2d(in_chans, out_chans, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_chans),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout2d(drop_prob),
            nn.Conv2d(out_chans, out_chans, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_chans),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout2d(drop_prob),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: Input 4D tensor of shape `(N, in_chans, H, W)`.

        Returns:
            Output tensor of shape `(N, out_chans, H, W)`.
        """
        return self.layers(image)


class TransposeConvBlock(nn.Module):
    """
    A Transpose Convolutional Block that consists of one convolution transpose
    layers followed by instance normalization and LeakyReLU activation.
    """

    def __init__(self, in_chans: int, out_chans: int):
        """
        Args:
            in_chans: Number of channels in the input.
            out_chans: Number of channels in the output.
        """
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans

        self.layers = nn.Sequential(
            nn.ConvTranspose2d(
                in_chans, out_chans, kernel_size=2, stride=2, bias=False
            ),
            nn.InstanceNorm2d(out_chans),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: Input 4D tensor of shape `(N, in_chans, H, W)`.

        Returns:
            Output tensor of shape `(N, out_chans, H*2, W*2)`.
        """
        return self.layers(image)
    


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        padding = kernel_size // 2
        # self.roi_weight = roi_weight
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        attention_map = self.sigmoid(x)

        # if roi is not None:
        #     for r in roi:
        #         x, y, w, h = r
        #         roi_attention = torch.ones_like(attention_map)
        #         roi_attention[:, :, y:y+h, x:x+w] *= self.roi_weight
        #         attention_map = attention_map * roi_attention

        return attention_map


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)
    

class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x
    
"""
Attention Gate Module for U-Net
This module implements attention gates for skip connections in the U-Net architecture.
"""

class AttentionGate(nn.Module):
    """
    Attention Gate module for focusing on relevant features in skip connections.
    
    Implements the attention mechanism described in the Attention U-Net paper:
    https://arxiv.org/abs/1804.03999
    """
    def __init__(self, F_g, F_l, F_int):
        """
        Initialize the attention gate module.
        
        Args:
            F_g: Number of feature channels in the gating signal (from the decoder)
            F_l: Number of feature channels in the input feature map (from the encoder)
            F_int: Number of intermediate channels for dimension reduction
        """
        super(AttentionGate, self).__init__()
        
        # Gating path (decoder features)
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        # Skip connection path (encoder features)
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        # Attention coefficient computation
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, g, x):
        """
        Forward pass of the attention gate.
        
        Args:
            g: Gating signal (decoder features)
            x: Skip connection input (encoder features)
            
        Returns:
            Attended feature map with same dimensions as x
        """
        # Apply 1x1 convolutions for dimension reduction
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        
        # Element-wise sum followed by ReLU
        if g1.shape[2:] != x1.shape[2:]:
            # Upsample gating signal to match spatial dimensions of skip connection
            g1 = F.interpolate(g1, size=x1.shape[2:], mode='bilinear', align_corners=False)
        
        psi = self.relu(g1 + x1)
        
        # Compute attention map
        psi = self.psi(psi)
        
        # Element-wise multiplication of input feature map and attention map
        return x * psi