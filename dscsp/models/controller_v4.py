"""
Content-adaptive steganographic controller for DSCSP v4.

Replaces the tiled-grid controller (v1/v2) with a U-Net encoder-decoder
that produces image-dependent perturbations. Key differences:

  1. Takes BOTH cover image and payload as input -> content-adaptive delta
  2. No periodic grid structure -> smooth perturbation, no tile boundary artifacts
  3. Skip connections preserve spatial detail for targeted embedding

Architecture:
  Payload (B,256) -> MLP -> (B, ch, 4, 4) -> upsample to (B, ch, 256, 256)
  Concat with cover image -> (B, ch+3, 256, 256)
  3-level U-Net encoder-decoder with skip connections
  Output: tanh(delta) * max_perturbation -> (B, 3, 256, 256)

~3.5M parameters with base_channels=48.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _group_norm(channels: int) -> nn.GroupNorm:
    groups = min(8, channels)
    while groups > 1 and channels % groups != 0:
        groups -= 1
    return nn.GroupNorm(groups, channels)


class ConvBlock(nn.Module):
    """Conv + GroupNorm + ReLU block."""
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride, padding=padding),
            _group_norm(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class DownBlock(nn.Module):
    """Downsample by 2x with two conv layers."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 4, stride=2, padding=1),
            _group_norm(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            _group_norm(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UpBlock(nn.Module):
    """Upsample by 2x with ConvTranspose + conv."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, 4, stride=2, padding=1)
        self.bn_up = _group_norm(out_ch)
        self.conv = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            _group_norm(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = F.relu(self.bn_up(self.up(x)), inplace=True)
        return self.conv(x)


class ContentAdaptiveController(nn.Module):
    """
    Content-adaptive steganographic encoder.

    Takes cover image + payload bits as input. Outputs a perturbation
    delta that is spatially adapted to the image content, concentrating
    changes in textured/edge regions and minimizing changes in smooth areas.

    Args:
        num_bits: Number of payload bits to embed (default 256)
        max_perturbation: Maximum absolute perturbation magnitude
        base_channels: Base channel count for U-Net (affects model size)
    """

    def __init__(
        self,
        num_bits: int = 256,
        max_perturbation: float = 0.1,
        base_channels: int = 48,
        carrier_type: str = "nearest",
        spread_image_size: int = 256,
        spread_seed: int = 20260424,
    ):
        super().__init__()
        self.num_bits = num_bits
        self.max_perturbation = max_perturbation
        self.carrier_type = carrier_type
        ch = base_channels

        # Message preparation: bits -> spatial feature map
        self.msg_encoder = nn.Sequential(
            nn.Linear(num_bits, ch * 16),
            nn.ReLU(inplace=True),
        )
        # Output: (B, ch, 4, 4) -> upsampled to (B, ch, 256, 256)

        # Direct payload-to-delta pathway (major BER recovery path).
        # Gives the model an explicit, high-SNR route to carry bits even when
        # content-adaptive U-Net path is still under-trained.
        self.payload_direct = nn.Linear(num_bits, 3 * 16 * 16)
        self.payload_fixed_gain = nn.Parameter(torch.tensor(0.18))
        self.payload_grid_h, self.payload_grid_w = self._infer_payload_grid(num_bits)

        # Spread-spectrum carrier (Instruction 3 — translation-tolerant).
        # Each bit gets a pseudorandom binary mask in {-1, +1}^(H*W). The
        # carrier for payload b is sum_i (2*b_i - 1) * m_i * gain, which
        # distributes each bit's energy across the entire image rather than
        # concentrating it in a 16x16 grid region.
        if carrier_type == "spread_spectrum":
            gen = torch.Generator(device="cpu").manual_seed(int(spread_seed))
            Himg, Wimg = int(spread_image_size), int(spread_image_size)
            masks = torch.randint(0, 2, (num_bits, Himg * Wimg), generator=gen, dtype=torch.int8)
            masks = masks.float() * 2.0 - 1.0  # {-1, +1}
            self.register_buffer("spread_masks", masks)  # (num_bits, H*W)
            self.spread_image_size = Himg
            # Per-channel phase rolls so the 3 RGB channels use shifted masks.
            self.spread_channel_rolls = (0, Himg // 3, (2 * Himg) // 3)
        else:
            self.spread_masks = None
            self.spread_image_size = None
            self.spread_channel_rolls = None

        # Initial convolution: image (3ch) + message (ch) -> ch
        self.initial = ConvBlock(3 + ch, ch)

        # Encoder path (3 levels of downsampling)
        self.enc1 = DownBlock(ch, ch * 2)       # 256 -> 128
        self.enc2 = DownBlock(ch * 2, ch * 4)   # 128 -> 64
        self.enc3 = DownBlock(ch * 4, ch * 4)   # 64 -> 32

        # Decoder path with skip connections
        self.dec3 = UpBlock(ch * 4, ch * 4)     # 32 -> 64
        # After concat with enc2 skip: ch*4 + ch*4 = ch*8
        self.dec2_reduce = ConvBlock(ch * 8, ch * 4)
        self.dec2 = UpBlock(ch * 4, ch * 2)     # 64 -> 128
        # After concat with enc1 skip: ch*2 + ch*2 = ch*4
        self.dec1_reduce = ConvBlock(ch * 4, ch * 2)
        self.dec1 = UpBlock(ch * 2, ch)         # 128 -> 256

        # Final output: concat with initial skip (ch + ch = ch*2) -> 3
        self.final = nn.Sequential(
            ConvBlock(ch * 2, ch),
            nn.Conv2d(ch, 3, kernel_size=1),
        )

    @staticmethod
    def _infer_payload_grid(num_bits: int):
        side = int(num_bits ** 0.5)
        for h in range(side, 0, -1):
            if num_bits % h == 0:
                return h, num_bits // h
        return 1, num_bits

    def forward(self, image: torch.Tensor, payload: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: (B, 3, 256, 256) cover image in [-1, 1]
            payload: (B, num_bits) payload bits in {0, 1}

        Returns:
            delta: (B, 3, 256, 256) perturbation in [-max_pert, +max_pert]
        """
        B = image.shape[0]
        H, W = image.shape[2], image.shape[3]

        # Prepare message as spatial feature map
        msg = self.msg_encoder(payload)             # (B, ch*16)
        ch = msg.shape[1] // 16
        msg = msg.view(B, ch, 4, 4)                 # (B, ch, 4, 4)
        msg = F.interpolate(msg, size=(H, W), mode='bilinear', align_corners=False)

        # Concatenate image + message and initial conv
        d0 = self.initial(torch.cat([image, msg], dim=1))   # (B, ch, 256, 256)

        # Encoder
        d1 = self.enc1(d0)      # (B, ch*2, 128, 128)
        d2 = self.enc2(d1)      # (B, ch*4, 64, 64)
        d3 = self.enc3(d2)      # (B, ch*4, 32, 32)

        # Decoder with skip connections
        u3 = self.dec3(d3)                                      # (B, ch*4, 64, 64)
        u3 = self.dec2_reduce(torch.cat([u3, d2], dim=1))       # (B, ch*4, 64, 64)
        u2 = self.dec2(u3)                                      # (B, ch*2, 128, 128)
        u2 = self.dec1_reduce(torch.cat([u2, d1], dim=1))       # (B, ch*2, 128, 128)
        u1 = self.dec1(u2)                                      # (B, ch, 256, 256)

        # Final with skip from initial
        delta = self.final(torch.cat([u1, d0], dim=1))  # (B, 3, 256, 256)

        # Carrier branch: either (legacy) nearest-neighbour grid or
        # (Instruction 3) spread-spectrum — each bit's energy distributed
        # across the whole image via a pseudorandom ±1 mask.
        if self.carrier_type == "spread_spectrum":
            signed = payload * 2.0 - 1.0                         # (B, num_bits)
            masks = self.spread_masks                            # (num_bits, Himg*Wimg)
            carrier_flat = signed @ masks                        # (B, Himg*Wimg)
            carrier_flat = carrier_flat / (self.num_bits ** 0.5) # normalise energy
            Himg = self.spread_image_size
            carrier = carrier_flat.view(B, 1, Himg, Himg)
            if (H, W) != (Himg, Himg):
                carrier = F.interpolate(carrier, size=(H, W), mode="bilinear",
                                        align_corners=False)
            r0, r1, r2 = self.spread_channel_rolls
            carrier_rgb = torch.cat([
                torch.roll(carrier, shifts=r0, dims=2),
                torch.roll(carrier, shifts=r1, dims=3),
                torch.roll(carrier, shifts=r2, dims=2),
            ], dim=1)
        else:
            signed = payload[:, : self.payload_grid_h * self.payload_grid_w] * 2.0 - 1.0
            carrier = signed.view(B, 1, self.payload_grid_h, self.payload_grid_w)
            carrier = F.interpolate(carrier, size=(H, W), mode='nearest')
            sh = max(1, H // max(1, 2 * self.payload_grid_h))
            sw = max(1, W // max(1, 2 * self.payload_grid_w))
            carrier_rgb = torch.cat([
                carrier,
                torch.roll(carrier, shifts=sh, dims=2),
                torch.roll(carrier, shifts=sw, dims=3),
            ], dim=1)

        direct = self.payload_direct(payload).view(B, 3, 16, 16)
        direct = F.interpolate(direct, size=(H, W), mode='bilinear', align_corners=False)

        # Blend content-adaptive and direct payload channels.
        delta = delta + self.payload_fixed_gain * carrier_rgb + 0.5 * direct

        return torch.tanh(delta) * self.max_perturbation
