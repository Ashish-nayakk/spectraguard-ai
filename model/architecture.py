"""
model/architecture.py
=====================
Dual-Stream Deepfake Detector

Stream 1 — Spatial (EfficientNet-B3 backbone, ImageNet PRETRAINED via timm)
    Learns visual manipulation artefacts: blending seams, skin texture
    anomalies, eye reflections, hair boundary inconsistencies.

Stream 2 — Frequency (2-D FFT magnitude spectrum)
    Detects generation artefacts invisible to the human eye:
    GAN upsampling grid noise, diffusion model spectral signatures,
    periodic patterns from decoder networks.

Both streams are fused and classified by a shared head.

✅ PRETRAINED: EfficientNet-B3 is loaded with ImageNet weights via timm.
   The first run downloads ~49 MB from HuggingFace Hub automatically.
"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional

try:
    import timm
    _TIMM_AVAILABLE = True
except ImportError:
    _TIMM_AVAILABLE = False
    import torchvision.models as tv_models

# ---------------------------------------------------------------------------

WEIGHTS_PATH = Path(__file__).parent / "weights" / "deepfake_detector.pt"

# EfficientNet-B3 output channels when used as a feature extractor
_EFF_B3_CHANNELS = 1536


# ---------------------------------------------------------------------------
# Sub-modules
# ---------------------------------------------------------------------------

class FrequencyStream(nn.Module):
    """
    Extracts spectral features from the log-magnitude FFT of the
    grayscale input.  GAN decoders and diffusion upsamplers leave
    characteristic periodic patterns in this representation.
    """

    def __init__(self, out_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            # (B, 1, H, W) → (B, 32, H, W)
            nn.Conv2d(1, 32, 3, padding=1, bias=False),
            nn.GELU(),
            # → (B, 64, H/2, W/2)
            nn.Conv2d(32, 64, 3, padding=1, stride=2, bias=False),
            nn.GELU(),
            # → (B, out_dim, H/4, W/4)
            nn.Conv2d(64, out_dim, 3, padding=1, stride=2, bias=False),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),          # → (B, out_dim, 1, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Convert RGB → grayscale
        gray = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
        # Log-magnitude FFT (shift zero-freq to centre)
        fft = torch.fft.fft2(gray, norm="ortho")
        mag = torch.log(torch.abs(fft) + 1e-8)
        mag = torch.roll(
            mag,
            shifts=(mag.shape[-2] // 2, mag.shape[-1] // 2),
            dims=(-2, -1),
        )
        return self.net(mag).flatten(1)       # (B, out_dim)


class ClassifierHead(nn.Module):
    """Fusion head that combines spatial + frequency feature vectors."""

    def __init__(self, spatial_dim: int, freq_dim: int):
        super().__init__()
        in_dim = spatial_dim + freq_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.40),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.30),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(0.20),
            nn.Linear(128, 1),
        )

    def forward(self, spatial: torch.Tensor, freq: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(torch.cat([spatial, freq], dim=1)))


# ---------------------------------------------------------------------------
# Backbone loader
# ---------------------------------------------------------------------------

def _build_spatial_backbone(pretrained: bool = True):
    """
    Build EfficientNet-B3 spatial backbone.

    Uses timm (preferred) which pulls from HuggingFace Hub.
    Falls back to torchvision if timm is not installed.

    Returns (body, pool, out_channels).
    """
    if _TIMM_AVAILABLE:
        # timm returns a model whose forward_features() gives (B, C, H, W)
        backbone = timm.create_model(
            "efficientnet_b3",
            pretrained=pretrained,
            features_only=False,
            num_classes=0,      # removes classifier head → outputs pooled (B, 1536)
            global_pool="avg",
        )
        # num_features is the channel count before the head
        out_channels = backbone.num_features   # 1536 for B3
        return backbone, None, out_channels
    else:
        backbone = tv_models.efficientnet_b3(
            weights=tv_models.EfficientNet_B3_Weights.IMAGENET1K_V1
            if pretrained else None
        )
        body = backbone.features
        pool = nn.AdaptiveAvgPool2d(1)
        return body, pool, _EFF_B3_CHANNELS


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class DeepfakeDetector(nn.Module):
    """
    Dual-stream deepfake detector with pretrained EfficientNet-B3 backbone.

    Input  : (B, 3, 224, 224) float32, ImageNet-normalised
    Output : (B, 1) float32 — probability that the sample is FAKE

    The EfficientNet-B3 spatial stream is initialised with ImageNet
    pretrained weights (downloaded automatically on first run via timm).
    """

    FREQ_DIM = 128

    def __init__(self, pretrained: bool = True):
        super().__init__()

        # --- Spatial stream: EfficientNet-B3 (pretrained) -----------
        self._use_timm, self.spatial_pool = False, None
        backbone, pool, self.SPATIAL_DIM = _build_spatial_backbone(pretrained)

        if pool is None:
            # timm path — backbone(x) already returns pooled vector
            self.spatial_body = backbone
            self._use_timm = True
        else:
            # torchvision path
            self.spatial_body = backbone
            self.spatial_pool = pool

        # --- Frequency stream ----------------------------------------
        self.freq_stream = FrequencyStream(out_dim=self.FREQ_DIM)

        # --- Classifier head -----------------------------------------
        self.head = ClassifierHead(self.SPATIAL_DIM, self.FREQ_DIM)

        # Initialise non-pretrained layers
        self._init_new_weights()

    # ------------------------------------------------------------------

    def _init_new_weights(self):
        """Apply Kaiming/Xavier init to frequency stream and classifier head."""
        for name, m in list(self.freq_stream.named_modules()) + \
                        list(self.head.named_modules()):
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._use_timm:
            spatial = self.spatial_body(x)                          # (B, 1536)
        else:
            spatial = self.spatial_pool(self.spatial_body(x)).flatten(1)

        freq = self.freq_stream(x)                                   # (B, 128)
        return self.head(spatial, freq)                              # (B, 1)

    # ------------------------------------------------------------------

    def freeze_backbone(self):
        """Phase 1: freeze EfficientNet body, train head + freq stream only."""
        for p in self.spatial_body.parameters():
            p.requires_grad = False

    def unfreeze_backbone(self, from_block: int = 5):
        """Phase 2: unfreeze top blocks for fine-tuning."""
        for p in self.spatial_body.parameters():
            p.requires_grad = False
        if self._use_timm:
            # timm EfficientNet-B3 blocks are in model.blocks (list of 7 stages)
            blocks = list(self.spatial_body.blocks)
            for block in blocks[from_block:]:
                for p in block.parameters():
                    p.requires_grad = True
        else:
            for i in range(from_block, 9):
                for p in self.spatial_body[i].parameters():
                    p.requires_grad = True

    def trainable_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def total_params(self):
        return sum(p.numel() for p in self.parameters())


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

_cached_model: Optional[DeepfakeDetector] = None


def load_model(
    weights: Optional[str] = None,
    device: str = "cpu",
    force_reload: bool = False,
) -> "DeepfakeDetector":
    """
    Load (and cache) the deepfake detector model.

    Parameters
    ----------
    weights      : Path to .pt weights file. Defaults to WEIGHTS_PATH.
    device       : 'cpu' or 'cuda'.
    force_reload : Bypass cache and reload from disk.
    """
    global _cached_model
    if _cached_model is not None and not force_reload and weights is None:
        return _cached_model

    path = Path(weights) if weights else WEIGHTS_PATH
    model = DeepfakeDetector(pretrained=True)

    if path.exists():
        state = torch.load(str(path), map_location=device, weights_only=True)
        model.load_state_dict(state, strict=True)
        size_mb = path.stat().st_size // (1024 * 1024)
        print(f"[Model] Loaded weights from '{path.name}'  ({size_mb} MB)")
    else:
        print(f"[Model] No saved weights at '{path}'.")
        print(f"[Model] Using pretrained EfficientNet-B3 backbone (ImageNet).")
        print(f"[Model] Run  python scripts/build_weights.py  to save weights.")

    model.to(device).eval()
    _cached_model = model
    return model
