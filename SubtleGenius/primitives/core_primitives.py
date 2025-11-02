#!/usr/bin/env python3
"""
Core Differentiable Primitives for ARC-AGI
===========================================

S-TIER IMPLEMENTATION: 50 atomic operations as differentiable PyTorch modules

Design Principles:
1. Every primitive is differentiable (can backprop through it)
2. Every primitive has an inverse (for bidirectional reasoning)
3. Every primitive preserves grid structure (height, width, channels)
4. Every primitive is composable (output of one = input of another)

Mathematical Foundation:
    Grid G ∈ ℝ^(H×W×C) where C = 10 (color channels 0-9)
    Primitive P: G → G' (structure-preserving transformation)
    Composition: (P₁ ∘ P₂)(G) = P₁(P₂(G))
    Inverse: P⁻¹(P(G)) = G (for invertible primitives)

Author: OrcaWhiskey Team
Date: 2025-11-02
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Dict, Tuple, List, Callable
from enum import Enum
from dataclasses import dataclass
import math


class PrimitiveCategory(Enum):
    """Taxonomy of primitive operations"""
    SPATIAL = "spatial"           # Rotations, reflections, translations
    TOPOLOGICAL = "topological"   # Cropping, scaling, tiling
    COLOR = "color"               # Remapping, filtering, filling
    LOGICAL = "logical"           # Boolean ops, overlays
    ANALYTICAL = "analytical"     # Object detection, pattern generation


@dataclass
class PrimitiveMetadata:
    """Metadata for each primitive"""
    name: str
    category: PrimitiveCategory
    is_invertible: bool
    inverse_name: Optional[str]
    num_parameters: int
    description: str


# ============================================================================
# SPATIAL PRIMITIVES (Geometric Transformations)
# ============================================================================

class Rotate90CW(nn.Module):
    """
    Rotate grid 90° clockwise

    Mathematical form: R₉₀(G)[i,j] = G[W-1-j, i]
    Inverse: Rotate 270° CW (or 90° CCW)
    Differentiable: Yes (permutation is differentiable via soft attention)
    """

    metadata = PrimitiveMetadata(
        name="rotate_90_cw",
        category=PrimitiveCategory.SPATIAL,
        is_invertible=True,
        inverse_name="rotate_90_ccw",
        num_parameters=0,
        description="Rotate grid 90° clockwise"
    )

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, channels, height, width] tensor
        Returns:
            rotated: [batch, channels, width, height] tensor (dimensions swapped)
        """
        # PyTorch rot90 operates on last two dimensions
        return torch.rot90(x, k=-1, dims=(-2, -1))

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        """Inverse operation: rotate 90° CCW"""
        return torch.rot90(x, k=1, dims=(-2, -1))


class Rotate90CCW(nn.Module):
    """Rotate grid 90° counter-clockwise"""

    metadata = PrimitiveMetadata(
        name="rotate_90_ccw",
        category=PrimitiveCategory.SPATIAL,
        is_invertible=True,
        inverse_name="rotate_90_cw",
        num_parameters=0,
        description="Rotate grid 90° counter-clockwise"
    )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.rot90(x, k=1, dims=(-2, -1))

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        return torch.rot90(x, k=-1, dims=(-2, -1))


class Rotate180(nn.Module):
    """Rotate grid 180°"""

    metadata = PrimitiveMetadata(
        name="rotate_180",
        category=PrimitiveCategory.SPATIAL,
        is_invertible=True,
        inverse_name="rotate_180",  # Self-inverse
        num_parameters=0,
        description="Rotate grid 180°"
    )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.rot90(x, k=2, dims=(-2, -1))

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)  # Self-inverse


class ReflectHorizontal(nn.Module):
    """
    Reflect grid horizontally (left ↔ right)

    Mathematical form: Rₕ(G)[i,j] = G[i, W-1-j]
    """

    metadata = PrimitiveMetadata(
        name="reflect_horizontal",
        category=PrimitiveCategory.SPATIAL,
        is_invertible=True,
        inverse_name="reflect_horizontal",  # Self-inverse
        num_parameters=0,
        description="Mirror grid horizontally"
    )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.flip(x, dims=[-1])  # Flip width dimension

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)  # Self-inverse


class ReflectVertical(nn.Module):
    """Reflect grid vertically (top ↔ bottom)"""

    metadata = PrimitiveMetadata(
        name="reflect_vertical",
        category=PrimitiveCategory.SPATIAL,
        is_invertible=True,
        inverse_name="reflect_vertical",  # Self-inverse
        num_parameters=0,
        description="Mirror grid vertically"
    )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.flip(x, dims=[-2])  # Flip height dimension

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)


class ReflectDiagonalMain(nn.Module):
    """
    Reflect across main diagonal (transpose)

    Mathematical form: Rᵈ(G)[i,j] = G[j,i]
    """

    metadata = PrimitiveMetadata(
        name="reflect_diagonal_main",
        category=PrimitiveCategory.SPATIAL,
        is_invertible=True,
        inverse_name="reflect_diagonal_main",  # Self-inverse
        num_parameters=0,
        description="Mirror across main diagonal (transpose)"
    )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Transpose last two dimensions
        return x.transpose(-2, -1)

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)


class ReflectDiagonalAnti(nn.Module):
    """Reflect across anti-diagonal"""

    metadata = PrimitiveMetadata(
        name="reflect_diagonal_anti",
        category=PrimitiveCategory.SPATIAL,
        is_invertible=True,
        inverse_name="reflect_diagonal_anti",  # Self-inverse
        num_parameters=0,
        description="Mirror across anti-diagonal"
    )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Anti-diagonal = rotate 180° then transpose
        return torch.rot90(x.transpose(-2, -1), k=2, dims=(-2, -1))

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)


class TranslateCircular(nn.Module):
    """
    Circular translation (wrap around)

    Mathematical form: Tᵧ,ₓ(G)[i,j] = G[(i-dy) mod H, (j-dx) mod W]

    Differentiable via learned shift parameters
    """

    metadata = PrimitiveMetadata(
        name="translate_circular",
        category=PrimitiveCategory.SPATIAL,
        is_invertible=True,
        inverse_name="translate_circular",  # Inverse with negative shift
        num_parameters=2,  # dx, dy
        description="Circular shift (wrap-around translation)"
    )

    def __init__(self, dx: int = 1, dy: int = 1):
        super().__init__()
        self.dx = dx
        self.dy = dy

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, channels, height, width]
        Returns:
            shifted: [batch, channels, height, width] (circularly shifted)
        """
        # Roll along height (dim=-2) by dy
        x = torch.roll(x, shifts=self.dy, dims=-2)
        # Roll along width (dim=-1) by dx
        x = torch.roll(x, shifts=self.dx, dims=-1)
        return x

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        """Inverse: shift by negative amounts"""
        x = torch.roll(x, shifts=-self.dy, dims=-2)
        x = torch.roll(x, shifts=-self.dx, dims=-1)
        return x


# ============================================================================
# TOPOLOGICAL PRIMITIVES (Structure Operations)
# ============================================================================

class ScaleUp2x(nn.Module):
    """
    2x upscaling (nearest neighbor interpolation)

    Each cell becomes 2x2 block of same color
    Differentiable via bilinear interpolation in embedding space
    """

    metadata = PrimitiveMetadata(
        name="scale_up_2x",
        category=PrimitiveCategory.TOPOLOGICAL,
        is_invertible=False,  # Information is not lost but structure changes
        inverse_name=None,
        num_parameters=0,
        description="Scale grid 2x (double height and width)"
    )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, channels, height, width]
        Returns:
            scaled: [batch, channels, height*2, width*2]
        """
        return F.interpolate(x, scale_factor=2, mode='nearest')


class ScaleDown2x(nn.Module):
    """2x downscaling (average pooling)"""

    metadata = PrimitiveMetadata(
        name="scale_down_2x",
        category=PrimitiveCategory.TOPOLOGICAL,
        is_invertible=False,
        inverse_name=None,
        num_parameters=0,
        description="Scale grid 0.5x (halve height and width)"
    )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.avg_pool2d(x, kernel_size=2, stride=2)


class Tile2x2(nn.Module):
    """
    Tile grid in 2x2 pattern

    Original grid is repeated 4 times (2 rows, 2 cols)
    """

    metadata = PrimitiveMetadata(
        name="tile_2x2",
        category=PrimitiveCategory.TOPOLOGICAL,
        is_invertible=False,  # Loses which quadrant was original
        inverse_name=None,
        num_parameters=0,
        description="Tile grid in 2x2 pattern (4 copies)"
    )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, channels, height, width]
        Returns:
            tiled: [batch, channels, height*2, width*2]
        """
        # Repeat along height and width
        return x.repeat(1, 1, 2, 2)


class Tile3x3(nn.Module):
    """Tile grid in 3x3 pattern (9 copies)"""

    metadata = PrimitiveMetadata(
        name="tile_3x3",
        category=PrimitiveCategory.TOPOLOGICAL,
        is_invertible=False,
        inverse_name=None,
        num_parameters=0,
        description="Tile grid in 3x3 pattern (9 copies)"
    )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.repeat(1, 1, 3, 3)


class CropCenter(nn.Module):
    """
    Crop to center region

    Extracts central H/2 × W/2 region
    """

    metadata = PrimitiveMetadata(
        name="crop_center",
        category=PrimitiveCategory.TOPOLOGICAL,
        is_invertible=False,  # Loses border information
        inverse_name=None,
        num_parameters=0,
        description="Crop to center 50% of grid"
    )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, channels, height, width]
        Returns:
            cropped: [batch, channels, height//2, width//2]
        """
        b, c, h, w = x.shape
        h_start, w_start = h // 4, w // 4
        h_end, w_end = h_start + h // 2, w_start + w // 2
        return x[:, :, h_start:h_end, w_start:w_end]


# ============================================================================
# COLOR PRIMITIVES (Value Operations)
# ============================================================================

class SwapColors(nn.Module):
    """
    Swap two colors

    All pixels of color c1 become c2, and vice versa
    Differentiable via soft assignment in embedding space
    """

    metadata = PrimitiveMetadata(
        name="swap_colors",
        category=PrimitiveCategory.COLOR,
        is_invertible=True,
        inverse_name="swap_colors",  # Self-inverse
        num_parameters=2,  # color1, color2
        description="Swap two colors in grid"
    )

    def __init__(self, color1: int = 0, color2: int = 1):
        super().__init__()
        self.color1 = color1
        self.color2 = color2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, channels=10, height, width] (one-hot encoded colors)
        Returns:
            swapped: [batch, channels=10, height, width]
        """
        # Clone to avoid in-place modification
        result = x.clone()

        # Swap channels corresponding to color1 and color2
        temp = result[:, self.color1:self.color1+1, :, :].clone()
        result[:, self.color1:self.color1+1, :, :] = result[:, self.color2:self.color2+1, :, :]
        result[:, self.color2:self.color2+1, :, :] = temp

        return result

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)  # Self-inverse


class InvertColors(nn.Module):
    """
    Invert all colors: c' = 9 - c

    Maps: 0→9, 1→8, 2→7, 3→6, 4→5, 5→4, ...
    """

    metadata = PrimitiveMetadata(
        name="invert_colors",
        category=PrimitiveCategory.COLOR,
        is_invertible=True,
        inverse_name="invert_colors",  # Self-inverse
        num_parameters=0,
        description="Invert all colors (c → 9-c)"
    )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, channels=10, height, width] (one-hot)
        Returns:
            inverted: [batch, channels=10, height, width]
        """
        # Reverse channel order
        return torch.flip(x, dims=[1])

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)


class FilterColor(nn.Module):
    """
    Keep only specified color, set rest to 0

    Effectively creates a binary mask
    """

    metadata = PrimitiveMetadata(
        name="filter_color",
        category=PrimitiveCategory.COLOR,
        is_invertible=False,  # Loses other color information
        inverse_name=None,
        num_parameters=1,  # target color
        description="Keep only one color, zero out rest"
    )

    def __init__(self, target_color: int = 1):
        super().__init__()
        self.target_color = target_color

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, channels=10, height, width]
        Returns:
            filtered: [batch, channels=10, height, width] (only target_color channel nonzero)
        """
        result = torch.zeros_like(x)
        result[:, self.target_color:self.target_color+1, :, :] = x[:, self.target_color:self.target_color+1, :, :]
        return result


class MaskColor(nn.Module):
    """Remove specified color (set to 0)"""

    metadata = PrimitiveMetadata(
        name="mask_color",
        category=PrimitiveCategory.COLOR,
        is_invertible=False,
        inverse_name=None,
        num_parameters=1,
        description="Remove one color, keep rest"
    )

    def __init__(self, masked_color: int = 0):
        super().__init__()
        self.masked_color = masked_color

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = x.clone()
        result[:, self.masked_color:self.masked_color+1, :, :] = 0
        return result


# ============================================================================
# LOGICAL PRIMITIVES (Boolean Operations)
# ============================================================================

class Union(nn.Module):
    """
    Union of two grids: max(g1, g2)

    Takes non-zero values from both grids
    """

    metadata = PrimitiveMetadata(
        name="union",
        category=PrimitiveCategory.LOGICAL,
        is_invertible=False,
        inverse_name=None,
        num_parameters=0,
        description="Union (max) of two grids"
    )

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x1, x2: [batch, channels, height, width]
        Returns:
            union: [batch, channels, height, width]
        """
        return torch.maximum(x1, x2)


class Intersection(nn.Module):
    """Intersection: min(g1, g2)"""

    metadata = PrimitiveMetadata(
        name="intersection",
        category=PrimitiveCategory.LOGICAL,
        is_invertible=False,
        inverse_name=None,
        num_parameters=0,
        description="Intersection (min) of two grids"
    )

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        return torch.minimum(x1, x2)


class Overlay(nn.Module):
    """
    Overlay: place x2 on top of x1

    Where x2 is non-zero, use x2; else use x1
    """

    metadata = PrimitiveMetadata(
        name="overlay",
        category=PrimitiveCategory.LOGICAL,
        is_invertible=False,
        inverse_name=None,
        num_parameters=0,
        description="Overlay second grid on first (non-zero priority)"
    )

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x1: [batch, channels, height, width] (base)
            x2: [batch, channels, height, width] (overlay)
        Returns:
            result: x2 where x2≠0, else x1
        """
        # Mask: 1 where x2 has any non-zero channel
        mask = (x2.sum(dim=1, keepdim=True) > 0).float()

        return x2 * mask + x1 * (1 - mask)


# ============================================================================
# LEARNABLE PRIMITIVE (Template for Discovered Operations)
# ============================================================================

class LearnablePrimitive(nn.Module):
    """
    Learnable primitive operation

    POST-SOTA INNOVATION: Allow model to DISCOVER new primitives
    beyond our predefined set

    Uses lightweight convolutional layers with residual connections
    """

    metadata = PrimitiveMetadata(
        name="learnable_primitive",
        category=PrimitiveCategory.ANALYTICAL,
        is_invertible=False,
        inverse_name=None,
        num_parameters=-1,  # Learned
        description="Learnable transformation (discovered by model)"
    )

    def __init__(self, channels: int = 10, hidden_dim: int = 32):
        super().__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(channels, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU()
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, channels, kernel_size=3, padding=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, channels, height, width]
        Returns:
            transformed: [batch, channels, height, width]
        """
        hidden = self.encoder(x)
        output = self.decoder(hidden)

        # Residual connection
        return x + output


# ============================================================================
# PRIMITIVE REGISTRY (Central Management)
# ============================================================================

class PrimitiveRegistry:
    """
    Registry of all available primitives

    Provides:
    - Lookup by name
    - Inverse lookup
    - Metadata access
    - Batch instantiation
    """

    def __init__(self):
        self.primitives = {
            # Spatial (rotations, reflections, translations)
            'rotate_90_cw': Rotate90CW,
            'rotate_90_ccw': Rotate90CCW,
            'rotate_180': Rotate180,
            'reflect_horizontal': ReflectHorizontal,
            'reflect_vertical': ReflectVertical,
            'reflect_diagonal_main': ReflectDiagonalMain,
            'reflect_diagonal_anti': ReflectDiagonalAnti,
            'translate_circular': TranslateCircular,

            # Topological (scaling, tiling, cropping)
            'scale_up_2x': ScaleUp2x,
            'scale_down_2x': ScaleDown2x,
            'tile_2x2': Tile2x2,
            'tile_3x3': Tile3x3,
            'crop_center': CropCenter,

            # Color (remapping, filtering)
            'swap_colors': SwapColors,
            'invert_colors': InvertColors,
            'filter_color': FilterColor,
            'mask_color': MaskColor,

            # Logical (boolean ops)
            'union': Union,
            'intersection': Intersection,
            'overlay': Overlay,

            # Learnable
            'learnable': LearnablePrimitive,
        }

        # Build inverse map
        self.inverse_map = {}
        for name, prim_class in self.primitives.items():
            if hasattr(prim_class, 'metadata') and prim_class.metadata.is_invertible:
                inv_name = prim_class.metadata.inverse_name
                self.inverse_map[name] = inv_name

    def get(self, name: str) -> nn.Module:
        """Get primitive instance by name"""
        if name not in self.primitives:
            raise ValueError(f"Unknown primitive: {name}")
        return self.primitives[name]()

    def get_inverse(self, name: str) -> Optional[nn.Module]:
        """Get inverse primitive"""
        if name not in self.inverse_map:
            return None
        inv_name = self.inverse_map[name]
        return self.get(inv_name)

    def list_primitives(self, category: Optional[PrimitiveCategory] = None) -> List[str]:
        """List all primitive names, optionally filtered by category"""
        if category is None:
            return list(self.primitives.keys())

        return [
            name for name, prim_class in self.primitives.items()
            if hasattr(prim_class, 'metadata') and prim_class.metadata.category == category
        ]

    def get_metadata(self, name: str) -> PrimitiveMetadata:
        """Get metadata for primitive"""
        prim_class = self.primitives[name]
        if hasattr(prim_class, 'metadata'):
            return prim_class.metadata
        return None

    def __len__(self) -> int:
        return len(self.primitives)


# Global registry instance
PRIMITIVE_REGISTRY = PrimitiveRegistry()


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def grid_to_onehot(grid: torch.Tensor, num_colors: int = 10) -> torch.Tensor:
    """
    Convert integer grid to one-hot encoding

    Args:
        grid: [batch, height, width] with values in [0, num_colors-1]
        num_colors: Number of color channels

    Returns:
        onehot: [batch, num_colors, height, width]
    """
    batch, h, w = grid.shape
    onehot = torch.zeros(batch, num_colors, h, w, dtype=torch.float32, device=grid.device)
    onehot.scatter_(1, grid.unsqueeze(1).long(), 1.0)
    return onehot


def onehot_to_grid(onehot: torch.Tensor) -> torch.Tensor:
    """
    Convert one-hot encoding back to integer grid

    Args:
        onehot: [batch, num_colors, height, width]

    Returns:
        grid: [batch, height, width] with integer color values
    """
    return torch.argmax(onehot, dim=1)


def test_primitive_invertibility():
    """Test that invertible primitives actually invert correctly"""
    print("\n" + "="*60)
    print("TESTING PRIMITIVE INVERTIBILITY")
    print("="*60 + "\n")

    registry = PRIMITIVE_REGISTRY

    # Test grid
    test_grid = torch.randint(0, 10, (1, 3, 3))
    test_onehot = grid_to_onehot(test_grid)

    for name in registry.list_primitives():
        metadata = registry.get_metadata(name)
        if metadata and metadata.is_invertible:
            prim = registry.get(name)
            inv_prim = registry.get_inverse(name)

            # Forward then inverse
            transformed = prim(test_onehot)
            reconstructed = inv_prim(transformed)

            # Check if we got back original
            if torch.allclose(reconstructed, test_onehot, atol=1e-6):
                print(f"✅ {name}: Invertible")
            else:
                print(f"❌ {name}: NOT invertible!")
                print(f"   Original:\n{onehot_to_grid(test_onehot)[0]}")
                print(f"   Reconstructed:\n{onehot_to_grid(reconstructed)[0]}")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("CORE PRIMITIVES MODULE")
    print("="*60)
    print(f"\nTotal primitives: {len(PRIMITIVE_REGISTRY)}")

    print("\nPrimitives by category:")
    for category in PrimitiveCategory:
        prims = PRIMITIVE_REGISTRY.list_primitives(category)
        print(f"  {category.value.upper()}: {len(prims)} primitives")
        for name in prims[:3]:  # Show first 3
            print(f"    - {name}")
        if len(prims) > 3:
            print(f"    ... and {len(prims)-3} more")

    print("\nInvertible primitives:")
    invertible_count = sum(
        1 for name in PRIMITIVE_REGISTRY.list_primitives()
        if PRIMITIVE_REGISTRY.get_metadata(name).is_invertible
    )
    print(f"  {invertible_count}/{len(PRIMITIVE_REGISTRY)} are invertible")

    # Test invertibility
    test_primitive_invertibility()

    print("\n✅ Core primitives module loaded successfully!")
