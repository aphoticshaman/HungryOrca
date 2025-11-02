#!/usr/bin/env python3
"""
Neural Primitive Composition Network
=====================================

S-TIER POST-SOTA ARCHITECTURE: Differentiable program synthesis for ARC-AGI

Core Innovation: Learn to compose primitives through:
1. Program execution (forward reasoning)
2. Program inference (inverse reasoning)
3. Beam search + reinforcement learning
4. Interpretable program traces

Mathematical Framework:
    Program P = [p₁, p₂, ..., pₙ] where pᵢ ∈ PRIMITIVES
    Execution: exec(G, P) = pₙ(...p₂(p₁(G)))
    Inference: infer(G_in, G_out) = argmax_P P(exec(G_in, P) = G_out)

Architecture:
    Grid → Encoder → Latent → Program Controller → Primitive Execution → Decoder → Grid

Author: OrcaWhiskey Team
Date: 2025-11-02
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
from dataclasses import dataclass
from collections import defaultdict

from core_primitives import (
    PRIMITIVE_REGISTRY,
    grid_to_onehot,
    onehot_to_grid,
    PrimitiveCategory
)


@dataclass
class ProgramExecutionTrace:
    """Complete trace of program execution"""
    program: List[int]                    # Primitive indices
    intermediate_grids: List[torch.Tensor]  # Grid after each step
    primitive_names: List[str]             # Human-readable names
    confidence: float                      # Execution confidence
    num_steps: int                         # Program length


# ============================================================================
# GRID ENCODER: Raw Grid → Learned Embedding
# ============================================================================

class GridEncoder(nn.Module):
    """
    Encode raw integer grid to learned latent representation

    Architecture:
    - One-hot encoding (10 color channels)
    - Convolutional feature extraction
    - Positional encoding
    - Output: dense feature map

    Design: Similar to ViT but for grids, not images
    """

    def __init__(self, num_colors: int = 10, hidden_dim: int = 256, num_layers: int = 3):
        super().__init__()
        self.num_colors = num_colors
        self.hidden_dim = hidden_dim

        # Initial projection from one-hot to hidden_dim
        self.input_proj = nn.Conv2d(num_colors, hidden_dim, kernel_size=1)

        # Convolutional layers for spatial reasoning
        self.conv_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
                nn.LayerNorm([hidden_dim, 30, 30]),  # Assuming max 30x30 grids
                nn.GELU(),
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
                nn.LayerNorm([hidden_dim, 30, 30]),
                nn.GELU()
            ) for _ in range(num_layers)
        ])

        # Positional encoding
        self.register_buffer('pos_encoding', self._create_positional_encoding(30, 30, hidden_dim))

    def _create_positional_encoding(self, max_h: int, max_w: int, dim: int) -> torch.Tensor:
        """Create 2D positional encoding"""
        pe = torch.zeros(dim, max_h, max_w)

        # Create position indices
        h_pos = torch.arange(0, max_h).unsqueeze(1).repeat(1, max_w)  # [H, W]
        w_pos = torch.arange(0, max_w).unsqueeze(0).repeat(max_h, 1)  # [H, W]

        # Sinusoidal encoding
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-np.log(10000.0) / dim))

        for i in range(0, dim, 2):
            pe[i, :, :] = torch.sin(h_pos * div_term[i // 2])
            if i + 1 < dim:
                pe[i + 1, :, :] = torch.cos(h_pos * div_term[i // 2])

        return pe.unsqueeze(0)  # [1, dim, H, W]

    def forward(self, grid: torch.Tensor) -> torch.Tensor:
        """
        Args:
            grid: [batch, height, width] integer grid (values 0-9)

        Returns:
            embedding: [batch, hidden_dim, height, width] dense feature map
        """
        # One-hot encode
        onehot = grid_to_onehot(grid, self.num_colors)  # [batch, 10, H, W]

        # Initial projection
        x = self.input_proj(onehot)  # [batch, hidden_dim, H, W]

        # Add positional encoding
        h, w = x.shape[-2:]
        x = x + self.pos_encoding[:, :, :h, :w]

        # Convolutional feature extraction with residual connections
        for conv_layer in self.conv_layers:
            x = x + conv_layer(x)

        return x


# ============================================================================
# GRID DECODER: Learned Embedding → Raw Grid
# ============================================================================

class GridDecoder(nn.Module):
    """
    Decode latent embedding back to integer grid

    Architecture:
    - Convolutional upsampling if needed
    - Project to num_colors channels
    - Softmax over color dimension
    - Argmax to get discrete colors
    """

    def __init__(self, hidden_dim: int = 256, num_colors: int = 10):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_colors = num_colors

        # Decoder layers
        self.decoder = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim // 2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim // 2, hidden_dim // 4, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim // 4, num_colors, kernel_size=1)
        )

    def forward(self, embedding: torch.Tensor, return_probs: bool = False) -> torch.Tensor:
        """
        Args:
            embedding: [batch, hidden_dim, height, width]
            return_probs: If True, return softmax probabilities instead of argmax

        Returns:
            grid: [batch, height, width] integer grid (if return_probs=False)
                  [batch, num_colors, height, width] probabilities (if return_probs=True)
        """
        # Decode to color logits
        logits = self.decoder(embedding)  # [batch, num_colors, H, W]

        if return_probs:
            return F.softmax(logits, dim=1)

        # Argmax to get discrete colors
        grid = torch.argmax(logits, dim=1)  # [batch, H, W]
        return grid


# ============================================================================
# PROGRAM CONTROLLER: Predict Next Primitive
# ============================================================================

class ProgramController(nn.Module):
    """
    Controller that predicts which primitive to execute next

    Architecture:
    - Observes current grid embedding
    - Predicts distribution over primitives
    - Uses LSTM for sequential decisions
    - Outputs: (primitive_logits, primitive_parameters)

    POST-SOTA: Uses attention over execution history
    """

    def __init__(self, num_primitives: int, hidden_dim: int = 256, max_length: int = 10):
        super().__init__()
        self.num_primitives = num_primitives
        self.hidden_dim = hidden_dim
        self.max_length = max_length

        # Spatial pooling to get global grid representation
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # LSTM for sequential program generation
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True
        )

        # Primitive prediction head
        self.primitive_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_primitives + 1)  # +1 for EOS token
        )

        # Parameter prediction heads (for parameterized primitives)
        self.param_heads = nn.ModuleDict({
            'translation': nn.Linear(hidden_dim, 2),  # dx, dy
            'color_swap': nn.Linear(hidden_dim, 2),   # color1, color2
            'color_filter': nn.Linear(hidden_dim, 1), # target_color
        })

        self.EOS_TOKEN = num_primitives  # Special token for end-of-sequence

    def forward(self, grid_embedding: torch.Tensor, step: int,
                lstm_state: Optional[Tuple] = None) -> Tuple[torch.Tensor, Dict, Tuple]:
        """
        Predict next primitive given current grid state

        Args:
            grid_embedding: [batch, hidden_dim, height, width]
            step: Current program step (0-indexed)
            lstm_state: Optional LSTM hidden state from previous step

        Returns:
            primitive_logits: [batch, num_primitives+1] (includes EOS)
            parameters: Dict of predicted parameters for each primitive type
            new_lstm_state: Updated LSTM state
        """
        batch_size = grid_embedding.shape[0]

        # Global pooling to get vector representation
        global_repr = self.global_pool(grid_embedding).view(batch_size, -1)  # [batch, hidden_dim]

        # LSTM for sequential decision
        lstm_input = global_repr.unsqueeze(1)  # [batch, 1, hidden_dim]
        lstm_output, new_lstm_state = self.lstm(lstm_input, lstm_state)
        lstm_output = lstm_output.squeeze(1)  # [batch, hidden_dim]

        # Predict primitive
        primitive_logits = self.primitive_head(lstm_output)  # [batch, num_primitives+1]

        # Predict parameters
        parameters = {
            'translation': torch.tanh(self.param_heads['translation'](lstm_output)) * 5,  # dx, dy in [-5, 5]
            'color_swap': torch.sigmoid(self.param_heads['color_swap'](lstm_output)) * 9,  # colors in [0, 9]
            'color_filter': torch.sigmoid(self.param_heads['color_filter'](lstm_output)) * 9,
        }

        return primitive_logits, parameters, new_lstm_state


# ============================================================================
# NEURAL PRIMITIVE COMPOSITION NETWORK (Main Architecture)
# ============================================================================

class NeuralPrimitiveCompositionNetwork(nn.Module):
    """
    POST-SOTA: Complete system for learning to compose primitives

    Capabilities:
    1. FORWARD: Execute programs (inference mode)
    2. INVERSE: Infer programs from (input, output) pairs (training mode)
    3. SEARCH: Beam search + RL for program discovery
    4. INTERPRET: Generate human-readable program traces

    This is the CORE innovation that enables compositional generalization
    """

    def __init__(self,
                 num_colors: int = 10,
                 hidden_dim: int = 256,
                 max_program_length: int = 10,
                 beam_width: int = 5):
        super().__init__()

        self.num_colors = num_colors
        self.hidden_dim = hidden_dim
        self.max_program_length = max_program_length
        self.beam_width = beam_width

        # Initialize primitive registry
        self.primitives = PRIMITIVE_REGISTRY
        self.num_primitives = len(self.primitives)

        # Encoder/Decoder
        self.encoder = GridEncoder(num_colors, hidden_dim)
        self.decoder = GridDecoder(hidden_dim, num_colors)

        # Program controller
        self.controller = ProgramController(self.num_primitives, hidden_dim, max_program_length)

        # Primitive modules (differentiable implementations)
        self.primitive_modules = nn.ModuleList([
            self.primitives.get(name) for name in self.primitives.list_primitives()
        ])

        # Value network (for RL-based program search)
        self.value_network = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, input_grid: torch.Tensor,
                target_grid: Optional[torch.Tensor] = None,
                mode: str = 'inference') -> Dict[str, Any]:
        """
        Main forward pass with two modes:

        INFERENCE MODE: input_grid → predict program → execute → output_grid
        TRAINING MODE: (input_grid, target_grid) → infer program → supervise

        Args:
            input_grid: [batch, height, width] integer grid
            target_grid: [batch, height, width] integer grid (only for training)
            mode: 'inference' or 'training'

        Returns:
            Dict with:
                - output_grid: Predicted output
                - program: List of primitive indices
                - trace: Complete execution trace
                - confidence: Program confidence score
        """
        if mode == 'inference':
            return self._inference_mode(input_grid)
        elif mode == 'training':
            assert target_grid is not None, "target_grid required for training mode"
            return self._training_mode(input_grid, target_grid)
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def _inference_mode(self, input_grid: torch.Tensor) -> Dict[str, Any]:
        """
        INFERENCE: Generate program and execute it

        Uses greedy decoding for speed (can use beam search for better quality)
        """
        batch_size = input_grid.shape[0]

        # Encode input
        current_embedding = self.encoder(input_grid)  # [batch, hidden_dim, H, W]

        # Program generation
        program = []
        intermediate_embeddings = [current_embedding]
        lstm_state = None

        for step in range(self.max_program_length):
            # Predict next primitive
            prim_logits, params, lstm_state = self.controller(current_embedding, step, lstm_state)

            # Greedy selection
            prim_idx = torch.argmax(prim_logits, dim=-1).item()

            # Check for EOS
            if prim_idx == self.controller.EOS_TOKEN:
                break

            program.append(prim_idx)

            # Execute primitive
            prim_onehot = grid_to_onehot(self.decoder(current_embedding), self.num_colors)
            transformed = self.primitive_modules[prim_idx](prim_onehot)
            current_embedding = self.encoder(onehot_to_grid(transformed))

            intermediate_embeddings.append(current_embedding)

        # Decode final output
        output_grid = self.decoder(current_embedding)

        # Build trace
        trace = ProgramExecutionTrace(
            program=program,
            intermediate_grids=[self.decoder(emb) for emb in intermediate_embeddings],
            primitive_names=[self.primitives.list_primitives()[i] for i in program],
            confidence=1.0,  # TODO: compute actual confidence
            num_steps=len(program)
        )

        return {
            'output_grid': output_grid,
            'program': program,
            'trace': trace,
            'confidence': trace.confidence
        }

    def _training_mode(self, input_grid: torch.Tensor, target_grid: torch.Tensor) -> Dict[str, Any]:
        """
        TRAINING: Infer program from (input, target) pair

        POST-SOTA INNOVATION: Beam search + RL for program inference
        """
        # Encode input and target
        input_emb = self.encoder(input_grid)
        target_emb = self.encoder(target_grid)

        # Run beam search to find best program
        best_program, best_score = self._beam_search_program(input_emb, target_emb)

        # Execute best program to get output
        output_grid = self._execute_program(input_grid, best_program)

        # Compute loss
        reconstruction_loss = F.cross_entropy(
            self.decoder(self.encoder(output_grid), return_probs=True),
            target_grid
        )

        return {
            'output_grid': output_grid,
            'program': best_program,
            'loss': reconstruction_loss,
            'program_score': best_score
        }

    def _beam_search_program(self, input_emb: torch.Tensor, target_emb: torch.Tensor,
                             beam_width: Optional[int] = None) -> Tuple[List[int], float]:
        """
        POST-SOTA: Beam search for program inference

        Searches over program space to find sequence that transforms input → target

        Returns:
            best_program: List of primitive indices
            best_score: Score of best program (higher = better)
        """
        if beam_width is None:
            beam_width = self.beam_width

        # Initialize beam: [(embedding, program, score, lstm_state)]
        beam = [(input_emb, [], 0.0, None)]

        for step in range(self.max_program_length):
            candidates = []

            for current_emb, program, score, lstm_state in beam:
                # Predict next primitives
                prim_logits, params, new_lstm_state = self.controller(current_emb, step, lstm_state)
                prim_probs = F.softmax(prim_logits, dim=-1)[0]  # [num_primitives+1]

                # Top-k primitives
                topk_probs, topk_indices = torch.topk(prim_probs, k=min(beam_width, len(prim_probs)))

                for k in range(len(topk_probs)):
                    prim_idx = topk_indices[k].item()
                    prim_prob = topk_probs[k].item()

                    # Stop if EOS
                    if prim_idx == self.controller.EOS_TOKEN:
                        candidates.append((current_emb, program, score + np.log(prim_prob), new_lstm_state))
                        continue

                    # Execute primitive
                    try:
                        prim_onehot = grid_to_onehot(self.decoder(current_emb), self.num_colors)
                        transformed = self.primitive_modules[prim_idx](prim_onehot)
                        next_emb = self.encoder(onehot_to_grid(transformed))

                        # Compute reward: distance to target
                        distance = F.mse_loss(next_emb, target_emb, reduction='mean').item()
                        reward = -distance  # Negative distance (higher = better)

                        # Update score: log_prob + reward
                        new_score = score + np.log(prim_prob + 1e-10) + reward * 0.1

                        # Add to candidates
                        candidates.append((next_emb, program + [prim_idx], new_score, new_lstm_state))

                    except Exception as e:
                        # Skip if primitive execution fails
                        continue

            # Keep top beam_width candidates
            if not candidates:
                break

            candidates.sort(key=lambda x: x[2], reverse=True)
            beam = candidates[:beam_width]

            # Early stop if best candidate is very close to target
            best_emb = beam[0][0]
            if F.mse_loss(best_emb, target_emb) < 0.01:
                break

        # Return best program
        if not beam:
            return [], -float('inf')

        best_emb, best_program, best_score, _ = beam[0]
        return best_program, best_score

    def _execute_program(self, grid: torch.Tensor, program: List[int]) -> torch.Tensor:
        """
        Execute program on grid

        Args:
            grid: [batch, height, width] integer grid
            program: List of primitive indices

        Returns:
            output_grid: [batch, height, width] transformed grid
        """
        current_grid = grid

        for prim_idx in program:
            # Convert to one-hot
            onehot = grid_to_onehot(current_grid, self.num_colors)

            # Execute primitive
            transformed = self.primitive_modules[prim_idx](onehot)

            # Convert back to integer grid
            current_grid = onehot_to_grid(transformed)

        return current_grid

    def interpret_program(self, program: List[int]) -> str:
        """
        Convert program to human-readable description

        POST-SOTA: FULL INTERPRETABILITY

        Args:
            program: List of primitive indices

        Returns:
            description: Human-readable program trace
        """
        primitive_names = self.primitives.list_primitives()

        steps = []
        for i, prim_idx in enumerate(program):
            if prim_idx < len(primitive_names):
                name = primitive_names[prim_idx]
                steps.append(f"{i+1}. {name}")
            else:
                steps.append(f"{i+1}. UNKNOWN[{prim_idx}]")

        return "\n".join(steps)


# ============================================================================
# TRAINING UTILITIES
# ============================================================================

def train_on_atomic_tasks(model: NeuralPrimitiveCompositionNetwork,
                          atomic_tasks: List[Dict],
                          epochs: int = 10,
                          lr: float = 1e-4,
                          device: str = 'cuda'):
    """
    Train on atomic tasks (single primitive each)

    This establishes the foundation before compositional learning
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    model = model.to(device)
    model.train()

    print(f"\n{'='*60}")
    print(f"TRAINING ON ATOMIC TASKS")
    print(f"{'='*60}")
    print(f"Tasks: {len(atomic_tasks)}")
    print(f"Epochs: {epochs}")
    print(f"Device: {device}\n")

    for epoch in range(epochs):
        total_loss = 0.0
        correct = 0

        for task in atomic_tasks:
            input_grid = torch.LongTensor(task['input']).unsqueeze(0).to(device)
            target_grid = torch.LongTensor(task['output']).unsqueeze(0).to(device)

            # Forward pass
            result = model(input_grid, target_grid, mode='training')

            # Loss
            loss = result['loss']

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            # Metrics
            total_loss += loss.item()

            # Check if output matches target
            if torch.equal(result['output_grid'], target_grid):
                correct += 1

        # Epoch summary
        avg_loss = total_loss / len(atomic_tasks)
        accuracy = correct / len(atomic_tasks)

        print(f"Epoch {epoch+1}/{epochs}: Loss = {avg_loss:.4f}, Accuracy = {accuracy*100:.1f}%")

    print(f"\n✅ Atomic training complete!\n")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("NEURAL PRIMITIVE COMPOSITION NETWORK")
    print("="*60 + "\n")

    # Initialize model
    model = NeuralPrimitiveCompositionNetwork(
        num_colors=10,
        hidden_dim=256,
        max_program_length=10,
        beam_width=5
    )

    print(f"Model initialized:")
    print(f"  Primitives: {model.num_primitives}")
    print(f"  Hidden dim: {model.hidden_dim}")
    print(f"  Max program length: {model.max_program_length}")
    print(f"  Beam width: {model.beam_width}")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total_params:,} ({total_params/1e6:.2f}M)")

    # Test inference mode
    print("\n" + "="*60)
    print("TESTING INFERENCE MODE")
    print("="*60 + "\n")

    test_grid = torch.randint(0, 10, (1, 5, 5))
    print("Input grid:")
    print(test_grid[0])

    with torch.no_grad():
        result = model(test_grid, mode='inference')

    print(f"\nGenerated program ({result['trace'].num_steps} steps):")
    print(model.interpret_program(result['program']))

    print("\nOutput grid:")
    print(result['output_grid'][0])

    print("\n✅ Composition network module loaded successfully!")
