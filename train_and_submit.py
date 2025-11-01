#!/usr/bin/env python3
"""
OrcaSword v3 - Complete Training & Submission Pipeline
=======================================================

End-to-end pipeline for ARC Prize 2025:
1. Load ARC training/evaluation/test datasets
2. Train OrcaSword v3 model
3. Evaluate on validation set
4. Generate submission.json for test set

Usage:
    python3 train_and_submit.py --data_dir /kaggle/input/arc-prize-2025 --train --submit

Requirements:
    - PyTorch
    - NumPy
    - JSON files from ARC Prize 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import os
import sys
import time
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Import core components from orcaswordv3
# Note: In production, these would be imported from the main module
# For now, we'll define simplified versions or assume they exist

print("=" * 80)
print("OrcaSword v3 - Training & Submission Pipeline")
print("=" * 80)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}")

# =============================================================================
# 1. DATA LOADING FOR ARC PRIZE 2025
# =============================================================================

class ARCDataset:
    """ARC dataset loader for training/evaluation/test sets"""

    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.train_tasks = {}
        self.eval_tasks = {}
        self.test_tasks = {}
        self.train_solutions = {}
        self.eval_solutions = {}

    def load_all(self):
        """Load all datasets"""
        print("\nLoading ARC datasets...")

        # Training data
        train_path = self.data_dir / "arc-agi_training_challenges.json"
        train_sol_path = self.data_dir / "arc-agi_training_solutions.json"
        if train_path.exists():
            with open(train_path, 'r') as f:
                self.train_tasks = json.load(f)
            print(f"  ✓ Loaded {len(self.train_tasks)} training tasks")

        if train_sol_path.exists():
            with open(train_sol_path, 'r') as f:
                self.train_solutions = json.load(f)

        # Evaluation data
        eval_path = self.data_dir / "arc-agi_evaluation_challenges.json"
        eval_sol_path = self.data_dir / "arc-agi_evaluation_solutions.json"
        if eval_path.exists():
            with open(eval_path, 'r') as f:
                self.eval_tasks = json.load(f)
            print(f"  ✓ Loaded {len(self.eval_tasks)} evaluation tasks")

        if eval_sol_path.exists():
            with open(eval_sol_path, 'r') as f:
                self.eval_solutions = json.load(f)

        # Test data
        test_path = self.data_dir / "arc-agi_test_challenges.json"
        if test_path.exists():
            with open(test_path, 'r') as f:
                self.test_tasks = json.load(f)
            print(f"  ✓ Loaded {len(self.test_tasks)} test tasks")

        return self

    def get_task_pairs(self, split='train') -> List[Tuple[str, Dict, List[List[int]]]]:
        """Get task pairs for training

        Returns: List of (task_id, task_data, solution)
        """
        if split == 'train':
            tasks = self.train_tasks
            solutions = self.train_solutions
        elif split == 'eval':
            tasks = self.eval_tasks
            solutions = self.eval_solutions
        else:
            return []

        pairs = []
        for task_id, task_data in tasks.items():
            if task_id in solutions:
                # Get first test case solution
                solution = solutions[task_id][0] if solutions[task_id] else None
                if solution:
                    pairs.append((task_id, task_data, solution))

        return pairs

def pad_grid(grid: List[List[int]], max_h: int = 30, max_w: int = 30) -> np.ndarray:
    """Pad grid to fixed size"""
    h, w = len(grid), len(grid[0]) if grid else 0

    # Truncate if too large
    h = min(h, max_h)
    w = min(w, max_w)

    padded = np.zeros((max_h, max_w), dtype=np.int64)
    for i in range(h):
        for j in range(min(w, len(grid[i]))):
            padded[i, j] = grid[i][j]

    return padded

# =============================================================================
# 2. SIMPLIFIED MODEL FOR TRAINING
# =============================================================================

class SimpleARCModel(nn.Module):
    """Simplified ARC model for quick training

    Uses core concepts from OrcaSword v3 but streamlined for efficiency
    """

    def __init__(self, grid_size=30, num_colors=10, embed_dim=128, num_layers=4):
        super().__init__()
        self.grid_size = grid_size
        self.num_colors = num_colors
        self.embed_dim = embed_dim

        # Embedding
        self.color_embed = nn.Embedding(num_colors, embed_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, grid_size * grid_size, embed_dim) * 0.02)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=8,
            dim_feedforward=embed_dim * 4,
            batch_first=True,
            dropout=0.1,
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        # Output head
        self.output_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim // 2, num_colors)
        )

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, H, W] with values in [0, 9]

        Returns:
            logits: [batch, H, W, num_colors]
        """
        batch, H, W = x.shape

        # Flatten and embed
        x_flat = x.view(batch, -1).long()
        x_emb = self.color_embed(x_flat)  # [batch, H*W, embed_dim]

        # Add positional encoding
        seq_len = x_flat.shape[1]
        x_emb = x_emb + self.pos_embed[:, :seq_len, :]

        # Transform
        encoded = self.transformer(x_emb)  # [batch, H*W, embed_dim]

        # Predict
        logits = self.output_head(encoded)  # [batch, H*W, num_colors]
        logits = logits.view(batch, H, W, self.num_colors)

        return logits

# =============================================================================
# 3. TRAINING LOOP
# =============================================================================

class ARCTrainer:
    """Trainer for ARC models"""

    def __init__(self, model, device, learning_rate=1e-4):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=100)
        self.best_accuracy = 0.0

    def train_epoch(self, task_pairs: List[Tuple], max_steps: int = None):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        total_acc = 0.0
        n_samples = 0

        for i, (task_id, task_data, solution) in enumerate(task_pairs):
            if max_steps and i >= max_steps:
                break

            # Get training examples
            train_examples = task_data.get('train', [])
            if not train_examples:
                continue

            # Use train examples as input-output pairs
            for example in train_examples[:3]:  # Use first 3 examples
                input_grid = pad_grid(example['input'])
                output_grid = pad_grid(example['output'])

                # Convert to tensors
                x = torch.from_numpy(input_grid).unsqueeze(0).to(self.device)
                y = torch.from_numpy(output_grid).to(self.device)

                # Forward pass
                logits = self.model(x)  # [1, H, W, num_colors]
                logits = logits.squeeze(0)  # [H, W, num_colors]

                # Compute loss
                loss = F.cross_entropy(
                    logits.reshape(-1, self.model.num_colors),
                    y.reshape(-1)
                )

                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

                # Compute accuracy
                pred = logits.argmax(dim=-1)
                acc = (pred == y).float().mean().item()

                total_loss += loss.item()
                total_acc += acc
                n_samples += 1

            if (i + 1) % 20 == 0:
                avg_loss = total_loss / max(n_samples, 1)
                avg_acc = total_acc / max(n_samples, 1)
                print(f"  Step {i+1}/{len(task_pairs)}: Loss={avg_loss:.4f}, Acc={avg_acc:.3f}")

        self.scheduler.step()

        avg_loss = total_loss / max(n_samples, 1)
        avg_acc = total_acc / max(n_samples, 1)

        return avg_loss, avg_acc

    def evaluate(self, task_pairs: List[Tuple]) -> float:
        """Evaluate on task pairs"""
        self.model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for task_id, task_data, solution in task_pairs:
                # Get test input
                test_examples = task_data.get('test', [])
                if not test_examples:
                    continue

                test_input = pad_grid(test_examples[0]['input'])
                expected_output = pad_grid(solution)

                # Predict
                x = torch.from_numpy(test_input).unsqueeze(0).to(self.device)
                logits = self.model(x)
                pred = logits.argmax(dim=-1).squeeze(0).cpu().numpy()

                # Compare (pixel-level accuracy)
                correct += (pred == expected_output).sum()
                total += expected_output.size

        accuracy = correct / max(total, 1)
        return accuracy

    def save_checkpoint(self, path: str):
        """Save model checkpoint"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_accuracy': self.best_accuracy,
        }, path)
        print(f"  ✓ Saved checkpoint to {path}")

    def load_checkpoint(self, path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.best_accuracy = checkpoint.get('best_accuracy', 0.0)
        print(f"  ✓ Loaded checkpoint from {path}")

# =============================================================================
# 4. SUBMISSION GENERATION
# =============================================================================

def generate_submission(model, test_tasks: Dict, output_path: str, device):
    """Generate submission.json for test set"""
    print("\nGenerating submission.json...")
    model.eval()

    submission = []

    with torch.no_grad():
        for i, (task_id, task_data) in enumerate(test_tasks.items(), 1):
            test_examples = task_data.get('test', [])
            if not test_examples:
                continue

            # Get test input
            test_input = test_examples[0]['input']
            padded_input = pad_grid(test_input)

            # Predict
            x = torch.from_numpy(padded_input).unsqueeze(0).to(device)
            logits = model(x)
            pred = logits.argmax(dim=-1).squeeze(0).cpu().numpy()

            # Unpad to original size (or keep reasonable size)
            H, W = len(test_input), len(test_input[0]) if test_input else 1
            pred_grid = pred[:H, :W].tolist()

            # Add to submission (2 attempts as per competition rules)
            submission.append({
                "task_id": task_id,
                "attempt_1": pred_grid,
                "attempt_2": pred_grid  # Same prediction for both attempts
            })

            if i % 50 == 0:
                print(f"  Progress: {i}/{len(test_tasks)}")

    # Write submission
    with open(output_path, 'w') as f:
        json.dump(submission, f)

    print(f"  ✓ Submission saved to {output_path}")
    print(f"  ✓ Total tasks: {len(submission)}")

    return submission

# =============================================================================
# 5. MAIN PIPELINE
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='OrcaSword v3 Training & Submission')
    parser.add_argument('--data_dir', type=str, default='/kaggle/input/arc-prize-2025',
                        help='Path to ARC Prize 2025 data directory')
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--embed_dim', type=int, default=128, help='Embedding dimension')
    parser.add_argument('--num_layers', type=int, default=4, help='Number of transformer layers')
    parser.add_argument('--checkpoint', type=str, default='orcasword_checkpoint.pt',
                        help='Checkpoint file path')
    parser.add_argument('--submit', action='store_true', help='Generate submission.json')
    parser.add_argument('--output', type=str, default='submission.json',
                        help='Output submission file path')

    args = parser.parse_args()

    print("\n" + "=" * 80)
    print("Configuration:")
    print(f"  Data directory: {args.data_dir}")
    print(f"  Device: {DEVICE}")
    print(f"  Train: {args.train}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Embed dim: {args.embed_dim}")
    print(f"  Num layers: {args.num_layers}")
    print("=" * 80)

    # Load datasets
    dataset = ARCDataset(args.data_dir)
    dataset.load_all()

    # Initialize model
    model = SimpleARCModel(
        embed_dim=args.embed_dim,
        num_layers=args.num_layers
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n✓ Model initialized: {total_params:,} parameters")

    # Initialize trainer
    trainer = ARCTrainer(model, DEVICE, learning_rate=args.lr)

    # Load checkpoint if exists
    if os.path.exists(args.checkpoint):
        try:
            trainer.load_checkpoint(args.checkpoint)
        except Exception as e:
            print(f"  Warning: Could not load checkpoint: {e}")

    # Training
    if args.train:
        print("\n" + "=" * 80)
        print("TRAINING")
        print("=" * 80)

        train_pairs = dataset.get_task_pairs('train')
        eval_pairs = dataset.get_task_pairs('eval')

        print(f"Training samples: {len(train_pairs)}")
        print(f"Evaluation samples: {len(eval_pairs)}")

        best_eval_acc = 0.0

        for epoch in range(args.epochs):
            print(f"\nEpoch {epoch + 1}/{args.epochs}")
            print("-" * 40)

            # Train
            train_loss, train_acc = trainer.train_epoch(train_pairs, max_steps=100)
            print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.3f}")

            # Evaluate
            if eval_pairs:
                eval_acc = trainer.evaluate(eval_pairs[:50])  # Evaluate on subset
                print(f"  Eval Acc: {eval_acc:.3f}")

                if eval_acc > best_eval_acc:
                    best_eval_acc = eval_acc
                    trainer.best_accuracy = eval_acc
                    trainer.save_checkpoint(args.checkpoint)
                    print(f"  ✓ New best accuracy: {eval_acc:.3f}")

        print("\n" + "=" * 80)
        print(f"Training complete! Best accuracy: {best_eval_acc:.3f}")
        print("=" * 80)

    # Generate submission
    if args.submit:
        if not args.train and os.path.exists(args.checkpoint):
            trainer.load_checkpoint(args.checkpoint)

        submission = generate_submission(
            model,
            dataset.test_tasks,
            args.output,
            DEVICE
        )

        print("\n" + "=" * 80)
        print("SUBMISSION COMPLETE")
        print("=" * 80)
        print(f"File: {args.output}")
        print(f"Tasks: {len(submission)}")
        print("=" * 80)

if __name__ == "__main__":
    main()
