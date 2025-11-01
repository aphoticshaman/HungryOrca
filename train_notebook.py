"""
OrcaSword v3 - Notebook-Friendly Training Script
=================================================

Copy-paste this entire file into a Kaggle notebook cell and run it!
No command-line arguments needed.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import os
import time
from pathlib import Path

print("=" * 80)
print("OrcaSword v3 - Training & Submission Pipeline (Notebook Version)")
print("=" * 80)

# =============================================================================
# CONFIGURATION - EDIT THESE VALUES
# =============================================================================

CONFIG = {
    # Data paths
    'data_dir': '/kaggle/input/arc-prize-2025',  # Change if needed

    # Training
    'do_training': True,          # Set to False to skip training
    'epochs': 10,                 # Number of training epochs
    'learning_rate': 1e-4,        # Learning rate

    # Model architecture
    'embed_dim': 128,             # Embedding dimension (64, 128, 256)
    'num_layers': 4,              # Number of transformer layers (2-8)
    'grid_size': 30,              # Max grid size

    # Output
    'checkpoint_path': 'orcasword_checkpoint.pt',
    'do_submission': True,        # Generate submission.json
    'output_path': 'submission.json'
}

print("\nConfiguration:")
for key, value in CONFIG.items():
    print(f"  {key}: {value}")

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nDevice: {DEVICE}")

# =============================================================================
# DATA LOADING
# =============================================================================

def pad_grid(grid, max_h=30, max_w=30):
    """Pad grid to fixed size"""
    h, w = len(grid), len(grid[0]) if grid else 0
    h = min(h, max_h)
    w = min(w, max_w)

    padded = np.zeros((max_h, max_w), dtype=np.int64)
    for i in range(h):
        for j in range(min(w, len(grid[i]))):
            padded[i, j] = grid[i][j]

    return padded

print("\nLoading ARC datasets...")
data_dir = Path(CONFIG['data_dir'])

# Load training data
with open(data_dir / "arc-agi_training_challenges.json", 'r') as f:
    train_tasks = json.load(f)
print(f"  ✓ Loaded {len(train_tasks)} training tasks")

# Load evaluation data
with open(data_dir / "arc-agi_evaluation_challenges.json", 'r') as f:
    eval_tasks = json.load(f)
with open(data_dir / "arc-agi_evaluation_solutions.json", 'r') as f:
    eval_solutions = json.load(f)
print(f"  ✓ Loaded {len(eval_tasks)} evaluation tasks")

# Load test data
with open(data_dir / "arc-agi_test_challenges.json", 'r') as f:
    test_tasks = json.load(f)
print(f"  ✓ Loaded {len(test_tasks)} test tasks")

# =============================================================================
# MODEL
# =============================================================================

class SimpleARCModel(nn.Module):
    """Simplified ARC solver"""

    def __init__(self, grid_size=30, num_colors=10, embed_dim=128, num_layers=4):
        super().__init__()
        self.grid_size = grid_size
        self.num_colors = num_colors
        self.embed_dim = embed_dim

        # Embedding
        self.color_embed = nn.Embedding(num_colors, embed_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, grid_size * grid_size, embed_dim) * 0.02)

        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=8,
            dim_feedforward=embed_dim * 4,
            batch_first=True,
            dropout=0.1,
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        # Output
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

    def forward(self, x):
        batch, H, W = x.shape

        # Flatten and embed
        x_flat = x.view(batch, -1).long()
        x_emb = self.color_embed(x_flat)

        # Add positional encoding
        seq_len = x_flat.shape[1]
        x_emb = x_emb + self.pos_embed[:, :seq_len, :]

        # Transform
        encoded = self.transformer(x_emb)

        # Predict
        logits = self.output_head(encoded)
        logits = logits.view(batch, H, W, self.num_colors)

        return logits

print("\nInitializing model...")
model = SimpleARCModel(
    grid_size=CONFIG['grid_size'],
    embed_dim=CONFIG['embed_dim'],
    num_layers=CONFIG['num_layers']
).to(DEVICE)

total_params = sum(p.numel() for p in model.parameters())
print(f"  Total parameters: {total_params:,}")

# =============================================================================
# TRAINING
# =============================================================================

if CONFIG['do_training']:
    print("\n" + "=" * 80)
    print("TRAINING")
    print("=" * 80)

    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG['learning_rate'], weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG['epochs'])

    # Prepare training pairs
    train_pairs = []
    for task_id, task_data in train_tasks.items():
        for example in task_data.get('train', [])[:3]:  # First 3 examples
            train_pairs.append((example['input'], example['output']))

    print(f"Training samples: {len(train_pairs)}")

    best_loss = float('inf')

    for epoch in range(CONFIG['epochs']):
        model.train()
        total_loss = 0.0
        total_acc = 0.0
        n_samples = 0

        for i, (input_grid, output_grid) in enumerate(train_pairs[:100]):  # Limit per epoch
            # Prepare data
            x = torch.from_numpy(pad_grid(input_grid)).unsqueeze(0).to(DEVICE)
            y = torch.from_numpy(pad_grid(output_grid)).to(DEVICE)

            # Forward
            logits = model(x).squeeze(0)
            loss = F.cross_entropy(
                logits.reshape(-1, model.num_colors),
                y.reshape(-1)
            )

            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            # Metrics
            pred = logits.argmax(dim=-1)
            acc = (pred == y).float().mean().item()

            total_loss += loss.item()
            total_acc += acc
            n_samples += 1

        scheduler.step()

        avg_loss = total_loss / n_samples
        avg_acc = total_acc / n_samples

        print(f"Epoch {epoch+1}/{CONFIG['epochs']}: Loss={avg_loss:.4f}, Acc={avg_acc:.3f}")

        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), CONFIG['checkpoint_path'])
            print(f"  ✓ Saved checkpoint (loss: {avg_loss:.4f})")

    print("\n" + "=" * 80)
    print("Training complete!")
    print("=" * 80)

# =============================================================================
# GENERATE SUBMISSION
# =============================================================================

if CONFIG['do_submission']:
    print("\n" + "=" * 80)
    print("GENERATING SUBMISSION")
    print("=" * 80)

    # Load best model
    if os.path.exists(CONFIG['checkpoint_path']):
        model.load_state_dict(torch.load(CONFIG['checkpoint_path'], map_location=DEVICE))
        print(f"  ✓ Loaded checkpoint: {CONFIG['checkpoint_path']}")

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
            x = torch.from_numpy(padded_input).unsqueeze(0).to(DEVICE)
            logits = model(x)
            pred = logits.argmax(dim=-1).squeeze(0).cpu().numpy()

            # Unpad
            H, W = len(test_input), len(test_input[0]) if test_input else 1
            pred_grid = pred[:H, :W].tolist()

            # Add to submission
            submission.append({
                "task_id": task_id,
                "attempt_1": pred_grid,
                "attempt_2": pred_grid
            })

            if i % 50 == 0:
                print(f"  Progress: {i}/{len(test_tasks)}")

    # Save submission
    with open(CONFIG['output_path'], 'w') as f:
        json.dump(submission, f)

    print(f"\n  ✓ Submission saved: {CONFIG['output_path']}")
    print(f"  ✓ Total tasks: {len(submission)}")

    # Validate
    print("\n" + "=" * 80)
    print("VALIDATION")
    print("=" * 80)

    all_valid = all(
        'task_id' in task and
        'attempt_1' in task and
        'attempt_2' in task
        for task in submission
    )

    if all_valid:
        print("  ✓ Submission format is VALID!")
        print(f"  ✓ Ready to download: {CONFIG['output_path']}")
    else:
        print("  ✗ Submission format is INVALID")

print("\n" + "=" * 80)
print("COMPLETE! 🎉")
print("=" * 80)
print(f"\nNext step: Download {CONFIG['output_path']} and submit to competition!")
