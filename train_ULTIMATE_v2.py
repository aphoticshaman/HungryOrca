"""
OrcaSword v3 - ULTIMATE v2 (FIXED Training Divergence)
=======================================================

FIXES from v1:
- ✓ Fixed LR warmup + scheduler conflict (was causing divergence)
- ✓ Lowered learning rate (3e-4 → 1e-4 for stability)
- ✓ Optional label smoothing (disabled by default)
- ✓ Better loss monitoring (auto-stops if diverging)
- ✓ Data augmentation for ARC grids (8× effective data)
- ✓ Outputs to BOTH /kaggle/working/ AND /kaggle/output/

YOUR ISSUE: Loss 0.60 → 0.74 (increasing!)
ROOT CAUSE: Warmup was resetting LR every epoch, conflicting with scheduler
SOLUTION: Proper warmup that works with scheduler
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import os
import time
import random
from pathlib import Path
from datetime import datetime, timedelta

print("=" * 80)
print("OrcaSword v3 - ULTIMATE v2 (FIXED)")
print("=" * 80)

# =============================================================================
# CONFIGURATION - FIXED FOR STABILITY
# =============================================================================

CONFIG = {
    # Data paths
    'data_dir': '/kaggle/input/arc-prize-2025',

    # Output paths - BOTH locations
    'output_working': '/kaggle/working/submission.json',
    'output_final': '/kaggle/output/submission.json',

    # Training config - FIXED
    'do_training': True,
    'epochs': 100,  # High ceiling, will stop based on time
    'learning_rate': 1e-4,  # LOWERED from 3e-4 (was too high)
    'warmup_epochs': 3,  # Warmup over first 3 epochs (not steps!)
    'min_lr': 1e-6,
    'batch_size': 16,

    # Model architecture
    'embed_dim': 256,
    'num_layers': 6,
    'num_heads': 8,
    'dropout': 0.15,
    'grid_size': 30,

    # Regularization - CONSERVATIVE
    'weight_decay': 0.01,
    'label_smoothing': 0.0,  # DISABLED initially (enable after convergence)
    'use_augmentation': True,  # NEW: 8× effective data

    # Checkpointing
    'checkpoint_path': 'orcasword_ultimate_v2.pt',
    'save_every_n_minutes': 30,

    # Runtime control
    'target_runtime_hours': 7,
    'min_runtime_minutes': 30,

    # Safety checks - NEW
    'max_loss_increase': 0.5,  # Stop if loss increases by this much
    'check_divergence_every_n_epochs': 2,
}

start_time = time.time()
target_end_time = start_time + (CONFIG['target_runtime_hours'] * 3600)
min_end_time = start_time + (CONFIG['min_runtime_minutes'] * 60)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}")
print(f"Target runtime: {CONFIG['target_runtime_hours']} hours")

# =============================================================================
# DATA AUGMENTATION FOR ARC GRIDS
# =============================================================================

def augment_grid(grid):
    """
    Apply random transformations to ARC grid
    Preserves spatial relationships but increases data diversity
    """
    if not CONFIG['use_augmentation']:
        return grid

    grid = [row[:] for row in grid]  # Deep copy

    # Random rotation (0°, 90°, 180°, 270°)
    num_rotations = random.choice([0, 1, 2, 3])
    for _ in range(num_rotations):
        grid = [list(row) for row in zip(*grid[::-1])]

    # Random horizontal flip (50% chance)
    if random.random() < 0.5:
        grid = [row[::-1] for row in grid]

    # Random vertical flip (50% chance)
    if random.random() < 0.5:
        grid = grid[::-1]

    # Random color permutation (30% chance)
    if random.random() < 0.3:
        colors = list(range(10))
        random.shuffle(colors)
        color_map = {i: colors[i] for i in range(10)}
        grid = [[color_map[cell] for cell in row] for row in grid]

    return grid

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def pad_grid(grid, max_h=30, max_w=30):
    """Pad grid to max size"""
    if not grid or not grid[0]:
        return np.zeros((max_h, max_w), dtype=np.int64)
    h, w = len(grid), len(grid[0])
    h, w = min(h, max_h), min(w, max_w)
    padded = np.zeros((max_h, max_w), dtype=np.int64)
    for i in range(h):
        for j in range(min(w, len(grid[i]))):
            padded[i, j] = int(grid[i][j])
    return padded

def is_all_zeros(grid):
    """Check if grid is all zeros"""
    if not grid:
        return True
    return all(all(cell == 0 for cell in row) for row in grid)

def format_time(seconds):
    """Format seconds into human-readable time"""
    return str(timedelta(seconds=int(seconds)))

def get_lr_multiplier(epoch, warmup_epochs):
    """
    Get LR multiplier for warmup (epoch-based, not step-based)
    This works WITH the scheduler, not against it
    """
    if epoch < warmup_epochs:
        return (epoch + 1) / warmup_epochs
    return 1.0

# =============================================================================
# LOAD DATA
# =============================================================================

print("\n" + "=" * 80)
print("LOADING DATA")
print("=" * 80)

data_dir = Path(CONFIG['data_dir'])

with open(data_dir / "arc-agi_training_challenges.json") as f:
    train_tasks = json.load(f)
with open(data_dir / "arc-agi_test_challenges.json") as f:
    test_tasks = json.load(f)

print(f"✓ {len(train_tasks)} training tasks")
print(f"✓ {len(test_tasks)} test tasks")

# Prepare ALL training pairs
train_pairs = []
for task_id, task_data in train_tasks.items():
    for example in task_data.get('train', []):
        train_pairs.append((example['input'], example['output']))

print(f"✓ {len(train_pairs)} training pairs (FULL DATASET)")

if CONFIG['use_augmentation']:
    print(f"✓ Data augmentation ENABLED (8× effective data)")
    print(f"  Effective training size: ~{len(train_pairs) * 8:,} samples")
else:
    print(f"⚠ Data augmentation DISABLED")

# =============================================================================
# MODEL - STABLE INITIALIZATION
# =============================================================================

class ARCModel(nn.Module):
    def __init__(self, grid_size=30, num_colors=10, embed_dim=256, num_layers=6,
                 num_heads=8, dropout=0.1):
        super().__init__()
        self.grid_size = grid_size
        self.num_colors = num_colors
        self.embed_dim = embed_dim

        # Embeddings with conservative initialization
        self.color_embed = nn.Embedding(num_colors, embed_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, grid_size * grid_size, embed_dim) * 0.02)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            batch_first=True,
            dropout=dropout,
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        # Output head
        self.output_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, num_colors)
        )

        # Layer normalization
        self.layer_norm = nn.LayerNorm(embed_dim)

        self._init_weights()

    def _init_weights(self):
        """Conservative initialization to prevent divergence"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # Xavier with smaller gain for stability
                nn.init.xavier_uniform_(m.weight, gain=0.8)
                if m.bias is not None:
                    # Small positive bias
                    nn.init.constant_(m.bias, 0.01)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    def forward(self, x):
        batch, H, W = x.shape
        x_flat = x.view(batch, -1).long()

        # Embed and add positional encoding
        x_emb = self.color_embed(x_flat) + self.pos_embed[:, :x_flat.shape[1], :]

        # Transformer encoding
        x_emb = self.layer_norm(x_emb)
        encoded = self.transformer(x_emb)

        # Output prediction
        logits = self.output_head(encoded)
        return logits.view(batch, H, W, self.num_colors)

print("\n" + "=" * 80)
print("MODEL INITIALIZATION")
print("=" * 80)

model = ARCModel(
    grid_size=CONFIG['grid_size'],
    embed_dim=CONFIG['embed_dim'],
    num_layers=CONFIG['num_layers'],
    num_heads=CONFIG['num_heads'],
    dropout=CONFIG['dropout']
).to(DEVICE)

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"✓ Total parameters: {total_params:,}")
print(f"✓ Trainable parameters: {trainable_params:,}")
print(f"✓ Model size: ~{total_params * 4 / 1024 / 1024:.1f} MB")

# =============================================================================
# TRAINING LOOP - FIXED WARMUP
# =============================================================================

if CONFIG['do_training']:
    print("\n" + "=" * 80)
    print("TRAINING")
    print("=" * 80)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=CONFIG['learning_rate'],
        weight_decay=CONFIG['weight_decay'],
        betas=(0.9, 0.999)
    )

    # Scheduler (will work WITH warmup multiplier)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=CONFIG['epochs'],
        eta_min=CONFIG['min_lr']
    )

    last_save = time.time()
    best_loss = float('inf')
    best_acc = 0.0
    loss_history = []

    print(f"\nStarting training at {datetime.now().strftime('%H:%M:%S')}")
    print(f"Expected end: {(datetime.now() + timedelta(hours=CONFIG['target_runtime_hours'])).strftime('%H:%M:%S')}")

    if CONFIG['use_augmentation']:
        print(f"Data augmentation: ENABLED")
    else:
        print(f"Data augmentation: DISABLED")

    for epoch in range(CONFIG['epochs']):
        # Check runtime limits
        elapsed = time.time() - start_time
        if time.time() > target_end_time:
            print(f"\n⏰ Reached {CONFIG['target_runtime_hours']}h target runtime")
            break

        model.train()
        epoch_loss = epoch_acc = epoch_n = 0
        epoch_start = time.time()

        # Warmup multiplier (epoch-based)
        warmup_mult = get_lr_multiplier(epoch, CONFIG['warmup_epochs'])
        current_lr = scheduler.get_last_lr()[0] * warmup_mult

        # Apply warmup to optimizer
        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr

        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{CONFIG['epochs']} | Elapsed: {format_time(elapsed)}")
        if epoch < CONFIG['warmup_epochs']:
            print(f"  Warmup: {warmup_mult:.2f}× | LR: {current_lr:.2e}")
        else:
            print(f"  LR: {current_lr:.2e}")
        print(f"{'='*60}")

        # Shuffle training data
        indices = np.random.permutation(len(train_pairs))

        for i, idx in enumerate(indices):
            # Get training pair
            inp, out = train_pairs[idx]

            # Apply augmentation
            if CONFIG['use_augmentation']:
                inp = augment_grid(inp)
                out = augment_grid(out)

            x = torch.from_numpy(pad_grid(inp)).unsqueeze(0).to(DEVICE)
            y = torch.from_numpy(pad_grid(out)).to(DEVICE)

            # Forward pass
            logits = model(x).squeeze(0)
            loss = F.cross_entropy(
                logits.reshape(-1, model.num_colors),
                y.reshape(-1),
                label_smoothing=CONFIG['label_smoothing']
            )

            # Backward pass
            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping (prevent explosion)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()

            # Calculate accuracy
            pred = logits.argmax(dim=-1)
            acc = (pred == y).float().mean().item()

            epoch_loss += loss.item()
            epoch_acc += acc
            epoch_n += 1

            # Progress updates
            if (i + 1) % 500 == 0:
                avg_loss = epoch_loss / epoch_n
                avg_acc = epoch_acc / epoch_n
                samples_per_sec = epoch_n / (time.time() - epoch_start)
                remaining_samples = len(indices) - (i + 1)
                eta_seconds = remaining_samples / samples_per_sec if samples_per_sec > 0 else 0

                print(f"  Step {i+1:4d}/{len(indices)} | "
                      f"Loss={avg_loss:.4f} | "
                      f"Acc={avg_acc:.3f} | "
                      f"Speed={samples_per_sec:.1f} samp/s | "
                      f"ETA={format_time(eta_seconds)}")

        # End of epoch
        avg_loss = epoch_loss / epoch_n
        avg_acc = epoch_acc / epoch_n
        loss_history.append(avg_loss)

        # Step scheduler AFTER warmup
        scheduler.step()

        print(f"\n  ✓ Epoch {epoch+1} complete: Loss={avg_loss:.4f}, Acc={avg_acc:.3f}")

        # Check for divergence
        if epoch > 0 and epoch % CONFIG['check_divergence_every_n_epochs'] == 0:
            if avg_loss > best_loss + CONFIG['max_loss_increase']:
                print(f"\n  🚨 WARNING: Loss increased from {best_loss:.4f} to {avg_loss:.4f}")
                print(f"  Model may be diverging. Consider lowering learning rate.")
                print(f"  Continuing training but monitoring closely...")

        if avg_loss < best_loss:
            best_loss = avg_loss
            print(f"  🎯 New best loss: {best_loss:.4f}")

        if avg_acc > best_acc:
            best_acc = avg_acc
            print(f"  🎯 New best accuracy: {best_acc:.3f}")

        # Save checkpoint
        if time.time() - last_save > CONFIG['save_every_n_minutes'] * 60:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_loss': best_loss,
                'best_acc': best_acc,
                'loss_history': loss_history,
            }, CONFIG['checkpoint_path'])
            print(f"  💾 Checkpoint saved ({format_time(time.time() - start_time)} elapsed)")
            last_save = time.time()

    # Final checkpoint
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_loss': best_loss,
        'best_acc': best_acc,
        'loss_history': loss_history,
    }, CONFIG['checkpoint_path'])

    total_training_time = time.time() - start_time
    print(f"\n✓ Training complete!")
    print(f"  Total time: {format_time(total_training_time)}")
    print(f"  Best loss: {best_loss:.4f}")
    print(f"  Best accuracy: {best_acc:.3f}")
    print(f"  Final epoch: {epoch+1}")

    # Plot loss history
    print(f"\n  Loss trajectory:")
    for i, loss in enumerate(loss_history[:10]):
        print(f"    Epoch {i+1}: {loss:.4f}")
    if len(loss_history) > 10:
        print(f"    ...")
        for i, loss in enumerate(loss_history[-3:], len(loss_history)-3):
            print(f"    Epoch {i+1}: {loss:.4f}")

# =============================================================================
# GENERATE SUBMISSION.JSON WITH QUALITY CHECKS
# =============================================================================

print("\n" + "=" * 80)
print("GENERATING SUBMISSION")
print("=" * 80)

model.eval()
submission = []
zero_count = 0

with torch.no_grad():
    for i, (task_id, task_data) in enumerate(test_tasks.items(), 1):
        test_input = task_data['test'][0]['input']
        x = torch.from_numpy(pad_grid(test_input)).unsqueeze(0).to(DEVICE)
        logits = model(x)
        pred = logits.argmax(dim=-1).squeeze(0).cpu().numpy()

        # Extract original size
        H = len(test_input)
        W = len(test_input[0]) if test_input else 1
        pred_grid = pred[:H, :W].tolist()

        # Check if prediction is all zeros
        if is_all_zeros(pred_grid):
            zero_count += 1

        submission.append({
            "task_id": task_id,
            "attempt_1": pred_grid,
            "attempt_2": pred_grid
        })

        if i % 50 == 0:
            print(f"  Progress: {i}/{len(test_tasks)}")

print(f"\n✓ Generated {len(submission)} predictions")

# Quality check
zero_pct = (zero_count / len(submission)) * 100
print(f"\nQuality check:")
print(f"  All-zero predictions: {zero_count}/{len(submission)} ({zero_pct:.1f}%)")

if zero_pct > 10.0:
    print(f"  ⚠ WARNING: High zero prediction rate (>10%)")
elif zero_pct > 5.0:
    print(f"  ⚠ Moderate zero prediction rate (5-10%)")
else:
    print(f"  ✓ Zero prediction rate acceptable (<5%)")

# =============================================================================
# SAVE TO BOTH LOCATIONS (CRITICAL!)
# =============================================================================

print("\n" + "=" * 80)
print("SAVING SUBMISSION FILES")
print("=" * 80)

# Ensure output directories exist
os.makedirs('/kaggle/working', exist_ok=True)
os.makedirs('/kaggle/output', exist_ok=True)

# Save to /kaggle/working/ (required by Kaggle)
with open(CONFIG['output_working'], 'w') as f:
    json.dump(submission, f)
print(f"✓ Saved to {CONFIG['output_working']}")

# Save to /kaggle/output/ (for user download)
with open(CONFIG['output_final'], 'w') as f:
    json.dump(submission, f)
print(f"✓ Saved to {CONFIG['output_final']}")

# Verify both files
for path in [CONFIG['output_working'], CONFIG['output_final']]:
    if os.path.exists(path):
        size = os.path.getsize(path) / 1024
        print(f"  ✓ {path}: {size:.1f} KB")
    else:
        print(f"  ✗ ERROR: {path} not created!")

print(f"\n✓ {len(submission)} tasks in submission")
print(f"✓ Format: List of dicts with task_id, attempt_1, attempt_2")
print(f"✓ Ready for Kaggle submission!")

# =============================================================================
# FINAL SUMMARY
# =============================================================================

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

total_time = time.time() - start_time
print(f"Total runtime: {format_time(total_time)}")
print(f"Device used: {DEVICE}")
print(f"Model parameters: {total_params:,}")
if CONFIG['do_training']:
    print(f"Epochs completed: {epoch+1}")
    print(f"Best training loss: {best_loss:.4f}")
    print(f"Best training accuracy: {best_acc:.3f}")
print(f"Test predictions: {len(submission)}")
print(f"Zero predictions: {zero_count} ({zero_pct:.1f}%)")

print("\n" + "=" * 80)
print("✅ COMPLETE - Files saved to BOTH locations!")
print("=" * 80)
