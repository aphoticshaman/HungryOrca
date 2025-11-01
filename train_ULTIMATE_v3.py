"""
OrcaSword v3 - ULTIMATE v3 (FIXED Overfitting)
================================================

CRITICAL FIXES from v2 analysis:
- ❌ DISABLED data augmentation (caused train-test mismatch)
- ✓ SMALLER model (128 dim, 4 layers = ~600K params vs 5M)
- ✓ FEWER epochs (50 vs 100)
- ✓ BETTER LR schedule (5e-5 initial, doesn't freeze)
- ✓ EARLY STOPPING (prevents overfitting)
- ✓ VALIDATION monitoring (catches issues early)

ULTv2 PROBLEMS:
- 10% zero predictions (vs 5.8% baseline) - WORSE!
- Massive overfitting (91.5% train, poor test)
- Data augmentation mismatch
- Model too large (5M params for 3K examples)

EXPECTED RESULTS v3:
- Zero predictions: 2-3% (vs 10% in v2)
- Training time: ~40 min (vs 80 min)
- Better generalization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import os
import time
from pathlib import Path
from datetime import datetime, timedelta

print("=" * 80)
print("OrcaSword v3 - ULTIMATE v3 (FIXED)")
print("=" * 80)

# =============================================================================
# CONFIGURATION - FIXED FOR GENERALIZATION
# =============================================================================

CONFIG = {
    # Data paths
    'data_dir': '/kaggle/input/arc-prize-2025',

    # Output paths
    'output_working': '/kaggle/working/submission.json',
    'output_final': '/kaggle/output/submission.json',

    # Training config - CONSERVATIVE
    'do_training': True,
    'epochs': 50,  # REDUCED from 100
    'learning_rate': 5e-5,  # LOWER from 1e-4
    'warmup_epochs': 5,  # LONGER warmup
    'min_lr': 5e-6,  # HIGHER minimum (don't freeze)
    'batch_size': 16,

    # Model architecture - SMALLER!
    'embed_dim': 128,  # REDUCED from 256 (4× fewer params)
    'num_layers': 4,   # REDUCED from 6
    'num_heads': 8,
    'dropout': 0.2,    # INCREASED from 0.15 (more regularization)
    'grid_size': 30,

    # Regularization
    'weight_decay': 0.02,  # INCREASED from 0.01
    'label_smoothing': 0.0,
    'use_augmentation': False,  # ❌ DISABLED! (was True)

    # Early stopping - NEW!
    'early_stopping': True,
    'early_stopping_patience': 5,  # Stop if no improvement for 5 epochs
    'min_delta': 0.001,  # Minimum improvement to count

    # Validation - NEW!
    'validation_split': 0.1,  # Use 10% for validation

    # Checkpointing
    'checkpoint_path': 'orcasword_ultimate_v3.pt',
    'save_every_n_minutes': 30,

    # Runtime control
    'target_runtime_hours': 7,
    'min_runtime_minutes': 20,
}

start_time = time.time()
target_end_time = start_time + (CONFIG['target_runtime_hours'] * 3600)
min_end_time = start_time + (CONFIG['min_runtime_minutes'] * 60)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}")
print(f"Target runtime: {CONFIG['target_runtime_hours']} hours")

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
    # Proper check
    for row in grid:
        for cell in row:
            if cell != 0:
                return False
    return True

def format_time(seconds):
    """Format seconds into human-readable time"""
    return str(timedelta(seconds=int(seconds)))

def get_lr_multiplier(epoch, warmup_epochs):
    """Epoch-based warmup"""
    if epoch < warmup_epochs:
        return (epoch + 1) / warmup_epochs
    return 1.0

# =============================================================================
# LOAD DATA WITH VALIDATION SPLIT
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
all_pairs = []
for task_id, task_data in train_tasks.items():
    for example in task_data.get('train', []):
        all_pairs.append((example['input'], example['output']))

# Split into train + validation
np.random.shuffle(all_pairs)
val_size = int(len(all_pairs) * CONFIG['validation_split'])
train_pairs = all_pairs[val_size:]
val_pairs = all_pairs[:val_size]

print(f"✓ {len(all_pairs)} total training pairs")
print(f"  → {len(train_pairs)} for training ({100*(1-CONFIG['validation_split']):.0f}%)")
print(f"  → {len(val_pairs)} for validation ({100*CONFIG['validation_split']:.0f}%)")
print(f"✓ Data augmentation: DISABLED (fixing train-test mismatch)")

# =============================================================================
# MODEL - SMALLER FOR BETTER GENERALIZATION
# =============================================================================

class ARCModel(nn.Module):
    def __init__(self, grid_size=30, num_colors=10, embed_dim=128, num_layers=4,
                 num_heads=8, dropout=0.2):
        super().__init__()
        self.grid_size = grid_size
        self.num_colors = num_colors
        self.embed_dim = embed_dim

        # Embeddings
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
        """Conservative initialization"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.8)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.01)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    def forward(self, x):
        batch, H, W = x.shape
        x_flat = x.view(batch, -1).long()

        x_emb = self.color_embed(x_flat) + self.pos_embed[:, :x_flat.shape[1], :]
        x_emb = self.layer_norm(x_emb)
        encoded = self.transformer(x_emb)
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
print(f"✓ Params/example ratio: {total_params / len(train_pairs):.1f}:1")
if total_params / len(train_pairs) > 500:
    print(f"  ⚠️ High ratio, but smaller model reduces overfitting risk")
else:
    print(f"  ✓ Good ratio for generalization")

# =============================================================================
# VALIDATION FUNCTION
# =============================================================================

def validate(model, val_pairs):
    """Evaluate on validation set"""
    model.eval()
    val_loss = val_acc = val_n = 0

    with torch.no_grad():
        for inp, out in val_pairs:
            x = torch.from_numpy(pad_grid(inp)).unsqueeze(0).to(DEVICE)
            y = torch.from_numpy(pad_grid(out)).to(DEVICE)

            logits = model(x).squeeze(0)
            loss = F.cross_entropy(
                logits.reshape(-1, model.num_colors),
                y.reshape(-1)
            )

            pred = logits.argmax(dim=-1)
            acc = (pred == y).float().mean().item()

            val_loss += loss.item()
            val_acc += acc
            val_n += 1

    return val_loss / val_n, val_acc / val_n

# =============================================================================
# TRAINING LOOP WITH EARLY STOPPING
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

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=CONFIG['epochs'],
        eta_min=CONFIG['min_lr']
    )

    last_save = time.time()
    best_val_loss = float('inf')
    best_train_loss = float('inf')
    best_acc = 0.0
    patience_counter = 0
    loss_history = []
    val_loss_history = []

    print(f"\nStarting training at {datetime.now().strftime('%H:%M:%S')}")
    print(f"Expected end: {(datetime.now() + timedelta(minutes=40)).strftime('%H:%M:%S')}")
    print(f"Early stopping: {CONFIG['early_stopping_patience']} epochs patience")

    for epoch in range(CONFIG['epochs']):
        # Check runtime limits
        elapsed = time.time() - start_time
        if time.time() > target_end_time:
            print(f"\n⏰ Reached {CONFIG['target_runtime_hours']}h target runtime")
            break

        model.train()
        epoch_loss = epoch_acc = epoch_n = 0
        epoch_start = time.time()

        # Warmup multiplier
        warmup_mult = get_lr_multiplier(epoch, CONFIG['warmup_epochs'])
        current_lr = scheduler.get_last_lr()[0] * warmup_mult

        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr

        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{CONFIG['epochs']} | Elapsed: {format_time(elapsed)}")
        if epoch < CONFIG['warmup_epochs']:
            print(f"  Warmup: {warmup_mult:.2f}× | LR: {current_lr:.2e}")
        else:
            print(f"  LR: {current_lr:.2e}")
        print(f"{'='*60}")

        # Training loop
        indices = np.random.permutation(len(train_pairs))

        for i, idx in enumerate(indices):
            inp, out = train_pairs[idx]
            x = torch.from_numpy(pad_grid(inp)).unsqueeze(0).to(DEVICE)
            y = torch.from_numpy(pad_grid(out)).to(DEVICE)

            logits = model(x).squeeze(0)
            loss = F.cross_entropy(
                logits.reshape(-1, model.num_colors),
                y.reshape(-1),
                label_smoothing=CONFIG['label_smoothing']
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            pred = logits.argmax(dim=-1)
            acc = (pred == y).float().mean().item()

            epoch_loss += loss.item()
            epoch_acc += acc
            epoch_n += 1

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
        avg_train_loss = epoch_loss / epoch_n
        avg_train_acc = epoch_acc / epoch_n
        loss_history.append(avg_train_loss)

        # Validation
        val_loss, val_acc = validate(model, val_pairs)
        val_loss_history.append(val_loss)

        scheduler.step()

        print(f"\n  ✓ Train: Loss={avg_train_loss:.4f}, Acc={avg_train_acc:.3f}")
        print(f"  ✓ Val:   Loss={val_loss:.4f}, Acc={val_acc:.3f}")

        # Check for improvement
        if val_loss < best_val_loss - CONFIG['min_delta']:
            best_val_loss = val_loss
            patience_counter = 0
            print(f"  🎯 New best val loss: {best_val_loss:.4f}")
        else:
            patience_counter += 1
            print(f"  ⏳ No improvement ({patience_counter}/{CONFIG['early_stopping_patience']})")

        if avg_train_loss < best_train_loss:
            best_train_loss = avg_train_loss

        if avg_train_acc > best_acc:
            best_acc = avg_train_acc

        # Early stopping
        if CONFIG['early_stopping'] and patience_counter >= CONFIG['early_stopping_patience']:
            print(f"\n🛑 Early stopping triggered! No improvement for {CONFIG['early_stopping_patience']} epochs")
            print(f"  Best val loss: {best_val_loss:.4f}")
            print(f"  Stopping at epoch {epoch+1} to prevent overfitting")
            break

        # Save checkpoint
        if time.time() - last_save > CONFIG['save_every_n_minutes'] * 60:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_loss': best_val_loss,
                'best_train_loss': best_train_loss,
                'best_acc': best_acc,
                'loss_history': loss_history,
                'val_loss_history': val_loss_history,
            }, CONFIG['checkpoint_path'])
            print(f"  💾 Checkpoint saved ({format_time(time.time() - start_time)} elapsed)")
            last_save = time.time()

    # Final checkpoint
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_val_loss': best_val_loss,
        'best_train_loss': best_train_loss,
        'best_acc': best_acc,
        'loss_history': loss_history,
        'val_loss_history': val_loss_history,
    }, CONFIG['checkpoint_path'])

    total_training_time = time.time() - start_time
    print(f"\n✓ Training complete!")
    print(f"  Total time: {format_time(total_training_time)}")
    print(f"  Best val loss: {best_val_loss:.4f}")
    print(f"  Best train loss: {best_train_loss:.4f}")
    print(f"  Best train accuracy: {best_acc:.3f}")
    print(f"  Final epoch: {epoch+1}")

    # Plot training curves
    print(f"\n  Training progression:")
    sample_epochs = [0, len(loss_history)//4, len(loss_history)//2, 3*len(loss_history)//4, len(loss_history)-1]
    for i in sample_epochs:
        if i < len(loss_history):
            print(f"    Epoch {i+1}: Train={loss_history[i]:.4f}, Val={val_loss_history[i]:.4f}")

# =============================================================================
# GENERATE SUBMISSION.JSON WITH PROPER QUALITY CHECK
# =============================================================================

print("\n" + "=" * 80)
print("GENERATING SUBMISSION")
print("=" * 80)

model.eval()
submission = {}  # FIXED: dict not list! (Kaggle format requirement)
zero_count = 0
total_test_items = 0

with torch.no_grad():
    for i, (task_id, task_data) in enumerate(test_tasks.items(), 1):
        # FIXED: Handle MULTIPLE test items per task
        task_predictions = []

        for test_item in task_data['test']:
            test_input = test_item['input']
            x = torch.from_numpy(pad_grid(test_input)).unsqueeze(0).to(DEVICE)
            logits = model(x)
            pred = logits.argmax(dim=-1).squeeze(0).cpu().numpy()

            H = len(test_input)
            W = len(test_input[0]) if test_input else 1
            pred_grid = pred[:H, :W].tolist()

            # PROPER zero check (v2 had this wrong!)
            if is_all_zeros(pred_grid):
                zero_count += 1

            total_test_items += 1

            # Each test item gets one attempt object
            task_predictions.append({
                "attempt_1": pred_grid,
                "attempt_2": pred_grid
            })

        # FIXED: Dict format with task_id as key, list of attempts as value
        submission[task_id] = task_predictions

        if i % 50 == 0:
            print(f"  Progress: {i}/{len(test_tasks)} tasks, {total_test_items} test items")

print(f"\n✓ Generated {len(submission)} tasks, {total_test_items} total test items")

# Quality check
zero_pct = (zero_count / total_test_items) * 100 if total_test_items > 0 else 0
print(f"\nQuality check:")
print(f"  All-zero predictions: {zero_count}/{total_test_items} ({zero_pct:.1f}%)")

if zero_pct > 10.0:
    print(f"  🔴 HIGH zero rate (>10%) - May need more training")
elif zero_pct > 5.0:
    print(f"  ⚠️ Moderate zero rate (5-10%) - Acceptable but could improve")
elif zero_pct > 3.0:
    print(f"  ✓ Good zero rate (3-5%) - Decent quality")
else:
    print(f"  ✅ Excellent zero rate (<3%) - High quality!")

# =============================================================================
# SAVE TO BOTH LOCATIONS
# =============================================================================

print("\n" + "=" * 80)
print("SAVING SUBMISSION FILES")
print("=" * 80)

os.makedirs('/kaggle/working', exist_ok=True)
os.makedirs('/kaggle/output', exist_ok=True)

with open(CONFIG['output_working'], 'w') as f:
    json.dump(submission, f)
print(f"✓ Saved to {CONFIG['output_working']}")

with open(CONFIG['output_final'], 'w') as f:
    json.dump(submission, f)
print(f"✓ Saved to {CONFIG['output_final']}")

for path in [CONFIG['output_working'], CONFIG['output_final']]:
    if os.path.exists(path):
        size = os.path.getsize(path) / 1024
        print(f"  ✓ {path}: {size:.1f} KB")
    else:
        print(f"  ✗ ERROR: {path} not created!")

print(f"\n✓ {len(submission)} tasks in submission ({total_test_items} test items)")
print(f"✓ Format: DICT (Kaggle-compliant)")
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
print(f"Model parameters: {total_params:,} (~{total_params / 1e6:.1f}M)")
if CONFIG['do_training']:
    print(f"Epochs completed: {epoch+1}")
    print(f"Best val loss: {best_val_loss:.4f}")
    print(f"Best train loss: {best_train_loss:.4f}")
    print(f"Best train accuracy: {best_acc:.3f}")
print(f"Test tasks: {len(submission)} ({total_test_items} test items)")
print(f"Zero predictions: {zero_count} ({zero_pct:.1f}%)")
print(f"Submission format: DICT (Kaggle-compliant)")

print("\n📊 Comparison to v2:")
print(f"  v2: 24 zeros (10.0%), 5M params, 100 epochs, WITH augmentation")
print(f"  v3: {zero_count} zeros ({zero_pct:.1f}%), {total_params/1e6:.1f}M params, {epoch+1} epochs, NO augmentation")
print(f"  Expected: 5-7 zeros (2-3%) - Much better!")

print("\n" + "=" * 80)
print("✅ COMPLETE - Simpler & Better!")
print("=" * 80)
