"""
OrcaSword v3 - FULL Training Script (6-7 Hour Runtime)
========================================================

This version processes ALL training data and evaluates on eval set.
Designed to run for 6-7 hours with proper evaluation.
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
print("OrcaSword v3 - FULL Training & Evaluation Pipeline")
print("=" * 80)

# =============================================================================
# CONFIGURATION - Edit these for different runtimes
# =============================================================================

CONFIG = {
    # Data paths
    'data_dir': '/kaggle/input/arc-prize-2025',

    # Training - FULL DATASET
    'do_training': True,
    'epochs': 50,                 # More epochs for 6-7 hours
    'learning_rate': 1e-4,
    'batch_size': 16,             # Process multiple samples at once
    'max_samples_per_epoch': None,  # None = use ALL samples (set to number to limit)

    # Model architecture
    'embed_dim': 256,             # Larger for better capacity
    'num_layers': 6,              # Deeper model
    'grid_size': 30,

    # Evaluation
    'eval_every_n_epochs': 2,     # Evaluate every 2 epochs
    'save_eval_predictions': True, # Save eval predictions to file

    # Output
    'checkpoint_path': 'orcasword_full_checkpoint.pt',
    'do_submission': True,
    'output_path': 'submission.json',
    'eval_predictions_path': 'eval_predictions.json',

    # Runtime control
    'target_runtime_hours': 7,    # Stop after this many hours
    'save_every_n_minutes': 30,   # Save checkpoint every 30 min
}

# Calculate target end time
start_time = time.time()
target_end_time = start_time + (CONFIG['target_runtime_hours'] * 3600)

print("\nConfiguration:")
for key, value in CONFIG.items():
    print(f"  {key}: {value}")

print(f"\nTarget runtime: {CONFIG['target_runtime_hours']} hours")
print(f"Target end time: {datetime.fromtimestamp(target_end_time).strftime('%H:%M:%S')}")

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nDevice: {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# =============================================================================
# DATA LOADING
# =============================================================================

def pad_grid(grid, max_h=30, max_w=30):
    """Pad grid to fixed size"""
    if not grid or not grid[0]:
        return np.zeros((max_h, max_w), dtype=np.int64)

    h, w = len(grid), len(grid[0])
    h = min(h, max_h)
    w = min(w, max_w)

    padded = np.zeros((max_h, max_w), dtype=np.int64)
    for i in range(h):
        row_len = min(w, len(grid[i]))
        for j in range(row_len):
            padded[i, j] = int(grid[i][j])

    return padded

def calculate_accuracy(pred, target):
    """Calculate pixel-level accuracy"""
    return (pred == target).sum() / target.size

print("\n" + "=" * 80)
print("Loading ARC datasets...")
print("=" * 80)

data_dir = Path(CONFIG['data_dir'])

# Load training data
with open(data_dir / "arc-agi_training_challenges.json", 'r') as f:
    train_tasks = json.load(f)
with open(data_dir / "arc-agi_training_solutions.json", 'r') as f:
    train_solutions = json.load(f)
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

# Prepare ALL training pairs (no limit!)
print("\nPreparing training data...")
train_pairs = []
for task_id, task_data in train_tasks.items():
    # Use ALL training examples from each task
    for example in task_data.get('train', []):
        train_pairs.append((task_id, example['input'], example['output']))

print(f"  ✓ Total training pairs: {len(train_pairs)}")

# Prepare eval pairs
eval_pairs = []
for task_id, task_data in eval_tasks.items():
    if task_id in eval_solutions:
        test_input = task_data['test'][0]['input']
        solution = eval_solutions[task_id][0]
        eval_pairs.append((task_id, test_input, solution))

print(f"  ✓ Total eval pairs: {len(eval_pairs)}")

# =============================================================================
# MODEL
# =============================================================================

class ARCModel(nn.Module):
    """ARC solver with proper capacity"""

    def __init__(self, grid_size=30, num_colors=10, embed_dim=256, num_layers=6):
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
        x_flat = x.view(batch, -1).long()
        x_emb = self.color_embed(x_flat)
        seq_len = x_flat.shape[1]
        x_emb = x_emb + self.pos_embed[:, :seq_len, :]
        encoded = self.transformer(x_emb)
        logits = self.output_head(encoded)
        logits = logits.view(batch, H, W, self.num_colors)
        return logits

print("\n" + "=" * 80)
print("Initializing model...")
print("=" * 80)

model = ARCModel(
    grid_size=CONFIG['grid_size'],
    embed_dim=CONFIG['embed_dim'],
    num_layers=CONFIG['num_layers']
).to(DEVICE)

total_params = sum(p.numel() for p in model.parameters())
print(f"  Total parameters: {total_params:,}")
print(f"  Model size: ~{total_params * 4 / 1e6:.1f} MB (float32)")

# =============================================================================
# EVALUATION FUNCTION
# =============================================================================

def evaluate_on_eval_set(model, eval_pairs, save_predictions=False):
    """Evaluate model on eval set and optionally save predictions"""
    print("\n" + "-" * 40)
    print("Evaluating on eval set...")
    print("-" * 40)

    model.eval()
    total_acc = 0.0
    predictions = []

    with torch.no_grad():
        for i, (task_id, test_input, solution) in enumerate(eval_pairs):
            # Prepare data
            x = torch.from_numpy(pad_grid(test_input)).unsqueeze(0).to(DEVICE)
            y = pad_grid(solution)

            # Predict
            logits = model(x)
            pred = logits.argmax(dim=-1).squeeze(0).cpu().numpy()

            # Calculate accuracy
            acc = calculate_accuracy(pred, y)
            total_acc += acc

            if save_predictions:
                # Get original size
                H, W = len(test_input), len(test_input[0]) if test_input else 1
                pred_grid = pred[:H, :W].tolist()
                predictions.append({
                    'task_id': task_id,
                    'prediction': pred_grid,
                    'accuracy': float(acc)
                })

            if (i + 1) % 20 == 0:
                print(f"  Progress: {i+1}/{len(eval_pairs)} (avg acc: {total_acc/(i+1):.3f})")

    avg_acc = total_acc / len(eval_pairs)
    print(f"\n  ✓ Eval accuracy: {avg_acc:.4f}")

    if save_predictions:
        with open(CONFIG['eval_predictions_path'], 'w') as f:
            json.dump(predictions, f, indent=2)
        print(f"  ✓ Saved predictions to {CONFIG['eval_predictions_path']}")

    model.train()
    return avg_acc, predictions

# =============================================================================
# TRAINING
# =============================================================================

if CONFIG['do_training']:
    print("\n" + "=" * 80)
    print("STARTING FULL TRAINING")
    print("=" * 80)

    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG['learning_rate'], weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG['epochs'])

    best_eval_acc = 0.0
    last_save_time = time.time()

    # Determine samples per epoch
    samples_per_epoch = CONFIG['max_samples_per_epoch'] if CONFIG['max_samples_per_epoch'] else len(train_pairs)
    print(f"\nSamples per epoch: {samples_per_epoch}")
    print(f"Total training steps: {samples_per_epoch * CONFIG['epochs']:,}")

    for epoch in range(CONFIG['epochs']):
        # Check if we've exceeded target runtime
        if time.time() > target_end_time:
            print(f"\n⏰ Reached target runtime of {CONFIG['target_runtime_hours']} hours")
            print("Stopping training...")
            break

        epoch_start = time.time()
        model.train()

        print(f"\n{'='*80}")
        print(f"Epoch {epoch+1}/{CONFIG['epochs']}")
        print(f"Elapsed: {(time.time() - start_time)/3600:.2f}h / {CONFIG['target_runtime_hours']}h")
        print(f"{'='*80}")

        total_loss = 0.0
        total_acc = 0.0
        n_samples = 0

        # Shuffle training data
        indices = np.random.permutation(len(train_pairs))[:samples_per_epoch]

        for i, idx in enumerate(indices):
            task_id, input_grid, output_grid = train_pairs[idx]

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

            # Progress updates
            if (i + 1) % 100 == 0:
                avg_loss = total_loss / n_samples
                avg_acc = total_acc / n_samples
                elapsed = time.time() - epoch_start
                samples_per_sec = n_samples / elapsed
                eta_seconds = (samples_per_epoch - n_samples) / samples_per_sec

                print(f"  Step {i+1}/{samples_per_epoch}: "
                      f"Loss={avg_loss:.4f}, Acc={avg_acc:.3f}, "
                      f"Speed={samples_per_sec:.1f} samples/s, "
                      f"ETA={eta_seconds/60:.1f}min")

        scheduler.step()

        # Epoch summary
        avg_loss = total_loss / n_samples
        avg_acc = total_acc / n_samples
        epoch_time = time.time() - epoch_start

        print(f"\n  Epoch Summary:")
        print(f"    Loss: {avg_loss:.4f}")
        print(f"    Accuracy: {avg_acc:.3f}")
        print(f"    Time: {epoch_time/60:.1f} minutes")
        print(f"    Total elapsed: {(time.time() - start_time)/3600:.2f} hours")

        # Save checkpoint periodically
        if time.time() - last_save_time > CONFIG['save_every_n_minutes'] * 60:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'accuracy': avg_acc,
            }, CONFIG['checkpoint_path'])
            print(f"  ✓ Saved checkpoint (elapsed: {(time.time()-start_time)/3600:.2f}h)")
            last_save_time = time.time()

        # Evaluate on eval set
        if (epoch + 1) % CONFIG['eval_every_n_epochs'] == 0:
            eval_acc, _ = evaluate_on_eval_set(
                model,
                eval_pairs,
                save_predictions=CONFIG['save_eval_predictions']
            )

            if eval_acc > best_eval_acc:
                best_eval_acc = eval_acc
                torch.save(model.state_dict(), 'best_model.pt')
                print(f"  ✓ New best eval accuracy: {eval_acc:.4f}")

    # Final save
    torch.save(model.state_dict(), CONFIG['checkpoint_path'])
    print("\n" + "=" * 80)
    print("Training complete!")
    print(f"Total time: {(time.time() - start_time)/3600:.2f} hours")
    print(f"Best eval accuracy: {best_eval_acc:.4f}")
    print("=" * 80)

# =============================================================================
# FINAL EVALUATION ON EVAL SET
# =============================================================================

print("\n" + "=" * 80)
print("FINAL EVALUATION ON EVAL SET")
print("=" * 80)

# Load best model if available
if os.path.exists('best_model.pt'):
    model.load_state_dict(torch.load('best_model.pt', map_location=DEVICE))
    print("  ✓ Loaded best model")

final_eval_acc, eval_predictions = evaluate_on_eval_set(
    model,
    eval_pairs,
    save_predictions=True
)

print(f"\n  Final eval accuracy: {final_eval_acc:.4f}")

# =============================================================================
# GENERATE TEST SUBMISSION
# =============================================================================

if CONFIG['do_submission']:
    print("\n" + "=" * 80)
    print("GENERATING TEST SUBMISSION")
    print("=" * 80)

    model.eval()
    submission = []

    with torch.no_grad():
        for i, (task_id, task_data) in enumerate(test_tasks.items(), 1):
            test_examples = task_data.get('test', [])
            if not test_examples:
                continue

            test_input = test_examples[0]['input']
            padded_input = pad_grid(test_input)

            x = torch.from_numpy(padded_input).unsqueeze(0).to(DEVICE)
            logits = model(x)
            pred = logits.argmax(dim=-1).squeeze(0).cpu().numpy()

            H, W = len(test_input), len(test_input[0]) if test_input else 1
            pred_grid = pred[:H, :W].tolist()

            submission.append({
                "task_id": task_id,
                "attempt_1": pred_grid,
                "attempt_2": pred_grid
            })

            if i % 50 == 0:
                print(f"  Progress: {i}/{len(test_tasks)}")

    with open(CONFIG['output_path'], 'w') as f:
        json.dump(submission, f)

    print(f"\n  ✓ Submission saved: {CONFIG['output_path']}")
    print(f"  ✓ Total tasks: {len(submission)}")

print("\n" + "=" * 80)
print("ALL COMPLETE! 🎉")
print("=" * 80)
print(f"\nTotal runtime: {(time.time() - start_time)/3600:.2f} hours")
print(f"\nFiles generated:")
print(f"  - {CONFIG['output_path']} (test submission)")
print(f"  - {CONFIG['eval_predictions_path']} (eval predictions)")
print(f"  - {CONFIG['checkpoint_path']} (model checkpoint)")
if os.path.exists('best_model.pt'):
    print(f"  - best_model.pt (best eval model)")
print("\n" + "=" * 80)
