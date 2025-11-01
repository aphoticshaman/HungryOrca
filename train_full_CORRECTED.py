"""
OrcaSword v3 - CORRECTED: Only outputs submission.json
========================================================

CRITICAL FIX for Kaggle competition submission:
- ONLY creates submission.json (no eval_predictions.json)
- Saves to /kaggle/working/submission.json (required location)
- 240 test tasks in correct format
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import os
import time
from pathlib import Path
from datetime import datetime

print("=" * 80)
print("OrcaSword v3 - CORRECTED Training (submission.json only)")
print("=" * 80)

# =============================================================================
# CONFIGURATION
# =============================================================================

CONFIG = {
    'data_dir': '/kaggle/input/arc-prize-2025',
    'do_training': True,
    'epochs': 50,
    'learning_rate': 1e-4,
    'batch_size': 16,
    'embed_dim': 256,
    'num_layers': 6,
    'grid_size': 30,
    'checkpoint_path': 'orcasword_checkpoint.pt',
    'output_path': '/kaggle/working/submission.json',  # ← CORRECTED PATH
    'target_runtime_hours': 7,
    'save_every_n_minutes': 30,
}

start_time = time.time()
target_end_time = start_time + (CONFIG['target_runtime_hours'] * 3600)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}")

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def pad_grid(grid, max_h=30, max_w=30):
    if not grid or not grid[0]:
        return np.zeros((max_h, max_w), dtype=np.int64)
    h, w = len(grid), len(grid[0])
    h, w = min(h, max_h), min(w, max_w)
    padded = np.zeros((max_h, max_w), dtype=np.int64)
    for i in range(h):
        for j in range(min(w, len(grid[i]))):
            padded[i, j] = int(grid[i][j])
    return padded

# =============================================================================
# LOAD DATA
# =============================================================================

print("\nLoading datasets...")
data_dir = Path(CONFIG['data_dir'])

with open(data_dir / "arc-agi_training_challenges.json") as f:
    train_tasks = json.load(f)
with open(data_dir / "arc-agi_test_challenges.json") as f:
    test_tasks = json.load(f)

print(f"  ✓ {len(train_tasks)} training tasks")
print(f"  ✓ {len(test_tasks)} test tasks")

# Prepare training pairs
train_pairs = []
for task_id, task_data in train_tasks.items():
    for example in task_data.get('train', []):
        train_pairs.append((example['input'], example['output']))

print(f"  ✓ {len(train_pairs)} training pairs")

# =============================================================================
# MODEL
# =============================================================================

class ARCModel(nn.Module):
    def __init__(self, grid_size=30, num_colors=10, embed_dim=256, num_layers=6):
        super().__init__()
        self.grid_size = grid_size
        self.num_colors = num_colors
        self.embed_dim = embed_dim
        
        self.color_embed = nn.Embedding(num_colors, embed_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, grid_size * grid_size, embed_dim) * 0.02)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=8, dim_feedforward=embed_dim * 4,
            batch_first=True, dropout=0.1, activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        self.output_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(embed_dim // 2, num_colors)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)
    
    def forward(self, x):
        batch, H, W = x.shape
        x_flat = x.view(batch, -1).long()
        x_emb = self.color_embed(x_flat) + self.pos_embed[:, :x_flat.shape[1], :]
        encoded = self.transformer(x_emb)
        logits = self.output_head(encoded)
        return logits.view(batch, H, W, self.num_colors)

model = ARCModel(
    grid_size=CONFIG['grid_size'],
    embed_dim=CONFIG['embed_dim'],
    num_layers=CONFIG['num_layers']
).to(DEVICE)

print(f"\nModel: {sum(p.numel() for p in model.parameters()):,} parameters")

# =============================================================================
# TRAINING
# =============================================================================

if CONFIG['do_training']:
    print("\n" + "=" * 80)
    print("TRAINING")
    print("=" * 80)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG['learning_rate'], weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG['epochs'])
    
    last_save = time.time()
    
    for epoch in range(CONFIG['epochs']):
        if time.time() > target_end_time:
            print(f"\n⏰ Reached {CONFIG['target_runtime_hours']}h target")
            break
        
        model.train()
        total_loss = total_acc = n = 0
        
        print(f"\nEpoch {epoch+1}/{CONFIG['epochs']}")
        
        indices = np.random.permutation(len(train_pairs))
        for i, idx in enumerate(indices):
            inp, out = train_pairs[idx]
            x = torch.from_numpy(pad_grid(inp)).unsqueeze(0).to(DEVICE)
            y = torch.from_numpy(pad_grid(out)).to(DEVICE)
            
            logits = model(x).squeeze(0)
            loss = F.cross_entropy(logits.reshape(-1, model.num_colors), y.reshape(-1))
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            pred = logits.argmax(dim=-1)
            acc = (pred == y).float().mean().item()
            
            total_loss += loss.item()
            total_acc += acc
            n += 1
            
            if (i + 1) % 100 == 0:
                print(f"  Step {i+1}: Loss={total_loss/n:.4f}, Acc={total_acc/n:.3f}")
        
        scheduler.step()
        print(f"  Epoch done: Loss={total_loss/n:.4f}, Acc={total_acc/n:.3f}")
        
        # Save checkpoint
        if time.time() - last_save > CONFIG['save_every_n_minutes'] * 60:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, CONFIG['checkpoint_path'])
            print(f"  ✓ Checkpoint saved")
            last_save = time.time()

# =============================================================================
# GENERATE SUBMISSION.JSON (ONLY OUTPUT!)
# =============================================================================

print("\n" + "=" * 80)
print("GENERATING submission.json")
print("=" * 80)

model.eval()
submission = []

with torch.no_grad():
    for i, (task_id, task_data) in enumerate(test_tasks.items(), 1):
        test_input = task_data['test'][0]['input']
        x = torch.from_numpy(pad_grid(test_input)).unsqueeze(0).to(DEVICE)
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
            print(f"  {i}/{len(test_tasks)}")

# SAVE TO /kaggle/working/submission.json (REQUIRED!)
with open(CONFIG['output_path'], 'w') as f:
    json.dump(submission, f)

print(f"\n✓ submission.json saved to {CONFIG['output_path']}")
print(f"✓ {len(submission)} tasks")

# Verify
if os.path.exists(CONFIG['output_path']):
    size = os.path.getsize(CONFIG['output_path']) / 1024
    print(f"✓ File size: {size:.1f} KB")
    print(f"✓ Ready for Kaggle submission!")
else:
    print("✗ ERROR: File not created!")

print("\n" + "=" * 80)
print("COMPLETE")
print("=" * 80)
