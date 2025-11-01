# OrcaSword v3 - Training & Submission Guide

Complete guide for training OrcaSword v3 and generating submissions for ARC Prize 2025.

## Quick Start (Kaggle)

### Option 1: Simple Training Script (Recommended)

```python
# In Kaggle Notebook
!python train_and_submit.py \
    --data_dir /kaggle/input/arc-prize-2025 \
    --train \
    --epochs 10 \
    --submit \
    --output submission.json
```

### Option 2: Step-by-Step

#### 1. Training Only
```bash
python train_and_submit.py \
    --data_dir /kaggle/input/arc-prize-2025 \
    --train \
    --epochs 20 \
    --lr 1e-4 \
    --checkpoint orcasword_v3.pt
```

#### 2. Generate Submission Only
```bash
python train_and_submit.py \
    --data_dir /kaggle/input/arc-prize-2025 \
    --checkpoint orcasword_v3.pt \
    --submit \
    --output submission.json
```

#### 3. Both Training and Submission
```bash
python train_and_submit.py \
    --data_dir /kaggle/input/arc-prize-2025 \
    --train \
    --epochs 10 \
    --submit
```

---

## Configuration Options

### Model Architecture
- `--embed_dim`: Embedding dimension (default: 128, range: 64-512)
- `--num_layers`: Transformer layers (default: 4, range: 2-12)

### Training
- `--train`: Enable training mode
- `--epochs`: Number of training epochs (default: 10)
- `--lr`: Learning rate (default: 1e-4)
- `--checkpoint`: Checkpoint file path (default: orcasword_checkpoint.pt)

### Submission
- `--submit`: Generate submission.json
- `--output`: Output file path (default: submission.json)
- `--data_dir`: Path to ARC Prize 2025 data

---

## Expected Runtime

### On Kaggle GPU (T4/P100):
- **Training** (10 epochs): ~30-45 minutes
- **Submission generation**: ~2-5 minutes
- **Total**: <1 hour

### On CPU:
- **Training** (10 epochs): ~2-3 hours
- **Submission generation**: ~5-10 minutes

---

## File Structure

```
/kaggle/input/arc-prize-2025/
├── arc-agi_training_challenges.json
├── arc-agi_training_solutions.json
├── arc-agi_evaluation_challenges.json
├── arc-agi_evaluation_solutions.json
└── arc-agi_test_challenges.json

/kaggle/working/
├── train_and_submit.py          # Main training script
├── orcaswordv3.py               # Core model code
├── orcasword_checkpoint.pt      # Saved model (after training)
└── submission.json               # Final submission
```

---

## Submission Format

The `submission.json` follows ARC Prize 2025 format:

```json
[
  {
    "task_id": "00576224",
    "attempt_1": [[0, 1, 2], [3, 4, 5]],
    "attempt_2": [[0, 1, 2], [3, 4, 5]]
  },
  ...
]
```

---

## Performance Metrics

During training, you'll see:

```
Epoch 1/10
----------------------------------------
  Step 20/400: Loss=1.2345, Acc=0.456
  ...
  Train Loss: 1.1234, Train Acc: 0.567
  Eval Acc: 0.234
  ✓ New best accuracy: 0.234
```

- **Loss**: Cross-entropy loss (lower is better)
- **Acc**: Pixel-level accuracy (0-1, higher is better)
- **Eval Acc**: Accuracy on evaluation set

---

## Advanced Usage

### Custom Model Size

For more capacity (if you have GPU memory):
```bash
python train_and_submit.py \
    --embed_dim 256 \
    --num_layers 8 \
    --epochs 20 \
    --train \
    --submit
```

### Fast Prototype (CPU-friendly):
```bash
python train_and_submit.py \
    --embed_dim 64 \
    --num_layers 2 \
    --epochs 5 \
    --train \
    --submit
```

---

## Troubleshooting

### Issue: CUDA Out of Memory
**Solution**: Reduce model size
```bash
--embed_dim 64 --num_layers 2
```

### Issue: Training too slow
**Solution**: Reduce epochs or use GPU
```bash
--epochs 5
```

### Issue: Low accuracy
**Solution**: Train longer with larger model
```bash
--epochs 30 --embed_dim 256 --num_layers 6
```

### Issue: Submission file too large
**Solution**: Normal - ARC grids are small, file should be <10MB

---

## Integration with Full OrcaSword v3

The `train_and_submit.py` uses a **simplified** model for speed. To use the full OrcaSword v3 with all features:

1. Import from `orcaswordv3.py`:
```python
from orcaswordv3 import OrcaSwordV3Model, ProductionPipeline

model = OrcaSwordV3Model(config)
pipeline = ProductionPipeline(model, config)
pipeline.solve_arc_dataset(dataset_path, "submission.json")
```

2. Note: Full model is slower but more accurate (uses all 38 theorems, full Phi lattice, etc.)

---

## Validation Before Submission

Always validate your submission:

```python
import json

with open('submission.json', 'r') as f:
    submission = json.load(f)

print(f"Total tasks: {len(submission)}")
print(f"First task: {submission[0]['task_id']}")

# Check format
assert all('task_id' in task for task in submission)
assert all('attempt_1' in task for task in submission)
assert all('attempt_2' in task for task in submission)

print("✓ Submission format valid!")
```

---

## Tips for Best Results

1. **Train longer**: More epochs = better accuracy
2. **Use GPU**: 10-20x faster than CPU
3. **Monitor eval accuracy**: Stop when it plateaus
4. **Save checkpoints**: Resume training if interrupted
5. **Test on evaluation set**: Predict your score before submitting

---

## Expected Scores

Based on baseline testing:

- **After 5 epochs**: ~10-15% accuracy
- **After 10 epochs**: ~20-30% accuracy
- **After 20 epochs**: ~30-40% accuracy
- **Full OrcaSword v3**: Target >50% accuracy

Note: These are pixel-level accuracies. Task-level accuracy will be lower.

---

## Questions?

- Check logs for errors
- Ensure data paths are correct
- Verify GPU is enabled (if using Kaggle)
- Review checkpoint files are saving

Good luck! 🚀
