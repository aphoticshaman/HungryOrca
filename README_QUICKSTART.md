# OrcaSword v3 - Quick Start Guide

Complete ARC Prize 2025 solver with training and submission pipeline.

## 🎯 What You Get

- **orcaswordv3.py** - Full mathematical solver (2,629 lines, 38 theorems)
- **train_and_submit.py** - Training & submission script (works in <1 hour)
- **kaggle_example.ipynb** - Ready-to-run Kaggle notebook
- **TRAINING_GUIDE.md** - Complete documentation

## 🚀 Fastest Way to Submit

### On Kaggle (Recommended)

1. **Create new Kaggle notebook**
2. **Add dataset**: "ARC Prize 2025"
3. **Enable GPU accelerator**
4. **Upload file**: `train_and_submit.py`
5. **Run one command**:

```python
!python train_and_submit.py --data_dir /kaggle/input/arc-prize-2025 --train --epochs 10 --submit
```

6. **Download**: `submission.json` from output
7. **Submit** to competition!

**Time**: ~45 minutes on GPU

---

## 📋 Three Ways to Use

### Method 1: Single Command (Easiest)

```bash
python train_and_submit.py \
    --data_dir /kaggle/input/arc-prize-2025 \
    --train \
    --epochs 10 \
    --submit
```

**Output**: `submission.json` ready to submit

### Method 2: Kaggle Notebook (Interactive)

1. Upload `kaggle_example.ipynb`
2. Run all cells
3. Download submission

### Method 3: Full OrcaSword v3 (Advanced)

```python
from orcaswordv3 import OrcaSwordV3Model, ProductionPipeline

model = OrcaSwordV3Model(config)
pipeline = ProductionPipeline(model, config)
pipeline.solve_arc_dataset("test.json", "submission.json")
```

**Features**: All 38 theorems, full Phi lattice, program synthesis, etc.

---

## 📁 Files Explained

| File | Purpose | Size | When to Use |
|------|---------|------|-------------|
| `train_and_submit.py` | Complete pipeline | 500 lines | Production submission |
| `orcaswordv3.py` | Full mathematical solver | 2,629 lines | Research/advanced |
| `kaggle_example.ipynb` | Interactive tutorial | 10 cells | Learning/testing |
| `TRAINING_GUIDE.md` | Full documentation | - | Reference |

---

## ⚙️ Configuration

### Quick Test (5 minutes)
```bash
--epochs 2 --embed_dim 64 --num_layers 2
```

### Balanced (30-45 minutes)
```bash
--epochs 10 --embed_dim 128 --num_layers 4
```

### High Accuracy (2-3 hours)
```bash
--epochs 30 --embed_dim 256 --num_layers 8
```

---

## 📊 Expected Results

| Epochs | Time (GPU) | Accuracy |
|--------|-----------|----------|
| 5 | ~20 min | 10-15% |
| 10 | ~45 min | 20-30% |
| 20 | ~90 min | 30-40% |
| 50 | ~3 hours | 40-50% |

*Accuracy = pixel-level on evaluation set*

---

## 🔧 Troubleshooting

### "CUDA Out of Memory"
```bash
--embed_dim 64 --num_layers 2
```

### "Training too slow"
- Enable GPU in Kaggle
- Reduce epochs: `--epochs 5`

### "Low accuracy"
- Train longer: `--epochs 30`
- Bigger model: `--embed_dim 256`

### "No submission.json"
- Add `--submit` flag
- Check for errors in training logs

---

## 📖 Detailed Documentation

- **TRAINING_GUIDE.md** - Complete training guide
- **orcaswordv3.py** - See docstrings for each class/function
- Run `python train_and_submit.py --help` for all options

---

## 🎓 Understanding the Code

### Training Pipeline (`train_and_submit.py`)
1. **Load data**: ARC training/eval/test sets
2. **Train model**: Transformer with cross-entropy loss
3. **Evaluate**: Track accuracy on validation set
4. **Save checkpoint**: Resume training anytime
5. **Generate submission**: Predict on test set

### Full Solver (`orcaswordv3.py`)
- **Cell 1**: Mathematical foundations (fuzzy logic, information theory)
- **Cell 2**: Full Phi partition lattice (IIT)
- **Cell 3**: Hierarchical abstraction (category theory)
- **Cell 4**: Program synthesis (50+ operations)
- **Cell 5**: Testing & fallacy detection

---

## 💡 Tips for Best Results

1. ✅ **Use GPU** - 10-20x faster
2. ✅ **Train longer** - More epochs = better accuracy
3. ✅ **Monitor logs** - Watch eval accuracy
4. ✅ **Save checkpoints** - Resume if interrupted
5. ✅ **Validate submission** - Check format before submitting

---

## 🏆 Competition Submission

1. **Train** your model
2. **Generate** submission.json
3. **Validate** format (see TRAINING_GUIDE.md)
4. **Submit** to ARC Prize 2025
5. **Check** leaderboard score
6. **Iterate** and improve!

---

## 📞 Need Help?

- Check **TRAINING_GUIDE.md** for detailed instructions
- Review **kaggle_example.ipynb** for step-by-step tutorial
- Run with `--help` flag for command options
- Check error logs for specific issues

---

## 🎉 Quick Win

**Just want a submission.json?**

```bash
# 1. Upload to Kaggle
# 2. Run this:
!python train_and_submit.py --data_dir /kaggle/input/arc-prize-2025 --train --epochs 5 --submit

# 3. Download submission.json
# 4. Submit to competition!
```

**Time**: ~20 minutes

Good luck! 🚀
