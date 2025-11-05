# ARC Prize 2025 - Visual Testing Interface

## 🎨 Interactive Task Viewer & Solver

This is the **official ARC testing interface** from [fchollet/ARC-AGI](https://github.com/fchollet/ARC-AGI), adapted to work with the **ARC Prize 2025 dataset** stored locally.

### Features

✅ **Browse all ARC tasks** from Training, Evaluation, and Test datasets
✅ **Visual grid interface** with color palette and drawing tools
✅ **Study training examples** to understand task patterns
✅ **Try solving test cases manually** to build intuition
✅ **Immediate feedback** when you submit your solution

### Why Use This?

Before building AI solvers, it's critical to:
1. **Manually solve tasks** to understand what reasoning is required
2. **Identify task categories** (object manipulation, pattern completion, etc.)
3. **Build intuition** for what makes tasks easy vs. hard
4. **Debug solver outputs** by comparing to manual solutions

Many AI researchers underestimate this step and build solvers that can't even match human intuition!

## 🚀 Quick Start

### 1. Run the server

```bash
cd /home/user/HungryOrca
python3 run_arc_interface.py
```

The interface will automatically open in your browser at: **http://localhost:8000**

### 2. Select a dataset

- **Training** (400 tasks): Full training set with solutions available
- **Evaluation** (400 tasks): Public evaluation set (semi-private solutions)
- **Test** (Competition): The actual competition test set

### 3. Browse and select a task

The interface will show all task IDs. Each shows:
- Task ID (e.g., `00576224`)
- Number of training examples
- Number of test cases

Click on a task to select it, then click **"Start Selected Task"**.

### 4. Study the training examples

The left panel shows training examples:
- **Input** → **Output** pairs
- Study the pattern: What transformation is being applied?

### 5. Solve the test case

The right panel shows:
- **Test input**: The grid you need to transform
- **Output editor**: Draw your solution here

Tools:
- **Edit mode**: Click to draw pixels one at a time
- **Select mode**: Click and drag to select rectangular regions, then copy/paste
- **Flood fill mode**: Click to fill all connected cells of the same color

### 6. Submit your solution

Click **"Submit!"** to check if your solution is correct.

## 🎨 Color Palette

ARC uses 10 colors (0-9):

| Number | Color | RGB |
|--------|-------|-----|
| 0 | Black | Background |
| 1 | Blue | #0074D9 |
| 2 | Red | #FF4136 |
| 3 | Green | #2ECC40 |
| 4 | Yellow | #FFDC00 |
| 5 | Gray | #AAAAAA |
| 6 | Magenta | #F012BE |
| 7 | Orange | #FF851B |
| 8 | Sky Blue | #7FDBFF |
| 9 | Brown | #870C25 |

## 💡 Pro Tips

### Understanding Task Patterns

As you solve tasks, look for these common patterns:

**L1 (Pixel-level):**
- Rotation (90°, 180°, 270°)
- Reflection (horizontal, vertical, diagonal)
- Scaling (2x, 3x, etc.)
- Cropping

**L2 (Object-level):**
- Moving objects
- Recoloring objects
- Scaling individual objects
- Deleting certain objects
- Duplicating objects

**L3 (Pattern-level):**
- "Keep only the largest object"
- "Arrange objects in a grid"
- "Fill the background with a pattern"
- "Color objects based on their size"
- "Connect objects with lines"

**L4 (Abstract rules):**
- Count-based rules ("repeat N times")
- Conditional rules ("if object is red, then...")
- Compositional rules ("apply rule A, then rule B")

### Common Pitfalls

❌ **Don't assume symmetry!** Many tasks look symmetric but have subtle asymmetries.

❌ **Don't assume fixed grid size!** Output can be larger, smaller, or same size as input.

❌ **Don't assume color preservation!** Objects may change colors as part of the transformation.

✅ **Do study ALL training examples!** The pattern must hold across all examples.

✅ **Do look for object-level patterns first!** Most tasks involve object manipulation, not pixel operations.

✅ **Do consider spatial relationships!** Position, size, and neighbor relationships matter.

## 🧠 Building Better AI Solvers

After manually solving 20-50 tasks, you'll notice:

1. **Object decomposition is critical**: Most tasks require identifying distinct objects first
2. **Random mutation won't work**: You need structured reasoning (L1 → L2 → L3 → L4)
3. **Pattern inference beats evolution**: Learning explicit rules from training examples is better than evolving operation sequences
4. **Multi-stage reasoning**: Successful solvers use hierarchical reasoning, not flat operation lists

This is why `arc_multi_stage_reasoner.py` (in this repo) outperforms random evolution approaches!

## 📁 Files

```
/home/user/HungryOrca/
├── arc_testing_interface/          # Visual interface files
│   ├── local_data_loader.html      # Main interface (loads local data)
│   ├── testing_interface.html      # Original interface (GitHub loader)
│   ├── css/                        # Styles
│   ├── js/                         # JavaScript
│   └── img/                        # Images
├── run_arc_interface.py            # Server script (run this!)
├── arc-agi_training_challenges.json    # Training data
├── arc-agi_evaluation_challenges.json  # Evaluation data
└── arc-agi_test_challenges.json        # Test data
```

## 🔧 Troubleshooting

### Port already in use

If port 8000 is taken, edit `run_arc_interface.py` and change:
```python
PORT = 8000  # Change to 8001, 8002, etc.
```

### Data files not found

Make sure you're running from the `/home/user/HungryOrca` directory:
```bash
cd /home/user/HungryOrca
python3 run_arc_interface.py
```

### Browser doesn't open automatically

Manually open: **http://localhost:8000/local_data_loader.html**

## 🎯 Challenge

Try to manually solve 10 tasks from the training set. Track your accuracy and time per task.

Most humans achieve:
- **Easy tasks**: 100% accuracy, 1-5 minutes
- **Medium tasks**: 70-90% accuracy, 5-15 minutes
- **Hard tasks**: 30-60% accuracy, 15-30+ minutes

Can you beat the current SOTA AI (20-40% overall accuracy)?

## 📚 References

- [ARC Prize 2025](https://arcprize.org/)
- [ARC-AGI Dataset](https://github.com/fchollet/ARC-AGI)
- [Original Testing Interface](https://github.com/fchollet/ARC-AGI/tree/master/apps)
- [François Chollet's Paper](https://arxiv.org/abs/1911.01547)

---

**Happy Solving!** 🧩🔍🤖
