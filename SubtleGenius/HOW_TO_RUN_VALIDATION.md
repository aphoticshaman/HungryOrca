# How to Run Validation Harness - For Humans
## Dead Simple Guide (No Technical BS)

**Goal:** Find out if your symmetry predictions are actually correct (66% vs 18%)

**Time:** 5-10 minutes

**Difficulty:** Easy (copy-paste)

---

## 🎯 THE SIMPLE ANSWER

**You DON'T upload validation_harness.py to Kaggle.**

**You run it LOCALLY on your computer** (or in a Colab notebook if you don't have Python locally).

**Why?** Because validation harness needs **training data** (which has the correct answers). Kaggle doesn't give you training data access in competition notebooks.

---

## 📋 OPTION 1: Run Locally (If You Have Python)

### Step 1: Get the Training Data
```
Go to: https://www.kaggle.com/competitions/arc-prize-2025/data
Download: arc-agi_training_challenges.json
Save to: HungryOrca/SubtleGenius/data/
```

### Step 2: Copy the Validation Harness
```
The file is already in your repo:
HungryOrca/SubtleGenius/infrastructure/validation_harness.py

Just make sure it's there.
```

### Step 3: Create a Test Script
Create a new file: `HungryOrca/SubtleGenius/run_validation.py`

Paste this:
```python
# Run validation harness on training data
import sys
sys.path.append('infrastructure')
sys.path.append('notebooks')

from validation_harness import ValidationHarness

# Your v5-Lite solver (we need to extract it)
# For now, let's just validate on a simple solver

print("🔍 Running validation harness...")
print("This will tell us if symmetry is actually accurate!")

harness = ValidationHarness('data/arc-agi_training_challenges.json')

# We'll need to build a wrapper for your v5-Lite solver
# Coming in next step...
```

### Step 4: Wait, This is Getting Complicated...

**STOP.** This approach requires extracting your solver from the notebook, which is annoying.

**Let me give you the EASIER way...**

---

## 📋 OPTION 2: Quick Analysis (Easiest - Do This First)

**Instead of running the full validation harness, let's do a QUICK CHECK:**

### Step 1: Pick a Few Training Tasks Manually
```
Go to: https://www.kaggle.com/competitions/arc-prize-2025/data
Download: arc-agi_training_challenges.json
Open it in a text editor or online JSON viewer
```

### Step 2: Find Tasks with Symmetry
Look for a task where the output looks symmetric.

Example: Task "007bbfb7" (just search for it in the file)

You'll see:
- Input grid
- Output grid (the CORRECT answer)

### Step 3: Test Your Symmetry Logic
**Does your symmetry detection say:**
- "This output has 60-95% horizontal symmetry" when it ACTUALLY DOES?
- OR does it detect symmetry when there ISN'T any?

### Step 4: Manual Spot Check (5-10 tasks)
Pick 10 random training tasks.
For each one:
- Look at the output
- Ask: "Would my symmetry detector (60-95% similar) trigger here?"
- Ask: "Is the actual answer symmetric?"
- Count how many times you're right vs wrong

**Rough accuracy estimate in 10 minutes!**

---

## 📋 OPTION 3: Google Colab (Recommended for Non-Programmers)

**This is the EASIEST way to actually run code.**

### Step 1: Go to Google Colab
```
https://colab.research.google.com/
Click "New Notebook"
```

### Step 2: Upload Training Data
```
In Colab sidebar, click the folder icon
Upload: arc-agi_training_challenges.json
```

### Step 3: Paste This Entire Code Block

I'll create a self-contained validation script you can just paste...

---

## 🚀 ULTRA-SIMPLE COLAB SCRIPT (Copy This Whole Thing)

Create a new file: `validation_colab.py` and I'll make it COMPLETE:

```python
# ========================================
# ULTRA-SIMPLE VALIDATION SCRIPT
# Paste this into Google Colab
# ========================================

import json
import numpy as np
from collections import defaultdict

print("🐋 Symmetry Validation - v5-Lite Edition")
print("=" * 60)

# === STEP 1: Load Training Data ===
print("\n📂 Loading training data...")

# Upload arc-agi_training_challenges.json to Colab first!
with open('arc-agi_training_challenges.json', 'r') as f:
    training_data = json.load(f)

print(f"✅ Loaded {len(training_data)} training tasks")

# === STEP 2: Copy Your Symmetry Detection Logic ===
def detect_symmetry(grid):
    """
    This is YOUR symmetry detection from v5-Lite
    (copied from the notebook)
    """
    arr = np.array(grid)
    h, w = arr.shape

    # Horizontal symmetry
    left = arr[:, :w//2]
    right = np.fliplr(arr[:, w//2:])
    if left.shape == right.shape:
        score = np.mean(left == right)
        if 0.6 <= score < 0.95:
            return 'h'

    # Vertical symmetry
    top = arr[:h//2, :]
    bottom = np.flipud(arr[h//2:, :])
    if top.shape == bottom.shape:
        score = np.mean(top == bottom)
        if 0.6 <= score < 0.95:
            return 'v'

    return None

def complete_symmetry(grid, sym_type):
    """Complete partial symmetry"""
    arr = np.array(grid)
    h, w = arr.shape

    if sym_type == 'h':
        left = arr[:, :w//2]
        return np.hstack([left, np.fliplr(left)]).tolist()
    elif sym_type == 'v':
        top = arr[:h//2, :]
        return np.vstack([top, np.flipud(top)]).tolist()

    return grid

# === STEP 3: Test on Training Data ===
print("\n🔍 Testing symmetry detection...")

results = {
    'total_tasks': 0,
    'symmetry_detected': 0,
    'correct': 0,
    'incorrect': 0,
    'details': []
}

for task_id, task_data in list(training_data.items())[:50]:  # Test first 50 tasks
    results['total_tasks'] += 1

    # Get the FIRST test case (we have the answer in training data)
    if 'test' not in task_data or len(task_data['test']) == 0:
        continue

    test_input = task_data['test'][0]['input']
    correct_output = task_data['test'][0]['output']

    # Try symmetry detection
    sym_type = detect_symmetry(test_input)

    if sym_type:
        results['symmetry_detected'] += 1

        # Apply symmetry completion
        predicted_output = complete_symmetry(test_input, sym_type)

        # Check if prediction matches correct answer
        if np.array_equal(np.array(predicted_output), np.array(correct_output)):
            results['correct'] += 1
            results['details'].append({
                'task_id': task_id,
                'sym_type': sym_type,
                'result': 'CORRECT ✅'
            })
        else:
            results['incorrect'] += 1
            results['details'].append({
                'task_id': task_id,
                'sym_type': sym_type,
                'result': 'WRONG ❌'
            })

# === STEP 4: Calculate Accuracy ===
print("\n" + "=" * 60)
print("📊 VALIDATION RESULTS")
print("=" * 60)

print(f"\nTotal tasks tested: {results['total_tasks']}")
print(f"Symmetry detected:  {results['symmetry_detected']} ({results['symmetry_detected']/results['total_tasks']*100:.1f}%)")

if results['symmetry_detected'] > 0:
    accuracy = results['correct'] / results['symmetry_detected']
    print(f"\nWhen symmetry detected:")
    print(f"  Correct:   {results['correct']}")
    print(f"  Incorrect: {results['incorrect']}")
    print(f"  ACCURACY:  {accuracy*100:.1f}%")

    # Calculate contribution
    coverage = results['symmetry_detected'] / results['total_tasks']
    contribution = coverage * accuracy
    print(f"\nContribution to score:")
    print(f"  Coverage:      {coverage*100:.1f}%")
    print(f"  Accuracy:      {accuracy*100:.1f}%")
    print(f"  CONTRIBUTION:  {contribution*100:.1f}%")

    # Predict full score
    if contribution > 0:
        print(f"\n🎯 PREDICTED KAGGLE SCORE (symmetry only):")
        print(f"   {contribution*100:.0f}% - {(contribution*100)+5:.0f}%")

print("\n" + "=" * 60)
print("✅ VALIDATION COMPLETE")
print("=" * 60)

# Show a few examples
print("\nFirst 5 detections:")
for detail in results['details'][:5]:
    print(f"  {detail['task_id']}: {detail['sym_type']} → {detail['result']}")
```

---

## 📝 STEP-BY-STEP COLAB INSTRUCTIONS

### 1. Open Google Colab
```
Go to: https://colab.research.google.com/
Click: New Notebook
```

### 2. Upload Training Data
```
Click folder icon on left sidebar
Click upload button
Upload: arc-agi_training_challenges.json
(Download from Kaggle first if you don't have it)
```

### 3. Paste the Script
```
Copy the ENTIRE script above (the big Python block)
Paste into a Colab cell
```

### 4. Run It
```
Click the Play button (or press Shift+Enter)
Wait 10-30 seconds
See results!
```

### 5. Read the Results
```
You'll see:
- Total tasks tested: 50
- Symmetry detected: XX (XX%)
- Accuracy: XX%
- PREDICTED KAGGLE SCORE: XX-XX%

This tells you if symmetry is working!
```

---

## 🎯 WHAT THE RESULTS MEAN

### If Accuracy is 70-85%:
```
🏆 CHAMPIONSHIP PERFORMANCE!
Your symmetry is working great
Predicted score: 60-70%
Action: Keep it, build more patterns
```

### If Accuracy is 40-60%:
```
😊 SOLID PERFORMANCE
Symmetry is decent, room for improvement
Predicted score: 40-50%
Action: Tune thresholds, add more patterns
```

### If Accuracy is 10-30%:
```
😐 NEEDS WORK
Symmetry detection too sensitive
Predicted score: 15-25%
Action: Fix thresholds, validate logic
```

---

## ❓ TROUBLESHOOTING

### "I don't have Python installed"
→ Use Google Colab (Option 3 above)

### "I can't download training data from Kaggle"
→ You need a Kaggle account (free)
→ Accept competition rules
→ Then you can download data

### "The script has errors"
→ Make sure you uploaded the JSON file to Colab
→ Make sure numpy is available (Colab has it by default)
→ Share the error and I'll help

### "I'm on mobile"
→ Colab works on mobile browser!
→ Or wait until you're on a computer

---

## 🚀 THE SIMPLE VERSION

**If all this is too much:**

1. Go to Colab
2. Upload training data JSON
3. Paste the big Python script
4. Click Run
5. Read the accuracy percentage
6. Tell me what it says!

**That's it. That's the whole thing.**

---

## 💡 WHY WE'RE DOING THIS

Right now we have:
- ✅ Code that runs (v5-Lite works!)
- ❓ Unknown accuracy (is it 66% or 18%?)

After validation:
- ✅ Code that runs
- ✅ KNOWN accuracy (now we know the truth!)
- ✅ Data-driven next steps

**Validation = turning on the lights to see what we built.**

---

**Bottom line:** Use the Colab script (easiest). Just copy-paste-run. Get your answer in 30 seconds.

🎨 Let's see if that ceiling is a masterpiece or needs touch-ups!
