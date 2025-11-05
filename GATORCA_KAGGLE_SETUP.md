# PROJECT GATORCA - KAGGLE DEPLOYMENT GUIDE
## 20-Minute Test Run Setup

---

## 📦 QUICK START (20-Minute Test)

### Step 1: Upload Files to Kaggle

1. Go to Kaggle Notebook: https://www.kaggle.com/code
2. Create new notebook
3. Upload `gatorca_submission_compressed.py` (29.5 KB)

### Step 2: Set Time Limit (20 Minutes)

In your Kaggle notebook, add this at the top:

```python
import time
import signal

# Set 20-minute timeout
TIMEOUT_SECONDS = 20 * 60  # 20 minutes
START_TIME = time.time()

def timeout_handler(signum, frame):
    raise TimeoutError("20-minute limit reached!")

signal.signal(signal.SIGALRM, timeout_handler)
signal.alarm(TIMEOUT_SECONDS)

print(f"⏱️  20-minute timer started at {time.ctime()}")
print(f"⏱️  Will timeout at {time.ctime(START_TIME + TIMEOUT_SECONDS)}")
```

### Step 3: Import GatORCA

```python
# Import the compressed solver
exec(open('gatorca_submission_compressed.py').read())

# Or import directly if you uploaded it as a module
# from gatorca_submission_compressed import solve_arc_task
```

### Step 4: Load ARC Dataset

```python
import json

# Load ARC training data (included in Kaggle competition)
with open('/kaggle/input/arc-prize-2024/arc-agi_training_challenges.json') as f:
    training_data = json.load(f)

print(f"📁 Loaded {len(training_data)} training tasks")
```

### Step 5: Run 20-Minute Test

```python
import random

# Select random subset for 20-minute test
num_tasks = 50  # Adjust based on time available
task_ids = random.sample(list(training_data.keys()), min(num_tasks, len(training_data)))

print(f"🎲 Testing on {len(task_ids)} random tasks")
print(f"⏱️  Time limit: 20 minutes")

results = []
solved_count = 0

for i, task_id in enumerate(task_ids):
    # Check time remaining
    elapsed = time.time() - START_TIME
    remaining = TIMEOUT_SECONDS - elapsed

    if remaining < 60:  # Less than 1 minute left
        print(f"⏱️  Time running out! Stopping early.")
        break

    print(f"\n[{i+1}/{len(task_ids)}] Task {task_id} (Time: {elapsed/60:.1f}min / {TIMEOUT_SECONDS/60}min)")

    try:
        task = training_data[task_id]

        # Solve with per-task timeout (to ensure we test multiple tasks)
        per_task_timeout = min(60, remaining / (len(task_ids) - i))

        result = solve_arc_task(task)

        if result.get('solved'):
            solved_count += 1
            print(f"   ✓ SOLVED! ({result['best_fitness']:.1%})")
        else:
            print(f"   ✗ Unsolved ({result['best_fitness']:.1%})")

        results.append({
            'task_id': task_id,
            'solved': result.get('solved', False),
            'fitness': result.get('best_fitness', 0.0),
            'time': result.get('time_elapsed', 0.0)
        })

    except TimeoutError:
        print("⏱️  20-minute limit reached!")
        break
    except Exception as e:
        print(f"   ✗ Error: {e}")

# Summary
total_tested = len(results)
accuracy = solved_count / total_tested if total_tested > 0 else 0
avg_fitness = sum(r['fitness'] for r in results) / total_tested if total_tested > 0 else 0

print(f"\n" + "="*80)
print(f"📊 20-MINUTE TEST RESULTS")
print(f"="*80)
print(f"Tasks Tested: {total_tested}")
print(f"Tasks Solved: {solved_count}")
print(f"Accuracy: {accuracy:.1%}")
print(f"Avg Fitness: {avg_fitness:.1%}")
print(f"Time Used: {(time.time() - START_TIME)/60:.1f} minutes")
```

---

## 🎯 ALTERNATIVE: Quick 5-Task Test

For fastest testing, use this minimal version:

```python
import json
import time

# Load
exec(open('gatorca_submission_compressed.py').read())

with open('/kaggle/input/arc-prize-2024/arc-agi_training_challenges.json') as f:
    data = json.load(f)

# Test on first 5 tasks
for i, (task_id, task) in enumerate(list(data.items())[:5]):
    print(f"[{i+1}/5] {task_id}...")
    result = solve_arc_task(task)
    print(f"  Fitness: {result['best_fitness']:.1%}")
```

---

## 📊 COMPETITION SUBMISSION FORMAT

When you're ready to submit to the competition:

```python
import json

# Load test data
with open('/kaggle/input/arc-prize-2024/arc-agi_test_challenges.json') as f:
    test_data = json.load(f)

# Solve all test tasks
submission = {}
for task_id, task in test_data.items():
    predictions = solve_arc_task(task)
    submission[task_id] = predictions

# Save submission
with open('submission.json', 'w') as f:
    json.dump(submission, f)

print("✅ Submission file created: submission.json")
```

---

## ⚙️ TUNING PARAMETERS

Adjust these in the solver for different time/accuracy tradeoffs:

```python
# In the solve_arc_task function, modify:

result = solver.solve_task(
    task,
    max_generations=30,      # Reduce for faster (was 50)
    timeout_seconds=30       # Reduce for faster (was 60)
)
```

**Time Estimates**:
- `max_generations=10, timeout=10s`: ~10-15 tasks in 20 min
- `max_generations=30, timeout=30s`: ~30-40 tasks in 20 min
- `max_generations=50, timeout=60s`: ~20-30 tasks in 20 min

---

## 🚀 RECOMMENDED 20-MINUTE TEST PROTOCOL

```python
import json
import time
import random
import signal

# ===== SETUP =====
TIMEOUT = 20 * 60
START = time.time()

signal.signal(signal.SIGALRM, lambda s,f: (_ for _ in ()).throw(TimeoutError()))
signal.alarm(TIMEOUT)

# ===== LOAD =====
exec(open('gatorca_submission_compressed.py').read())

with open('/kaggle/input/arc-prize-2024/arc-agi_training_challenges.json') as f:
    data = json.load(f)

# ===== TEST ON 40 TASKS =====
tasks = random.sample(list(data.items()), 40)
results = []

for i, (tid, task) in enumerate(tasks):
    if time.time() - START > TIMEOUT - 60:
        break

    print(f"[{i+1}/40] {tid} ({(time.time()-START)/60:.1f}min)")

    try:
        r = solve_arc_task(task)
        results.append({
            'id': tid,
            'solved': r['solved'],
            'fitness': r['best_fitness']
        })
        print(f"  {'✓' if r['solved'] else '✗'} {r['best_fitness']:.1%}")
    except:
        pass

# ===== REPORT =====
solved = sum(1 for r in results if r['solved'])
avg_fit = sum(r['fitness'] for r in results) / len(results)

print(f"\n📊 RESULTS: {solved}/{len(results)} solved ({solved/len(results):.1%})")
print(f"📊 Avg Fitness: {avg_fit:.1%}")
print(f"⏱️  Time: {(time.time()-START)/60:.1f}min")
```

---

## 💡 TIPS FOR KAGGLE

1. **Start Small**: Test on 5-10 tasks first to verify it works
2. **Monitor Time**: Check `time.time() - START` frequently
3. **Adjust Params**: Reduce `max_generations` if running out of time
4. **Save Progress**: Print results for each task so you don't lose data
5. **Use Random Sample**: Test on diverse tasks, not just first N
6. **Check Memory**: Monitor with `import psutil; psutil.virtual_memory()`

---

## 📝 FULL KAGGLE NOTEBOOK TEMPLATE

See `gatorca_kaggle_notebook.ipynb` for complete ready-to-upload notebook!

---

## 🎖️ COMPETITION STRATEGY

**Phase 1: Validation (20 min)**
- Test on 30-50 training tasks
- Verify system works on Kaggle
- Measure actual performance

**Phase 2: Optimization (if time permits)**
- Tune parameters based on Phase 1
- Focus on puzzle types that perform well
- Reduce timeout for broader coverage

**Phase 3: Full Submission**
- Run on all test tasks
- Use learned optimal parameters
- Submit to competition

---

**Good luck! 🐊 GATORCA ready for battle! 🎖️**
