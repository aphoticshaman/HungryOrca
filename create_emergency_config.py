#!/usr/bin/env python3
"""
EMERGENCY: Create properly configured notebook for production run

The user is running the wrong notebook. Create a CLEAN version with:
- MAX_PROGRAM_DEPTH = 300 (go MUCH deeper - depth=150 still failing)
- BEAM_SEARCH_WIDTH = 10 (wider search)
- DIAGNOSTIC_RUN = False (production mode)
- Training on full dataset with proper depth
"""

import json
import sys

# Start from the redesigned notebook
with open('/home/user/HungryOrca/lucidorcax_redesigned.ipynb', 'r') as f:
    nb = json.load(f)

print("🚨 EMERGENCY CONFIGURATION - Going MUCH deeper")
print("="*70)

# Apply AGGRESSIVE settings
changes = []

for i, cell in enumerate(nb['cells']):
    source = cell.get('source', [])
    new_source = []

    for line in source:
        # Change depth to 300 (2x our redesign)
        if 'MAX_PROGRAM_DEPTH: int = 150' in line:
            line = line.replace('150', '300')
            print(f"✅ Cell {i}: MAX_PROGRAM_DEPTH = 300 (was 150)")
            changes.append(f"depth: 150 → 300")

        # Change beam to 10
        if 'BEAM_SEARCH_WIDTH: int = 8' in line:
            line = line.replace('8', '10')
            print(f"✅ Cell {i}: BEAM_SEARCH_WIDTH = 10 (was 8)")
            changes.append(f"beam: 8 → 10")

        # FORCE diagnostic off
        if 'DIAGNOSTIC_RUN: bool = True' in line:
            line = line.replace('True', 'False')
            print(f"✅ Cell {i}: DIAGNOSTIC_RUN = False")
            changes.append("diagnostic: off")

        new_source.append(line)

    cell['source'] = new_source

output_file = '/home/user/HungryOrca/lucidorcax_EMERGENCY_v300.ipynb'
with open(output_file, 'w') as f:
    json.dump(nb, f, indent=1)

print(f"\n✅ Created: {output_file}")
print(f"Changes applied: {changes}")
print("\n" + "="*70)
print("⚡ EMERGENCY CONFIGURATION SUMMARY")
print("="*70)
print("""
MAX_PROGRAM_DEPTH = 300   (was 20 → 100 → 150, NOW 300!)
BEAM_SEARCH_WIDTH = 10    (was 5 → 8, NOW 10!)
DIAGNOSTIC_RUN = False    (production mode)

RATIONALE:
- 300 tasks in 8 min = still hitting depth limit immediately
- Even depth=150 is insufficient
- Going nuclear: depth=300, beam=10
- Expected time/task: ~90s (still only 2.5 hours for 100 tasks)

UPLOAD THIS FILE IMMEDIATELY AND RE-RUN!
""")

