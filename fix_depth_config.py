#!/usr/bin/env python3
"""
Fix MAX_PROGRAM_DEPTH in lucidorcax.ipynb

This script updates the MAX_PROGRAM_DEPTH configuration from 20 to 100
to fix the performance issue where the solver runs in 3 minutes instead of 30.
"""

import json
import re
import sys

def fix_notebook_depth(input_file, output_file, new_depth=100):
    """
    Fix the MAX_PROGRAM_DEPTH value in the Jupyter notebook
    """
    print(f"Reading notebook: {input_file}")

    with open(input_file, 'r') as f:
        notebook = json.load(f)

    changes_made = 0

    # Find Cell 2 and update the MAX_PROGRAM_DEPTH value
    for cell_idx, cell in enumerate(notebook.get('cells', [])):
        if cell.get('cell_type') == 'code':
            source = cell.get('source', [])

            # Check if this is Cell 2 (Configuration cell)
            cell_source_text = ''.join(source)

            if 'MAX_PROGRAM_DEPTH' in cell_source_text and 'ChampionshipConfig' in cell_source_text:
                print(f"\nFound configuration cell (index {cell_idx})")

                # Update the source
                new_source = []
                for line in source:
                    # Match the MAX_PROGRAM_DEPTH line (more flexible pattern)
                    if 'MAX_PROGRAM_DEPTH' in line and 'int' in line and '=' in line and '#' not in line[:line.find('MAX_PROGRAM_DEPTH')] if 'MAX_PROGRAM_DEPTH' in line else False:
                        old_line = line
                        # Replace the value - more robust pattern
                        new_line = re.sub(
                            r'(MAX_PROGRAM_DEPTH:\s*int\s*=\s*)\d+',
                            r'\g<1>' + str(new_depth),
                            line
                        )
                        # Update comment
                        if '# Increased from' in new_line:
                            new_line = re.sub(
                                r'# Increased from \d+',
                                f'# FIXED: Increased from 20 to {new_depth}',
                                new_line
                            )

                        print(f"  Old: {old_line[:100]}...")
                        print(f"  New: {new_line[:100]}...")
                        new_source.append(new_line)
                        changes_made += 1
                    else:
                        new_source.append(line)

                cell['source'] = new_source

    if changes_made == 0:
        print("\n⚠️  WARNING: No changes were made. MAX_PROGRAM_DEPTH not found or already fixed.")
        return False

    print(f"\n✅ Made {changes_made} change(s)")
    print(f"Writing updated notebook to: {output_file}")

    with open(output_file, 'w') as f:
        json.dump(notebook, f, indent=1)

    print("✅ Notebook updated successfully!")
    return True

def main():
    input_notebook = '/home/user/HungryOrca/lucidorcax.ipynb'
    output_notebook = '/home/user/HungryOrca/lucidorcax_fixed.ipynb'

    print("=" * 70)
    print("🔧 Fixing MAX_PROGRAM_DEPTH Configuration")
    print("=" * 70)

    success = fix_notebook_depth(input_notebook, output_notebook, new_depth=100)

    if success:
        print("\n" + "=" * 70)
        print("✨ FIX APPLIED SUCCESSFULLY!")
        print("=" * 70)
        print(f"""
Next Steps:
1. Review the changes in: {output_notebook}
2. Test the fixed notebook locally
3. If tests pass, rename to lucidorcax.ipynb
4. Commit and push changes

Expected Results After Fix:
- Runtime: 30+ minutes (instead of 3 minutes)
- Some tasks will solve successfully
- LTM cache will be populated
- Fewer Synthesizer.Fail.MaxDepth failures
""")
    else:
        print("\n❌ Fix failed. Please review the notebook manually.")
        sys.exit(1)

if __name__ == "__main__":
    main()
