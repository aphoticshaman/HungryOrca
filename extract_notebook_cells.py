#!/usr/bin/env python3
"""
Extract and compile Python code cells from Jupyter notebooks

This tool allows you to:
1. Extract specific cells by index or tag
2. Compile all code cells into a single .py file
3. Work with .py files instead of full notebooks for faster iteration

Usage:
  python extract_notebook_cells.py <notebook.ipynb> [options]

Examples:
  # Extract all code cells to a .py file
  python extract_notebook_cells.py orcaswordv3.ipynb -o extracted.py

  # Extract only cells 0-2
  python extract_notebook_cells.py orcaswordv3.ipynb --cells 0,1,2 -o cells_0_2.py

  # List all cells
  python extract_notebook_cells.py orcaswordv3.ipynb --list
"""

import json
import sys
import argparse
from pathlib import Path

def list_cells(notebook_path):
    """List all cells in notebook with indices"""
    with open(notebook_path, 'r') as f:
        nb = json.load(f)

    cells = nb.get('cells', [])
    print(f"Notebook: {notebook_path}")
    print(f"Total cells: {len(cells)}\n")

    for i, cell in enumerate(cells):
        cell_type = cell.get('cell_type', 'unknown')
        source = cell.get('source', [])

        # Get first line
        if isinstance(source, list):
            first_line = source[0] if source else ""
        else:
            first_line = source.split('\n')[0] if source else ""

        # Truncate if too long
        if len(first_line) > 70:
            first_line = first_line[:67] + "..."

        # Count lines
        if isinstance(source, list):
            num_lines = len(source)
        else:
            num_lines = len(source.split('\n'))

        print(f"Cell {i:3d} [{cell_type:8s}] {num_lines:4d} lines | {first_line}")

def extract_cells(notebook_path, output_path, cell_indices=None):
    """Extract code cells to Python file"""
    with open(notebook_path, 'r') as f:
        nb = json.load(f)

    cells = nb.get('cells', [])

    # Filter to requested cells
    if cell_indices is not None:
        cells = [cells[i] for i in cell_indices if i < len(cells)]

    # Extract code cells
    code_cells = [c for c in cells if c.get('cell_type') == 'code']

    print(f"Extracting {len(code_cells)} code cells from {notebook_path}...")

    # Compile into Python file
    with open(output_path, 'w') as f:
        f.write(f'"""Extracted from {notebook_path}"""\n\n')

        for i, cell in enumerate(code_cells):
            source = cell.get('source', [])

            # Write cell header
            f.write(f"# {'=' * 77}\n")
            f.write(f"# CELL {i}\n")
            f.write(f"# {'=' * 77}\n\n")

            # Write source
            if isinstance(source, list):
                for line in source:
                    f.write(line)
                    if not line.endswith('\n'):
                        f.write('\n')
            else:
                f.write(source)
                if not source.endswith('\n'):
                    f.write('\n')

            f.write('\n')

    print(f"✓ Extracted to: {output_path}")

def push_cells_to_notebook(py_path, notebook_path, cell_indices):
    """Push code from .py file back into notebook cells"""
    # Read Python file
    with open(py_path, 'r') as f:
        py_content = f.read()

    # Split by cell markers
    cell_marker = "# " + "=" * 77 + "\n# CELL "
    parts = py_content.split(cell_marker)

    # First part is header
    cells_content = []
    for part in parts[1:]:  # Skip header
        # Extract cell code (skip header lines)
        lines = part.split('\n')
        # Find where actual code starts (after the === line)
        code_start = 0
        for i, line in enumerate(lines):
            if line.startswith('# ' + '=' * 77):
                code_start = i + 1
                break

        code = '\n'.join(lines[code_start:])
        cells_content.append(code.strip() + '\n')

    # Read notebook
    with open(notebook_path, 'r') as f:
        nb = json.load(f)

    # Update cells
    for i, cell_idx in enumerate(cell_indices):
        if i < len(cells_content) and cell_idx < len(nb['cells']):
            nb['cells'][cell_idx]['source'] = cells_content[i].split('\n')

    # Write back
    with open(notebook_path, 'w') as f:
        json.dump(nb, f, indent=1)

    print(f"✓ Updated {len(cell_indices)} cells in {notebook_path}")

def main():
    parser = argparse.ArgumentParser(
        description='Extract/compile Python code cells from Jupyter notebooks',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument('notebook', help='Path to .ipynb file')
    parser.add_argument('-o', '--output', help='Output .py file')
    parser.add_argument('--cells', help='Comma-separated cell indices (e.g., 0,1,2)')
    parser.add_argument('--list', action='store_true', help='List all cells')
    parser.add_argument('--push', help='Push .py file back to notebook (requires --cells)')

    args = parser.parse_args()

    notebook_path = Path(args.notebook)
    if not notebook_path.exists():
        print(f"ERROR: Notebook not found: {notebook_path}")
        sys.exit(1)

    # List mode
    if args.list:
        list_cells(notebook_path)
        return

    # Parse cell indices
    cell_indices = None
    if args.cells:
        try:
            cell_indices = [int(x.strip()) for x in args.cells.split(',')]
        except ValueError:
            print(f"ERROR: Invalid cell indices: {args.cells}")
            sys.exit(1)

    # Push mode
    if args.push:
        if not cell_indices:
            print("ERROR: --push requires --cells to specify which cells to update")
            sys.exit(1)

        py_path = Path(args.push)
        if not py_path.exists():
            print(f"ERROR: Python file not found: {py_path}")
            sys.exit(1)

        push_cells_to_notebook(py_path, notebook_path, cell_indices)
        return

    # Extract mode
    if not args.output:
        output_path = notebook_path.with_suffix('.py')
    else:
        output_path = Path(args.output)

    extract_cells(notebook_path, output_path, cell_indices)

if __name__ == '__main__':
    main()
