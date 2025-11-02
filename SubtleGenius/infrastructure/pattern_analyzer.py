"""
Pattern Frequency Analyzer - Know What Patterns Actually Exist

This analyzes training data to identify common transformation patterns.
Prevents building patterns that don't exist (grid arithmetic)
and missing obvious patterns (crop, tile).

LAELD Rule #2: ANALYZE TRAINING DATA BEFORE BUILDING PATTERNS

Usage:
    from pattern_analyzer import PatternAnalyzer

    analyzer = PatternAnalyzer('data/arc-agi_training_challenges.json')
    patterns = analyzer.analyze()
    analyzer.print_report(patterns)
"""

import json
import numpy as np
from typing import Dict, List, Tuple, Any
from collections import Counter, defaultdict
from pathlib import Path


class PatternAnalyzer:
    """Analyze training data to identify common transformation patterns"""

    def __init__(self, training_data_path: str):
        """Initialize with training data path"""
        self.training_data_path = training_data_path
        self.training_data = self._load_training_data()

    def _load_training_data(self) -> Dict:
        """Load training challenges"""
        path = Path(self.training_data_path)
        if not path.exists():
            raise FileNotFoundError(f"Training data not found: {self.training_data_path}")

        with open(path, 'r') as f:
            return json.load(f)

    def analyze(self, verbose: bool = True) -> Dict:
        """
        Analyze all training tasks to identify pattern frequencies

        Returns:
            Dict with pattern frequencies and examples
        """
        if verbose:
            print("\n" + "="*70)
            print("PATTERN FREQUENCY ANALYSIS")
            print("="*70)
            print(f"Analyzing {len(self.training_data)} training tasks...\n")

        results = {
            'total_tasks': len(self.training_data),
            'shape_patterns': defaultdict(int),
            'geometric_patterns': defaultdict(int),
            'color_patterns': defaultdict(int),
            'size_patterns': defaultdict(int),
            'examples': defaultdict(list)
        }

        for task_id, task_data in self.training_data.items():
            # Analyze each training pair
            for pair in task_data['train']:
                inp = np.array(pair['input'])
                out = np.array(pair['output'])

                # Shape patterns
                shape_pattern = self._analyze_shape_change(inp, out)
                if shape_pattern:
                    results['shape_patterns'][shape_pattern] += 1
                    if len(results['examples'][shape_pattern]) < 3:
                        results['examples'][shape_pattern].append(task_id)

                # Geometric patterns
                geom_pattern = self._analyze_geometric(inp, out)
                if geom_pattern:
                    results['geometric_patterns'][geom_pattern] += 1
                    if len(results['examples'][geom_pattern]) < 3:
                        results['examples'][geom_pattern].append(task_id)

                # Color patterns
                color_pattern = self._analyze_color_change(inp, out)
                if color_pattern:
                    results['color_patterns'][color_pattern] += 1
                    if len(results['examples'][color_pattern]) < 3:
                        results['examples'][color_pattern].append(task_id)

                # Size patterns
                size_pattern = self._analyze_size_change(inp, out)
                if size_pattern:
                    results['size_patterns'][size_pattern] += 1
                    if len(results['examples'][size_pattern]) < 3:
                        results['examples'][size_pattern].append(task_id)

        return results

    def _analyze_shape_change(self, inp: np.ndarray, out: np.ndarray) -> str:
        """Analyze if shapes match or change"""
        if inp.shape == out.shape:
            return "same_shape"
        elif out.shape[0] < inp.shape[0] or out.shape[1] < inp.shape[1]:
            return "crop_smaller"
        elif out.shape[0] > inp.shape[0] or out.shape[1] > inp.shape[1]:
            return "expand_larger"
        return "shape_change"

    def _analyze_geometric(self, inp: np.ndarray, out: np.ndarray) -> str:
        """Detect geometric transformations"""
        if inp.shape != out.shape:
            return None

        # Rotate 90
        if np.array_equal(out, np.rot90(inp, k=-1)):
            return "rotate_90_cw"

        # Rotate 180
        if np.array_equal(out, np.rot90(inp, k=2)):
            return "rotate_180"

        # Rotate 270
        if np.array_equal(out, np.rot90(inp, k=1)):
            return "rotate_270_cw"

        # Flip horizontal
        if np.array_equal(out, np.fliplr(inp)):
            return "flip_horizontal"

        # Flip vertical
        if np.array_equal(out, np.flipud(inp)):
            return "flip_vertical"

        # Transpose
        if inp.shape[0] == inp.shape[1] and np.array_equal(out, inp.T):
            return "transpose"

        return None

    def _analyze_color_change(self, inp: np.ndarray, out: np.ndarray) -> str:
        """Analyze color transformations"""
        if inp.shape != out.shape:
            return None

        # Check if it's a color mapping
        mapping = {}
        is_mapping = True
        for i_val, o_val in zip(inp.flatten(), out.flatten()):
            if i_val in mapping:
                if mapping[i_val] != o_val:
                    is_mapping = False
                    break
            else:
                mapping[i_val] = o_val

        if is_mapping and mapping and not all(k == v for k, v in mapping.items()):
            return "color_mapping"

        # Check if all colors changed by same amount
        diff = out - inp
        if len(np.unique(diff)) == 1 and np.unique(diff)[0] != 0:
            return "color_shift"

        # Check if colors inverted/swapped
        inp_colors = set(inp.flatten())
        out_colors = set(out.flatten())
        if inp_colors == out_colors and len(inp_colors) <= 3:
            # Might be color swap
            return "color_swap"

        return None

    def _analyze_size_change(self, inp: np.ndarray, out: np.ndarray) -> str:
        """Analyze size changes"""
        inp_h, inp_w = inp.shape
        out_h, out_w = out.shape

        if inp_h == out_h and inp_w == out_w:
            return None

        # Check if it's a tiling pattern
        if out_h % inp_h == 0 and out_w % inp_w == 0:
            h_factor = out_h // inp_h
            w_factor = out_w // inp_w
            if h_factor > 1 or w_factor > 1:
                expected = np.tile(inp, (h_factor, w_factor))
                if np.array_equal(out, expected):
                    return f"tile_{h_factor}x{w_factor}"

        # Check if it's cropping
        if out_h <= inp_h and out_w <= inp_w:
            # Could be crop to bounding box, crop to center, etc.
            return "crop_pattern"

        # Check if it's padding
        if out_h >= inp_h and out_w >= inp_w:
            return "pad_pattern"

        return "size_change"

    def print_report(self, results: Dict, top_n: int = 15):
        """Print pattern frequency report"""
        print("\n" + "="*70)
        print("PATTERN FREQUENCY REPORT")
        print("="*70)

        print(f"\nTotal tasks analyzed: {results['total_tasks']}")

        # Shape patterns
        print(f"\n📐 SHAPE PATTERNS")
        print(f"  {'Pattern':<25} {'Count':>8} {'Frequency':>12} {'Examples'}")
        print(f"  {'-'*25} {'-'*8} {'-'*12} {'-'*20}")
        for pattern, count in sorted(results['shape_patterns'].items(),
                                     key=lambda x: x[1],
                                     reverse=True)[:top_n]:
            freq = count / results['total_tasks']
            examples = ', '.join(results['examples'][pattern][:2])
            print(f"  {pattern:<25} {count:>8} {freq*100:>11.1f}% {examples}")

        # Geometric patterns
        if results['geometric_patterns']:
            print(f"\n🔄 GEOMETRIC PATTERNS")
            print(f"  {'Pattern':<25} {'Count':>8} {'Frequency':>12} {'Examples'}")
            print(f"  {'-'*25} {'-'*8} {'-'*12} {'-'*20}")
            for pattern, count in sorted(results['geometric_patterns'].items(),
                                         key=lambda x: x[1],
                                         reverse=True)[:top_n]:
                freq = count / results['total_tasks']
                examples = ', '.join(results['examples'][pattern][:2])
                print(f"  {pattern:<25} {count:>8} {freq*100:>11.1f}% {examples}")

        # Color patterns
        if results['color_patterns']:
            print(f"\n🎨 COLOR PATTERNS")
            print(f"  {'Pattern':<25} {'Count':>8} {'Frequency':>12} {'Examples'}")
            print(f"  {'-'*25} {'-'*8} {'-'*12} {'-'*20}")
            for pattern, count in sorted(results['color_patterns'].items(),
                                         key=lambda x: x[1],
                                         reverse=True)[:top_n]:
                freq = count / results['total_tasks']
                examples = ', '.join(results['examples'][pattern][:2])
                print(f"  {pattern:<25} {count:>8} {freq*100:>11.1f}% {examples}")

        # Size patterns
        if results['size_patterns']:
            print(f"\n📏 SIZE PATTERNS")
            print(f"  {'Pattern':<25} {'Count':>8} {'Frequency':>12} {'Examples'}")
            print(f"  {'-'*25} {'-'*8} {'-'*12} {'-'*20}")
            for pattern, count in sorted(results['size_patterns'].items(),
                                         key=lambda x: x[1],
                                         reverse=True)[:top_n]:
                freq = count / results['total_tasks']
                examples = ', '.join(results['examples'][pattern][:2])
                print(f"  {pattern:<25} {count:>8} {freq*100:>11.1f}% {examples}")

        # Priority recommendations
        print(f"\n🎯 TOP PRIORITIES (Frequency × Simplicity)")
        print(f"  Build these patterns first:")

        all_patterns = []
        for category in ['geometric_patterns', 'color_patterns', 'size_patterns']:
            for pattern, count in results[category].items():
                freq = count / results['total_tasks']
                # Simple heuristic: basic patterns are easier
                simplicity = 1.0  # Could refine this
                priority = freq * simplicity
                all_patterns.append((pattern, freq, simplicity, priority, category))

        all_patterns.sort(key=lambda x: x[3], reverse=True)

        print(f"  {'Rank':>4} {'Pattern':<25} {'Frequency':>12} {'Category':<20}")
        print(f"  {'-'*4} {'-'*25} {'-'*12} {'-'*20}")
        for rank, (pattern, freq, _, priority, category) in enumerate(all_patterns[:15], 1):
            print(f"  {rank:>4} {pattern:<25} {freq*100:>11.1f}% {category:<20}")

        print("\n" + "="*70)
        print("✅ ANALYSIS COMPLETE")
        print("="*70 + "\n")


# Example usage
if __name__ == "__main__":
    print("Pattern Frequency Analyzer - Infrastructure Test")
    print("="*70)
    print("\nThis analyzer identifies common transformation patterns in training data.")
    print("\nUsage:")
    print("  analyzer = PatternAnalyzer('data/arc-agi_training_challenges.json')")
    print("  patterns = analyzer.analyze()")
    print("  analyzer.print_report(patterns)")
    print("\n" + "="*70)
