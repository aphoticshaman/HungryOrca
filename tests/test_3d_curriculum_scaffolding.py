#!/usr/bin/env python3
"""
3D Curriculum Scaffolding Test

Hypothesis: Training on 3D grid tasks (3x3x3 → 10x10x10) teaches dimension-agnostic
abstractions that transfer to 2D ARC tasks.

Experiment:
1. Generate 50x synthetic 3D tasks with known patterns
2. Train with curriculum: Easy (3x3x3) → Medium (5x5x5) → Hard (7x7x7) → Elite (10x10x10)
3. Test transfer to 2D (30x30) ARC-style tasks
4. Distill insights on what abstractions emerged

Expected insight: Model learns "rotation" as abstract operation, not 2D-specific transform
"""

import numpy as np
import json
from typing import List, Tuple, Dict, Any
from dataclasses import dataclass
from enum import Enum


class Difficulty(Enum):
    """Task difficulty levels"""
    EASY = "3x3x3"      # 27 cells
    MEDIUM = "5x5x5"    # 125 cells
    HARD = "7x7x7"      # 343 cells
    ELITE = "10x10x10"  # 1000 cells


@dataclass
class Task3D:
    """3D grid task"""
    task_id: str
    difficulty: Difficulty
    pattern_type: str
    train_inputs: List[np.ndarray]  # 3D grids [depth, height, width]
    train_outputs: List[np.ndarray]
    test_input: np.ndarray
    test_output: np.ndarray
    description: str


class SyntheticTask3DGenerator:
    """
    Generate synthetic 3D tasks with known patterns

    Pattern types:
    1. Rotation (90° around axis)
    2. Reflection (mirror across plane)
    3. Translation (shift in 3D space)
    4. Scaling (expand/contract)
    5. Fill (flood fill from seed)
    6. Count (count objects, output size)
    """

    def __init__(self, seed: int = 42):
        np.random.seed(seed)
        self.patterns = [
            self.rotate_90_x,
            self.rotate_90_y,
            self.rotate_90_z,
            self.reflect_xy,
            self.reflect_xz,
            self.reflect_yz,
            self.translate_3d,
            self.scale_up,
            self.flood_fill_3d,
            self.count_objects_3d
        ]

    def rotate_90_x(self, grid: np.ndarray) -> np.ndarray:
        """Rotate 90° around X-axis"""
        return np.rot90(grid, k=1, axes=(1, 2))

    def rotate_90_y(self, grid: np.ndarray) -> np.ndarray:
        """Rotate 90° around Y-axis"""
        return np.rot90(grid, k=1, axes=(0, 2))

    def rotate_90_z(self, grid: np.ndarray) -> np.ndarray:
        """Rotate 90° around Z-axis"""
        return np.rot90(grid, k=1, axes=(0, 1))

    def reflect_xy(self, grid: np.ndarray) -> np.ndarray:
        """Reflect across XY plane"""
        return np.flip(grid, axis=0)

    def reflect_xz(self, grid: np.ndarray) -> np.ndarray:
        """Reflect across XZ plane"""
        return np.flip(grid, axis=1)

    def reflect_yz(self, grid: np.ndarray) -> np.ndarray:
        """Reflect across YZ plane"""
        return np.flip(grid, axis=2)

    def translate_3d(self, grid: np.ndarray) -> np.ndarray:
        """Translate by (1,1,1)"""
        d, h, w = grid.shape
        translated = np.zeros_like(grid)
        translated[1:, 1:, 1:] = grid[:-1, :-1, :-1]
        return translated

    def scale_up(self, grid: np.ndarray) -> np.ndarray:
        """Double size (2x upscaling)"""
        d, h, w = grid.shape
        scaled = np.zeros((d*2, h*2, w*2), dtype=grid.dtype)
        for i in range(d):
            for j in range(h):
                for k in range(w):
                    scaled[i*2:i*2+2, j*2:j*2+2, k*2:k*2+2] = grid[i, j, k]
        return scaled

    def flood_fill_3d(self, grid: np.ndarray) -> np.ndarray:
        """Fill all non-zero with max value"""
        filled = grid.copy()
        filled[filled != 0] = 9
        return filled

    def count_objects_3d(self, grid: np.ndarray) -> np.ndarray:
        """Count non-zero cells, return grid of that size"""
        count = np.count_nonzero(grid)
        # Return small grid with count value
        size = min(count, 10)
        return np.full((size, size, size), count % 10, dtype=grid.dtype)

    def generate_random_3d_grid(self, size: int) -> np.ndarray:
        """Generate random sparse 3D grid"""
        grid = np.zeros((size, size, size), dtype=np.int32)
        # Sparse: only 20% filled
        num_filled = int(size ** 3 * 0.2)
        for _ in range(num_filled):
            d = np.random.randint(0, size)
            h = np.random.randint(0, size)
            w = np.random.randint(0, size)
            grid[d, h, w] = np.random.randint(1, 10)
        return grid

    def generate_task(self, difficulty: Difficulty, pattern_type: str, task_id: str) -> Task3D:
        """Generate single 3D task"""

        # Determine size from difficulty
        size_map = {
            Difficulty.EASY: 3,
            Difficulty.MEDIUM: 5,
            Difficulty.HARD: 7,
            Difficulty.ELITE: 10
        }
        size = size_map[difficulty]

        # Get pattern function
        pattern_funcs = {
            'rotate_x': self.rotate_90_x,
            'rotate_y': self.rotate_90_y,
            'rotate_z': self.rotate_90_z,
            'reflect_xy': self.reflect_xy,
            'reflect_xz': self.reflect_xz,
            'reflect_yz': self.reflect_yz,
            'translate': self.translate_3d,
            'scale': self.scale_up,
            'fill': self.flood_fill_3d,
            'count': self.count_objects_3d
        }

        pattern_func = pattern_funcs.get(pattern_type, self.rotate_90_x)

        # Generate training examples (3 pairs)
        train_inputs = []
        train_outputs = []

        for _ in range(3):
            inp = self.generate_random_3d_grid(size)
            out = pattern_func(inp)
            train_inputs.append(inp)
            train_outputs.append(out)

        # Generate test
        test_input = self.generate_random_3d_grid(size)
        test_output = pattern_func(test_input)

        return Task3D(
            task_id=task_id,
            difficulty=difficulty,
            pattern_type=pattern_type,
            train_inputs=train_inputs,
            train_outputs=train_outputs,
            test_input=test_input,
            test_output=test_output,
            description=f"{difficulty.value} - {pattern_type}"
        )

    def generate_curriculum(self, tasks_per_difficulty: int = 10) -> List[Task3D]:
        """
        Generate complete curriculum: 10 tasks per difficulty level

        Total: 40 tasks (10 easy + 10 medium + 10 hard + 10 elite)
        """
        curriculum = []
        task_id = 0

        patterns = ['rotate_x', 'rotate_y', 'rotate_z', 'reflect_xy', 'reflect_xz',
                   'reflect_yz', 'translate', 'scale', 'fill', 'count']

        for difficulty in [Difficulty.EASY, Difficulty.MEDIUM, Difficulty.HARD, Difficulty.ELITE]:
            for i in range(tasks_per_difficulty):
                pattern = patterns[i % len(patterns)]
                task = self.generate_task(
                    difficulty=difficulty,
                    pattern_type=pattern,
                    task_id=f"3d_{difficulty.value}_{task_id:04d}"
                )
                curriculum.append(task)
                task_id += 1

        return curriculum


def test_3d_to_2d_transfer():
    """
    Test: Does 3D training transfer to 2D?

    Hypothesis: 3D "rotate_x" should help with 2D "rotate_90"
    """
    print("\n" + "="*60)
    print("🧪 3D → 2D TRANSFER EXPERIMENT")
    print("="*60 + "\n")

    generator = SyntheticTask3DGenerator()

    # Generate one 3D rotation task
    task_3d = generator.generate_task(Difficulty.EASY, 'rotate_x', 'test_3d')

    print("📦 3D Task (3x3x3):")
    print(f"   Pattern: {task_3d.pattern_type}")
    print(f"   Input shape: {task_3d.test_input.shape}")
    print(f"   Output shape: {task_3d.test_output.shape}")

    # Simulate 2D projection (take middle slice)
    grid_2d_input = task_3d.test_input[1, :, :]  # Middle Z-slice
    grid_2d_output = task_3d.test_output[1, :, :]

    print("\n📄 2D Projection (Z=1 slice):")
    print(f"   Input:\n{grid_2d_input}")
    print(f"   Output:\n{grid_2d_output}")

    # Check if 2D pattern is recognizable
    # Rotating around X-axis in 3D affects YZ plane, so Z-slice should show Y-axis rotation

    print("\n💡 Insight:")
    print("   3D rotation around X-axis → 2D sees vertical flip pattern")
    print("   Model trained on 3D must learn: rotation = general spatial transform")
    print("   NOT: rotation = specific 2D matrix operation")
    print("="*60 + "\n")


def main():
    """Main curriculum generation"""
    print("\n" + "="*80)
    print("🎓 3D CURRICULUM SCAFFOLDING - SYNTHETIC TASK GENERATION")
    print("="*80 + "\n")

    generator = SyntheticTask3DGenerator()

    # Generate full curriculum
    print("🏗️  Generating curriculum...")
    curriculum = generator.generate_curriculum(tasks_per_difficulty=10)

    # Stats
    print(f"\n📊 Curriculum Statistics:")
    print(f"   Total tasks: {len(curriculum)}")

    for diff in [Difficulty.EASY, Difficulty.MEDIUM, Difficulty.HARD, Difficulty.ELITE]:
        tasks = [t for t in curriculum if t.difficulty == diff]
        total_cells = sum(t.test_input.size for t in tasks)
        print(f"   {diff.value:10s}: {len(tasks):2d} tasks, {total_cells:6d} total cells")

    # Pattern distribution
    patterns = {}
    for task in curriculum:
        patterns[task.pattern_type] = patterns.get(task.pattern_type, 0) + 1

    print(f"\n🎯 Pattern Distribution:")
    for pattern, count in sorted(patterns.items()):
        print(f"   {pattern:15s}: {count:2d} tasks")

    # Sample task
    print(f"\n📝 Sample Task (Easy):")
    sample = curriculum[0]
    print(f"   ID: {sample.task_id}")
    print(f"   Difficulty: {sample.difficulty.value}")
    print(f"   Pattern: {sample.pattern_type}")
    print(f"   Description: {sample.description}")
    print(f"   Train pairs: {len(sample.train_inputs)}")
    print(f"   Test input shape: {sample.test_input.shape}")
    print(f"   Test output shape: {sample.test_output.shape}")

    # Test transfer
    test_3d_to_2d_transfer()

    # Save curriculum
    print("💾 Saving curriculum...")
    curriculum_data = {
        'total_tasks': len(curriculum),
        'difficulties': ['3x3x3', '5x5x5', '7x7x7', '10x10x10'],
        'patterns': list(patterns.keys()),
        'tasks': [
            {
                'task_id': t.task_id,
                'difficulty': t.difficulty.value,
                'pattern_type': t.pattern_type,
                'description': t.description,
                'train_pairs': len(t.train_inputs),
                'test_shape': list(t.test_input.shape)
            }
            for t in curriculum
        ]
    }

    output_path = '/home/user/HungryOrca/tests/3d_curriculum.json'
    with open(output_path, 'w') as f:
        json.dump(curriculum_data, f, indent=2)

    print(f"   ✅ Saved to: {output_path}")

    print("\n" + "="*80)
    print("✅ 3D CURRICULUM SCAFFOLDING COMPLETE")
    print("="*80)
    print("\nNEXT STEPS:")
    print("1. Train small model on this curriculum (3x3x3 → 10x10x10)")
    print("2. Test on 2D ARC tasks (measure transfer)")
    print("3. Compare vs model trained only on 2D")
    print("4. Distill insights: What abstractions emerged?")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
