"""
Generate 1000 atomic training tasks for NSPSA

Each task demonstrates a single primitive transformation.
Ground truth programs are known, enabling supervised learning.

Purpose:
- Train PrimitiveRanker to predict useful primitives
- Train ProgramEncoder to create meaningful latent representations
- Compare zero-shot (symbolic only) vs few-shot (after neural training)
"""

import numpy as np
import json
from typing import List, Dict, Tuple
from dataclasses import dataclass, asdict
from symbolic_solver import SymbolicPrimitiveExecutor, GridState

@dataclass
class AtomicTask:
    """Single primitive demonstration"""
    task_id: str
    primitive: str
    input_grid: List[List[int]]
    output_grid: List[List[int]]
    difficulty: str  # 'easy', 'medium', 'hard'


class AtomicTaskGenerator:
    """Generate synthetic tasks for each primitive"""

    def __init__(self):
        self.executor = SymbolicPrimitiveExecutor()
        self.primitives = list(self.executor.primitives.keys())

        # Grid templates of varying complexity
        self.easy_grids = self._generate_easy_grids()
        self.medium_grids = self._generate_medium_grids()
        self.hard_grids = self._generate_hard_grids()

    def _generate_easy_grids(self) -> List[np.ndarray]:
        """2x2 and 3x3 grids"""
        grids = []

        # Simple patterns
        grids.append(np.array([[1, 0], [0, 2]]))
        grids.append(np.array([[1, 2], [3, 4]]))
        grids.append(np.array([[0, 1, 0], [1, 2, 1], [0, 1, 0]]))
        grids.append(np.array([[1, 0, 2], [0, 3, 0], [4, 0, 5]]))
        grids.append(np.array([[2, 2], [2, 2]]))

        # Diagonal patterns
        grids.append(np.array([[1, 0, 0], [0, 2, 0], [0, 0, 3]]))
        grids.append(np.array([[0, 0, 1], [0, 2, 0], [1, 0, 0]]))

        # Random sparse
        for _ in range(10):
            grid = np.zeros((3, 3), dtype=int)
            num_nonzero = np.random.randint(1, 4)
            positions = np.random.choice(9, num_nonzero, replace=False)
            for pos in positions:
                grid[pos // 3, pos % 3] = np.random.randint(1, 5)
            grids.append(grid)

        return grids

    def _generate_medium_grids(self) -> List[np.ndarray]:
        """4x4 and 5x5 grids"""
        grids = []

        # Structured patterns
        grids.append(np.array([
            [1, 0, 0, 1],
            [0, 2, 2, 0],
            [0, 2, 2, 0],
            [1, 0, 0, 1]
        ]))

        grids.append(np.array([
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [9, 1, 2, 3],
            [4, 5, 6, 7]
        ]))

        # Cross patterns
        grid = np.zeros((5, 5), dtype=int)
        grid[2, :] = 1
        grid[:, 2] = 1
        grids.append(grid)

        # Borders
        grid = np.zeros((5, 5), dtype=int)
        grid[0, :] = 1
        grid[-1, :] = 1
        grid[:, 0] = 1
        grid[:, -1] = 1
        grids.append(grid)

        # Random dense
        for _ in range(10):
            grid = np.random.randint(0, 6, (4, 4))
            grids.append(grid)

        return grids

    def _generate_hard_grids(self) -> List[np.ndarray]:
        """6x6 to 10x10 grids"""
        grids = []

        # Checkerboard
        grid = np.zeros((6, 6), dtype=int)
        grid[::2, ::2] = 1
        grid[1::2, 1::2] = 2
        grids.append(grid)

        # Concentric squares
        grid = np.zeros((7, 7), dtype=int)
        grid[0, :] = 1
        grid[-1, :] = 1
        grid[:, 0] = 1
        grid[:, -1] = 1
        grid[1:-1, 1:-1] = np.full((5, 5), 2)
        grid[2:-2, 2:-2] = np.full((3, 3), 3)
        grids.append(grid)

        # Random complex
        for size in [6, 7, 8, 10]:
            for _ in range(3):
                grid = np.random.randint(0, 10, (size, size))
                grids.append(grid)

        return grids

    def generate_task(self, primitive: str, grid: np.ndarray, difficulty: str, task_id: int) -> AtomicTask:
        """Generate single task by applying primitive to grid"""

        grid_state = GridState.from_array(grid)
        output_state = self.executor.execute(grid_state, primitive)

        if output_state is None:
            # Primitive not applicable to this grid - return identity
            output_grid = grid
        else:
            output_grid = output_state.to_array()

        return AtomicTask(
            task_id=f"atomic_{primitive}_{difficulty}_{task_id:04d}",
            primitive=primitive,
            input_grid=grid.tolist(),
            output_grid=output_grid.tolist(),
            difficulty=difficulty
        )

    def generate_dataset(self, tasks_per_primitive: int = 50) -> List[AtomicTask]:
        """
        Generate full dataset of atomic tasks.

        Args:
            tasks_per_primitive: Number of tasks per primitive (default 50)
                                With 15 primitives → 750 total tasks

        Returns:
            List of AtomicTask objects
        """
        tasks = []
        task_id_counter = 0

        print(f"Generating {tasks_per_primitive} tasks for each of {len(self.primitives)} primitives...")

        for prim in self.primitives:
            print(f"  Generating tasks for: {prim}")

            # Distribution: 30% easy, 40% medium, 30% hard
            num_easy = int(tasks_per_primitive * 0.3)
            num_medium = int(tasks_per_primitive * 0.4)
            num_hard = tasks_per_primitive - num_easy - num_medium

            # Easy tasks
            for i in range(num_easy):
                grid = self.easy_grids[i % len(self.easy_grids)]
                task = self.generate_task(prim, grid, 'easy', task_id_counter)
                tasks.append(task)
                task_id_counter += 1

            # Medium tasks
            for i in range(num_medium):
                grid = self.medium_grids[i % len(self.medium_grids)]
                task = self.generate_task(prim, grid, 'medium', task_id_counter)
                tasks.append(task)
                task_id_counter += 1

            # Hard tasks
            for i in range(num_hard):
                grid = self.hard_grids[i % len(self.hard_grids)]
                task = self.generate_task(prim, grid, 'hard', task_id_counter)
                tasks.append(task)
                task_id_counter += 1

        print(f"\n✅ Generated {len(tasks)} atomic tasks")
        return tasks

    def save_dataset(self, tasks: List[AtomicTask], filepath: str):
        """Save tasks to JSON file"""
        data = {
            'num_tasks': len(tasks),
            'num_primitives': len(self.primitives),
            'primitives': self.primitives,
            'tasks': [asdict(task) for task in tasks]
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"✅ Saved dataset to: {filepath}")

    def compute_statistics(self, tasks: List[AtomicTask]):
        """Print dataset statistics"""
        from collections import Counter

        print("\n" + "="*70)
        print("DATASET STATISTICS")
        print("="*70)

        # Total tasks
        print(f"Total tasks: {len(tasks)}")

        # Tasks per difficulty
        difficulties = [t.difficulty for t in tasks]
        diff_counts = Counter(difficulties)
        print(f"\nTasks by difficulty:")
        for diff, count in diff_counts.items():
            print(f"  {diff}: {count} ({100*count/len(tasks):.1f}%)")

        # Tasks per primitive
        primitives = [t.primitive for t in tasks]
        prim_counts = Counter(primitives)
        print(f"\nTasks per primitive:")
        for prim, count in sorted(prim_counts.items(), key=lambda x: -x[1])[:10]:
            print(f"  {prim}: {count}")

        # Grid size distribution
        sizes = [(len(t.input_grid), len(t.input_grid[0])) for t in tasks]
        size_counts = Counter(sizes)
        print(f"\nMost common grid sizes:")
        for size, count in size_counts.most_common(10):
            print(f"  {size[0]}x{size[1]}: {count}")

        print("="*70)


def test_atomic_tasks():
    """Quick test of atomic task generation"""
    print("="*70)
    print("ATOMIC TASK GENERATION TEST")
    print("="*70)

    generator = AtomicTaskGenerator()

    # Test single task
    test_grid = np.array([[1, 2], [3, 4]])
    task = generator.generate_task('rotate_90_cw', test_grid, 'easy', 0)

    print(f"\nTest task: {task.task_id}")
    print(f"Primitive: {task.primitive}")
    print(f"Input:")
    print(np.array(task.input_grid))
    print(f"Output:")
    print(np.array(task.output_grid))

    # Verify correctness
    expected_output = np.array([[3, 1], [4, 2]])
    assert np.array_equal(task.output_grid, expected_output), "Output mismatch!"

    print("\n✅ Single task generation working correctly")


def generate_full_dataset(tasks_per_primitive: int = 70):
    """
    Generate full training dataset.

    With 15 primitives × 70 tasks = 1050 total tasks
    Covers all primitives with diverse examples
    """
    print("\n" + "="*70)
    print("GENERATING FULL ATOMIC TASK DATASET")
    print("="*70)

    generator = AtomicTaskGenerator()

    # Generate tasks
    tasks = generator.generate_dataset(tasks_per_primitive=tasks_per_primitive)

    # Save to file
    output_path = '/home/user/HungryOrca/SubtleGenius/primitives/atomic_tasks_dataset.json'
    generator.save_dataset(tasks, output_path)

    # Statistics
    generator.compute_statistics(tasks)

    return tasks


if __name__ == '__main__':
    # Test first
    test_atomic_tasks()

    # Generate full dataset
    tasks = generate_full_dataset(tasks_per_primitive=70)

    print("\n" + "="*70)
    print("✅ ATOMIC TASK DATASET READY FOR TRAINING")
    print("="*70)
    print(f"Use this dataset to:")
    print(f"  1. Pre-train PrimitiveRanker (supervised learning)")
    print(f"  2. Pre-train ProgramEncoder (contrastive learning)")
    print(f"  3. Benchmark: Zero-shot (symbolic only) vs Few-shot (after training)")
    print("="*70)
