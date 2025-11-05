#!/usr/bin/env python3
"""
Evolutionary AGI ARC Solver - "Darwin's Orca"
Evolves solver functions over generations using genetic algorithms
Based on ctf.txt + OIS Framework principles

Architecture:
- Population of solver "DNA" (transformation pipelines)
- Fitness evaluation on training tasks
- Genetic operators: crossover, mutation, selection
- Evolves optimized solver code over generations
"""

import json
import random
import copy
from typing import List, Dict, Any, Callable, Tuple

# =====================================================
# ATOMIC OPERATIONS - THE GENE POOL
# =====================================================

class AtomicOp:
    """Atomic transformation operations - building blocks of DNA"""

    @staticmethod
    def copy(g):
        return [r[:] for r in g]

    @staticmethod
    def refl_y(g):
        return g[::-1]

    @staticmethod
    def refl_x(g):
        return [r[::-1] for r in g]

    @staticmethod
    def rot90(g):
        h, w = len(g), len(g[0]) if g else 0
        return [[g[h-1-y][x] for y in range(h)] for x in range(w)]

    @staticmethod
    def rot180(g):
        return AtomicOp.refl_y(AtomicOp.refl_x(g))

    @staticmethod
    def rot270(g):
        return AtomicOp.rot90(AtomicOp.rot90(AtomicOp.rot90(g)))

    @staticmethod
    def transpose(g):
        h, w = len(g), len(g[0]) if g else 0
        return [[g[y][x] for y in range(h)] for x in range(w)]

    @staticmethod
    def tile_2x2(g):
        h, w = len(g), len(g[0]) if g else 0
        r = [[0]*(w*2) for _ in range(h*2)]
        for ty in range(2):
            for tx in range(2):
                for y in range(h):
                    for x in range(w):
                        r[ty*h+y][tx*w+x] = g[y][x]
        return r

    @staticmethod
    def tile_3x3(g):
        h, w = len(g), len(g[0]) if g else 0
        r = [[0]*(w*3) for _ in range(h*3)]
        for ty in range(3):
            for tx in range(3):
                for y in range(h):
                    for x in range(w):
                        r[ty*h+y][tx*w+x] = g[y][x]
        return r

    @staticmethod
    def color_increment(g):
        """Increment all non-zero colors by 1"""
        h, w = len(g), len(g[0]) if g else 0
        return [[(g[y][x]+1) if g[y][x]>0 else 0 for x in range(w)] for y in range(h)]

    @staticmethod
    def color_decrement(g):
        """Decrement all non-zero colors by 1"""
        h, w = len(g), len(g[0]) if g else 0
        return [[max(0,g[y][x]-1) if g[y][x]>0 else 0 for x in range(w)] for y in range(h)]

    @staticmethod
    def scale_2x(g):
        """Scale up by 2x"""
        h, w = len(g), len(g[0]) if g else 0
        r = [[0]*(w*2) for _ in range(h*2)]
        for y in range(h):
            for x in range(w):
                r[y*2][x*2] = g[y][x]
                r[y*2][x*2+1] = g[y][x]
                r[y*2+1][x*2] = g[y][x]
                r[y*2+1][x*2+1] = g[y][x]
        return r

# Gene pool - all available operations
GENE_POOL = [
    ('copy', AtomicOp.copy),
    ('refl_y', AtomicOp.refl_y),
    ('refl_x', AtomicOp.refl_x),
    ('rot90', AtomicOp.rot90),
    ('rot180', AtomicOp.rot180),
    ('rot270', AtomicOp.rot270),
    ('transpose', AtomicOp.transpose),
    ('tile_2x2', AtomicOp.tile_2x2),
    ('tile_3x3', AtomicOp.tile_3x3),
    ('color_inc', AtomicOp.color_increment),
    ('color_dec', AtomicOp.color_decrement),
    ('scale_2x', AtomicOp.scale_2x),
]

# =====================================================
# DNA - TRANSFORMATION PIPELINE
# =====================================================

class SolverDNA:
    """
    DNA of a solver - a sequence of transformation operations
    Represents a complete solver as an executable pipeline
    """

    def __init__(self, genes: List[str] = None, max_length: int = 5):
        self.max_length = max_length
        if genes is None:
            # Random initialization
            length = random.randint(1, max_length)
            self.genes = [random.choice(GENE_POOL)[0] for _ in range(length)]
        else:
            self.genes = genes[:max_length]

        self.fitness = 0.0

    def execute(self, grid: List[List[int]]) -> List[List[int]]:
        """Execute the transformation pipeline"""
        try:
            result = grid
            for gene in self.genes:
                # Find operation by name
                op_func = next((func for name, func in GENE_POOL if name == gene), None)
                if op_func:
                    result = op_func(result)
            return result
        except:
            return grid

    def mutate(self, mutation_rate: float = 0.1):
        """Mutate DNA - modify, add, or remove genes"""
        i = 0
        while i < len(self.genes):
            if random.random() < mutation_rate:
                # Random mutation
                mutation_type = random.choice(['modify', 'insert', 'delete'])

                if mutation_type == 'modify':
                    self.genes[i] = random.choice(GENE_POOL)[0]
                    i += 1
                elif mutation_type == 'insert' and len(self.genes) < self.max_length:
                    self.genes.insert(i, random.choice(GENE_POOL)[0])
                    i += 2  # Skip the inserted gene
                elif mutation_type == 'delete' and len(self.genes) > 1:
                    self.genes.pop(i)
                    # Don't increment i, as we removed an element
                else:
                    i += 1
            else:
                i += 1

    def __repr__(self):
        return f"DNA({' -> '.join(self.genes)}) [fitness: {self.fitness:.3f}]"

# =====================================================
# EVOLUTIONARY ENGINE
# =====================================================

class EvolutionaryEngine:
    """
    Main evolutionary engine
    Evolves population of solvers over generations
    """

    def __init__(self, population_size: int = 50, max_generations: int = 20):
        self.population_size = population_size
        self.max_generations = max_generations
        self.population: List[SolverDNA] = []
        self.best_solver: SolverDNA = None
        self.generation = 0

    def initialize_population(self):
        """Create initial random population"""
        self.population = [SolverDNA() for _ in range(self.population_size)]
        print(f"Initialized population of {self.population_size} solvers")

    def evaluate_fitness(self, tasks: Dict[str, Any], solutions: Dict[str, Any], max_tasks: int = 20):
        """
        Evaluate fitness of all solvers in population
        Fitness = number of training tasks solved correctly
        """
        task_list = list(tasks.items())[:max_tasks]

        for solver in self.population:
            correct = 0
            total = 0

            for task_id, task in task_list:
                try:
                    if 'train' not in task or len(task['train']) == 0:
                        continue

                    # Test on first training example
                    train_input = task['train'][0]['input']
                    train_output = task['train'][0]['output']

                    predicted = solver.execute(train_input)

                    if predicted == train_output:
                        correct += 1

                    total += 1
                except:
                    pass

            solver.fitness = correct / total if total > 0 else 0.0

    def selection(self) -> List[SolverDNA]:
        """
        Tournament selection - select best performers
        """
        # Sort by fitness
        sorted_pop = sorted(self.population, key=lambda s: s.fitness, reverse=True)

        # Keep top 50%
        elite_size = self.population_size // 2
        return sorted_pop[:elite_size]

    def crossover(self, parent1: SolverDNA, parent2: SolverDNA) -> SolverDNA:
        """
        Crossover - combine DNA from two parents
        """
        # Random crossover point
        if len(parent1.genes) > 0 and len(parent2.genes) > 0:
            point1 = random.randint(0, len(parent1.genes))
            point2 = random.randint(0, len(parent2.genes))

            child_genes = parent1.genes[:point1] + parent2.genes[point2:]
            return SolverDNA(genes=child_genes)
        else:
            return SolverDNA(genes=parent1.genes[:])

    def evolve_generation(self):
        """Evolve one generation"""
        # Selection
        elite = self.selection()

        # Generate new population through crossover and mutation
        new_population = elite[:]  # Keep elite

        while len(new_population) < self.population_size:
            # Select two random parents from elite
            parent1 = random.choice(elite)
            parent2 = random.choice(elite)

            # Crossover
            child = self.crossover(parent1, parent2)

            # Mutation
            child.mutate(mutation_rate=0.15)

            new_population.append(child)

        self.population = new_population
        self.generation += 1

        # Track best solver
        self.best_solver = max(self.population, key=lambda s: s.fitness)

    def run_evolution(self, tasks: Dict, solutions: Dict, max_tasks: int = 20):
        """
        Run complete evolutionary cycle
        """
        print("="*60)
        print("EVOLUTIONARY AGI ARC SOLVER - Darwin's Orca")
        print("="*60)

        self.initialize_population()

        for gen in range(self.max_generations):
            # Evaluate fitness
            self.evaluate_fitness(tasks, solutions, max_tasks)

            # Get best solver
            best = max(self.population, key=lambda s: s.fitness)
            avg_fitness = sum(s.fitness for s in self.population) / len(self.population)

            print(f"\nGeneration {gen+1}/{self.max_generations}")
            print(f"  Best Fitness: {best.fitness:.1%}")
            print(f"  Avg Fitness:  {avg_fitness:.1%}")
            print(f"  Best DNA: {best}")

            # Check for perfect solution
            if best.fitness >= 0.95:
                print(f"\n🎯 Found near-perfect solver at generation {gen+1}!")
                break

            # Evolve next generation
            if gen < self.max_generations - 1:
                self.evolve_generation()

        self.best_solver = max(self.population, key=lambda s: s.fitness)

        print("\n" + "="*60)
        print("EVOLUTION COMPLETE")
        print("="*60)
        print(f"Best Solver: {self.best_solver}")
        print(f"Final Fitness: {self.best_solver.fitness:.1%}")

        return self.best_solver

# =====================================================
# META-SOLVER - TASK-SPECIFIC EVOLUTION
# =====================================================

class MetaSolver:
    """
    Meta-solver that evolves task-specific solvers
    Uses evolutionary engine to find best transformation for each task
    """

    def __init__(self):
        self.task_solvers: Dict[str, SolverDNA] = {}

    def evolve_for_task(self, task_id: str, task: Dict, generations: int = 10):
        """
        Evolve a solver specifically for one task
        """
        print(f"\nEvolving solver for task {task_id}...")

        engine = EvolutionaryEngine(population_size=30, max_generations=generations)
        engine.initialize_population()

        for gen in range(generations):
            # Evaluate on this specific task
            for solver in engine.population:
                try:
                    if 'train' in task and len(task['train']) > 0:
                        correct = 0
                        total = len(task['train'])

                        for example in task['train']:
                            predicted = solver.execute(example['input'])
                            if predicted == example['output']:
                                correct += 1

                        solver.fitness = correct / total
                except:
                    solver.fitness = 0.0

            best = max(engine.population, key=lambda s: s.fitness)

            if gen % 3 == 0:
                print(f"  Gen {gen+1}: Best = {best.fitness:.1%}")

            if best.fitness >= 0.99:
                print(f"  ✓ Solved at generation {gen+1}!")
                break

            engine.evolve_generation()

        best_solver = max(engine.population, key=lambda s: s.fitness)
        self.task_solvers[task_id] = best_solver

        return best_solver

    def solve_task(self, task_id: str, task: Dict) -> List[List[int]]:
        """Solve a task using evolved solver"""
        if task_id in self.task_solvers:
            solver = self.task_solvers[task_id]
        else:
            # Evolve on the fly
            solver = self.evolve_for_task(task_id, task, generations=5)

        if 'test' in task and len(task['test']) > 0:
            return solver.execute(task['test'][0]['input'])

        return [[0]]

# =====================================================
# MAIN EXECUTION
# =====================================================

def main():
    print("="*60)
    print("EVOLUTIONARY AGI ARC SOLVER")
    print("Genetic Programming for ARC-AGI Challenge")
    print("="*60)

    # Load training data
    with open('arc-agi_training_challenges.json', 'r') as f:
        training = json.load(f)

    with open('arc-agi_training_solutions.json', 'r') as f:
        solutions = json.load(f)

    # Run evolution on a subset of tasks
    print("\n[1] Global Evolution - Finding universal patterns")
    engine = EvolutionaryEngine(population_size=100, max_generations=30)
    best_global = engine.run_evolution(training, solutions, max_tasks=30)

    # Test best solver
    print("\n[2] Testing best global solver...")
    correct = 0
    total = 0

    for task_id, task in list(training.items())[:50]:
        try:
            if 'train' in task and len(task['train']) > 0:
                predicted = best_global.execute(task['train'][0]['input'])
                expected = task['train'][0]['output']

                if predicted == expected:
                    correct += 1
                    print(f"✓ {task_id}")
                else:
                    print(f"✗ {task_id}")

                total += 1
        except:
            total += 1

    accuracy = correct / total if total > 0 else 0
    print(f"\nGlobal Solver Accuracy: {correct}/{total} = {accuracy:.1%}")

    # Task-specific evolution for hard tasks
    print("\n[3] Task-specific evolution for difficult tasks...")
    meta = MetaSolver()

    difficult_tasks = list(training.items())[:5]  # Try first 5 tasks

    for task_id, task in difficult_tasks:
        solver = meta.evolve_for_task(task_id, task, generations=20)

    print("\n" + "="*60)
    print("✅ Evolutionary solver development complete!")
    print("="*60)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Evolution interrupted by user")
    except FileNotFoundError as e:
        print(f"\n❌ Data files not found: {e}")
        print("Make sure arc-agi training files are in current directory")
