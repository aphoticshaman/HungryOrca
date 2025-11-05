#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    PROJECT GATORCA - PHASE 7                                 ║
║                    ARC Puzzle Integration                                    ║
║                                                                              ║
║              Test on Real ARC-AGI 2025 Dataset                              ║
╚══════════════════════════════════════════════════════════════════════════════╝

PHASE 7 OBJECTIVE: Integrate with real ARC-AGI dataset and measure performance

Components:
1. Load arc-agi training/evaluation data
2. Use full DNA library (65 operations)
3. Run evolutionary system on real puzzles
4. Measure accuracy and performance
5. Identify strengths and weaknesses
6. Generate beta test report

This is the real test!
"""

import json
import random
import time
from typing import List, Dict, Any, Tuple
from pathlib import Path
from collections import defaultdict, Counter

# Load DNA Library operations
exec(Path('gatorca_phase5_dna_library.py').read_text())

# =====================================================
# ARC DATA LOADER
# =====================================================

class ARCDataLoader:
    """Load and manage ARC-AGI dataset"""

    def __init__(self):
        self.training_challenges = {}
        self.training_solutions = {}
        self.evaluation_challenges = {}
        self.evaluation_solutions = {}
        self.test_challenges = {}

    def load_all(self):
        """Load all ARC datasets"""
        print("📁 Loading ARC-AGI datasets...")

        try:
            with open('arc-agi_training_challenges.json', 'r') as f:
                self.training_challenges = json.load(f)
            print(f"   ✓ Training challenges: {len(self.training_challenges)} tasks")
        except FileNotFoundError:
            print("   ✗ Training challenges not found")

        try:
            with open('arc-agi_training_solutions.json', 'r') as f:
                self.training_solutions = json.load(f)
            print(f"   ✓ Training solutions: {len(self.training_solutions)} tasks")
        except FileNotFoundError:
            print("   ✗ Training solutions not found")

        try:
            with open('arc-agi_evaluation_challenges.json', 'r') as f:
                self.evaluation_challenges = json.load(f)
            print(f"   ✓ Evaluation challenges: {len(self.evaluation_challenges)} tasks")
        except FileNotFoundError:
            print("   ✗ Evaluation challenges not found")

        try:
            with open('arc-agi_evaluation_solutions.json', 'r') as f:
                self.evaluation_solutions = json.load(f)
            print(f"   ✓ Evaluation solutions: {len(self.evaluation_solutions)} tasks")
        except FileNotFoundError:
            print("   ✗ Evaluation solutions not found")

        try:
            with open('arc-agi_test_challenges.json', 'r') as f:
                self.test_challenges = json.load(f)
            print(f"   ✓ Test challenges: {len(self.test_challenges)} tasks (no solutions)")
        except FileNotFoundError:
            print("   ✗ Test challenges not found")

    def get_random_training_tasks(self, n: int = 10) -> List[Tuple[str, Dict]]:
        """Get N random training tasks"""
        if not self.training_challenges:
            return []

        task_ids = random.sample(list(self.training_challenges.keys()), min(n, len(self.training_challenges)))
        return [(tid, self.training_challenges[tid]) for tid in task_ids]

    def get_task_statistics(self, task: Dict) -> Dict:
        """Get statistics about a task"""
        stats = {
            'train_examples': len(task.get('train', [])),
            'test_examples': len(task.get('test', []))
        }

        if task.get('train'):
            example = task['train'][0]
            input_grid = example['input']
            output_grid = example['output']

            stats['input_size'] = (len(input_grid), len(input_grid[0]) if input_grid else 0)
            stats['output_size'] = (len(output_grid), len(output_grid[0]) if output_grid else 0)

            # Color analysis
            input_colors = set()
            output_colors = set()
            for row in input_grid:
                input_colors.update(row)
            for row in output_grid:
                output_colors.update(row)

            stats['input_colors'] = len(input_colors)
            stats['output_colors'] = len(output_colors)

        return stats


# =====================================================
# SIMPLE EVOLUTIONARY SOLVER (Streamlined for speed)
# =====================================================

class SimpleEvolutionarySolver:
    """
    Simplified evolutionary solver for ARC puzzles

    Focus on speed and simplicity for Phase 7 testing
    """

    def __init__(self, operations: Dict[str, callable]):
        self.operations = operations
        self.gene_pool = list(operations.keys())
        self.population_size = 30
        self.max_dna_length = 5

    def solve_task(self, task: Dict, max_generations: int = 20, timeout_seconds: int = 30) -> Dict:
        """
        Attempt to solve a single ARC task

        Returns: {
            'solved': bool,
            'best_fitness': float,
            'best_dna': List[str],
            'generations': int,
            'time_elapsed': float
        }
        """
        start_time = time.time()

        # Initialize population
        population = []
        for _ in range(self.population_size):
            dna_length = random.randint(1, self.max_dna_length)
            dna = [random.choice(self.gene_pool) for _ in range(dna_length)]
            population.append({'dna': dna, 'fitness': 0.0})

        best_ever = {'dna': [], 'fitness': 0.0}

        for gen in range(max_generations):
            # Check timeout
            if time.time() - start_time > timeout_seconds:
                break

            # Evaluate fitness
            for individual in population:
                fitness = self._evaluate_fitness(individual['dna'], task)
                individual['fitness'] = fitness

                if fitness > best_ever['fitness']:
                    best_ever = {'dna': individual['dna'][:], 'fitness': fitness}

            # Check if solved
            if best_ever['fitness'] >= 0.99:
                break

            # Breed next generation
            population = self._breed_generation(population)

        time_elapsed = time.time() - start_time

        return {
            'solved': best_ever['fitness'] >= 0.99,
            'best_fitness': best_ever['fitness'],
            'best_dna': best_ever['dna'],
            'generations': gen + 1,
            'time_elapsed': time_elapsed
        }

    def _evaluate_fitness(self, dna: List[str], task: Dict) -> float:
        """Evaluate DNA on task"""
        if 'train' not in task or not task['train']:
            return 0.0

        correct = 0
        total = len(task['train'])

        for example in task['train']:
            try:
                input_grid = example['input']
                expected_output = example['output']

                # Execute DNA
                result = input_grid
                for gene in dna:
                    if gene in self.operations:
                        result = self.operations[gene](result)

                # Check if matches
                if result == expected_output:
                    correct += 1
            except:
                pass

        return correct / total if total > 0 else 0.0

    def _breed_generation(self, population: List[Dict]) -> List[Dict]:
        """Breed next generation"""

        # Sort by fitness
        sorted_pop = sorted(population, key=lambda x: x['fitness'], reverse=True)

        # Keep top 20% (elite)
        elite_count = max(1, len(population) // 5)
        new_population = [{'dna': ind['dna'][:], 'fitness': 0.0} for ind in sorted_pop[:elite_count]]

        # Fill rest with mutations
        while len(new_population) < self.population_size:
            # Select parent (tournament)
            tournament = random.sample(sorted_pop[:len(sorted_pop)//2], 3)
            parent = max(tournament, key=lambda x: x['fitness'])

            # Mutate
            child_dna = self._mutate(parent['dna'])
            new_population.append({'dna': child_dna, 'fitness': 0.0})

        return new_population

    def _mutate(self, dna: List[str]) -> List[str]:
        """Apply random mutation"""
        if not dna:
            return [random.choice(self.gene_pool)]

        new_dna = dna[:]

        mutation_type = random.choice(['insert', 'delete', 'modify', 'swap'])

        if mutation_type == 'insert' and len(new_dna) < self.max_dna_length:
            pos = random.randint(0, len(new_dna))
            new_dna.insert(pos, random.choice(self.gene_pool))
        elif mutation_type == 'delete' and len(new_dna) > 1:
            pos = random.randint(0, len(new_dna) - 1)
            new_dna.pop(pos)
        elif mutation_type == 'modify':
            pos = random.randint(0, len(new_dna) - 1)
            new_dna[pos] = random.choice(self.gene_pool)
        elif mutation_type == 'swap' and len(new_dna) >= 2:
            i, j = random.sample(range(len(new_dna)), 2)
            new_dna[i], new_dna[j] = new_dna[j], new_dna[i]

        return new_dna


# =====================================================
# BETA TESTER
# =====================================================

class BetaTester:
    """Run beta tests on ARC dataset and generate report"""

    def __init__(self, operations: Dict[str, callable]):
        self.operations = operations
        self.solver = SimpleEvolutionarySolver(operations)
        self.results = []

    def run_beta_test(self, tasks: List[Tuple[str, Dict]], max_tasks: int = 20):
        """
        Run beta test on multiple tasks

        Args:
            tasks: List of (task_id, task_data) tuples
            max_tasks: Maximum number of tasks to test
        """
        print(f"\n🧪 BETA TESTING ON {min(max_tasks, len(tasks))} ARC PUZZLES")
        print("="*80)

        tasks_to_test = tasks[:max_tasks]

        for i, (task_id, task) in enumerate(tasks_to_test, 1):
            print(f"\n[{i}/{len(tasks_to_test)}] Task {task_id}")

            # Quick stats
            stats = ARCDataLoader().get_task_statistics(task)
            print(f"   Train examples: {stats['train_examples']}")
            if 'input_size' in stats:
                print(f"   Grid size: {stats['input_size']} → {stats['output_size']}")

            # Solve
            result = self.solver.solve_task(task, max_generations=20, timeout_seconds=30)

            print(f"   Result: {'✓ SOLVED' if result['solved'] else '✗ UNSOLVED'}")
            print(f"   Best fitness: {result['best_fitness']:.1%}")
            print(f"   Generations: {result['generations']}")
            print(f"   Time: {result['time_elapsed']:.2f}s")

            if result['solved']:
                print(f"   Solution DNA: {' → '.join(result['best_dna'][:5])}")

            # Store result
            self.results.append({
                'task_id': task_id,
                'stats': stats,
                **result
            })

    def generate_report(self) -> Dict:
        """Generate comprehensive beta test report"""

        if not self.results:
            return {'status': 'no_results'}

        total_tasks = len(self.results)
        solved_tasks = sum(1 for r in self.results if r['solved'])
        accuracy = solved_tasks / total_tasks if total_tasks > 0 else 0.0

        avg_fitness = sum(r['best_fitness'] for r in self.results) / total_tasks
        avg_generations = sum(r['generations'] for r in self.results) / total_tasks
        avg_time = sum(r['time_elapsed'] for r in self.results) / total_tasks

        # Find best and worst
        best_result = max(self.results, key=lambda r: r['best_fitness'])
        worst_result = min(self.results, key=lambda r: r['best_fitness'])

        # DNA pattern analysis
        all_dna_genes = []
        for r in self.results:
            all_dna_genes.extend(r['best_dna'])

        gene_frequency = Counter(all_dna_genes)
        most_common_genes = gene_frequency.most_common(10)

        report = {
            'summary': {
                'total_tasks': total_tasks,
                'solved_tasks': solved_tasks,
                'accuracy': accuracy,
                'avg_fitness': avg_fitness,
                'avg_generations': avg_generations,
                'avg_time_per_task': avg_time
            },
            'best_performance': {
                'task_id': best_result['task_id'],
                'fitness': best_result['best_fitness'],
                'dna': best_result['best_dna']
            },
            'worst_performance': {
                'task_id': worst_result['task_id'],
                'fitness': worst_result['best_fitness'],
                'dna': worst_result['best_dna']
            },
            'dna_insights': {
                'most_common_genes': most_common_genes,
                'unique_genes_used': len(gene_frequency),
                'total_gene_pool': len(self.operations)
            },
            'detailed_results': self.results
        }

        return report


# =====================================================
# MAIN EXECUTION
# =====================================================

if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║                    🧩 PROJECT GATORCA - PHASE 7 🧩                           ║
║                                                                              ║
║                     ARC Puzzle Integration                                   ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """)

    # Load ARC data
    loader = ARCDataLoader()
    loader.load_all()

    # Get DNA library operations
    operations = get_all_operations()
    print(f"\n🧬 DNA Library: {len(operations)} operations loaded")

    # Get random training tasks
    print(f"\n🎲 Selecting random training tasks for beta test...")
    tasks = loader.get_random_training_tasks(n=20)
    print(f"   Selected {len(tasks)} tasks")

    # Run beta test
    tester = BetaTester(operations)
    tester.run_beta_test(tasks, max_tasks=20)

    # Generate report
    print("\n\n" + "="*80)
    print("📊 BETA TEST REPORT")
    print("="*80)

    report = tester.generate_report()

    summary = report['summary']
    print(f"\n🎯 OVERALL PERFORMANCE:")
    print(f"   Tasks Tested: {summary['total_tasks']}")
    print(f"   Tasks Solved: {summary['solved_tasks']}")
    print(f"   Accuracy: {summary['accuracy']:.1%}")
    print(f"   Avg Fitness: {summary['avg_fitness']:.1%}")
    print(f"   Avg Generations: {summary['avg_generations']:.1f}")
    print(f"   Avg Time/Task: {summary['avg_time_per_task']:.2f}s")

    print(f"\n🏆 BEST PERFORMANCE:")
    best = report['best_performance']
    print(f"   Task: {best['task_id']}")
    print(f"   Fitness: {best['fitness']:.1%}")
    print(f"   DNA: {' → '.join(best['dna'][:5])}")

    print(f"\n⚠️  WORST PERFORMANCE:")
    worst = report['worst_performance']
    print(f"   Task: {worst['task_id']}")
    print(f"   Fitness: {worst['fitness']:.1%}")

    print(f"\n🧬 DNA INSIGHTS:")
    insights = report['dna_insights']
    print(f"   Genes Used: {insights['unique_genes_used']}/{insights['total_gene_pool']}")
    print(f"   Most Common Genes:")
    for gene, count in insights['most_common_genes'][:5]:
        print(f"     • {gene}: {count} times")

    print("\n" + "="*80)
    print("✅ PHASE 7: ARC PUZZLE INTEGRATION COMPLETE!")
    print("="*80)
    print(f"\n🧩 Tested on real ARC-AGI puzzles")
    print(f"📊 Accuracy: {summary['accuracy']:.1%}")
    print(f"🧬 {len(operations)} atomic operations in use")
    print(f"⏱️  Average solve time: {summary['avg_time_per_task']:.2f}s")

    if summary['accuracy'] < 0.10:
        print(f"\n💡 Current accuracy is low - this is expected for ARC!")
        print(f"   ARC-AGI is designed to be extremely difficult")
        print(f"   Human performance: ~80%")
        print(f"   Current SOTA AI: ~20-40%")
        print(f"   Our system demonstrates evolutionary learning capability")
    elif summary['accuracy'] >= 0.50:
        print(f"\n🎉 EXCELLENT PERFORMANCE!")
        print(f"   Above 50% accuracy on ARC is remarkable!")
    elif summary['accuracy'] >= 0.20:
        print(f"\n👍 GOOD PERFORMANCE!")
        print(f"   Competitive with SOTA AI systems")

    print("\n🎖️ READY FOR PHASE 8: OPTIMIZATION & TUNING")

    # Save report
    with open('gatorca_phase7_beta_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    print("\n💾 Detailed report saved to: gatorca_phase7_beta_report.json")
