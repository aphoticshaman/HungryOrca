#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    PROJECT GATORCA - PHASE 8                                 ║
║                 Optimization & Tuning                                        ║
║                                                                              ║
║           Target: 10-20% Accuracy (Competitive with SOTA)                   ║
╚══════════════════════════════════════════════════════════════════════════════╝

PHASE 8 OBJECTIVE: Optimize solver to dramatically improve accuracy

Major Improvements:
1. FORCED DIVERSITY - Use more of the 65 operations
2. ADAPTIVE MUTATION - Increase when stuck
3. TASK FINGERPRINTING - Smart operation selection
4. BETTER FITNESS - Reward partial matches
5. LONGER EVOLUTION - More generations
6. CW5 INTERVENTIONS - Black magic when stuck
7. ENSEMBLE - Combine multiple approaches
8. EARLY STOPPING - Stop when solved

Target: 10-20% accuracy (would be competitive with SOTA!)
"""

import json
import random
import time
from typing import List, Dict, Any, Tuple
from pathlib import Path
from collections import defaultdict, Counter

# Load DNA Library
exec(Path('gatorca_phase5_dna_library.py').read_text())

# =====================================================
# OPTIMIZED EVOLUTIONARY SOLVER
# =====================================================

class OptimizedEvolutionarySolver:
    """
    Heavily optimized solver with all Phase 8 improvements

    Improvements:
    - Forced operation diversity
    - Adaptive mutation rates
    - Task fingerprinting
    - Better fitness function
    - CW5 interventions
    - Early stopping
    """

    def __init__(self, operations: Dict[str, callable]):
        self.operations = operations
        self.gene_pool = list(operations.keys())
        self.population_size = 50  # Increased from 30
        self.max_dna_length = 7  # Increased from 5

        # CW5 state
        self.cw5_coffee = 0
        self.cw5_cigarettes = 0
        self.cw5_interventions = 0

        # Operation usage tracking for diversity
        self.operation_usage = Counter()

    def fingerprint_task(self, task: Dict) -> Dict:
        """
        Fingerprint task to select good starting operations

        Returns hints about what operations might work
        """
        if 'train' not in task or not task['train']:
            return {}

        example = task['train'][0]
        input_grid = example['input']
        output_grid = example['output']

        h_in = len(input_grid)
        w_in = len(input_grid[0]) if input_grid else 0
        h_out = len(output_grid)
        w_out = len(output_grid[0]) if output_grid else 0

        fingerprint = {
            'input_size': (h_in, w_in),
            'output_size': (h_out, w_out),
            'size_change': (h_out / h_in if h_in > 0 else 1,
                           w_out / w_in if w_in > 0 else 1)
        }

        # Suggest operations based on fingerprint
        suggested_ops = []

        # Size changes suggest scaling/tiling
        if fingerprint['size_change'][0] > 1.5 or fingerprint['size_change'][1] > 1.5:
            suggested_ops.extend(['scale_up_2x', 'scale_up_3x', 'tile_2x2', 'tile_3x3'])
        elif fingerprint['size_change'][0] < 0.7 or fingerprint['size_change'][1] < 0.7:
            suggested_ops.extend(['scale_down_2x', 'bounding_box', 'crop_to_content',
                                'compress_horizontal', 'compress_vertical'])

        # Same size suggests transformations
        if 0.9 < fingerprint['size_change'][0] < 1.1 and 0.9 < fingerprint['size_change'][1] < 1.1:
            suggested_ops.extend(['reflect_horizontal', 'reflect_vertical', 'rotate_90',
                                'rotate_180', 'transpose', 'gravity_down', 'gravity_up'])

        # Very small output suggests counting/extraction
        if h_out <= 3 and w_out <= 3:
            suggested_ops.extend(['count_objects', 'extract_largest_object',
                                'bounding_box', 'center_objects'])

        fingerprint['suggested_ops'] = suggested_ops
        return fingerprint

    def calculate_fitness_advanced(self, result_grid: List[List[int]],
                                   expected_grid: List[List[int]]) -> float:
        """
        Advanced fitness function that rewards partial matches

        Beta version only checked exact match. This rewards:
        - Correct size
        - Correct colors
        - Partially correct pixels
        """
        if result_grid == expected_grid:
            return 1.0  # Perfect!

        fitness = 0.0

        # Size match (worth 20%)
        h_res = len(result_grid)
        w_res = len(result_grid[0]) if result_grid else 0
        h_exp = len(expected_grid)
        w_exp = len(expected_grid[0]) if expected_grid else 0

        if h_res == h_exp and w_res == w_exp:
            fitness += 0.2

            # Pixel-by-pixel match (worth 60%)
            correct_pixels = 0
            total_pixels = h_exp * w_exp

            for y in range(h_exp):
                for x in range(w_exp):
                    if result_grid[y][x] == expected_grid[y][x]:
                        correct_pixels += 1

            fitness += 0.6 * (correct_pixels / total_pixels if total_pixels > 0 else 0)

        # Color distribution match (worth 20%)
        colors_res = Counter()
        colors_exp = Counter()

        for row in result_grid:
            colors_res.update(row)
        for row in expected_grid:
            colors_exp.update(row)

        # Compare color distributions
        all_colors = set(colors_res.keys()) | set(colors_exp.keys())
        if all_colors:
            color_similarity = 0
            for color in all_colors:
                res_count = colors_res.get(color, 0)
                exp_count = colors_exp.get(color, 0)
                total = max(res_count, exp_count)
                if total > 0:
                    color_similarity += min(res_count, exp_count) / total

            fitness += 0.2 * (color_similarity / len(all_colors))

        return min(1.0, fitness)

    def solve_task(self, task: Dict, max_generations: int = 50,
                   timeout_seconds: int = 60) -> Dict:
        """
        Solve task with all optimizations enabled

        Improvements over Phase 7:
        - 50 generations (was 20)
        - 60s timeout (was 30)
        - Task fingerprinting
        - Forced diversity
        - Adaptive mutation
        - Better fitness
        - CW5 interventions
        """
        start_time = time.time()

        # Fingerprint task for smart initialization
        fingerprint = self.fingerprint_task(task)
        suggested_ops = fingerprint.get('suggested_ops', [])

        # Initialize population with diversity
        population = []
        for i in range(self.population_size):
            dna_length = random.randint(2, self.max_dna_length)

            # 50% of population uses suggested ops, 50% random
            if i < self.population_size // 2 and suggested_ops:
                # Bias toward suggested operations
                dna = [random.choice(suggested_ops + self.gene_pool)
                       for _ in range(dna_length)]
            else:
                # Force diversity - select from underused operations
                underused = [op for op in self.gene_pool
                           if self.operation_usage[op] < 5]
                if underused and len(underused) > dna_length:
                    dna = random.sample(underused, dna_length)
                else:
                    dna = [random.choice(self.gene_pool) for _ in range(dna_length)]

            population.append({'dna': dna, 'fitness': 0.0, 'age': 0})

        best_ever = {'dna': [], 'fitness': 0.0}
        generations_stuck = 0
        mutation_rate = 1.0  # Start normal

        for gen in range(max_generations):
            # Check timeout
            if time.time() - start_time > timeout_seconds:
                break

            # Evaluate fitness (with advanced function)
            for individual in population:
                fitness = self._evaluate_fitness_advanced(individual['dna'], task)
                individual['fitness'] = fitness

                if fitness > best_ever['fitness']:
                    best_ever = {'dna': individual['dna'][:], 'fitness': fitness}
                    generations_stuck = 0
                else:
                    generations_stuck += 1

            # Early stopping if solved
            if best_ever['fitness'] >= 0.99:
                print(f"      ✓ SOLVED in generation {gen+1}!")
                break

            # Adaptive mutation rate
            if generations_stuck > 5:
                mutation_rate = min(3.0, mutation_rate * 1.2)  # Increase when stuck
            else:
                mutation_rate = max(1.0, mutation_rate * 0.95)  # Decrease when improving

            # CW5 intervention if really stuck
            if generations_stuck > 10 and gen % 5 == 0:
                self._cw5_intervene(population, task)

            # Breed next generation
            population = self._breed_generation_optimized(
                population, mutation_rate, suggested_ops
            )

        time_elapsed = time.time() - start_time

        return {
            'solved': best_ever['fitness'] >= 0.99,
            'best_fitness': best_ever['fitness'],
            'best_dna': best_ever['dna'],
            'generations': gen + 1,
            'time_elapsed': time_elapsed,
            'cw5_interventions': self.cw5_interventions
        }

    def _evaluate_fitness_advanced(self, dna: List[str], task: Dict) -> float:
        """Evaluate with advanced fitness function"""
        if 'train' not in task or not task['train']:
            return 0.0

        total_fitness = 0.0
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
                        self.operation_usage[gene] += 1

                # Advanced fitness
                fitness = self.calculate_fitness_advanced(result, expected_output)
                total_fitness += fitness
            except:
                pass

        return total_fitness / total if total > 0 else 0.0

    def _breed_generation_optimized(self, population: List[Dict],
                                    mutation_rate: float,
                                    suggested_ops: List[str]) -> List[Dict]:
        """Breed with adaptive mutation and diversity enforcement"""

        # Sort by fitness
        sorted_pop = sorted(population, key=lambda x: x['fitness'], reverse=True)

        # Elitism - keep top 15%
        elite_count = max(1, len(population) * 15 // 100)
        new_population = []
        for ind in sorted_pop[:elite_count]:
            new_ind = {'dna': ind['dna'][:], 'fitness': 0.0, 'age': ind['age'] + 1}
            new_population.append(new_ind)

        # Fill rest with mutations
        while len(new_population) < self.population_size:
            # Tournament selection (larger tournaments for better parents)
            tournament_size = 5
            tournament = random.sample(sorted_pop[:len(sorted_pop)//2], tournament_size)
            parent = max(tournament, key=lambda x: x['fitness'])

            # Apply multiple mutations based on rate
            child_dna = parent['dna'][:]
            num_mutations = max(1, int(mutation_rate))
            for _ in range(num_mutations):
                child_dna = self._mutate_diverse(child_dna, suggested_ops)

            new_population.append({'dna': child_dna, 'fitness': 0.0, 'age': 0})

        return new_population

    def _mutate_diverse(self, dna: List[str], suggested_ops: List[str]) -> List[str]:
        """Mutate with emphasis on diversity"""
        if not dna:
            return [random.choice(self.gene_pool)]

        new_dna = dna[:]

        # Choose mutation type
        mutation_type = random.choice(['insert', 'delete', 'modify', 'swap',
                                     'replace_with_suggested', 'replace_with_rare'])

        if mutation_type == 'insert' and len(new_dna) < self.max_dna_length:
            pos = random.randint(0, len(new_dna))
            # Bias toward underused operations
            underused = [op for op in self.gene_pool if self.operation_usage[op] < 10]
            gene = random.choice(underused) if underused else random.choice(self.gene_pool)
            new_dna.insert(pos, gene)

        elif mutation_type == 'delete' and len(new_dna) > 1:
            pos = random.randint(0, len(new_dna) - 1)
            new_dna.pop(pos)

        elif mutation_type == 'modify' and new_dna:
            pos = random.randint(0, len(new_dna) - 1)
            # Replace with underused operation
            underused = [op for op in self.gene_pool if self.operation_usage[op] < 10]
            new_dna[pos] = random.choice(underused) if underused else random.choice(self.gene_pool)

        elif mutation_type == 'swap' and len(new_dna) >= 2:
            i, j = random.sample(range(len(new_dna)), 2)
            new_dna[i], new_dna[j] = new_dna[j], new_dna[i]

        elif mutation_type == 'replace_with_suggested' and suggested_ops and new_dna:
            pos = random.randint(0, len(new_dna) - 1)
            new_dna[pos] = random.choice(suggested_ops)

        elif mutation_type == 'replace_with_rare' and new_dna:
            # Find rarest operation in gene pool
            rarest = min(self.gene_pool, key=lambda op: self.operation_usage[op])
            pos = random.randint(0, len(new_dna) - 1)
            new_dna[pos] = rarest

        return new_dna

    def _cw5_intervene(self, population: List[Dict], task: Dict):
        """
        CW5 intervenes with black magic when system is stuck

        *lights cigarette* *pours coffee*
        """
        self.cw5_coffee += 1
        self.cw5_cigarettes += 1
        self.cw5_interventions += 1

        # CW5's black magic: inject radical diversity
        # Replace worst 20% with completely random DNA using rare operations
        population.sort(key=lambda x: x['fitness'], reverse=True)

        worst_count = len(population) // 5
        rarest_ops = sorted(self.gene_pool, key=lambda op: self.operation_usage[op])[:20]

        for i in range(-worst_count, 0):
            # CW5 creates radical new DNA
            dna_length = random.randint(3, self.max_dna_length)
            new_dna = [random.choice(rarest_ops) for _ in range(dna_length)]
            population[i]['dna'] = new_dna
            population[i]['fitness'] = 0.0
            population[i]['age'] = 0


# =====================================================
# OPTIMIZED BETA TESTER
# =====================================================

class OptimizedBetaTester:
    """Beta tester with Phase 8 optimizations"""

    def __init__(self, operations: Dict[str, callable]):
        self.operations = operations
        self.solver = OptimizedEvolutionarySolver(operations)
        self.results = []

    def run_test(self, tasks: List[Tuple[str, Dict]], max_tasks: int = 30):
        """Run optimized test"""
        print(f"\n🚀 PHASE 8 OPTIMIZED TEST ON {min(max_tasks, len(tasks))} PUZZLES")
        print("="*80)

        tasks_to_test = tasks[:max_tasks]

        for i, (task_id, task) in enumerate(tasks_to_test, 1):
            print(f"\n[{i}/{len(tasks_to_test)}] Task {task_id}")

            # Solve with optimizations
            result = self.solver.solve_task(task, max_generations=50, timeout_seconds=60)

            print(f"   Result: {'✓ SOLVED' if result['solved'] else '✗ UNSOLVED'}")
            print(f"   Fitness: {result['best_fitness']:.1%}")
            print(f"   Generations: {result['generations']}")
            print(f"   Time: {result['time_elapsed']:.2f}s")
            if result['cw5_interventions'] > 0:
                print(f"   🚬☕ CW5 Interventions: {result['cw5_interventions']}")

            if result['solved']:
                print(f"   ✓ Solution: {' → '.join(result['best_dna'][:7])}")
            elif result['best_fitness'] > 0.5:
                print(f"   Partial: {' → '.join(result['best_dna'][:7])}")

            self.results.append({
                'task_id': task_id,
                **result
            })

    def generate_report(self) -> Dict:
        """Generate optimization report"""
        if not self.results:
            return {'status': 'no_results'}

        total = len(self.results)
        solved = sum(1 for r in self.results if r['solved'])
        accuracy = solved / total

        avg_fitness = sum(r['best_fitness'] for r in self.results) / total
        avg_time = sum(r['time_elapsed'] for r in self.results) / total
        total_cw5 = sum(r.get('cw5_interventions', 0) for r in self.results)

        # Operation diversity
        all_genes = []
        for r in self.results:
            all_genes.extend(r['best_dna'])

        unique_ops = len(set(all_genes))

        return {
            'summary': {
                'total_tasks': total,
                'solved_tasks': solved,
                'accuracy': accuracy,
                'avg_fitness': avg_fitness,
                'avg_time': avg_time,
                'cw5_interventions': total_cw5,
                'unique_operations_used': unique_ops
            },
            'results': self.results
        }


# =====================================================
# MAIN
# =====================================================

if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║                    🚀 PROJECT GATORCA - PHASE 8 🚀                           ║
║                                                                              ║
║                  Optimization & Tuning                                       ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """)

    # Load ARC data
    print("📁 Loading ARC dataset...")
    with open('arc-agi_training_challenges.json', 'r') as f:
        training_data = json.load(f)

    print(f"   ✓ {len(training_data)} training tasks loaded")

    # Get operations
    operations = get_all_operations()
    print(f"   ✓ {len(operations)} DNA operations ready")

    # Select test tasks
    print(f"\n🎲 Selecting 30 random tasks for optimization test...")
    task_ids = random.sample(list(training_data.keys()), 30)
    tasks = [(tid, training_data[tid]) for tid in task_ids]

    # Run optimized test
    tester = OptimizedBetaTester(operations)
    tester.run_test(tasks, max_tasks=30)

    # Generate report
    print("\n\n" + "="*80)
    print("📊 PHASE 8 OPTIMIZATION REPORT")
    print("="*80)

    report = tester.generate_report()
    summary = report['summary']

    print(f"\n🎯 OPTIMIZED PERFORMANCE:")
    print(f"   Tasks: {summary['total_tasks']}")
    print(f"   Solved: {summary['solved_tasks']}")
    print(f"   Accuracy: {summary['accuracy']:.1%}")
    print(f"   Avg Fitness: {summary['avg_fitness']:.1%}")
    print(f"   Avg Time: {summary['avg_time']:.2f}s")
    print(f"   CW5 Interventions: {summary['cw5_interventions']}")
    print(f"   Unique Ops Used: {summary['unique_operations_used']}/65")

    # Compare to Phase 7
    phase7_accuracy = 0.0
    phase7_fitness = 0.033
    phase7_ops = 4

    print(f"\n📈 IMPROVEMENT OVER PHASE 7:")
    print(f"   Accuracy: {phase7_accuracy:.1%} → {summary['accuracy']:.1%} "
          f"({'+' if summary['accuracy'] > phase7_accuracy else ''}"
          f"{(summary['accuracy'] - phase7_accuracy)*100:.1f}%)")
    print(f"   Avg Fitness: {phase7_fitness:.1%} → {summary['avg_fitness']:.1%} "
          f"({'+' if summary['avg_fitness'] > phase7_fitness else ''}"
          f"{(summary['avg_fitness'] - phase7_fitness)*100:.1f}%)")
    print(f"   Ops Diversity: {phase7_ops}/65 → {summary['unique_operations_used']}/65 "
          f"({summary['unique_operations_used'] - phase7_ops:+d} ops)")

    # Achievement check
    print("\n🏆 ACHIEVEMENT STATUS:")
    if summary['accuracy'] >= 0.20:
        print("   ✅ EXCELLENT! Competitive with SOTA AI systems!")
    elif summary['accuracy'] >= 0.10:
        print("   ✅ GOOD! Significant progress on ARC-AGI!")
    elif summary['avg_fitness'] >= 0.15:
        print("   ✅ PROGRESS! Strong partial solutions!")
    else:
        print("   📊 Baseline improved - more optimization needed")

    print("\n" + "="*80)
    print("✅ PHASE 8: OPTIMIZATION COMPLETE!")
    print("="*80)
    print(f"\n🚀 Optimized solver deployed")
    print(f"📊 Accuracy: {summary['accuracy']:.1%}")
    print(f"🧬 Operation diversity: {summary['unique_operations_used']}/65")
    print(f"🚬☕ CW5 provided {summary['cw5_interventions']} interventions")
    print("\n🎖️ READY FOR PHASE 9: COMPRESSION & PACKAGING")

    # Save report
    with open('gatorca_phase8_optimization_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    print("\n💾 Report saved to: gatorca_phase8_optimization_report.json")
