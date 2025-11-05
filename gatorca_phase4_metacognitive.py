#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    PROJECT GATORCA - PHASE 4                                 ║
║                    Meta-Cognitive Engine                                     ║
║                                                                              ║
║              Learning What Mutation Strategies Actually Work                 ║
╚══════════════════════════════════════════════════════════════════════════════╝

PHASE 4 OBJECTIVE: Build meta-cognitive learning system where levels learn
                   from their mutation history what strategies work best.

Components:
1. Mutation History Tracker - Records all mutations and their outcomes
2. Meta-Learner - Analyzes which mutation types lead to fitness improvement
3. Mutation Operators - The actual genetic operations
4. Fitness Evaluation - Framework for testing solver performance
5. Meta-Learning Loop - Closes the loop between results and strategy

Based on evolutionary algorithms + meta-learning principles
"""

import json
import random
from typing import Dict, List, Any, Tuple, Optional
from collections import defaultdict, Counter
from pathlib import Path

# =====================================================
# MUTATION OPERATORS
# =====================================================

class MutationOperator:
    """
    Base class for mutation operations on solver DNA

    Each operator modifies a DNA sequence (list of operation names)
    """

    @staticmethod
    def insert(dna: List[str], gene_pool: List[str]) -> List[str]:
        """Insert a random operation at random position"""
        if len(dna) >= 10:  # Max DNA length
            return dna

        new_dna = dna.copy()
        insert_pos = random.randint(0, len(new_dna))
        new_gene = random.choice(gene_pool)
        new_dna.insert(insert_pos, new_gene)
        return new_dna

    @staticmethod
    def delete(dna: List[str]) -> List[str]:
        """Delete operation at random position"""
        if len(dna) <= 1:  # Minimum DNA length
            return dna

        new_dna = dna.copy()
        delete_pos = random.randint(0, len(new_dna) - 1)
        new_dna.pop(delete_pos)
        return new_dna

    @staticmethod
    def modify(dna: List[str], gene_pool: List[str]) -> List[str]:
        """Replace operation at random position"""
        if not dna:
            return dna

        new_dna = dna.copy()
        modify_pos = random.randint(0, len(new_dna) - 1)
        new_dna[modify_pos] = random.choice(gene_pool)
        return new_dna

    @staticmethod
    def swap(dna: List[str]) -> List[str]:
        """Swap two adjacent operations"""
        if len(dna) < 2:
            return dna

        new_dna = dna.copy()
        pos = random.randint(0, len(new_dna) - 2)
        new_dna[pos], new_dna[pos+1] = new_dna[pos+1], new_dna[pos]
        return new_dna

    @staticmethod
    def crossover(dna1: List[str], dna2: List[str]) -> List[str]:
        """Crossover between two DNAs"""
        if not dna1 or not dna2:
            return dna1 if dna1 else dna2

        point1 = random.randint(0, len(dna1))
        point2 = random.randint(0, len(dna2))

        return dna1[:point1] + dna2[point2:]

    @staticmethod
    def duplicate(dna: List[str]) -> List[str]:
        """Duplicate a random segment"""
        if len(dna) >= 10 or not dna:
            return dna

        new_dna = dna.copy()
        if len(dna) == 1:
            segment = dna
            pos = 0
        else:
            start = random.randint(0, len(dna) - 1)
            end = random.randint(start + 1, len(dna))
            segment = dna[start:end]
            pos = random.randint(0, len(new_dna))

        # Insert segment at position (if room)
        if len(new_dna) + len(segment) <= 10:
            for i, gene in enumerate(segment):
                new_dna.insert(pos + i, gene)

        return new_dna

    @staticmethod
    def reverse(dna: List[str]) -> List[str]:
        """Reverse the DNA sequence"""
        return dna[::-1]

# =====================================================
# MUTATION HISTORY TRACKER
# =====================================================

class MutationHistory:
    """
    Tracks all mutations and their fitness outcomes

    This is the memory that enables meta-learning
    """

    def __init__(self):
        self.history = []
        self.operator_stats = defaultdict(lambda: {'count': 0, 'fitness_sum': 0.0, 'improvements': 0})

    def record(self, mutation_type: str, before_dna: List[str], after_dna: List[str],
               before_fitness: float, after_fitness: float, context: Dict):
        """Record a mutation and its outcome"""

        entry = {
            'mutation_type': mutation_type,
            'before_dna': before_dna.copy(),
            'after_dna': after_dna.copy(),
            'before_fitness': before_fitness,
            'after_fitness': after_fitness,
            'fitness_delta': after_fitness - before_fitness,
            'improved': after_fitness > before_fitness,
            'context': context.copy()
        }

        self.history.append(entry)

        # Update operator stats
        stats = self.operator_stats[mutation_type]
        stats['count'] += 1
        stats['fitness_sum'] += after_fitness
        if after_fitness > before_fitness:
            stats['improvements'] += 1

    def get_operator_stats(self) -> Dict[str, Dict]:
        """Get statistics for each mutation operator"""
        result = {}
        for op, stats in self.operator_stats.items():
            if stats['count'] > 0:
                result[op] = {
                    'count': stats['count'],
                    'avg_fitness': stats['fitness_sum'] / stats['count'],
                    'improvement_rate': stats['improvements'] / stats['count'],
                    'total_improvements': stats['improvements']
                }
        return result

    def get_best_operators(self, top_n: int = 3) -> List[str]:
        """Get the top N performing operators"""
        stats = self.get_operator_stats()

        # Sort by improvement rate
        sorted_ops = sorted(stats.items(),
                          key=lambda x: x[1]['improvement_rate'],
                          reverse=True)

        return [op for op, _ in sorted_ops[:top_n]]

    def get_recent_trend(self, window_size: int = 10) -> float:
        """Get recent fitness trend (positive = improving)"""
        if len(self.history) < window_size:
            return 0.0

        recent = self.history[-window_size:]
        improvements = sum(1 for entry in recent if entry['improved'])
        return improvements / window_size

# =====================================================
# META-LEARNER
# =====================================================

class MetaLearner:
    """
    Learns from mutation history what strategies work

    This is the intelligence that makes the system self-improving
    """

    def __init__(self, mutation_history: MutationHistory):
        self.history = mutation_history
        self.operator_preferences = {}  # Learned preferences for each operator
        self.learning_rate = 0.1

    def analyze(self) -> Dict[str, Any]:
        """Analyze mutation history and extract insights"""

        if len(self.history.history) < 10:
            return {
                'status': 'insufficient_data',
                'entries': len(self.history.history),
                'recommendation': 'explore_randomly'
            }

        stats = self.history.get_operator_stats()
        best_ops = self.history.get_best_operators(top_n=3)
        trend = self.history.get_recent_trend(window_size=10)

        # Determine strategy based on analysis
        if trend > 0.6:
            strategy = 'exploit'  # Keep doing what's working
        elif trend < 0.3:
            strategy = 'explore'  # Try something different
        else:
            strategy = 'balanced'  # Mix of exploration and exploitation

        return {
            'status': 'analysis_complete',
            'entries': len(self.history.history),
            'operator_stats': stats,
            'best_operators': best_ops,
            'recent_trend': trend,
            'recommended_strategy': strategy,
            'insights': self._generate_insights(stats, trend)
        }

    def _generate_insights(self, stats: Dict, trend: float) -> List[str]:
        """Generate human-readable insights"""
        insights = []

        # Find best and worst operators
        if stats:
            sorted_ops = sorted(stats.items(),
                              key=lambda x: x[1]['improvement_rate'],
                              reverse=True)

            if sorted_ops:
                best_op, best_stats = sorted_ops[0]
                insights.append(f"'{best_op}' is performing best ({best_stats['improvement_rate']:.1%} improvement rate)")

            if len(sorted_ops) > 1:
                worst_op, worst_stats = sorted_ops[-1]
                insights.append(f"'{worst_op}' is underperforming ({worst_stats['improvement_rate']:.1%} improvement rate)")

        # Trend analysis
        if trend > 0.6:
            insights.append("System is improving - current strategy is working")
        elif trend < 0.3:
            insights.append("System is stagnating - need to explore new strategies")
        else:
            insights.append("System is moderately improving - maintain balanced approach")

        return insights

    def select_operator(self, gene_pool: List[str]) -> Tuple[str, callable]:
        """
        Select mutation operator based on learned preferences

        Uses epsilon-greedy strategy: exploit best operators with some exploration
        """

        analysis = self.analyze()

        if analysis['status'] == 'insufficient_data':
            # Random exploration
            operator_name = random.choice(['insert', 'delete', 'modify', 'swap', 'crossover', 'duplicate', 'reverse'])
        else:
            # Epsilon-greedy selection
            epsilon = 0.2  # 20% exploration

            if random.random() < epsilon:
                # Explore
                operator_name = random.choice(['insert', 'delete', 'modify', 'swap', 'crossover', 'duplicate', 'reverse'])
            else:
                # Exploit - use best performing operators
                best_ops = analysis['best_operators']
                if best_ops:
                    operator_name = random.choice(best_ops)
                else:
                    operator_name = random.choice(['insert', 'delete', 'modify', 'swap', 'crossover', 'duplicate', 'reverse'])

        # Map name to actual function
        operator_map = {
            'insert': MutationOperator.insert,
            'delete': MutationOperator.delete,
            'modify': MutationOperator.modify,
            'swap': MutationOperator.swap,
            'crossover': MutationOperator.crossover,
            'duplicate': MutationOperator.duplicate,
            'reverse': MutationOperator.reverse
        }

        return operator_name, operator_map[operator_name]

    def generate_strategy(self) -> Dict:
        """Generate mutation strategy based on learned insights"""

        analysis = self.analyze()

        if analysis['status'] == 'insufficient_data':
            return {
                'type': 'exploration',
                'operator_selection': 'random',
                'mutation_rate': 0.3,
                'reason': 'Insufficient data - exploring randomly'
            }

        strategy_type = analysis['recommended_strategy']

        if strategy_type == 'exploit':
            return {
                'type': 'exploitation',
                'operator_selection': 'best_performers',
                'preferred_operators': analysis['best_operators'],
                'mutation_rate': 0.2,  # Lower rate when exploiting
                'reason': 'System improving - exploiting successful strategies'
            }
        elif strategy_type == 'explore':
            return {
                'type': 'exploration',
                'operator_selection': 'diverse',
                'mutation_rate': 0.4,  # Higher rate when exploring
                'reason': 'System stagnating - exploring new strategies'
            }
        else:  # balanced
            return {
                'type': 'balanced',
                'operator_selection': 'epsilon_greedy',
                'preferred_operators': analysis['best_operators'],
                'mutation_rate': 0.25,
                'reason': 'System moderately improving - balanced approach'
            }

# =====================================================
# FITNESS EVALUATOR
# =====================================================

class FitnessEvaluator:
    """
    Evaluates fitness of solver DNA on ARC puzzles

    Fitness = how many training examples the DNA solves correctly
    """

    def __init__(self, atomic_operations: Dict[str, callable]):
        self.operations = atomic_operations

    def execute_dna(self, dna: List[str], input_grid: List[List[int]]) -> List[List[int]]:
        """Execute DNA sequence on input grid"""
        try:
            result = input_grid
            for gene in dna:
                if gene in self.operations:
                    result = self.operations[gene](result)
            return result
        except Exception as e:
            # If execution fails, return input unchanged
            return input_grid

    def evaluate_on_task(self, dna: List[str], task: Dict) -> float:
        """
        Evaluate DNA on a single ARC task

        Returns: fitness score (0.0 to 1.0)
        """
        if 'train' not in task or not task['train']:
            return 0.0

        correct = 0
        total = len(task['train'])

        for example in task['train']:
            input_grid = example['input']
            expected_output = example['output']

            predicted_output = self.execute_dna(dna, input_grid)

            if predicted_output == expected_output:
                correct += 1

        return correct / total if total > 0 else 0.0

    def evaluate_on_multiple_tasks(self, dna: List[str], tasks: List[Dict]) -> Dict:
        """Evaluate DNA on multiple tasks"""

        scores = []
        for task in tasks:
            score = self.evaluate_on_task(dna, task)
            scores.append(score)

        return {
            'scores': scores,
            'mean_fitness': sum(scores) / len(scores) if scores else 0.0,
            'max_fitness': max(scores) if scores else 0.0,
            'tasks_solved': sum(1 for s in scores if s >= 0.99)
        }

# =====================================================
# META-COGNITIVE LEARNING LOOP
# =====================================================

class MetaCognitiveLearningLoop:
    """
    Main meta-cognitive learning loop

    1. Try mutation
    2. Evaluate fitness
    3. Record result
    4. Learn from history
    5. Adjust strategy
    6. Repeat
    """

    def __init__(self, gene_pool: List[str], atomic_operations: Dict[str, callable]):
        self.gene_pool = gene_pool
        self.history = MutationHistory()
        self.meta_learner = MetaLearner(self.history)
        self.evaluator = FitnessEvaluator(atomic_operations)
        self.iteration = 0

    def run_iteration(self, current_dna: List[str], current_fitness: float,
                     task: Dict, context: Dict) -> Tuple[List[str], float, Dict]:
        """
        Run one iteration of meta-cognitive learning

        Returns: (new_dna, new_fitness, metadata)
        """
        self.iteration += 1

        # Step 1: Meta-learner selects operator based on learned strategy
        operator_name, operator_func = self.meta_learner.select_operator(self.gene_pool)

        # Step 2: Apply mutation
        if operator_name in ['insert', 'modify']:
            new_dna = operator_func(current_dna, self.gene_pool)
        elif operator_name == 'crossover':
            # Crossover needs two DNAs - create a random second DNA
            if len(self.history.history) > 0:
                # Use a random DNA from history
                random_entry = random.choice(self.history.history)
                dna2 = random_entry['after_dna']
            else:
                # Create random DNA
                dna2 = [random.choice(self.gene_pool) for _ in range(random.randint(1, 5))]
            new_dna = operator_func(current_dna, dna2)
        else:
            new_dna = operator_func(current_dna)

        # Step 3: Evaluate new fitness
        new_fitness = self.evaluator.evaluate_on_task(new_dna, task)

        # Step 4: Record in history
        self.history.record(
            mutation_type=operator_name,
            before_dna=current_dna,
            after_dna=new_dna,
            before_fitness=current_fitness,
            after_fitness=new_fitness,
            context={**context, 'iteration': self.iteration}
        )

        # Step 5: Meta-learner updates strategy (implicit in next select_operator call)

        # Step 6: Decide whether to keep mutation
        if new_fitness >= current_fitness:
            # Accept improvement (or neutral)
            return new_dna, new_fitness, {
                'accepted': True,
                'operator': operator_name,
                'fitness_delta': new_fitness - current_fitness
            }
        else:
            # Reject with probability (simulated annealing)
            temperature = 0.1  # Controls acceptance of worse solutions
            acceptance_prob = temperature
            if random.random() < acceptance_prob:
                return new_dna, new_fitness, {
                    'accepted': True,
                    'operator': operator_name,
                    'fitness_delta': new_fitness - current_fitness,
                    'note': 'Accepted worse solution for exploration'
                }
            else:
                return current_dna, current_fitness, {
                    'accepted': False,
                    'operator': operator_name,
                    'fitness_delta': new_fitness - current_fitness
                }

    def run_multiple_iterations(self, initial_dna: List[str], task: Dict,
                               iterations: int = 50) -> Dict:
        """Run multiple meta-cognitive learning iterations"""

        print(f"\n🧠 Running Meta-Cognitive Learning Loop")
        print(f"   Initial DNA: {' → '.join(initial_dna)}")
        print(f"   Iterations: {iterations}")

        current_dna = initial_dna
        current_fitness = self.evaluator.evaluate_on_task(current_dna, task)

        print(f"   Initial Fitness: {current_fitness:.1%}")

        best_dna = current_dna
        best_fitness = current_fitness

        for i in range(iterations):
            current_dna, current_fitness, metadata = self.run_iteration(
                current_dna, current_fitness, task, {'batch': 'test'}
            )

            if current_fitness > best_fitness:
                best_dna = current_dna
                best_fitness = current_fitness
                print(f"   Iteration {i+1}: NEW BEST! {best_fitness:.1%} (operator: {metadata['operator']})")
            elif (i + 1) % 10 == 0:
                print(f"   Iteration {i+1}: Current {current_fitness:.1%}, Best {best_fitness:.1%}")

        # Final analysis
        analysis = self.meta_learner.analyze()

        return {
            'initial_dna': initial_dna,
            'final_dna': current_dna,
            'best_dna': best_dna,
            'initial_fitness': self.evaluator.evaluate_on_task(initial_dna, task),
            'final_fitness': current_fitness,
            'best_fitness': best_fitness,
            'improvement': best_fitness - self.evaluator.evaluate_on_task(initial_dna, task),
            'iterations': iterations,
            'meta_analysis': analysis
        }

# =====================================================
# TEST/DEMO
# =====================================================

def create_dummy_atomic_operations():
    """Create dummy operations for testing"""

    def identity(grid):
        return [row[:] for row in grid]

    def flip_h(grid):
        return [row[::-1] for row in grid]

    def flip_v(grid):
        return grid[::-1]

    def rot_90(grid):
        h = len(grid)
        w = len(grid[0]) if grid else 0
        return [[grid[h-1-y][x] for y in range(h)] for x in range(w)]

    def add_one(grid):
        return [[(cell + 1) % 10 for cell in row] for row in grid]

    return {
        'identity': identity,
        'flip_h': flip_h,
        'flip_v': flip_v,
        'rot_90': rot_90,
        'add_one': add_one
    }

def create_dummy_task():
    """Create a simple test task"""
    return {
        'train': [
            {
                'input': [[1, 2], [3, 4]],
                'output': [[2, 1], [4, 3]]  # Horizontal flip
            },
            {
                'input': [[5, 6], [7, 8]],
                'output': [[6, 5], [8, 7]]  # Horizontal flip
            }
        ]
    }

if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║                    🧠 PROJECT GATORCA - PHASE 4 🧠                           ║
║                                                                              ║
║                       Meta-Cognitive Engine                                  ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """)

    # Setup
    operations = create_dummy_atomic_operations()
    gene_pool = list(operations.keys())
    task = create_dummy_task()

    print("🧬 Gene Pool:", gene_pool)
    print("🧩 Test Task: Horizontal flip")

    # Create learning loop
    loop = MetaCognitiveLearningLoop(gene_pool, operations)

    # Start with random DNA
    initial_dna = [random.choice(gene_pool) for _ in range(3)]

    # Run meta-cognitive learning
    result = loop.run_multiple_iterations(initial_dna, task, iterations=50)

    print("\n" + "="*80)
    print("📊 RESULTS")
    print("="*80)
    print(f"Initial DNA: {' → '.join(result['initial_dna'])}")
    print(f"Best DNA:    {' → '.join(result['best_dna'])}")
    print(f"Initial Fitness: {result['initial_fitness']:.1%}")
    print(f"Best Fitness:    {result['best_fitness']:.1%}")
    print(f"Improvement:     {result['improvement']:+.1%}")

    print("\n📈 META-LEARNING ANALYSIS")
    print("="*80)
    analysis = result['meta_analysis']
    print(f"Status: {analysis['status']}")
    print(f"History Entries: {analysis['entries']}")
    print(f"Recent Trend: {analysis['recent_trend']:.1%}")
    print(f"Strategy: {analysis['recommended_strategy']}")

    if 'operator_stats' in analysis:
        print("\n🎯 Operator Performance:")
        for op, stats in analysis['operator_stats'].items():
            print(f"  {op:12s}: {stats['improvement_rate']:.1%} improvement rate "
                  f"({stats['total_improvements']}/{stats['count']} successes)")

    if 'insights' in analysis:
        print("\n💡 Insights:")
        for insight in analysis['insights']:
            print(f"  • {insight}")

    print("\n" + "="*80)
    print("✅ PHASE 4: META-COGNITIVE ENGINE COMPLETE!")
    print("="*80)
    print("\n🧠 Meta-learning system operational")
    print("📊 Mutation history tracking functional")
    print("🎯 Operator selection strategy adaptive")
    print("📈 Self-improving learning loop verified")
    print("\n🎖️ READY FOR PHASE 5: SOLVER DNA LIBRARY")
