#!/usr/bin/env python3
"""
Ablation Test: TurboOrcav9 - 3 Iterations

Compare 3 different NSM+SDPM integration approaches:
- Iteration 1: Parallel strategies
- Iteration 2: Sequential chaining
- Iteration 3: Ensemble voting

Test on 10 training tasks, 2 minutes each.
"""

import subprocess
import json
import time
from datetime import datetime

ITERATIONS = {
    'iter1_parallel': 'TurboOrcav9_iter1.py',
    'iter2_sequential': 'TurboOrcav9_iter2.py',
    'iter3_ensemble': 'TurboOrcav9_iter3.py',
}

def run_test(iteration_name, script_path):
    """Run one iteration and collect metrics."""
    print(f"\n{'='*80}")
    print(f"TESTING: {iteration_name}")
    print(f"Script: {script_path}")
    print(f"{'='*80}")

    start = time.time()

    try:
        result = subprocess.run(
            ['python3', script_path],
            capture_output=True,
            text=True,
            timeout=180  # 3 min max
        )

        elapsed = time.time() - start

        # Extract metrics from output
        output = result.stdout

        # Look for metrics
        perfect = 0
        avg_sim = 0.0

        for line in output.split('\n'):
            if 'Perfect:' in line and '/' in line:
                try:
                    parts = line.split('(')[1].split(')')[0]
                    perfect = float(parts.strip('%')) / 100
                except:
                    pass
            if 'Avg:' in line and '%' in line:
                try:
                    avg_sim = float(line.split(':')[1].split('%')[0].strip()) / 100
                except:
                    pass

        return {
            'name': iteration_name,
            'elapsed': elapsed,
            'perfect': perfect,
            'avg_similarity': avg_sim,
            'success': True
        }

    except subprocess.TimeoutExpired:
        return {
            'name': iteration_name,
            'elapsed': 180,
            'perfect': 0.0,
            'avg_similarity': 0.0,
            'success': False,
            'error': 'Timeout'
        }
    except Exception as e:
        return {
            'name': iteration_name,
            'elapsed': 0,
            'perfect': 0.0,
            'avg_similarity': 0.0,
            'success': False,
            'error': str(e)
        }

def main():
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║            ABLATION TEST: TurboOrcav9 - 3 Iterations                        ║
║                                                                              ║
║  Iteration 1: Parallel strategies (3 approaches in parallel)                ║
║  Iteration 2: Sequential chaining (apply rules in sequence)                 ║
║  Iteration 3: Ensemble voting (programs vote on best solution)              ║
║                                                                              ║
║  Test: 2 minutes per iteration                                              ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """)

    results = []

    for iter_name, script_path in ITERATIONS.items():
        result = run_test(iter_name, script_path)
        results.append(result)

        print(f"\n✅ {iter_name} complete:")
        print(f"   Time: {result['elapsed']:.1f}s")
        print(f"   Perfect: {result['perfect']:.1%}")
        print(f"   Avg Similarity: {result['avg_similarity']:.1%}")
        if not result['success']:
            print(f"   ⚠️  Error: {result.get('error', 'Unknown')}")

    # Save results
    with open('ablation_v9_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    # Determine winner
    print(f"\n{'='*80}")
    print("ABLATION RESULTS")
    print(f"{'='*80}")

    for r in results:
        score = r['perfect'] * 0.5 + r['avg_similarity'] * 0.5
        print(f"{r['name']:20s} | Perfect: {r['perfect']:5.1%} | Avg: {r['avg_similarity']:5.1%} | Score: {score:.3f}")

    winner = max(results, key=lambda r: r['perfect'] * 0.5 + r['avg_similarity'] * 0.5)

    print(f"\n🏆 WINNER: {winner['name']}")
    print(f"   Perfect: {winner['perfect']:.1%}")
    print(f"   Avg Similarity: {winner['avg_similarity']:.1%}")
    print(f"\n💡 RECOMMENDATION: Use {winner['name'].replace('_', ' ')} for final TurboOrcav9.py")

if __name__ == '__main__':
    main()
