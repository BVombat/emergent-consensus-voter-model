
---

## 2. `scripts/run_all_experiments.py`

Этот скрипт запускает все 8 экспериментов из сессии #20, используя ваши существующие функции (предполагается, что они уже оформлены как вызываемые). Если какие-то функции ещё не обёрнуты – я дам универсальный шаблон, который вы можете адаптировать.

```python
#!/usr/bin/env python3
"""
Run all experiments from Session #20:
1. Majority rule on regular graphs (k=2..5, N=500, 30x100)
2. Asynchronous voter on k=5, N=500 (30x100, extended to 3M steps)
3. Synchronous voter on k=5, N=500 (30x100)
4. Synchronous voter on regular k=3,4,6 (N=200, 10x50)
5. Synchronous voter on irregular graph (degrees 2 and 6, N=200, 10x50)
6. Synchronous voter on Watts–Strogatz (N=200, 10x50)
7. Synchronous voter on Barabási–Albert (N=200, 10x50)
8. Synchronous voter on ring (N=200, 5x100)

Results are saved as CSV files in the 'results' folder.
"""

import os
import sys
import time
import pandas as pd
import numpy as np

# Add parent directory to path to import src modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import your existing modules (adjust names if needed)
# If you haven't modularised yet, replace these imports with the actual code.
try:
    from src.majority_rule import majority_rule_experiment
    from src.voter_asynch import async_voter_experiment
    from src.voter_sync import sync_voter_experiment
    from src.graph_generators import (
        regular_random_graph,
        irregular_degree_graph,
        watts_strogatz_graph,
        barabasi_albert_graph,
        ring_graph
    )
except ImportError as e:
    print("Error importing modules. Please ensure the 'src' folder contains the required files.")
    print("Alternatively, copy the experiment code directly into this script.")
    sys.exit(1)

# Create results directory
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# Common seeds for reproducibility
BASE_SEED = 42

def run_exp1():
    """Majority rule on regular graphs k=2..5, N=500, 30 runs × 100 initial conditions."""
    print("\n=== Experiment 1: Majority rule on regular graphs ===")
    results = []
    for k in [2,3,4,5]:
        print(f"  Running k={k} ...")
        # Assume majority_rule_experiment returns dict with at least 'consensus_rate'
        # Adjust parameters: runs=30, initial_conditions_per_run=100? Actually 30x100 means 3000 total.
        # We'll simulate 30 graph realizations, each with 100 random initial states.
        out = majority_rule_experiment(
            graph_generator=regular_random_graph,
            N=500, k=k,
            num_graphs=30,
            num_initial_per_graph=100,
            max_steps=100000,
            seed=BASE_SEED + k
        )
        results.append({
            'k': k,
            'consensus_rate': out['consensus_rate'],
            'mean_time': out.get('mean_time', np.nan),
            'std_time': out.get('std_time', np.nan)
        })
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(RESULTS_DIR, "exp1_majority_regular.csv"), index=False)
    print(df)
    return df

def run_exp2():
    """Asynchronous voter on k=5, N=500, 30x100, extended to 3M steps."""
    print("\n=== Experiment 2: Asynchronous voter (k=5, N=500) ===")
    out = async_voter_experiment(
        graph_generator=regular_random_graph,
        N=500, k=5,
        num_graphs=30,
        num_initial_per_graph=100,
        max_steps=3_000_000,
        seed=BASE_SEED+10
    )
    df = pd.DataFrame([{
        'consensus_rate': out['consensus_rate'],
        'median_time': out.get('median_time', np.nan),
        'mean_time': out.get('mean_time', np.nan)
    }])
    df.to_csv(os.path.join(RESULTS_DIR, "exp2_voter_asynch_k5_N500.csv"), index=False)
    print(df)
    return df

def run_exp3():
    """Synchronous voter on k=5, N=500, 30x100."""
    print("\n=== Experiment 3: Synchronous voter (k=5, N=500) ===")
    out = sync_voter_experiment(
        graph_generator=regular_random_graph,
        N=500, k=5,
        num_graphs=30,
        num_initial_per_graph=100,
        max_steps=100000,
        seed=BASE_SEED+20
    )
    df = pd.DataFrame([{
        'consensus_rate': out['consensus_rate'],
        'mean_time': out['mean_time'],
        'ci_lower': out.get('ci_lower', np.nan),
        'ci_upper': out.get('ci_upper', np.nan)
    }])
    df.to_csv(os.path.join(RESULTS_DIR, "exp3_voter_sync_k5_N500.csv"), index=False)
    print(df)
    return df

def run_exp4():
    """Synchronous voter on regular k=3,4,6, N=200, 10x50."""
    print("\n=== Experiment 4: Synchronous voter on regular graphs (k=3,4,6) ===")
    results = []
    for k in [3,4,6]:
        print(f"  Running k={k} ...")
        out = sync_voter_experiment(
            graph_generator=regular_random_graph,
            N=200, k=k,
            num_graphs=10,
            num_initial_per_graph=50,
            max_steps=100000,
            seed=BASE_SEED+30+k
        )
        results.append({
            'k': k,
            'consensus_rate': out['consensus_rate'],
            'mean_time': out['mean_time']
        })
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(RESULTS_DIR, "exp4_voter_sync_regular_k3_4_6_N200.csv"), index=False)
    print(df)
    return df

def run_exp5():
    """Synchronous voter on irregular graph (degrees 2 and 6), N=200, 10x50."""
    print("\n=== Experiment 5: Synchronous voter on irregular graph (deg 2 & 6) ===")
    out = sync_voter_experiment(
        graph_generator=irregular_degree_graph,
        N=200, deg_low=2, deg_high=6,
        num_graphs=10,
        num_initial_per_graph=50,
        max_steps=100000,
        seed=BASE_SEED+40
    )
    df = pd.DataFrame([{
        'consensus_rate': out['consensus_rate'],
        'mean_time': out['mean_time']
    }])
    df.to_csv(os.path.join(RESULTS_DIR, "exp5_voter_sync_irregular_deg2_6_N200.csv"), index=False)
    print(df)
    return df

def run_exp6():
    """Synchronous voter on Watts–Strogatz (N=200, 10x50)."""
    print("\n=== Experiment 6: Synchronous voter on Watts–Strogatz ===")
    out = sync_voter_experiment(
        graph_generator=watts_strogatz_graph,
        N=200, k=4, p=0.1,
        num_graphs=10,
        num_initial_per_graph=50,
        max_steps=100000,
        seed=BASE_SEED+50
    )
    df = pd.DataFrame([{
        'consensus_rate': out['consensus_rate'],
        'mean_time': out['mean_time']
    }])
    df.to_csv(os.path.join(RESULTS_DIR, "exp6_voter_sync_ws_N200.csv"), index=False)
    print(df)
    return df

def run_exp7():
    """Synchronous voter on Barabási–Albert (N=200, 10x50)."""
    print("\n=== Experiment 7: Synchronous voter on Barabási–Albert ===")
    out = sync_voter_experiment(
        graph_generator=barabasi_albert_graph,
        N=200, m=3,
        num_graphs=10,
        num_initial_per_graph=50,
        max_steps=100000,
        seed=BASE_SEED+60
    )
    df = pd.DataFrame([{
        'consensus_rate': out['consensus_rate'],
        'mean_time': out['mean_time']
    }])
    df.to_csv(os.path.join(RESULTS_DIR, "exp7_voter_sync_ba_N200.csv"), index=False)
    print(df)
    return df

def run_exp8():
    """Synchronous voter on ring (bipartite), N=200, 5x100."""
    print("\n=== Experiment 8: Synchronous voter on ring (bipartite) ===")
    out = sync_voter_experiment(
        graph_generator=ring_graph,
        N=200,
        num_graphs=5,
        num_initial_per_graph=100,
        max_steps=100000,
        seed=BASE_SEED+70
    )
    df = pd.DataFrame([{
        'consensus_rate': out['consensus_rate'],
        'mean_time_success': out.get('mean_time_success', np.nan),
        'two_cycle_fraction': out.get('two_cycle_fraction', np.nan)
    }])
    df.to_csv(os.path.join(RESULTS_DIR, "exp8_voter_sync_ring_N200.csv"), index=False)
    print(df)
    return df

def main():
    start = time.time()
    print("Starting all experiments from Session #20...")
    
    run_exp1()
    run_exp2()
    run_exp3()
    run_exp4()
    run_exp5()
    run_exp6()
    run_exp7()
    run_exp8()
    
    elapsed = time.time() - start
    print(f"\nAll experiments completed in {elapsed:.1f} seconds.")
    print(f"Results saved in '{RESULTS_DIR}/'")

if __name__ == "__main__":
    main()
