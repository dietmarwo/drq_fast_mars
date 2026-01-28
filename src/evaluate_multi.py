"""
Evaluate warriors using multi-warrior MARS simulation (original DRQ scoring).

Evaluates all warriors in a directory against opponents from another directory.
Optionally samples a subset of opponents multiple times for faster evaluation.

Usage:
    # Full evaluation against all opponents
    python evaluate_multi.py \
        --warrior_dir=results/explicit_16/final_niches \
        --opponent_dir=human_warriors \
        --output_csv=results/multi_scores.csv

    # Sampled evaluation (faster)
    python evaluate_multi.py \
        --warrior_dir=results/explicit_16/final_niches \
        --opponent_dir=human_warriors \
        --opponents_num=20 \
        --num_runs=5 \
        --output_csv=results/multi_scores_sampled.csv
"""

import os
import glob
import csv
import random
import numpy as np
from dataclasses import dataclass, field
from tqdm.auto import tqdm
import tyro

from corewar_util import (
    SimulationArgs, simargs_to_environment, parse_warrior_from_file, 
    run_multiple_rounds
)


@dataclass
class Args:
    # Input directories
    warrior_dir: str = "./warriors"  # Directory containing warriors to evaluate
    warrior_pattern: str = "*.red"
    opponent_dir: str = "./opponents"  # Directory containing opponent warriors
    opponent_pattern: str = "*.red"

    # Output
    output_csv: str = "multi_scores.csv"

    # Simulation settings
    simargs: SimulationArgs = field(default_factory=SimulationArgs)
    n_processes: int = 24
    timeout: int = 900

    # Sampling options (optional)
    opponents_num: int | None = None  # Number of opponents to sample (None = use all)
    num_runs: int = 1  # Number of sampling runs to average
    seed: int = 42  # Random seed for reproducibility


def load_warriors(warrior_dir: str, pattern: str, simargs: SimulationArgs) -> list:
    """Load warriors from directory."""
    full_pattern = os.path.join(warrior_dir, pattern)
    files = sorted(glob.glob(full_pattern))
    
    warriors = []
    for file in files:
        try:
            warrior_str, warrior = parse_warrior_from_file(simargs, file)
            warriors.append({
                'file': os.path.basename(file),
                'path': file,
                'code': warrior_str,
                'warrior': warrior
            })
        except Exception as e:
            print(f"Warning: Failed to load {file}: {e}")
    
    return warriors


def evaluate_warrior_multi(warrior, opponents, simargs, n_processes, timeout):
    """
    Evaluate a single warrior against opponents using multi-warrior MARS.
    Returns dict with score and BC features.
    """
    warriors = [warrior['warrior']] + [o['warrior'] for o in opponents]
    
    outputs = run_multiple_rounds(
        simargs, warriors,
        n_processes=n_processes,
        timeout=timeout
    )
    
    if outputs is None:
        return {
            'score': -np.inf,
            'alive_score': 0.0,
            'total_spawned_procs': 0.0,
            'memory_coverage': 0.0,
        }
    
    # Extract warrior's scores (index 0)
    return {
        'score': float(outputs['score'].mean(axis=-1)[0]),
        'alive_score': float(outputs['alive_score'].mean(axis=-1)[0]),
        'total_spawned_procs': float(outputs['total_spawned_procs'].mean(axis=-1)[0]),
        'memory_coverage': float(outputs['memory_coverage'].mean(axis=-1)[0]),
    }


def main(args: Args):
    print(f"Multi-Warrior Evaluation")
    print(f"=" * 60)
    print(f"Warrior dir: {args.warrior_dir}")
    print(f"Opponent dir: {args.opponent_dir}")
    print(f"Rounds per eval: {args.simargs.rounds}")
    
    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # Load warriors and opponents
    warriors = load_warriors(args.warrior_dir, args.warrior_pattern, args.simargs)
    opponents = load_warriors(args.opponent_dir, args.opponent_pattern, args.simargs)
    
    print(f"Loaded {len(warriors)} warriors to evaluate")
    print(f"Loaded {len(opponents)} opponents")
    
    if len(warriors) == 0:
        print("Error: No warriors found!")
        return
    
    if len(opponents) == 0:
        print("Error: No opponents found!")
        return
    
    # Determine evaluation mode
    if args.opponents_num is not None and args.opponents_num < len(opponents):
        sampled_mode = True
        print(f"\nSampled mode: {args.opponents_num} opponents × {args.num_runs} runs")
    else:
        sampled_mode = False
        args.num_runs = 1
        print(f"\nFull mode: all {len(opponents)} opponents")
    
    print(f"=" * 60)
    
    # Evaluate each warrior
    results = []
    
    for w in tqdm(warriors, desc="Evaluating warriors"):
        if sampled_mode:
            # Multiple runs with sampled opponents
            run_scores = []
            run_alive = []
            run_tsp = []
            run_mc = []
            
            for run_idx in range(args.num_runs):
                # Sample opponents (reproducible with seed + run_idx)
                random.seed(args.seed + run_idx)
                sampled_opps = random.sample(opponents, args.opponents_num)
                
                result = evaluate_warrior_multi(
                    w, sampled_opps, args.simargs, args.n_processes, args.timeout
                )
                
                run_scores.append(result['score'])
                run_alive.append(result['alive_score'])
                run_tsp.append(result['total_spawned_procs'])
                run_mc.append(result['memory_coverage'])
            
            # Average across runs
            results.append({
                'file': w['file'],
                'score': np.mean(run_scores),
                'score_std': np.std(run_scores),
                'alive_score': np.mean(run_alive),
                'total_spawned_procs': np.mean(run_tsp),
                'memory_coverage': np.mean(run_mc),
                'num_runs': args.num_runs,
                'opponents_per_run': args.opponents_num,
            })
        else:
            # Single run with all opponents
            result = evaluate_warrior_multi(
                w, opponents, args.simargs, args.n_processes, args.timeout
            )
            
            results.append({
                'file': w['file'],
                'score': result['score'],
                'score_std': 0.0,
                'alive_score': result['alive_score'],
                'total_spawned_procs': result['total_spawned_procs'],
                'memory_coverage': result['memory_coverage'],
                'num_runs': 1,
                'opponents_per_run': len(opponents),
            })
    
    # Sort by score descending
    results.sort(key=lambda x: -x['score'])
    
    # Write CSV
    print(f"\nWriting results to {args.output_csv}")
    os.makedirs(os.path.dirname(args.output_csv) or '.', exist_ok=True)
    
    with open(args.output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'rank', 'file', 'score', 'score_std', 'alive_score',
            'total_spawned_procs', 'memory_coverage', 
            'num_runs', 'opponents_per_run'
        ])
        
        for rank, r in enumerate(results, 1):
            writer.writerow([
                rank,
                r['file'],
                f"{r['score']:.6f}",
                f"{r['score_std']:.6f}",
                f"{r['alive_score']:.6f}",
                f"{r['total_spawned_procs']:.2f}",
                f"{r['memory_coverage']:.2f}",
                r['num_runs'],
                r['opponents_per_run'],
            ])
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"Results Summary")
    print(f"{'='*60}")
    print(f"Total warriors evaluated: {len(results)}")
    
    if len(results) > 0:
        print(f"\nTop 6 warriors:")
        for i, r in enumerate(results[:6], 1):
            print(f"  {i}. {r['file']}: {r['score']:.4f}" + 
                  (f" (±{r['score_std']:.4f})" if r['score_std'] > 0 else ""))
        
        print(f"\nScore statistics:")
        scores = [r['score'] for r in results if r['score'] > -np.inf]
        if scores:
            print(f"  Mean: {np.mean(scores):.4f}")
            print(f"  Std:  {np.std(scores):.4f}")
            print(f"  Min:  {np.min(scores):.4f}")
            print(f"  Max:  {np.max(scores):.4f}")


if __name__ == "__main__":
    main(tyro.cli(Args))
