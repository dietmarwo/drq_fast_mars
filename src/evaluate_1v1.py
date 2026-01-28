"""
Evaluate warriors using 1v1 pairwise battles with standard hill scoring.

Hill Scoring: Win=3, Tie=1, Loss=0

Evaluates all warriors in a directory against opponents from another directory.
Each warrior fights each opponent in multiple rounds (default 24).

Features symmetric caching: if A vs B is evaluated, B vs A is derived (not re-evaluated).

Usage:
    # Full evaluation against all opponents
    python evaluate_1v1.py \
        --warrior_dir=results/explicit_16/final_niches \
        --opponent_dir=human_warriors \
        --output_csv=results/hill_scores.csv

    # Round-robin (same dir for warriors and opponents)
    python evaluate_1v1.py \
        --warrior_dir=warriors \
        --opponent_dir=warriors \
        --output_csv=results/round_robin.csv

    # With more rounds for accuracy
    python evaluate_1v1.py \
        --warrior_dir=results/explicit_16/final_niches \
        --opponent_dir=human_warriors \
        --simargs.rounds=100 \
        --output_csv=results/hill_scores_100r.csv
"""

import os
import glob
import csv
import random
import hashlib
import numpy as np
from dataclasses import dataclass, field
from tqdm.auto import tqdm
from multiprocessing import Pool
from functools import partial
import tyro

from corewar_util import (
    SimulationArgs, simargs_to_environment, parse_warrior_from_file,
    run_1v1_multiple_rounds, compute_hill_score, compute_gradual_score
)


@dataclass
class Args:
    # Input directories
    warrior_dir: str = "./warriors"  # Directory containing warriors to evaluate
    warrior_pattern: str = "*.red"
    opponent_dir: str = "./opponents"  # Directory containing opponent warriors
    opponent_pattern: str = "*.red"

    # Output
    output_csv: str = "hill_scores.csv"
    pairings_csv: str | None = None  # Optional: save individual pairing results

    # Simulation settings
    simargs: SimulationArgs = field(default_factory=SimulationArgs)
    n_processes: int = 24

    # Scoring mode
    gradual_scoring: bool = False  # Use gradual scoring for smoother fitness

    # Random seed
    seed: int = 42


def compute_warrior_id(warrior) -> str:
    """Compute unique ID from warrior instructions."""
    instr_str = ""
    for instr in warrior.instructions:
        instr_str += f"{instr.opcode}:{instr.modifier}:{instr.a_mode}:{instr.a_number}:{instr.b_mode}:{instr.b_number}|"
    return hashlib.sha256(instr_str.encode()).hexdigest()[:16]


def get_canonical_cache_key(id1: str, id2: str) -> tuple:
    """
    Get canonical cache key for symmetric lookup.
    Returns (cache_key, is_reversed) where is_reversed indicates if ids were swapped.
    """
    if id1 <= id2:
        return f"{id1}:{id2}", False
    else:
        return f"{id2}:{id1}", True


def swap_pairing_result(result: dict) -> dict:
    """Swap a pairing result to get the reverse perspective."""
    return {
        'wins': result['losses'],
        'losses': result['wins'],
        'ties': result['ties'],
        'rounds': result['rounds'],
        'tsp_w': result.get('tsp_o', 0.0),
        'tsp_o': result.get('tsp_w', 0.0),
        'mc_w': result.get('mc_o', 0.0),
        'mc_o': result.get('mc_w', 0.0),
    }


def load_warriors(warrior_dir: str, pattern: str, simargs: SimulationArgs) -> list:
    """Load warriors from directory."""
    full_pattern = os.path.join(warrior_dir, pattern)
    files = sorted(glob.glob(full_pattern))
    
    warriors = []
    for file in files:
        try:
            warrior_str, warrior = parse_warrior_from_file(simargs, file)
            warrior_id = compute_warrior_id(warrior)
            warriors.append({
                'file': os.path.basename(file),
                'path': file,
                'code': warrior_str,
                'warrior': warrior,
                'id': warrior_id
            })
        except Exception as e:
            print(f"Warning: Failed to load {file}: {e}")
    
    return warriors


def evaluate_pairing(args_tuple):
    """Evaluate a single warrior vs opponent pairing. For parallel execution."""
    simargs, warrior, opponent, warrior_id, opponent_id, canonical_key = args_tuple
    
    result = run_1v1_multiple_rounds(simargs, warrior, opponent, n_processes=1)
    
    if result is None:
        return {
            'canonical_key': canonical_key,
            'warrior_id': warrior_id,
            'opponent_id': opponent_id,
            'wins': 0,
            'losses': simargs.rounds,
            'ties': 0,
            'rounds': simargs.rounds,
            'tsp_w': 0.0,
            'tsp_o': 0.0,
            'mc_w': 0.0,
            'mc_o': 0.0,
            'avg_win_cycles': 0.0,
            'avg_loss_cycles': float(simargs.cycles),
        }
    
    return {
        'canonical_key': canonical_key,
        'warrior_id': warrior_id,
        'opponent_id': opponent_id,
        'wins': result['w1_wins'],
        'losses': result['w2_wins'],
        'ties': result['ties'],
        'rounds': result['rounds'],
        'tsp_w': float(result['total_spawned_procs'][0]),
        'tsp_o': float(result['total_spawned_procs'][1]),
        'mc_w': float(result['memory_coverage'][0]),
        'mc_o': float(result['memory_coverage'][1]),
        'avg_win_cycles': float(result.get('avg_win_cycles', 0)),
        'avg_loss_cycles': float(result.get('avg_loss_cycles', 0)),
    }


def main(args: Args):
    print(f"1v1 Hill Score Evaluation (with symmetric caching)")
    print(f"=" * 60)
    print(f"Warrior dir: {args.warrior_dir}")
    print(f"Opponent dir: {args.opponent_dir}")
    print(f"Rounds per pairing: {args.simargs.rounds}")
    print(f"Scoring: {'gradual (Win~3.9, Tie=1, Loss~0.4)' if args.gradual_scoring else 'hill (Win=3, Tie=1, Loss=0)'}")
    
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
    
    # Build lookup dicts
    warrior_by_id = {w['id']: w for w in warriors}
    opponent_by_id = {o['id']: o for o in opponents}
    
    # Identify unique pairings (using canonical keys to avoid duplicates)
    all_pairings = []  # (warrior, opponent, warrior_id, opponent_id, canonical_key, is_reversed)
    unique_keys = set()
    unique_pairings = []  # Only the pairings we need to evaluate
    
    for w in warriors:
        for o in opponents:
            canonical_key, is_reversed = get_canonical_cache_key(w['id'], o['id'])
            all_pairings.append({
                'warrior': w,
                'opponent': o,
                'canonical_key': canonical_key,
                'is_reversed': is_reversed
            })
            
            if canonical_key not in unique_keys:
                unique_keys.add(canonical_key)
                # For unique pairings, always evaluate in canonical order (not reversed)
                if is_reversed:
                    # Swap: evaluate opponent vs warrior
                    unique_pairings.append((
                        args.simargs,
                        o['warrior'],  # opponent first
                        w['warrior'],  # warrior second
                        o['id'],
                        w['id'],
                        canonical_key
                    ))
                else:
                    # Normal: evaluate warrior vs opponent
                    unique_pairings.append((
                        args.simargs,
                        w['warrior'],
                        o['warrior'],
                        w['id'],
                        o['id'],
                        canonical_key
                    ))
    
    total_pairings = len(warriors) * len(opponents)
    unique_count = len(unique_pairings)
    saved_count = total_pairings - unique_count
    
    print(f"Total pairings needed: {total_pairings}")
    print(f"Unique pairings to evaluate: {unique_count}")
    print(f"Symmetric pairings (saved): {saved_count} ({100*saved_count/total_pairings:.1f}% reduction)")
    print(f"Max hill score per warrior: {len(opponents) * args.simargs.rounds * 3}")
    print(f"=" * 60)
    
    # Evaluate only unique pairings in parallel
    print(f"\nEvaluating {unique_count} unique pairings...")
    
    with Pool(processes=args.n_processes) as pool:
        unique_results = list(tqdm(
            pool.imap(evaluate_pairing, unique_pairings),
            total=unique_count,
            desc="Pairings"
        ))
    
    # Build cache from unique results
    cache = {}
    for result in unique_results:
        cache[result['canonical_key']] = result
    
    # Build full pairing results using cache + symmetry
    pairing_results = []
    for p in all_pairings:
        canonical_key = p['canonical_key']
        cached = cache[canonical_key]
        
        if p['is_reversed']:
            # Need to swap the result
            pr = {
                'warrior_file': p['warrior']['file'],
                'opponent_file': p['opponent']['file'],
                'wins': cached['losses'],  # swapped
                'losses': cached['wins'],  # swapped
                'ties': cached['ties'],
                'rounds': cached['rounds'],
                'tsp_w': cached['tsp_o'],  # swapped
                'mc_w': cached['mc_o'],    # swapped
                'avg_win_cycles': cached.get('avg_loss_cycles', 0),  # Their loss = our win
                'avg_loss_cycles': cached.get('avg_win_cycles', 0),  # Their win = our loss
            }
        else:
            pr = {
                'warrior_file': p['warrior']['file'],
                'opponent_file': p['opponent']['file'],
                'wins': cached['wins'],
                'losses': cached['losses'],
                'ties': cached['ties'],
                'rounds': cached['rounds'],
                'tsp_w': cached['tsp_w'],
                'mc_w': cached['mc_w'],
                'avg_win_cycles': cached.get('avg_win_cycles', 0),
                'avg_loss_cycles': cached.get('avg_loss_cycles', 0),
            }
        pairing_results.append(pr)
    
    # Aggregate results by warrior
    warrior_stats = {w['file']: {
        'wins': 0, 'losses': 0, 'ties': 0, 'total_rounds': 0,
        'tsp_sum': 0.0, 'mc_sum': 0.0, 'n_opponents': 0,
        'win_cycles': [], 'loss_cycles': []  # For gradual scoring
    } for w in warriors}
    
    for pr in pairing_results:
        wf = pr['warrior_file']
        warrior_stats[wf]['wins'] += pr['wins']
        warrior_stats[wf]['losses'] += pr['losses']
        warrior_stats[wf]['ties'] += pr['ties']
        warrior_stats[wf]['total_rounds'] += pr['rounds']
        warrior_stats[wf]['tsp_sum'] += pr['tsp_w']
        warrior_stats[wf]['mc_sum'] += pr['mc_w']
        warrior_stats[wf]['n_opponents'] += 1
        
        # Track timing for gradual scoring
        if pr['wins'] > 0 and pr.get('avg_win_cycles', 0) > 0:
            warrior_stats[wf]['win_cycles'].extend([pr['avg_win_cycles']] * pr['wins'])
        if pr['losses'] > 0 and pr.get('avg_loss_cycles', 0) > 0:
            warrior_stats[wf]['loss_cycles'].extend([pr['avg_loss_cycles']] * pr['losses'])
    
    # Compute hill scores
    results = []
    for w in warriors:
        stats = warrior_stats[w['file']]
        
        # Hill score
        hill_score, hill_max = compute_hill_score(
            stats['wins'], stats['ties'], stats['total_rounds']
        )
        
        # Timing averages for gradual scoring
        avg_win_cycles = sum(stats['win_cycles']) / len(stats['win_cycles']) if stats['win_cycles'] else 0
        avg_loss_cycles = sum(stats['loss_cycles']) / len(stats['loss_cycles']) if stats['loss_cycles'] else 0
        
        # Gradual score
        gradual_score, gradual_max = compute_gradual_score(
            stats['wins'], stats['losses'], stats['ties'], stats['total_rounds'],
            avg_win_cycles, avg_loss_cycles, args.simargs.cycles
        )
        
        # Select primary score based on args
        if args.gradual_scoring:
            primary_score = gradual_score
            primary_max = gradual_max
        else:
            primary_score = hill_score
            primary_max = hill_max
        
        results.append({
            'file': w['file'],
            'score': primary_score,
            'max_score': primary_max,
            'score_pct': (primary_score / primary_max * 100) if primary_max > 0 else 0,
            'hill_score': hill_score,
            'hill_max': hill_max,
            'hill_score_pct': (hill_score / hill_max * 100) if hill_max > 0 else 0,
            'gradual_score': gradual_score,
            'gradual_max': gradual_max,
            'gradual_score_pct': (gradual_score / gradual_max * 100) if gradual_max > 0 else 0,
            'wins': stats['wins'],
            'losses': stats['losses'],
            'ties': stats['ties'],
            'total_rounds': stats['total_rounds'],
            'win_pct': stats['wins'] / stats['total_rounds'] * 100 if stats['total_rounds'] > 0 else 0,
            'tie_pct': stats['ties'] / stats['total_rounds'] * 100 if stats['total_rounds'] > 0 else 0,
            'loss_pct': stats['losses'] / stats['total_rounds'] * 100 if stats['total_rounds'] > 0 else 0,
            'avg_win_cycles': avg_win_cycles,
            'avg_loss_cycles': avg_loss_cycles,
            'avg_tsp': stats['tsp_sum'] / stats['n_opponents'] if stats['n_opponents'] > 0 else 0,
            'avg_mc': stats['mc_sum'] / stats['n_opponents'] if stats['n_opponents'] > 0 else 0,
            'n_opponents': stats['n_opponents'],
        })
    
    # Sort by primary score descending
    results.sort(key=lambda x: -x['score'])
    
    # Write main results CSV
    print(f"\nWriting results to {args.output_csv}")
    os.makedirs(os.path.dirname(args.output_csv) or '.', exist_ok=True)
    
    with open(args.output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'rank', 'file', 'score', 'score_pct', 'max_score',
            'hill_score', 'hill_pct', 'gradual_score', 'gradual_pct',
            'wins', 'losses', 'ties', 'total_rounds',
            'win_pct', 'tie_pct', 'loss_pct',
            'avg_win_cycles', 'avg_loss_cycles',
            'avg_tsp', 'avg_mc', 'n_opponents'
        ])
        
        for rank, r in enumerate(results, 1):
            writer.writerow([
                rank,
                r['file'],
                f"{r['score']:.2f}",
                f"{r['score_pct']:.2f}",
                f"{r['max_score']:.2f}",
                f"{r['hill_score']:.2f}",
                f"{r['hill_score_pct']:.2f}",
                f"{r['gradual_score']:.2f}",
                f"{r['gradual_score_pct']:.2f}",
                r['wins'],
                r['losses'],
                r['ties'],
                r['total_rounds'],
                f"{r['win_pct']:.2f}",
                f"{r['tie_pct']:.2f}",
                f"{r['loss_pct']:.2f}",
                f"{r['avg_win_cycles']:.1f}",
                f"{r['avg_loss_cycles']:.1f}",
                f"{r['avg_tsp']:.2f}",
                f"{r['avg_mc']:.2f}",
                r['n_opponents'],
            ])
    
    # Optionally write pairings CSV
    if args.pairings_csv:
        print(f"Writing pairings to {args.pairings_csv}")
        os.makedirs(os.path.dirname(args.pairings_csv) or '.', exist_ok=True)
        
        with open(args.pairings_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'warrior_file', 'opponent_file', 
                'wins', 'losses', 'ties', 'rounds',
                'avg_win_cycles', 'avg_loss_cycles',
                'tsp_w', 'mc_w'
            ])
            
            for pr in pairing_results:
                writer.writerow([
                    pr['warrior_file'],
                    pr['opponent_file'],
                    pr['wins'],
                    pr['losses'],
                    pr['ties'],
                    pr['rounds'],
                    f"{pr.get('avg_win_cycles', 0):.1f}",
                    f"{pr.get('avg_loss_cycles', 0):.1f}",
                    f"{pr['tsp_w']:.2f}",
                    f"{pr['mc_w']:.2f}",
                ])
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"Results Summary")
    print(f"{'='*60}")
    print(f"Total warriors evaluated: {len(results)}")
    print(f"Opponents per warrior: {len(opponents)}")
    print(f"Rounds per pairing: {args.simargs.rounds}")
    print(f"Evaluations saved by symmetry: {saved_count}")
    print(f"Scoring mode: {'gradual' if args.gradual_scoring else 'hill'}")
    
    if len(results) > 0:
        score_type = "gradual" if args.gradual_scoring else "hill"
        print(f"\nTop 6 warriors (by {score_type} score):")
        for i, r in enumerate(results[:6], 1):
            print(f"  {i}. {r['file']}")
            print(f"     Score: {r['score']:.2f} ({r['score_pct']:.1f}%)")
            print(f"     Hill: {r['hill_score']:.1f} ({r['hill_score_pct']:.1f}%) | Gradual: {r['gradual_score']:.1f} ({r['gradual_score_pct']:.1f}%)")
            print(f"     W/T/L: {r['wins']}/{r['ties']}/{r['losses']} ({r['win_pct']:.1f}%/{r['tie_pct']:.1f}%/{r['loss_pct']:.1f}%)")
        
        print(f"\nScore statistics ({score_type}):")
        scores = [r['score'] for r in results]
        pcts = [r['score_pct'] for r in results]
        print(f"  Mean: {np.mean(scores):.1f} ({np.mean(pcts):.1f}%)")
        print(f"  Std:  {np.std(scores):.1f} ({np.std(pcts):.1f}%)")
        print(f"  Min:  {np.min(scores):.1f} ({np.min(pcts):.1f}%)")
        print(f"  Max:  {np.max(scores):.1f} ({np.max(pcts):.1f}%)")
        
        print(f"\nWin rate statistics:")
        win_pcts = [r['win_pct'] for r in results]
        print(f"  Mean: {np.mean(win_pcts):.1f}%")
        print(f"  Std:  {np.std(win_pcts):.1f}%")
        print(f"  Min:  {np.min(win_pcts):.1f}%")
        print(f"  Max:  {np.max(win_pcts):.1f}%")


if __name__ == "__main__":
    main(tyro.cli(Args))
