"""
Core War utilities with pairwise evaluation and hill scoring.

Key features:
- run_single_round_1v1: Returns win/loss/tie outcome
- Hill scoring: Win=3, Tie=1, Loss=0
- Pairwise caching support
"""

from dataclasses import dataclass
import random
import numpy as np
from functools import partial
from tqdm import tqdm
from multiprocessing import Pool

from corewar import MARS, Core, redcode


@dataclass
class SimulationArgs:
    rounds: int = 24  # Rounds to play
    size: int = 8000  # The core size
    cycles: int = 80000  # Cycles until tie
    processes: int = 8000  # Max processes
    length: int = 100  # Max warrior length
    distance: int = 100  # Minimum warrior distance


def simargs_to_environment(args):
    return dict(
        ROUNDS=args.rounds, CORESIZE=args.size, CYCLES=args.cycles,
        MAXPROCESSES=args.processes, MAXLENGTH=args.length, MINDISTANCE=args.distance
    )


def run_single_round(simargs, warriors, seed, pbar=False):
    """Run a single round of simulation with multiple warriors (original DRQ style)."""
    random.seed(seed)
    simulation = MARS(
        warriors=warriors, minimum_separation=simargs.distance,
        max_processes=simargs.processes, randomize=True, debug=False, track_coverage=True
    )
    score = np.zeros(len(warriors), dtype=float)
    alive_score = np.zeros(len(warriors), dtype=float)

    prev_nprocs = simulation._tq_sizes.copy()
    total_spawned_procs = np.zeros(len(simulation.warriors), dtype=int)

    for t in tqdm(range(simargs.cycles), disable=not pbar):
        simulation.step()

        nprocs = simulation._tq_sizes
        alive_flags = (nprocs > 0).astype(int)
        n_alive = alive_flags.sum()
        if n_alive == 0:
            break
        score += (alive_flags * (1. / n_alive)) / simargs.cycles
        alive_score += alive_flags / simargs.cycles

        total_spawned_procs = total_spawned_procs + np.maximum(0, nprocs - prev_nprocs)
        prev_nprocs = nprocs.copy()

    memory_coverage = simulation.get_coverage_sums()
    score = score * len(warriors)
    outputs = dict(
        score=score, alive_score=alive_score,
        total_spawned_procs=total_spawned_procs, memory_coverage=memory_coverage
    )
    return outputs


def run_single_round_1v1(simargs, warrior1, warrior2, seed):
    """
    Run a single round of 1v1 simulation.
    Returns outcome (win/loss/tie) plus BC features.
    
    Outcomes:
    - w1_wins: warrior1 survives, warrior2 dies
    - w2_wins: warrior2 survives, warrior1 dies
    - tie: both survive (timeout) or both die simultaneously
    """
    random.seed(seed)
    warriors = [warrior1, warrior2]

    simulation = MARS(
        warriors=warriors, minimum_separation=simargs.distance,
        max_processes=simargs.processes, randomize=True, debug=False, track_coverage=True
    )

    prev_nprocs = simulation._tq_sizes.copy()
    total_spawned_procs = np.zeros(2, dtype=int)

    # Track when each warrior dies (0 = still alive at end)
    death_cycle = [0, 0]

    for t in range(simargs.cycles):
        simulation.step()

        nprocs = simulation._tq_sizes

        # Check for deaths
        for i in range(2):
            if death_cycle[i] == 0 and nprocs[i] == 0:
                death_cycle[i] = t + 1  # Record death cycle (1-indexed)

        # Track spawned processes
        total_spawned_procs = total_spawned_procs + np.maximum(0, nprocs - prev_nprocs)
        prev_nprocs = nprocs.copy()

        # Both dead?
        if nprocs[0] == 0 and nprocs[1] == 0:
            break

    memory_coverage = simulation.get_coverage_sums()

    # Determine outcome
    w1_alive = death_cycle[0] == 0
    w2_alive = death_cycle[1] == 0

    if w1_alive and not w2_alive:
        # Warrior 1 wins
        outcome = (1, 0, 0)  # (w1_win, w2_win, tie)
    elif w2_alive and not w1_alive:
        # Warrior 2 wins
        outcome = (0, 1, 0)
    else:
        # Tie (both alive or both dead)
        outcome = (0, 0, 1)

    return dict(
        outcome=outcome,  # (w1_win, w2_win, tie)
        death_cycle=death_cycle,
        total_spawned_procs=total_spawned_procs,
        memory_coverage=memory_coverage
    )


def run_1v1_multiple_rounds(simargs, warrior1, warrior2, n_processes=1):
    """
    Run multiple rounds of 1v1 simulation.
    Returns aggregated win/loss/tie counts, BC features, and timing data for gradual scoring.
    """
    try:
        run_fn = partial(run_single_round_1v1, simargs, warrior1, warrior2)
        seeds = list(range(simargs.rounds))

        if n_processes > 1:
            with Pool(processes=n_processes) as pool:
                outputs = pool.map(run_fn, seeds)
        else:
            outputs = [run_fn(seed) for seed in seeds]

        # Aggregate outcomes
        w1_wins = sum(o['outcome'][0] for o in outputs)
        w2_wins = sum(o['outcome'][1] for o in outputs)
        ties = sum(o['outcome'][2] for o in outputs)

        # Average BC features
        avg_tsp = np.mean([o['total_spawned_procs'] for o in outputs], axis=0)
        avg_mc = np.mean([o['memory_coverage'] for o in outputs], axis=0)
        
        # Aggregate timing data for gradual scoring
        # death_cycle[i] = cycle when warrior i died (0 = survived full game)
        all_death_cycles = [o['death_cycle'] for o in outputs]
        
        # For wins: how quickly did we kill opponent? (lower = better)
        # For losses: how long did we survive? (higher = better)
        w1_win_cycles = []  # Cycles to kill opponent when we won
        w1_loss_cycles = []  # Cycles we survived when we lost
        
        for o in outputs:
            dc = o['death_cycle']
            if o['outcome'][0] == 1:  # w1 won
                # Opponent died at dc[1], we survived
                w1_win_cycles.append(dc[1] if dc[1] > 0 else simargs.cycles)
            elif o['outcome'][1] == 1:  # w1 lost
                # We died at dc[0]
                w1_loss_cycles.append(dc[0] if dc[0] > 0 else simargs.cycles)
        
        # Average timing (0 if no wins/losses)
        avg_win_cycles = np.mean(w1_win_cycles) if w1_win_cycles else 0
        avg_loss_cycles = np.mean(w1_loss_cycles) if w1_loss_cycles else 0

        return dict(
            w1_wins=w1_wins,
            w2_wins=w2_wins,
            ties=ties,
            rounds=simargs.rounds,
            total_spawned_procs=avg_tsp,  # shape: (2,)
            memory_coverage=avg_mc,  # shape: (2,)
            # Timing data for gradual scoring
            avg_win_cycles=avg_win_cycles,    # How fast we killed (when winning)
            avg_loss_cycles=avg_loss_cycles,  # How long we survived (when losing)
            max_cycles=simargs.cycles,
        )
    except Exception as e:
        print(f"Error in 1v1 evaluation: {e}")
        return None


def compute_hill_score(wins, ties, rounds):
    """
    Compute standard hill score from win/tie counts.
    Standard scoring: Win=3, Tie=1, Loss=0
    Returns (score, max_score).
    """
    score = wins * 3 + ties * 1
    max_score = rounds * 3
    return score, max_score


def compute_gradual_score(wins, losses, ties, rounds,
                          avg_win_cycles=0, avg_loss_cycles=0, 
                          max_cycles=80000):
    """
    Gradual scoring that provides continuous fitness signal.
    
    Base: Win=3, Tie=1, Loss=0 (standard hill scoring)
    
    Bonuses (provide gradient within win/loss categories):
    - Win speed bonus: +0 to +0.9 per win (faster kills = better)
    - Survival bonus: +0 to +0.4 per loss (longer survival = better)
    
    This gives continuous improvement signal:
    - Losing in 1000 cycles → ~0.005 per loss
    - Losing in 40000 cycles → ~0.2 per loss  
    - Winning in 40000 cycles → ~3.45 per win
    - Winning in 1000 cycles → ~3.89 per win
    
    The bonuses are small enough to not override the base scoring
    (a tie is always better than any loss, a win is always better than any tie)
    but large enough to provide gradient for optimization.
    """
    base_score = wins * 3 + ties * 1
    
    # Win speed bonus: faster wins are better (0 to 0.9 per win)
    # win_cycles close to 0 → bonus close to 0.9
    # win_cycles close to max → bonus close to 0
    if wins > 0 and avg_win_cycles > 0:
        speed_factor = 1.0 - (avg_win_cycles / max_cycles)
        win_speed_bonus = wins * 0.9 * speed_factor
    else:
        win_speed_bonus = 0.0
    
    # Survival bonus: surviving longer when losing is better (0 to 0.4 per loss)
    # loss_cycles close to 0 → bonus close to 0
    # loss_cycles close to max → bonus close to 0.4
    if losses > 0 and avg_loss_cycles > 0:
        survival_factor = avg_loss_cycles / max_cycles
        survival_bonus = losses * 0.4 * survival_factor
    else:
        survival_bonus = 0.0
    
    total_score = base_score + win_speed_bonus + survival_bonus
    max_score = rounds * 3.9  # Maximum possible with all fast wins
    
    return total_score, max_score


def _get_canonical_cache_key(id1: str, id2: str) -> tuple:
    """
    Get canonical cache key for symmetric lookup.
    Returns (cache_key, is_reversed) where is_reversed indicates if ids were swapped.
    """
    if id1 <= id2:
        return f"{id1}:{id2}", False
    else:
        return f"{id2}:{id1}", True


def _swap_pairing_result(result: dict) -> dict:
    """Swap a pairing result to get the reverse perspective."""
    return {
        'wins': result['losses'],
        'losses': result['wins'],
        'ties': result['ties'],
        'rounds': result['rounds'],
        'tsp_w': result['tsp_o'],
        'tsp_o': result['tsp_w'],
        'mc_w': result['mc_o'],
        'mc_o': result['mc_w'],
        # Timing swaps: our wins become their losses and vice versa
        'avg_win_cycles': result.get('avg_loss_cycles', 0),  # Their loss = our win
        'avg_loss_cycles': result.get('avg_win_cycles', 0),  # Their win = our loss
    }


def _evaluate_single_pairing(args_tuple):
    """
    Evaluate a single (warrior, opponent) pairing.
    Top-level function for multiprocessing Pool.
    """
    simargs, warrior, opponent, opp_idx = args_tuple
    
    result = run_1v1_multiple_rounds(simargs, warrior, opponent, n_processes=1)
    
    if result is None:
        return {
            'opp_idx': opp_idx,
            'wins': 0,
            'losses': simargs.rounds,
            'ties': 0,
            'rounds': simargs.rounds,
            'tsp_w': 0.0,
            'tsp_o': 0.0,
            'mc_w': 0.0,
            'mc_o': 0.0,
            'avg_win_cycles': 0.0,
            'avg_loss_cycles': float(simargs.cycles),  # Assume died at max
        }
    
    return {
        'opp_idx': opp_idx,
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


def run_pairwise_with_cache(simargs, warrior, warrior_id, opponents, opponent_ids,
                            pairing_cache, cache_lock=None, n_processes=1):
    """
    Evaluate warrior against all opponents using cached pairwise results.
    Uses hill scoring (Win=3, Tie=1, Loss=0).
    
    Now with:
    - Parallel evaluation of uncached pairings
    - Symmetric cache lookup (A vs B = B vs A with swapped results)

    Args:
        simargs: Simulation arguments
        warrior: The warrior being evaluated (Warrior object)
        warrior_id: Unique ID of the warrior
        opponents: List of opponent Warrior objects
        opponent_ids: List of opponent IDs (same order as opponents)
        pairing_cache: Shared dict for caching pairwise results
        cache_lock: Lock for thread-safe cache access (optional)
        n_processes: Number of processes for parallel evaluation of uncached pairings

    Returns:
        dict with hill score, BC features, and list of new pairings computed
    """
    total_wins = 0
    total_losses = 0
    total_ties = 0
    all_tsp = []
    all_mc = []
    new_pairings = []
    
    # Step 1: Check cache for all pairings (with symmetric lookup), identify uncached ones
    uncached_pairings = []  # (opp_idx, opponent, opp_id)
    cached_results = {}  # opp_idx -> result (from warrior's perspective)
    
    for opp_idx, (opp, opp_id) in enumerate(zip(opponents, opponent_ids)):
        # Use canonical key for symmetric cache lookup
        cache_key, is_reversed = _get_canonical_cache_key(warrior_id, opp_id)
        
        # Check cache
        cached = None
        if cache_lock is not None:
            with cache_lock:
                cached = pairing_cache.get(cache_key)
        else:
            cached = pairing_cache.get(cache_key)
        
        if cached is not None:
            # If reversed, swap the result to get warrior's perspective
            if is_reversed:
                cached_results[opp_idx] = _swap_pairing_result(cached)
            else:
                cached_results[opp_idx] = cached
        else:
            uncached_pairings.append((opp_idx, opp, opp_id))
    
    # Step 2: Evaluate uncached pairings in parallel
    if uncached_pairings:
        if n_processes > 1 and len(uncached_pairings) > 1:
            # Parallel evaluation
            eval_args = [
                (simargs, warrior, opp, opp_idx)
                for opp_idx, opp, opp_id in uncached_pairings
            ]
            
            with Pool(processes=min(n_processes, len(uncached_pairings))) as pool:
                results = pool.map(_evaluate_single_pairing, eval_args)
            
            # Process results
            for (opp_idx, opp, opp_id), result in zip(uncached_pairings, results):
                # Store result from warrior's perspective
                warrior_result = {
                    'wins': result['wins'],
                    'losses': result['losses'],
                    'ties': result['ties'],
                    'rounds': result['rounds'],
                    'tsp_w': result['tsp_w'],
                    'tsp_o': result['tsp_o'],
                    'mc_w': result['mc_w'],
                    'mc_o': result['mc_o'],
                    'avg_win_cycles': result.get('avg_win_cycles', 0),
                    'avg_loss_cycles': result.get('avg_loss_cycles', 0),
                }
                
                # Use canonical key for storage
                cache_key, is_reversed = _get_canonical_cache_key(warrior_id, opp_id)
                
                # Store in canonical form (may need to swap if reversed)
                if is_reversed:
                    cache_result = _swap_pairing_result(warrior_result)
                else:
                    cache_result = warrior_result
                
                # Store in cache
                if cache_lock is not None:
                    with cache_lock:
                        pairing_cache[cache_key] = cache_result
                else:
                    pairing_cache[cache_key] = cache_result
                
                # Track new pairing (from warrior's perspective for CSV)
                new_pairings.append({
                    'warrior_id': warrior_id,
                    'opponent_id': opp_id,
                    **warrior_result
                })
                
                cached_results[opp_idx] = warrior_result
        else:
            # Sequential evaluation (n_processes=1 or single pairing)
            for opp_idx, opp, opp_id in uncached_pairings:
                outputs = run_1v1_multiple_rounds(simargs, warrior, opp, n_processes=1)
                
                if outputs is None:
                    warrior_result = {
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
                else:
                    warrior_result = {
                        'wins': outputs['w1_wins'],
                        'losses': outputs['w2_wins'],
                        'ties': outputs['ties'],
                        'rounds': outputs['rounds'],
                        'tsp_w': float(outputs['total_spawned_procs'][0]),
                        'tsp_o': float(outputs['total_spawned_procs'][1]),
                        'mc_w': float(outputs['memory_coverage'][0]),
                        'mc_o': float(outputs['memory_coverage'][1]),
                        'avg_win_cycles': float(outputs.get('avg_win_cycles', 0)),
                        'avg_loss_cycles': float(outputs.get('avg_loss_cycles', 0)),
                    }
                
                # Use canonical key for storage
                cache_key, is_reversed = _get_canonical_cache_key(warrior_id, opp_id)
                
                # Store in canonical form
                if is_reversed:
                    cache_result = _swap_pairing_result(warrior_result)
                else:
                    cache_result = warrior_result
                
                # Store in cache
                if cache_lock is not None:
                    with cache_lock:
                        pairing_cache[cache_key] = cache_result
                else:
                    pairing_cache[cache_key] = cache_result
                
                new_pairings.append({
                    'warrior_id': warrior_id,
                    'opponent_id': opp_id,
                    **warrior_result
                })
                
                cached_results[opp_idx] = warrior_result
    
    # Step 3: Aggregate all results (all from warrior's perspective now)
    all_win_cycles = []
    all_loss_cycles = []
    
    for opp_idx in range(len(opponents)):
        result = cached_results[opp_idx]
        total_wins += result['wins']
        total_losses += result['losses']
        total_ties += result['ties']
        all_tsp.append(result['tsp_w'])
        all_mc.append(result['mc_w'])
        
        # Collect timing data (weighted by number of wins/losses in this pairing)
        if result['wins'] > 0 and result.get('avg_win_cycles', 0) > 0:
            all_win_cycles.extend([result['avg_win_cycles']] * result['wins'])
        if result['losses'] > 0 and result.get('avg_loss_cycles', 0) > 0:
            all_loss_cycles.extend([result['avg_loss_cycles']] * result['losses'])

    # Compute scores
    total_rounds = len(opponents) * simargs.rounds
    hill_score, hill_max = compute_hill_score(total_wins, total_ties, total_rounds)
    
    # Compute timing averages
    avg_win_cycles = sum(all_win_cycles) / len(all_win_cycles) if all_win_cycles else 0.0
    avg_loss_cycles = sum(all_loss_cycles) / len(all_loss_cycles) if all_loss_cycles else 0.0
    
    # Compute gradual score (provides continuous gradient)
    gradual_score, gradual_max = compute_gradual_score(
        total_wins, total_losses, total_ties, total_rounds,
        avg_win_cycles, avg_loss_cycles, simargs.cycles
    )

    # Win/loss/tie percentages
    win_pct = total_wins / total_rounds if total_rounds > 0 else 0
    tie_pct = total_ties / total_rounds if total_rounds > 0 else 0
    loss_pct = total_losses / total_rounds if total_rounds > 0 else 0

    # Average BC features
    avg_tsp = sum(all_tsp) / len(all_tsp) if all_tsp else 0.0
    avg_mc = sum(all_mc) / len(all_mc) if all_mc else 0.0

    return {
        'hill_score': hill_score,
        'max_score': hill_max,
        'hill_score_pct': (hill_score / hill_max * 100) if hill_max > 0 else 0,
        'gradual_score': gradual_score,
        'gradual_max': gradual_max,
        'gradual_score_pct': (gradual_score / gradual_max * 100) if gradual_max > 0 else 0,
        'wins': total_wins,
        'losses': total_losses,
        'ties': total_ties,
        'total_rounds': total_rounds,
        'win_pct': win_pct,
        'tie_pct': tie_pct,
        'loss_pct': loss_pct,
        'avg_win_cycles': avg_win_cycles,
        'avg_loss_cycles': avg_loss_cycles,
        'total_spawned_procs': avg_tsp,
        'memory_coverage': avg_mc,
        'n_opponents': len(opponents),
        'new_pairings': new_pairings
    }


def run_multiple_rounds(simargs, warriors, n_processes=1, timeout=None):
    """Original function for multi-warrior simulation (kept for compatibility)."""
    # Sequential execution when n_processes=1 (avoids daemon process issues)
    if n_processes == 1:
        outputs = [run_single_round(simargs, warriors, seed) for seed in range(simargs.rounds)]
    elif timeout is None:
        outputs = run_multiple_rounds_fast(simargs, warriors, n_processes)
    else:
        # Parallel with timeout
        try:
            run_single_round_fn = partial(run_single_round, simargs, warriors)
            seeds = list(range(simargs.rounds))
            with Pool(processes=n_processes) as pool:
                result = pool.map_async(run_single_round_fn, seeds)
                outputs = result.get(timeout=timeout)
        except Exception as e:
            print(e)
            return None
    return {k: np.stack([o[k] for o in outputs], axis=-1) for k in outputs[0].keys()}


def run_multiple_rounds_fast(simargs, warriors, n_processes=1):
    """Faster version without timeout (kept for compatibility)."""
    try:
        from multiprocessing import Process, Manager
        import multiprocessing as mp
        import ctypes as ct

        run_single_round_fn = partial(run_single_round, simargs, warriors)
        manager = Manager()
        outputs = manager.dict()
        mutex = mp.Lock()
        seed = mp.RawValue(ct.c_int, 0)

        def run():
            proc = [Process(target=loop, args=()) for p in range(n_processes)]
            for p in proc:
                p.start()
            for p in proc:
                p.join()
            return [outputs[seed] for seed in range(simargs.rounds)]

        def loop():
            import threadpoolctl
            with threadpoolctl.threadpool_limits(limits=1, user_api="blas"):
                while True:
                    with mutex:
                        if seed.value >= simargs.rounds:
                            return
                        seed_val = seed.value
                        seed.value += 1
                    outputs[seed_val] = run_single_round_fn(seed_val)

        return run()

    except Exception as e:
        print(e)
        return None


def parse_warrior_from_file(simargs, file):
    """Parse a warrior from a .red file."""
    environment = simargs_to_environment(simargs)
    with open(file, encoding="latin1") as f:
        warrior_str = f.read()
    warrior = redcode.parse(warrior_str.split("\n"), environment)
    return warrior_str, warrior
