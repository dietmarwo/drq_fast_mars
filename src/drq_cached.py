"""
DRQ Explicit with Pairwise Caching

Key features:
- ExplicitCoreMutator only (no LLM)
- Fixed opponent pool from directory
- Pairwise 1v1 evaluation with global caching
- Standard hill scoring: Win=3, Tie=1, Loss=0
- CSV output for pairings and warrior scores after each round
- Accumulating archive between rounds
"""

import re
import random
import os
import glob
import csv
import numpy as np
import time
import hashlib
import psutil
import copy
from multiprocessing import Manager, Lock

from dataclasses import dataclass, field
import tyro
import asyncio
from tqdm.auto import tqdm

from corewar_util import (
    SimulationArgs, simargs_to_environment, parse_warrior_from_file,
    run_pairwise_with_cache, run_1v1_multiple_rounds, compute_hill_score,
    compute_gradual_score, _get_canonical_cache_key, _swap_pairing_result
)
from corewar import MARS, Warrior
import util
from util import ExplicitWarrior
from multiprocessing import Pool


def _evaluate_warmstart_pairing(args_tuple):
    """
    Evaluate a single (warrior, opponent) pairing for warmstart.
    Top-level function for multiprocessing Pool.
    """
    simargs, warrior, opponent, warrior_idx, opponent_idx = args_tuple
    
    result = run_1v1_multiple_rounds(simargs, warrior, opponent, n_processes=1)
    
    if result is None:
        return {
            'warrior_idx': warrior_idx,
            'opponent_idx': opponent_idx,
            'wins': 0,
            'losses': simargs.rounds,
            'ties': 0,
            'rounds': simargs.rounds,
            'tsp_w': 0.0,
            'mc_w': 0.0,
            'avg_win_cycles': 0.0,
            'avg_loss_cycles': float(simargs.cycles),
        }
    
    return {
        'warrior_idx': warrior_idx,
        'opponent_idx': opponent_idx,
        'wins': result['w1_wins'],
        'losses': result['w2_wins'],
        'ties': result['ties'],
        'rounds': result['rounds'],
        'tsp_w': float(result['total_spawned_procs'][0]),
        'mc_w': float(result['memory_coverage'][0]),
        'avg_win_cycles': float(result.get('avg_win_cycles', 0)),
        'avg_loss_cycles': float(result.get('avg_loss_cycles', 0)),
    }

@dataclass
class Args:
    # General arguments
    seed: int = 0
    save_dir: str | None = 'results/explicit_16_1v1'
    n_processes: int = 24
    resume: bool | None = False
    job_timeout: int = 24 * 60 * 60

    # Core War arguments
    simargs: SimulationArgs = field(default_factory=SimulationArgs)
    timeout: int = 900

    # Opponent arguments
    opponent_dir: str = "../warrior_1v1"
    opponent_pattern: str = "*.red"
    max_opponents: int | None = None

    # DRQ arguments
    n_rounds: int = 120
    n_iters: int = 100
    log_every: int = 10
    sample_new_percent: float = 0.1
    bc_axes: str = "tsp,mc"
    n_init: int = 8
    n_mutate: int = 1
    fitness_threshold: float = 10000.0
    single_cell: bool | None = False
    
    # Scoring mode
    gradual_scoring: bool = True  # Use gradual scoring for smoother optimization

    # Archive accumulation
    carry_archive: bool = True
    reevaluate_carried: bool = False  # Not needed with fixed opponents and caching

    # CSV output
    pairings_csv: str = "pairings.csv"
    scores_csv: str = "warrior_scores.csv"
    
    # Warmstart from previous run
    #warmstart_dir: str | None = 'results/explicit_16/final_niches'  # Directory containing .red files to seed archive
    #warmstart_dir: str | None = 'results/drqres'  # Directory containing .red files to seed archive
    warmstart_dir: str | None = "../warrior_1v1"#"../human_warriors"  # Directory containing .red files to seed archive
    warmstart_pattern: str = "*.red"  # Glob pattern for warmstart files


class MapElites:
    def __init__(self):
        self.archive = {}
        self.history = []
        self.coverage_history = []
        self.fitness_history = []

    def sample(self) -> ExplicitWarrior:
        random_key = random.choice(list(self.archive.keys()))
        return self.archive[random_key]

    def place(self, warrior: ExplicitWarrior) -> bool:
        place = (warrior.bc is not None) and (warrior.fitness is not None) and (warrior.fitness > -np.inf)
        place = place and ((warrior.bc not in self.archive) or (warrior.fitness > self.archive[warrior.bc].fitness))
        if place:
            self.archive[warrior.bc] = warrior
        self.history.append(warrior)
        self.coverage_history.append(len(self.archive))
        self.fitness_history.append(self.get_best().fitness if len(self.archive) > 0 else -np.inf)
        return place

    def get_best(self) -> ExplicitWarrior | None:
        best_key, best_fitness = None, -np.inf
        for k, v in self.archive.items():
            if v.fitness > best_fitness:
                best_key, best_fitness = k, v.fitness
        return self.archive[best_key] if best_key in self.archive else None

    def get_all_entries(self) -> list:
        return list(self.archive.values())


class Main:
    """
    DRQ with explicit mutation, pairwise caching, and hill scoring.
    """

    def __init__(self, args: Args):
        self.args = args
        print(args)
        nproc = os.popen("nproc").read().strip()
        nproc_all = os.popen("nproc --all").read().strip()
        print(f"Number of cores: {nproc} / {nproc_all}")

        random.seed(args.seed)
        np.random.seed(args.seed)

        # Initialize explicit mutator (use 1v1 optimized version)
        from explicit_mutator_1v1 import ExplicitCoreMutator
        #from explicit_mutator2 import ExplicitCoreMutator
        # warmstart_mode=True for conservative mutations when warmstarting
        warmstart = args.warmstart_dir is not None
        self.mutator = ExplicitCoreMutator(
            environment=simargs_to_environment(args.simargs),
            warmstart_mode=warmstart
        )

        # Load opponents from directory
        self.init_opps = self._load_opponents()
        self.opponent_ids = [opp.id for opp in self.init_opps]
        print(f"Loaded {len(self.init_opps)} opponent warriors from {args.opponent_dir}")
        print(f"Rounds per pairing: {args.simargs.rounds}")
        print(f"Max hill score per opponent: {args.simargs.rounds * 3}")
        print(f"Max total hill score: {len(self.init_opps) * args.simargs.rounds * 3}")
        print(f"Scoring mode: {'gradual' if args.gradual_scoring else 'hill'}")

        # Initialize multiprocessing manager for shared cache
        self.manager = Manager()
        self.pairing_cache = self.manager.dict()
        self.cache_lock = self.manager.Lock()

        # Track all evaluated warriors for CSV output
        self.all_warriors = {}  # id -> ExplicitWarrior (with latest fitness)
        self.new_pairings_buffer = []  # Buffer for new pairings to write

        self.timestamps = []
        self.all_rounds_map_elites = {i_round: MapElites() for i_round in range(self.args.n_rounds)}

        # Load existing pairings if resuming
        if args.resume and args.save_dir:
            self._load_pairings_from_csv()
        
        # Warmstart from previous run's niches
        if args.warmstart_dir:
            if args.resume:
                print("Warning: Both --resume and --warmstart_dir specified. Using --resume only.")
            else:
                self._warmstart_from_directory()

    def _load_opponents(self) -> list:
        """Load opponent warriors from directory."""
        pattern = os.path.join(self.args.opponent_dir, self.args.opponent_pattern)
        files = sorted(glob.glob(pattern))

        if self.args.max_opponents is not None:
            files = files[:self.args.max_opponents]

        opponents = []
        for file in files:
            try:
                warrior_str, warrior = parse_warrior_from_file(self.args.simargs, file)
                ew = ExplicitWarrior(warrior=warrior, code=warrior_str)
                opponents.append(ew)
            except Exception as e:
                print(f"Warning: Failed to load {file}: {e}")

        return opponents

    def _load_pairings_from_csv(self):
        """Load existing pairings from CSV into cache."""
        csv_path = os.path.join(self.args.save_dir, self.args.pairings_csv)
        self._load_pairings_from_csv_path(csv_path)
    
    def _load_pairings_from_csv_path(self, csv_path: str):
        """Load pairings from a specific CSV path into cache."""
        if not os.path.exists(csv_path):
            return

        print(f"Loading cached pairings from {csv_path}...")
        count = 0
        with open(csv_path, 'r', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                warrior_id = row['warrior_id']
                opponent_id = row['opponent_id']
                
                # Use canonical key for symmetric caching
                cache_key, is_reversed = _get_canonical_cache_key(warrior_id, opponent_id)
                
                # Skip if already in cache (don't overwrite)
                if cache_key in self.pairing_cache:
                    continue
                
                # Result from CSV is from warrior's perspective
                warrior_result = {
                    'wins': int(row['wins']),
                    'losses': int(row['losses']),
                    'ties': int(row['ties']),
                    'rounds': int(row['rounds']),
                    'tsp_w': float(row['tsp_w']),
                    'tsp_o': float(row['tsp_o']),
                    'mc_w': float(row['mc_w']),
                    'mc_o': float(row['mc_o']),
                }
                
                # Store in canonical form (swap if reversed)
                if is_reversed:
                    self.pairing_cache[cache_key] = _swap_pairing_result(warrior_result)
                else:
                    self.pairing_cache[cache_key] = warrior_result
                    
                count += 1
        print(f"Loaded {count} cached pairings")

    def _warmstart_from_directory(self):
        """
        Load warriors from warmstart directory, evaluate them in parallel, and seed round 0 archive.
        
        Uses multiprocessing Pool to parallelize all pairings across all warriors,
        similar to evaluate_1v1.py approach.
        """
        pattern = os.path.join(self.args.warmstart_dir, self.args.warmstart_pattern)
        files = sorted(glob.glob(pattern))
        
        if not files:
            print(f"Warning: No files found matching {pattern}")
            return
        
        print(f"\n{'='*60}")
        print(f"WARMSTART: Loading {len(files)} warriors from {self.args.warmstart_dir}")
        print(f"{'='*60}")
        
        # Step 1: Load all warriors
        print("Loading warriors...")
        warmstart_warriors = []
        for file in files:
            try:
                warrior_str, warrior = parse_warrior_from_file(self.args.simargs, file)
                ew = ExplicitWarrior(warrior=warrior, code=warrior_str)
                warmstart_warriors.append({
                    'file': os.path.basename(file),
                    'warrior': warrior,
                    'code': warrior_str,
                    'ew': ew
                })
            except Exception as e:
                print(f"  Warning: Failed to load {os.path.basename(file)}: {e}")
        
        print(f"Loaded {len(warmstart_warriors)} warriors")
        
        if not warmstart_warriors:
            return
        
        # Step 2: Build all pairing arguments
        opponents = [opp.warrior for opp in self.init_opps]
        total_pairings = len(warmstart_warriors) * len(opponents)
        
        print(f"Building {total_pairings} pairings ({len(warmstart_warriors)} warriors × {len(opponents)} opponents)...")
        
        pairing_args = []
        for w_idx, w in enumerate(warmstart_warriors):
            for o_idx, opp in enumerate(opponents):
                pairing_args.append((
                    self.args.simargs,
                    w['warrior'],
                    opp,
                    w_idx,
                    o_idx
                ))
        
        # Step 3: Evaluate all pairings in parallel
        print(f"Evaluating {len(pairing_args)} pairings with {self.args.n_processes} processes...")
        
        with Pool(processes=self.args.n_processes) as pool:
            pairing_results = list(tqdm(
                pool.imap(_evaluate_warmstart_pairing, pairing_args),
                total=len(pairing_args),
                desc="Warmstart pairings"
            ))
        
        # Step 4: Aggregate results per warrior
        print("Aggregating results...")
        
        warrior_stats = [{
            'wins': 0, 'losses': 0, 'ties': 0, 'total_rounds': 0,
            'tsp_sum': 0.0, 'mc_sum': 0.0, 'n_opponents': 0,
            'win_cycles': [], 'loss_cycles': []  # For gradual scoring
        } for _ in warmstart_warriors]
        
        for pr in pairing_results:
            w_idx = pr['warrior_idx']
            warrior_stats[w_idx]['wins'] += pr['wins']
            warrior_stats[w_idx]['losses'] += pr['losses']
            warrior_stats[w_idx]['ties'] += pr['ties']
            warrior_stats[w_idx]['total_rounds'] += pr['rounds']
            warrior_stats[w_idx]['tsp_sum'] += pr['tsp_w']
            warrior_stats[w_idx]['mc_sum'] += pr['mc_w']
            warrior_stats[w_idx]['n_opponents'] += 1
            
            # Track timing for gradual scoring
            if pr['wins'] > 0 and pr.get('avg_win_cycles', 0) > 0:
                warrior_stats[w_idx]['win_cycles'].extend([pr['avg_win_cycles']] * pr['wins'])
            if pr['losses'] > 0 and pr.get('avg_loss_cycles', 0) > 0:
                warrior_stats[w_idx]['loss_cycles'].extend([pr['avg_loss_cycles']] * pr['losses'])
            
            # Also update cache for later use (with symmetric key)
            w_id = warmstart_warriors[w_idx]['ew'].id
            o_id = self.opponent_ids[pr['opponent_idx']]
            
            # Use canonical key for symmetric cache
            cache_key, is_reversed = _get_canonical_cache_key(w_id, o_id)
            
            if cache_key not in self.pairing_cache:
                # Result from warrior's perspective
                warrior_result = {
                    'wins': pr['wins'],
                    'losses': pr['losses'],
                    'ties': pr['ties'],
                    'rounds': pr['rounds'],
                    'tsp_w': pr['tsp_w'],
                    'tsp_o': 0.0,  # Not tracked for opponents in warmstart
                    'mc_w': pr['mc_w'],
                    'mc_o': 0.0,
                    'avg_win_cycles': pr.get('avg_win_cycles', 0),
                    'avg_loss_cycles': pr.get('avg_loss_cycles', 0),
                }
                
                # Store in canonical form (swap if reversed)
                if is_reversed:
                    self.pairing_cache[cache_key] = _swap_pairing_result(warrior_result)
                else:
                    self.pairing_cache[cache_key] = warrior_result
                
                # Add to pairings buffer for CSV (always from warrior's perspective)
                self.new_pairings_buffer.append({
                    'warrior_id': w_id,
                    'opponent_id': o_id,
                    **warrior_result
                })
        
        # Step 5: Create ExplicitWarriors with scores and place in archive
        loaded = 0
        failed = 0
        
        for w_idx, w in enumerate(warmstart_warriors):
            stats = warrior_stats[w_idx]
            ew = w['ew']
            
            if stats['total_rounds'] == 0:
                failed += 1
                continue
            
            # Compute hill score
            hill_score, max_score = compute_hill_score(
                stats['wins'], stats['ties'], stats['total_rounds']
            )
            
            # Compute timing averages for gradual scoring
            avg_win_cycles = sum(stats['win_cycles']) / len(stats['win_cycles']) if stats['win_cycles'] else 0
            avg_loss_cycles = sum(stats['loss_cycles']) / len(stats['loss_cycles']) if stats['loss_cycles'] else 0
            
            # Compute gradual score
            gradual_score, gradual_max = compute_gradual_score(
                stats['wins'], stats['losses'], stats['ties'], stats['total_rounds'],
                avg_win_cycles, avg_loss_cycles, self.args.simargs.cycles
            )
            
            # Compute averages
            avg_tsp = stats['tsp_sum'] / stats['n_opponents'] if stats['n_opponents'] > 0 else 0
            avg_mc = stats['mc_sum'] / stats['n_opponents'] if stats['n_opponents'] > 0 else 0
            
            # Update warrior with scores (use gradual if enabled)
            if self.args.gradual_scoring:
                ew.fitness = gradual_score
                ew.hill_score_pct = (gradual_score / gradual_max * 100) if gradual_max > 0 else 0
            else:
                ew.fitness = hill_score
                ew.hill_score_pct = (hill_score / max_score * 100) if max_score > 0 else 0
            
            ew.wins = stats['wins']
            ew.losses = stats['losses']
            ew.ties = stats['ties']
            ew.total_rounds = stats['total_rounds']
            
            ew.outputs = {
                'hill_score': hill_score,
                'hill_score_pct': (hill_score / max_score * 100) if max_score > 0 else 0,
                'gradual_score': gradual_score,
                'gradual_score_pct': (gradual_score / gradual_max * 100) if gradual_max > 0 else 0,
                'wins': stats['wins'],
                'losses': stats['losses'],
                'ties': stats['ties'],
                'avg_win_cycles': avg_win_cycles,
                'avg_loss_cycles': avg_loss_cycles,
                'total_spawned_procs': np.array([avg_tsp, 0]),
                'memory_coverage': np.array([avg_mc, 0]),
            }
            
            # Compute BC features
            ew.bc = self.get_bc_features(avg_tsp, avg_mc, ew.warrior)
            
            # Track in all_warriors
            self.all_warriors[ew.id] = ew
            
            # Place in round 0 archive
            if ew.fitness > -np.inf:
                self.all_rounds_map_elites[0].place(ew)
                loaded += 1
            else:
                failed += 1
        
        # Report results
        me = self.all_rounds_map_elites[0]
        print(f"\nWarmstart complete:")
        print(f"  Loaded: {loaded} warriors")
        print(f"  Failed: {failed} warriors")
        print(f"  Archive coverage: {len(me.archive)}/36 niches")
        print(f"  Cache entries: {len(self.pairing_cache)}")
        print(f"  Scoring mode: {'gradual' if self.args.gradual_scoring else 'hill'}")
        
        if len(me.archive) > 0:
            best = me.get_best()
            print(f"  Best fitness: {best.fitness:.2f} ({best.hill_score_pct:.1f}%)")
            print(f"  Best win rate: {best.win_pct:.1f}%")
        
        # Save initial state
        self.save_pairings_csv()
        self.save_scores_csv(0)
        print(f"{'='*60}\n")
        
        for bc, warrior in me.archive.items():
            code = re.sub(r"```.*", "", warrior.code)
            filename = f"niche_{bc[0]}_{bc[1]}.red"
            filepath = os.path.join(filename)
            
            with open(filepath, "w") as f:
                f.write(code)

    def get_bc_features(self, tsp: float, mc: float, warrior: Warrior) -> tuple:
        """Compute BC features from aggregated outputs."""
        if self.args.single_cell:
            return (0, 0)

        unique_opcodes = len({i.opcode for i in warrior.instructions})
        program_len = len(warrior.instructions)

        for bc, a in enumerate([1, 10, 100, 1000, 10000, np.inf]):
            if tsp < a:
                bc_tsp = bc
                break

        for bc, a in enumerate([10, 100, 500, 1000, 4000, np.inf]):
            if mc < a:
                bc_mc = bc
                break

        for bc, a in enumerate([4, 6, 8, 10, 14, np.inf]):
            if unique_opcodes < a:
                bc_uo = bc
                break

        for bc, a in enumerate([5, 12, 20, 35, 60, np.inf]):
            if program_len < a:
                bc_pl = bc
                break

        all_bcs = dict(tsp=bc_tsp, mc=bc_mc, uo=bc_uo, pl=bc_pl)
        bc1, bc2 = self.args.bc_axes.split(",")
        return (all_bcs[bc1], all_bcs[bc2])

    def evaluate_warrior(self, warrior: ExplicitWarrior) -> ExplicitWarrior:
        """Evaluate warrior against all opponents using pairwise cache with hill/gradual scoring."""
        warrior = copy.deepcopy(warrior)

        if warrior.warrior is None:
            warrior.bc, warrior.fitness = None, -np.inf
            return warrior

        # Ensure warrior has an ID
        if not warrior.id:
            warrior.id = warrior._compute_id()

        # Get opponents (fixed - no evolution)
        opponents = [opp.warrior for opp in self.init_opps]

        # Run pairwise evaluation with caching
        # n_processes enables parallel evaluation of uncached pairings
        result = run_pairwise_with_cache(
            simargs=self.args.simargs,
            warrior=warrior.warrior,
            warrior_id=warrior.id,
            opponents=opponents,
            opponent_ids=self.opponent_ids,
            pairing_cache=self.pairing_cache,
            cache_lock=self.cache_lock,
            n_processes=self.args.n_processes  # Enable parallel evaluation
        )

        # Store new pairings for CSV output
        self.new_pairings_buffer.extend(result['new_pairings'])

        # Update warrior with scoring results
        # Use gradual scoring for smoother optimization, or hill scoring for compatibility
        if self.args.gradual_scoring:
            warrior.fitness = result['gradual_score']
            warrior.hill_score_pct = result['gradual_score_pct']
        else:
            warrior.fitness = result['hill_score']
            warrior.hill_score_pct = result['hill_score_pct']
        
        warrior.wins = result['wins']
        warrior.losses = result['losses']
        warrior.ties = result['ties']
        warrior.total_rounds = result['total_rounds']
        
        warrior.outputs = {
            'hill_score': result['hill_score'],
            'hill_score_pct': result['hill_score_pct'],
            'gradual_score': result['gradual_score'],
            'gradual_score_pct': result['gradual_score_pct'],
            'wins': result['wins'],
            'losses': result['losses'],
            'ties': result['ties'],
            'avg_win_cycles': result.get('avg_win_cycles', 0),
            'avg_loss_cycles': result.get('avg_loss_cycles', 0),
            'total_spawned_procs': np.array(result['total_spawned_procs']),
            'memory_coverage': np.array(result['memory_coverage']),
        }
        warrior.bc = self.get_bc_features(
            result['total_spawned_procs'],
            result['memory_coverage'],
            warrior.warrior
        )

        # Track warrior for scores CSV
        self.all_warriors[warrior.id] = warrior

        return warrior

    def process_warrior(self, i_round: int, warrior: ExplicitWarrior) -> bool:
        """Evaluate and place warrior in archive."""
        warrior = self.evaluate_warrior(warrior)
        map_elites = self.all_rounds_map_elites[i_round]
        return map_elites.place(warrior)

    def carry_forward_archive(self, i_round: int):
        """Carry archive from previous round."""
        if i_round == 0:
            return

        prev_me = self.all_rounds_map_elites[i_round - 1]
        current_me = self.all_rounds_map_elites[i_round]

        carried = prev_me.get_all_entries()
        if len(carried) == 0:
            return

        print(f"Round {i_round}: Carrying {len(carried)} warriors from round {i_round - 1}")

        for warrior in carried:
            # With fixed opponents and caching, no need to re-evaluate
            # Fitness is the same, just copy to new archive
            current_me.place(warrior)

        print(f"Round {i_round}: Archive has {len(current_me.archive)} warriors after carryover")

    def init_round(self, i_round: int):
        """Initialize a round."""
        if self.args.carry_archive:
            self.carry_forward_archive(i_round)

        current_me = self.all_rounds_map_elites[i_round]
        n_new = max(0, self.args.n_init - len(current_me.archive))

        if n_new > 0:
            print(f"Round {i_round}: Generating {n_new} new warriors")
            new_warriors = asyncio.run(
                self.mutator.new_warrior_async(n_warriors=1, n_responses=n_new)
            ).flatten()

            for w in new_warriors:
                # Handle both .code and .llm_response attributes
                code = getattr(w, 'code', None) or getattr(w, 'llm_response', '')
                ew = ExplicitWarrior(warrior=w.warrior, code=code)
                self.process_warrior(i_round, ew)

        print(f"Round {i_round}: Initialized with {len(current_me.archive)} warriors")

    def step(self, i_round: int):
        """One evolution step."""
        current_me = self.all_rounds_map_elites[i_round]

        # Periodically update mutator with learned constants from archive
        if len(current_me.archive) > 0 and hasattr(self.mutator, 'learn_from_archive'):
            self.mutator.learn_from_archive(current_me.archive)

        if random.random() < self.args.sample_new_percent or len(current_me.archive) == 0:
            new_warriors = asyncio.run(
                self.mutator.new_warrior_async(n_warriors=1, n_responses=self.args.n_mutate)
            ).flatten()

            for w in new_warriors:
                code = getattr(w, 'code', None) or getattr(w, 'llm_response', '')
                ew = ExplicitWarrior(warrior=w.warrior, code=code)
                self.process_warrior(i_round, ew)
        else:
            parent = current_me.sample()

            # Convert to ExplicitWarrior format for mutator compatibility
            parent_warrior = ExplicitWarrior(
                code=parent.code, 
                warrior=parent.warrior
            )

            offspring = asyncio.run(
                self.mutator.mutate_warrior_async([parent_warrior], n_responses=self.args.n_mutate)
            ).flatten()

            for w in offspring:
                code = getattr(w, 'code', None) or getattr(w, 'llm_response', '')
                ew = ExplicitWarrior(warrior=w.warrior, code=code)
                self.process_warrior(i_round, ew)

    def save_pairings_csv(self):
        """Save all pairings to CSV (append new, or write all if file doesn't exist)."""
        if self.args.save_dir is None:
            return

        csv_path = os.path.join(self.args.save_dir, self.args.pairings_csv)
        file_exists = os.path.exists(csv_path)

        if len(self.new_pairings_buffer) == 0 and file_exists:
            return  # Nothing new to write

        # If file doesn't exist, write all cached pairings
        if not file_exists:
            print(f"Writing all {len(self.pairing_cache)} pairings to {csv_path}")
            with open(csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['warrior_id', 'opponent_id', 'wins', 'losses', 'ties', 'rounds', 
                                'tsp_w', 'tsp_o', 'mc_w', 'mc_o'])

                for cache_key, result in self.pairing_cache.items():
                    warrior_id, opponent_id = cache_key.split(':')
                    writer.writerow([
                        warrior_id, opponent_id,
                        result['wins'], result['losses'], result['ties'], result['rounds'],
                        result['tsp_w'], result['tsp_o'],
                        result['mc_w'], result['mc_o']
                    ])
        else:
            # Append only new pairings
            if len(self.new_pairings_buffer) > 0:
                print(f"Appending {len(self.new_pairings_buffer)} new pairings to {csv_path}")
                with open(csv_path, 'a', newline='') as f:
                    writer = csv.writer(f)
                    for p in self.new_pairings_buffer:
                        writer.writerow([
                            p['warrior_id'], p['opponent_id'],
                            p['wins'], p['losses'], p['ties'], p['rounds'],
                            p['tsp_w'], p['tsp_o'],
                            p['mc_w'], p['mc_o']
                        ])

        self.new_pairings_buffer = []  # Clear buffer

    def save_scores_csv(self, i_round: int):
        """Save all warrior scores to CSV, sorted by hill score (best first)."""
        if self.args.save_dir is None:
            return

        csv_path = os.path.join(self.args.save_dir, self.args.scores_csv)

        # Collect all warriors with valid scores
        warriors_with_scores = [
            (w_id, w) for w_id, w in self.all_warriors.items()
            if w.fitness is not None and w.fitness > -np.inf
        ]

        # Sort by fitness (hill score) descending
        warriors_with_scores.sort(key=lambda x: -x[1].fitness)

        print(f"Writing {len(warriors_with_scores)} warrior scores to {csv_path}")

        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['rank', 'warrior_id', 'name', 'hill_score', 'hill_score_pct',
                            'wins', 'losses', 'ties', 'total_rounds', 
                            'win_pct', 'tie_pct', 'loss_pct',
                            'bc_tsp', 'bc_mc', 'last_round'])

            for rank, (w_id, w) in enumerate(warriors_with_scores, 1):
                # Find which round this warrior last appeared in
                last_round = -1
                for r in range(i_round, -1, -1):
                    me = self.all_rounds_map_elites[r]
                    if any(arch_w.id == w_id for arch_w in me.archive.values()):
                        last_round = r
                        break

                bc = w.bc if w.bc else (None, None)
                writer.writerow([
                    rank, w_id, w.name, w.fitness, f"{w.hill_score_pct:.2f}",
                    w.wins, w.losses, w.ties, w.total_rounds,
                    f"{w.win_pct:.2f}", f"{w.tie_pct:.2f}", f"{w.loss_pct:.2f}",
                    bc[0], bc[1], last_round
                ])

    def run(self):
        this_job_start_time = time.time()

        if self.args.resume and os.path.exists(f"{self.args.save_dir}/args.pkl"):
            self.timestamps = util.load_pkl(self.args.save_dir, "timestamps")
            self.all_rounds_map_elites = util.load_pkl(self.args.save_dir, "all_rounds_map_elites")
            self.mutator.all_generations = util.load_pkl(self.args.save_dir, "all_generations")

            # Rebuild all_warriors from loaded archives
            for me in self.all_rounds_map_elites.values():
                for w in me.archive.values():
                    self.all_warriors[w.id] = w

            print(f"Resumed from {self.args.save_dir}")
            start_abs_iter = self.timestamps[-1]["abs_iter"] + 1
        else:
            start_abs_iter = 0

        pbar = tqdm(range(start_abs_iter, self.args.n_rounds * self.args.n_iters))
        for abs_iter in pbar:
            i_round = abs_iter // self.args.n_iters
            i_iter = abs_iter % self.args.n_iters
            start_time = time.time()

            me = self.all_rounds_map_elites[i_round]
            best = me.get_best()
            best_pct = best.hill_score_pct if best else 0
            should_skip = best_pct > self.args.fitness_threshold

            if not should_skip:
                if i_iter == 0:
                    self.init_round(i_round)
                self.step(i_round)

            process = psutil.Process(os.getpid())
            rss = process.memory_info().rss
            vms = process.memory_info().vms

            self.timestamps.append(dict(
                abs_iter=abs_iter,
                i_round=i_round,
                i_iter=i_iter,
                dt=time.time() - start_time,
                rss=rss,
                vms=vms,
                archive_size=len(me.archive),
                cache_size=len(self.pairing_cache)
            ))

            # Save at end of each round or at log_every
            if i_iter == self.args.n_iters - 1 or abs_iter % self.args.log_every == 0:
                self.save()
                self.save_pairings_csv()
                self.save_scores_csv(i_round)

            if len(me.archive) > 0:
                best = me.get_best()
                pbar.set_postfix(
                    score=f"{best.fitness}",
                    pct=f"{best.hill_score_pct:.1f}%",
                    win=f"{best.win_pct:.0f}%",
                    archive=len(me.archive),
                    cache=len(self.pairing_cache)
                )

            if (time.time() - this_job_start_time) > self.args.job_timeout:
                break

        self.save()
        self.save_pairings_csv()
        self.save_scores_csv(self.args.n_rounds - 1)
        self.save_final_summary()

    def save(self):
        if self.args.save_dir is None:
            return

        os.makedirs(self.args.save_dir, exist_ok=True)

        util.save_pkl(self.args.save_dir, "args", self.args)
        util.save_pkl(self.args.save_dir, "timestamps", self.timestamps)
        util.save_pkl(self.args.save_dir, "all_rounds_map_elites", self.all_rounds_map_elites)
        util.save_pkl(self.args.save_dir, "all_generations", self.mutator.all_generations)

        # Save champions
        for i_round, me in self.all_rounds_map_elites.items():
            if len(me.archive) > 0:
                champion = me.get_best()
                code = re.sub(r"```.*", "", champion.code)
                with open(f"{self.args.save_dir}/round_{i_round:03d}_champion.red", "w") as f:
                    f.write(code)

    def save_final_summary(self):
        """Save summary of final archive."""
        if self.args.save_dir is None:
            return

        final_round = -1
        for i_round, me in self.all_rounds_map_elites.items():
            if len(me.archive) > 0:
                final_round = i_round

        if final_round < 0:
            return

        final_me = self.all_rounds_map_elites[final_round]

        # Save all niche winners
        niche_dir = os.path.join(self.args.save_dir, "final_niches")
        os.makedirs(niche_dir, exist_ok=True)

        summary = []
        for bc, warrior in final_me.archive.items():
            code = re.sub(r"```.*", "", warrior.code)
            filename = f"niche_{bc[0]}_{bc[1]}.red"
            filepath = os.path.join(niche_dir, filename)

            with open(filepath, "w") as f:
                f.write(code)

            summary.append({
                'bc': bc,
                'hill_score': warrior.fitness,
                'hill_score_pct': warrior.hill_score_pct,
                'wins': warrior.wins,
                'losses': warrior.losses,
                'ties': warrior.ties,
                'win_pct': warrior.win_pct,
                'name': warrior.name,
                'id': warrior.id,
                'file': filename
            })

        # Save summary CSV
        with open(os.path.join(self.args.save_dir, "final_archive_summary.csv"), "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(['bc_tsp', 'bc_mc', 'hill_score', 'hill_score_pct', 
                            'wins', 'losses', 'ties', 'win_pct', 'name', 'id', 'file'])
            for s in sorted(summary, key=lambda x: -x['hill_score']):
                writer.writerow([
                    s['bc'][0], s['bc'][1], s['hill_score'], f"{s['hill_score_pct']:.2f}",
                    s['wins'], s['losses'], s['ties'], f"{s['win_pct']:.2f}",
                    s['name'], s['id'], s['file']
                ])

        best = final_me.get_best()
        print(f"\nFinal archive (round {final_round}):")
        print(f"  Coverage: {len(final_me.archive)}/36 niches")
        print(f"  Best hill score: {best.fitness} ({best.hill_score_pct:.1f}%)")
        print(f"  Best win rate: {best.win_pct:.1f}% ({best.wins}W/{best.ties}T/{best.losses}L)")
        print(f"  Total pairings cached: {len(self.pairing_cache)}")
        print(f"  Saved to: {niche_dir}")


if __name__ == "__main__":
    main = Main(tyro.cli(Args))
    main.run()
