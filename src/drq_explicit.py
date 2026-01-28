"""
DRQ Explicit - Simplified DRQ using only explicit mutation operators.

Key differences from original drq.py:
- No LLM/GPT - uses ExplicitCoreMutator only
- ExplicitWarrior class instead of GPTWarrior
- Opponents loaded from directory, fixed (no evolution of opponent pool)
- Accumulating archive between rounds with re-evaluation
- Warrior ID based on instruction hash
- Warmstart option to load warriors from a previous run
"""

import re
import random
import os
import glob
import csv
import numpy as np
import time
import psutil
import copy

from dataclasses import dataclass, field
import tyro
import asyncio
from tqdm.auto import tqdm

from corewar_util import SimulationArgs, simargs_to_environment, parse_warrior_from_file, run_multiple_rounds
from corewar import MARS, Warrior
import util
from util import ExplicitWarrior
from multiprocessing import Pool


def _evaluate_warmstart_warrior_multi(args_tuple):
    """Evaluate a single warrior in multi-warrior battle."""
    simargs, warrior, opponent_warriors, warrior_idx = args_tuple
    
    try:
        # Combine warrior with opponents (warrior is index 0)
        all_warriors = [warrior] + opponent_warriors
        
        # Run with n_processes=1 since we're parallelizing at warrior level
        outputs = run_multiple_rounds(simargs, all_warriors, n_processes=1, timeout=300)
        
        if outputs is None:
            return {'warrior_idx': warrior_idx, 'success': False, 'outputs': None}
        
        # Extract warrior's stats (index 0)
        return {
            'warrior_idx': warrior_idx,
            'success': True,
            'outputs': {k: v.mean(axis=-1)[0] for k, v in outputs.items()}
        }
    except Exception as e:
        print(f"Error evaluating warrior {warrior_idx}: {e}")
        return {'warrior_idx': warrior_idx, 'success': False, 'outputs': None}


@dataclass
class Args:
    # General arguments
    seed: int = 0
    save_dir: str | None = 'results/multi1'
    n_processes: int = 24
    resume: bool | None = False
    job_timeout: int = 24 * 60 * 60

    # Core War arguments
    simargs: SimulationArgs = field(default_factory=SimulationArgs)
    timeout: int = 900

    # Opponent arguments
    opponent_dir: str = "../warrior_multi"
    opponent_pattern: str = "*.red"  # Glob pattern for opponent files
    max_opponents: int | None = None  # Limit number of opponents (None = use all)

    # DRQ arguments
    n_rounds: int = 120
    n_iters: int = 100
    log_every: int = 10
    sample_new_percent: float = 0.1
    bc_axes: str = "tsp,mc"
    n_init: int = 8
    n_mutate: int = 1
    fitness_threshold: float = 10.0  # Set high to disable early stopping
    single_cell: bool | None = False

    # Archive accumulation
    carry_archive: bool = True
    reevaluate_carried: bool = True

    # Warmstart from previous run
    #warmstart_dir: str | None = 'results/explicit_16/final_niches'  # Directory containing .red files to seed archive
    warmstart_dir: str | None = None#"results/explicit_18/final_niches"  # Directory containing .red files to seed archive
    warmstart_pattern: str = "*.red"  # Glob pattern for warmstart files

    # Mutation stats logging
    log_mutation_stats_every: int = 100

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
    """DRQ with adaptive explicit mutation for multi-warrior battles."""

    def __init__(self, args: Args):
        self.args = args
        print(args)
        nproc = os.popen("nproc").read().strip()
        nproc_all = os.popen("nproc --all").read().strip()
        print(f"Number of cores: {nproc} / {nproc_all}")

        random.seed(args.seed)
        np.random.seed(args.seed)

        # Initialize adaptive melee mutator with all 9 improvements
        from mutator_multi_opus import ExplicitCoreMutator
        
        warmstart = args.warmstart_dir is not None
        self.mutator = ExplicitCoreMutator(
            environment=simargs_to_environment(args.simargs),
            warmstart_mode=warmstart,
            use_adaptive_weights=True  # Enable adaptive learning
        )
        print(f"Initialized adaptive melee mutator (warmstart={warmstart}, adaptive=True)")

        # Load opponents
        self.init_opps = self._load_opponents()
        print(f"Loaded {len(self.init_opps)} opponents from {args.opponent_dir}")

        self.timestamps = []
        self.all_rounds_map_elites = {i_round: MapElites() for i_round in range(self.args.n_rounds)}

        # Warmstart
        if args.warmstart_dir and not args.resume:
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

    def _warmstart_from_directory(self):
        """Load and evaluate warmstart warriors in parallel."""
        pattern = os.path.join(self.args.warmstart_dir, self.args.warmstart_pattern)
        files = sorted(glob.glob(pattern))
        
        if not files:
            print(f"Warning: No files found matching {pattern}")
            return
        
        print(f"\n{'='*60}")
        print(f"WARMSTART: Loading {len(files)} warriors from {self.args.warmstart_dir}")
        print(f"{'='*60}")
        
        print("Loading warriors...")
        warmstart_warriors = []
        for file in files:
            try:
                warrior_str, warrior = parse_warrior_from_file(self.args.simargs, file)
                ew = ExplicitWarrior(warrior=warrior, code=warrior_str)
                warmstart_warriors.append({'file': os.path.basename(file), 'warrior': warrior, 'code': warrior_str, 'ew': ew})
            except Exception as e:
                print(f"  Warning: Failed to load {os.path.basename(file)}: {e}")
        
        print(f"Loaded {len(warmstart_warriors)} warriors")
        if not warmstart_warriors:
            return
        
        # Step 2: Build evaluation arguments
        opponent_warriors = [opp.warrior for opp in self.init_opps]
        
        print(f"Evaluating {len(warmstart_warriors)} warriors against {len(opponent_warriors)} opponents...")
        with Pool(processes=self.args.n_processes) as pool:
            results = list(tqdm(pool.imap(_evaluate_warmstart_warrior_multi, eval_args), total=len(eval_args), desc="Warmstart"))
        
        # Step 4: Process results and place in archive
        loaded = 0

        for result in results:
            w_idx = result['warrior_idx']
            w = warmstart_warriors[w_idx]
            ew = w['ew']
            outputs = result['outputs']
            
            ew.fitness = outputs['score']
            ew.outputs = outputs
            ew.bc = self.get_bc_features(ew)
            
            if ew.fitness > -np.inf:
                self.all_rounds_map_elites[0].place(ew)
                loaded += 1
        
        # Learn from warmstart archive
        me = self.all_rounds_map_elites[0]
        if len(me.archive) > 0:
            self.mutator.learn_from_archive(me.archive)
        
        print(f"\nWarmstart complete: {loaded} warriors, {len(me.archive)} niches")
        if len(me.archive) > 0:
            best = me.get_best()
            print(f"Best fitness: {best.fitness:.4f}")
        print(f"{'='*60}\n")

        for bc, warrior in me.archive.items():
            code = re.sub(r"```.*", "", warrior.code)
            filename = f"niche_{bc[0]}_{bc[1]}.red"
            filepath = os.path.join(filename)
            
            with open(filepath, "w") as f:
                f.write(code)

    def get_bc_features(self, warrior: ExplicitWarrior) -> tuple:
        if self.args.single_cell:
            return (0, 0)
        if warrior.outputs is None or len(warrior.outputs) == 0:
            return (0, 0)
        
        tsp = warrior.outputs['total_spawned_procs'].item()
        mc = warrior.outputs['memory_coverage'].item()
        unique_opcodes = len({i.opcode for i in warrior.warrior.instructions})
        program_len = len(warrior.warrior.instructions)

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
        """Evaluate warrior in multi-warrior battle."""
        warrior = copy.deepcopy(warrior)

        if warrior.warrior is None:
            warrior.bc, warrior.fitness = None, -np.inf
            return warrior

        all_warriors = [warrior.warrior] + [opp.warrior for opp in self.used_opps]
        
        try:
            outputs = run_multiple_rounds(
                self.args.simargs, all_warriors,
                n_processes=self.args.n_processes,
                timeout=self.args.timeout
            )
        except Exception as e:
            print(f"Evaluation error: {e}")
            warrior.bc, warrior.fitness = None, -np.inf
            return warrior

        if outputs is None:
            warrior.bc, warrior.fitness = None, -np.inf
            return warrior

        warrior.outputs = {k: v.mean(axis=-1)[0] for k, v in outputs.items()}
        warrior.fitness = warrior.outputs['score']
        
        warrior.bc = self.get_bc_features(warrior)

        return warrior

    def process_warrior_with_feedback(self, i_round: int, warrior: ExplicitWarrior, parent_fitness: float = None) -> bool:
        """Evaluate warrior and provide feedback to mutator for adaptive learning."""
        warrior = self.evaluate_warrior(warrior)
        
        # Provide feedback to mutator if we have parent fitness
        if parent_fitness is not None and warrior.fitness is not None and warrior.fitness > -np.inf:
            warrior_hash = self.mutator.get_warrior_hash(warrior.code)
            self.mutator.record_fitness_feedback(warrior_hash, parent_fitness, warrior.fitness)
        
        map_elites = self.all_rounds_map_elites[i_round]
        return map_elites.place(warrior)

    def carry_forward_archive(self, i_round: int):
        """Carry archive from previous round, optionally re-evaluate."""
        if i_round == 0:
            return
        
        prev_me = self.all_rounds_map_elites[i_round - 1]
        current_me = self.all_rounds_map_elites[i_round]
        
        carried = prev_me.get_all_entries()
        if len(carried) == 0:
            return
        
        print(f"Round {i_round}: Carrying {len(carried)} warriors")
        
        for warrior in carried:
            if self.args.reevaluate_carried:
                self.process_warrior_with_feedback(i_round, warrior)
            else:
                current_me.place(warrior)
        
        print(f"Round {i_round}: Archive has {len(current_me.archive)} warriors")

    def init_round(self, i_round: int):
        """Initialize a round."""
        # Carry forward from previous round
        if self.args.carry_archive:
            self.carry_forward_archive(i_round)
        
        current_me = self.all_rounds_map_elites[i_round]
        
        # Generate new warriors if archive is sparse
        n_new = max(0, self.args.n_init - len(current_me.archive))
        
        if n_new > 0:
            print(f"Round {i_round}: Generating {n_new} new warriors")
            new_warriors = asyncio.run(self.mutator.new_warrior_async(n_warriors=1, n_responses=n_new)).flatten()
            
            for w in new_warriors:
                code = getattr(w, 'code', None) or getattr(w, 'llm_response', '')
                ew = ExplicitWarrior(warrior=w.warrior, code=code)
                self.process_warrior_with_feedback(i_round, ew, parent_fitness=None)
        
        print(f"Round {i_round}: Initialized with {len(current_me.archive)} warriors")

    def step(self, i_round: int):
        """One evolution step with adaptive feedback."""
        current_me = self.all_rounds_map_elites[i_round]
        
        # Update mutator with archive patterns
        if len(current_me.archive) > 0 and hasattr(self.mutator, 'learn_from_archive'):
            self.mutator.learn_from_archive(current_me.archive)
        
        if random.random() < self.args.sample_new_percent or len(current_me.archive) == 0:
            new_warriors = asyncio.run(self.mutator.new_warrior_async(n_warriors=1, n_responses=self.args.n_mutate)).flatten()
            
            for w in new_warriors:
                code = getattr(w, 'code', None) or getattr(w, 'llm_response', '')
                ew = ExplicitWarrior(warrior=w.warrior, code=code)
                self.process_warrior_with_feedback(i_round, ew, parent_fitness=None)
        else:
            # Sample parent from archive and mutate
            parent = current_me.sample()
            parent_fitness = parent.fitness  # Track for feedback
            
            parent_warrior = ExplicitWarrior(code=parent.code, warrior=parent.warrior)
            
            # Mutate - mutator will track the hash internally
            offspring = asyncio.run(self.mutator.mutate_warrior_async([parent_warrior], n_responses=self.args.n_mutate)).flatten()
            
            for w in offspring:
                code = getattr(w, 'code', None) or getattr(w, 'llm_response', '')
                ew = ExplicitWarrior(warrior=w.warrior, code=code)
                # Provide parent fitness for feedback
                self.process_warrior_with_feedback(i_round, ew, parent_fitness=parent_fitness)

    def save_mutation_stats(self):
        """Save mutation statistics to file."""
        if self.args.save_dir is None:
            return
        
        stats_path = os.path.join(self.args.save_dir, "mutation_stats.txt")
        with open(stats_path, 'w') as f:
            import sys
            old_stdout = sys.stdout
            sys.stdout = f
            self.mutator.print_mutation_stats()
            sys.stdout = old_stdout

    def run(self):
        this_job_start_time = time.time()
        
        if self.args.resume and os.path.exists(f"{self.args.save_dir}/args.pkl"):
            self.timestamps = util.load_pkl(self.args.save_dir, "timestamps")
            self.all_rounds_map_elites = util.load_pkl(self.args.save_dir, "all_rounds_map_elites")
            self.mutator.all_generations = util.load_pkl(self.args.save_dir, "all_generations")
            print(f"Resumed from {self.args.save_dir}")
            start_abs_iter = self.timestamps[-1]["abs_iter"] + 1
        else:
            start_abs_iter = 0

        pbar = tqdm(range(start_abs_iter, self.args.n_rounds * self.args.n_iters))
        for abs_iter in pbar:
            i_round = abs_iter // self.args.n_iters
            i_iter = abs_iter % self.args.n_iters
            start_time = time.time()
            self.used_opps = self.init_opps

            me = self.all_rounds_map_elites[i_round]
            best_fitness = me.get_best().fitness if len(me.archive) > 0 else -np.inf
            should_skip = best_fitness > self.args.fitness_threshold

            if not should_skip:
                if i_iter == 0:
                    self.init_round(i_round)
                self.step(i_round)

            process = psutil.Process(os.getpid())
            self.timestamps.append(dict(
                abs_iter=abs_iter, i_round=i_round, i_iter=i_iter,
                dt=time.time() - start_time,
                rss=process.memory_info().rss,
                archive_size=len(me.archive)
            ))

            if abs_iter % self.args.log_every == 0:
                self.save()
            
            # Log mutation stats periodically
            if abs_iter % self.args.log_mutation_stats_every == 0:
                self.save_mutation_stats()
            
            if len(me.archive) > 0:
                pbar.set_postfix(best=f"{me.get_best().fitness:.4f}", archive=len(me.archive))

            if (time.time() - this_job_start_time) > self.args.job_timeout:
                break
        
        self.save()
        self.save_mutation_stats()
        self.save_final_summary()
        
        # Print final mutation stats
        print("\n" + "="*60)
        print("FINAL MUTATION STATISTICS")
        print("="*60)
        self.mutator.print_mutation_stats()

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
        
        # Find last round with warriors
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
                'fitness': warrior.fitness,
                'name': warrior.name,
                'file': filename
            })
        
        # Save summary CSV
        import csv
        with open(os.path.join(self.args.save_dir, "final_archive_summary.csv"), "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(['bc_tsp', 'bc_mc', 'fitness', 'name', 'file'])
            for s in sorted(summary, key=lambda x: -x['fitness']):
                writer.writerow([s['bc'][0], s['bc'][1], s['fitness'], s['name'], s['file']])
        
        print(f"\nFinal archive (round {final_round}):")
        print(f"  Coverage: {len(final_me.archive)}/36 niches")
        print(f"  Best fitness: {final_me.get_best().fitness:.4f}")
        print(f"  Saved to: {niche_dir}")


if __name__ == "__main__":
    main = Main(tyro.cli(Args))
    main.run()
