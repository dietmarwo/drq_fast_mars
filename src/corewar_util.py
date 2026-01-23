from dataclasses import dataclass
import random
import numpy as np
from functools import partial
from tqdm import tqdm
from multiprocessing import Pool

from corewar import MARS, Core, redcode

@dataclass
class SimulationArgs:
    rounds: int = 24 # Rounds to play
    size: int = 8000 # The core size
    cycles: int = 80000 # Cycles until tie
    processes: int = 8000 # Max processes
    length: int = 100 # Max warrior length
    distance: int = 100 # Minimum warrior distance

def simargs_to_environment(args):
    return dict(ROUNDS=args.rounds, CORESIZE=args.size, CYCLES=args.cycles,
                MAXPROCESSES=args.processes, MAXLENGTH=args.length, MINDISTANCE=args.distance)

def run_single_round(simargs, warriors, seed, pbar=False):
    random.seed(seed)
    simulation = MARS(warriors=warriors, minimum_separation=simargs.distance, max_processes=simargs.processes, \
                      randomize=True, debug=False, track_coverage=True)
    score = np.zeros(len(warriors), dtype=float)
    alive_score = np.zeros(len(warriors), dtype=float)

    # Use numpy arrays directly for performance
    prev_nprocs = simulation._tq_sizes.copy()
    total_spawned_procs = np.zeros(len(simulation.warriors), dtype=int)

    for t in tqdm(range(simargs.cycles), disable=not pbar):
        simulation.step()

        # Direct access to numpy array - no Python loop
        nprocs = simulation._tq_sizes

        alive_flags = (nprocs > 0).astype(int)
        n_alive = alive_flags.sum()
        if n_alive == 0:
            break
        score += (alive_flags * (1./n_alive)) / simargs.cycles
        alive_score += alive_flags / simargs.cycles

        total_spawned_procs = total_spawned_procs + np.maximum(0, nprocs - prev_nprocs)
        prev_nprocs = nprocs.copy()

    # Get coverage directly from numpy array - no sync needed
    memory_coverage = simulation.get_coverage_sums()
    score = score * len(warriors)
    outputs = dict(score=score, alive_score=alive_score, total_spawned_procs=total_spawned_procs, memory_coverage=memory_coverage)
    return outputs

def run_multiple_rounds(simargs, warriors, n_processes=1, timeout=900):
    if timeout is None:
        return run_multiple_rounds_fast(simargs, warriors, n_processes) # slightly faster
    try:
        run_single_round_fn = partial(run_single_round, simargs, warriors)
        seeds = list(range(simargs.rounds))
        # print("Launching pool")
        with Pool(processes=n_processes) as pool:
            # outputs = pool.map(run_single_round_fn, seeds)
            result = pool.map_async(run_single_round_fn, seeds)
            # print("Blocking and waiting for results")
            outputs = result.get(timeout=timeout)  # Timeout in seconds
        outputs = {k: np.stack([o[k] for o in outputs], axis=-1) for k in outputs[0].keys()}
        # print("Got results!")
        return outputs # shape: (len(warriors), simargs.rounds)
    except Exception as e:
        print(e)
        return None
    

def run_multiple_rounds_fast(simargs, warriors, n_processes=1):
    # slightly faster, but no timeout
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
            for p in proc: p.start()
            for p in proc: p.join()   
            outs = [outputs[seed] for seed in range(simargs.rounds)]         
            return {k: np.stack([o[k] for o in outs], axis=-1) for k in outs[0].keys()}

        def loop():    
            while(True):
                with mutex:
                    if seed.value >= simargs.rounds:
                        return
                    seed_val = seed.value
                    seed.value += 1    
                outputs[seed_val] = run_single_round_fn(seed_val)
        return run() # shape: (len(warriors), simargs.rounds)
    
    except Exception as e:
        print(e)
        return None

def parse_warrior_from_file(simargs, file):
    environment = simargs_to_environment(simargs)
    with open(file, encoding="latin1") as f:
        warrior_str = f.read()
    warrior = redcode.parse(warrior_str.split("\n"), environment)
    return warrior_str, warrior

