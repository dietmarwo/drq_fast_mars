import os, sys
sys.path.append(os.path.abspath("corewar"))


import redcode
from core import Core
from mars import MARS

from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import numpy as np

import tyro
from dataclasses import dataclass
import random
from functools import partial

from multiprocessing import Pool

@dataclass
class Args:
    warriors: list[str] # List of warrior redcode filenames
    rounds: int = 1 # Rounds to play
    paused: bool | None = False # Start each round paused
    size: int = 8000 # The core size
    cycles: int = 80000 # Cycles until tie
    processes: int = 8000 # Max processes
    length: int = 100 # Max warrior length
    distance: int = 100 # Minimum warrior distance

    gpt_warrior: str | None = None
    save_dir: str | None = None

def run_single_round(args, warriors, seed):
    random.seed(seed)
    simulation = MARS(warriors=warriors, minimum_separation=args.distance, max_processes=args.processes, randomize=True)
    scores = np.zeros(len(warriors))
    for t in range(args.cycles):
        simulation.step()

        alive_flags = np.array([len(warrior.task_queue)>0 for warrior in simulation.warriors]).astype(int)
        n_alive = sum(alive_flags)
        if n_alive==0:
            break
        scores += (alive_flags * (1./n_alive))
    return scores

def run_multiple_rounds(args, warriors, n_processes=16):
    run_single_round_fn = partial(run_single_round, args, warriors)
    seeds = list(range(args.rounds))
    with Pool(processes=n_processes) as pool:
        scores = pool.map(run_single_round_fn, seeds)
    scores = np.array(scores)
    return scores


import pickle
def save_pkl(save_dir, name, item):
    if save_dir is not None:
        os.makedirs(f"{save_dir}/", exist_ok=True)
        with open(f"{save_dir}/{name}.pkl", "wb") as f:
            pickle.dump(item, f)


def load_pkl(load_dir, name):
    if load_dir is not None:
        with open(f"{load_dir}/{name}.pkl", "rb") as f:
            return pickle.load(f)
    else:
        return None

def main(args):
    environment = {'ROUNDS': args.rounds,
                'CORESIZE': args.size,
                'CYCLES': args.cycles,
                'MAXPROCESSES': args.processes,
                'MAXLENGTH': args.length,
                'MINDISTANCE': args.distance}

    warriors = []
    for file in args.warriors:
        with open(file, encoding="utf-8", errors="replace") as f:
            a = f.readlines()
        warrior = redcode.parse(a, environment)
        warriors.append(warrior)
    
    if args.gpt_warrior is not None:
        gpt_warrior = load_pkl("warriors", args.gpt_warrior)
        print(gpt_warrior)
        warriors.append(gpt_warrior)
    
    scores = run_multiple_rounds(args, warriors, n_processes=16)
    print(scores.shape, scores.mean())

    if args.save_dir is not None:
        os.makedirs(args.save_dir, exist_ok=True)
        save_pkl(args.save_dir, "scores", scores)

if __name__ == "__main__":
    args = tyro.cli(Args)
    # args = Args(warriors=[f"warriors/{file}" for file in os.listdir("warriors")])
    # args = Args(warriors=["warriors/validate.red", "warriors/validate.red", "warriors/validate.red", "warriors/validate.red"])

    main(args)

