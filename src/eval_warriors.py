
import re
import random
import os
import numpy as np
import glob

from dataclasses import dataclass, field
import tyro
import asyncio
from tqdm.auto import tqdm
from collections import defaultdict

from llm_corewar import CorewarGPT, GPTWarrior

from corewar_util import SimulationArgs, simargs_to_environment, parse_warrior_from_file, run_multiple_rounds
from corewar import MARS, Warrior
import util

@dataclass
class Args:
    # General arguments
    seed: int = 0
    save_dir: str | None = None
    n_processes: int = 24

    # Core War arguments
    simargs: SimulationArgs = field(default_factory=SimulationArgs) # Simulation arguments
    timeout: int = 900 # timeout for each simulation in seconds

    warrior_path: str | None = None
    opponents_path_glob: str | None = None

def main(args: Args):
    print(args)
    nproc = os.popen("nproc").read().strip()
    nproc_all = os.popen("nproc --all").read().strip()
    print(f"Number of cores: {nproc} / {nproc_all}")

    random.seed(args.seed)
    np.random.seed(args.seed)

    _, warrior = parse_warrior_from_file(args.simargs, args.warrior_path)
    results = {}

    files = sorted(glob.glob(args.opponents_path_glob))
    for i, file in enumerate(tqdm(files)):
        _, warrior2 = parse_warrior_from_file(args.simargs, file)
        outputs = run_multiple_rounds(args.simargs, [warrior, warrior2], n_processes=args.n_processes, timeout=args.timeout)
        results[(args.warrior_path, file)] = outputs
        if args.save_dir is not None and i % 10 == 0:
            util.save_pkl(args.save_dir, "results", results)
    if args.save_dir is not None:
        util.save_pkl(args.save_dir, "results", results)

if __name__ == "__main__":
    main(tyro.cli(Args))
    
