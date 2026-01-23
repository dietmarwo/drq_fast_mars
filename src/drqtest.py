from corewar_util import SimulationArgs, parse_warrior_from_file, run_multiple_rounds
from corewar.redcode import *
    
def main():

    simargs = SimulationArgs()
    simargs.rounds = 480
    
    _, warrior1 = parse_warrior_from_file(simargs, "./human_warriors/imp.red") 
    _, warrior2 = parse_warrior_from_file(simargs, "./human_warriors/dwarf.red") 
    _, warrior3 = parse_warrior_from_file(simargs, "./human_warriors/burp.red") 
    _, warrior4 = parse_warrior_from_file(simargs, "./human_warriors/mice.red") 
    _, warrior5 = parse_warrior_from_file(simargs, "./human_warriors/rato.red") 
    warriors = [warrior1, warrior2, warrior3, warrior4, warrior5] 
    # timeout=None uses optimized parallelization code. 
    battle_results = run_multiple_rounds(simargs, warriors, n_processes=24, timeout=None)
    print(battle_results['score'].mean(axis=-1))

if __name__ == '__main__':
    main()
