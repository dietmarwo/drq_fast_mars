"""
Explicit Core War Mutator optimized for 1v1 Hill Scoring and Warmstart

Design Philosophy:
=================
Multi-warrior rewards SURVIVAL - papers with many SPL processes excel.
1v1 rewards KILLING - ties give only 1 point vs 3 for wins.

In warmstart, we have already-good warriors. We need REFINEMENT, not exploration.
This means: fewer mutations, smaller changes, preserve what works.

Key Differences from Multi-Warrior Mutator:
==========================================
1. KILL-FOCUSED: Add finishing moves, core clears, kill loops
2. ANTI-TIE: Ensure opponent is fully destroyed, not just damaged
3. FINE-TUNING: Small constant changes (+/-1 to +/-5) for hill climbing  
4. CONSERVATIVE: Usually 1 mutation, preserve proven structure
5. ARCHIVE LEARNING: Extract successful constants from good warriors
6. REDUCED SPL: SPL fans help survival but not killing
7. STEP SIZE FOCUS: Most critical parameter, needs careful tuning

1v1 Scoring: Win=3, Tie=1, Loss=0
A tie is only 33% as valuable as a win - we must actively KILL!
"""

import re
import random
import numpy as np
import hashlib
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Set
from corewar import redcode, Warrior

from util import ExplicitWarrior


class ExplicitCoreMutator:
    """
    Mutator optimized for 1v1 hill scoring and warmstart refinement.
    
    Key features:
    - Fine-grained constant tuning (hill climbing)
    - Archive learning (extract successful constants)
    - Kill-focused mutations (core clear, finishing moves)
    - Anti-tie strategies
    - Conservative mutations for warmstart
    """
    
    CORE_SIZE = 8000
    MAX_RETRIES = 5
    
    # Proven step sizes
    GOOD_STEPS = [
        # Small primes - good for scanners and tight bombing
        3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47,
        53, 59, 61, 67, 71, 73, 79, 83, 89, 97,
        # Large steps for fast core coverage
        2339, 2663, 2671, 3037, 3041, 3049, 3061, 3067, 3079, 3083,
        3089, 3109, 3119, 3121, 3137, 3163, 3167, 3169, 3181, 3187,
        3191, 3203, 3209, 3217, 3221, 3229, 3251, 3253, 3257, 3259,
        3271, 3299, 3301, 3307, 3313, 3319, 3323, 3329, 3331, 3343,
        3347, 3359, 3361, 3371, 3373, 3389, 3391, 3407, 3413, 3433,
        3449, 3457, 3461, 3463, 3467, 3469, 3491, 3499, 3511, 3517,
        3527, 3529, 3533, 3539, 3541, 3547, 3557, 3559, 3571,
        # Popular KOTH constants (proven effective)
        2667, 5334, 1471, 1777, 3044, 4001, 4003, 4007,
        # Core fractions
        2000, 2666, 4000, 1333, 1600, 800, 400,
    ]
    
    # Deltas for fine-tuning (warmstart hill climbing)
    FINE_DELTAS = [-2, -1, 1, 2]
    SMALL_DELTAS = [-5, -3, 3, 5]
    MEDIUM_DELTAS = [-20, -10, 10, 20]
    
    BOOT_DISTANCES = [100, 200, 500, 800, 1000, 1500, 2000, 2500, 3000, 3500, 4000]
    
    # ==================== 1v1 OPTIMIZED TEMPLATES ====================
    # These templates emphasize KILLING over survival
    
    TEMPLATES = {
        # Aggressive stone - fast bombing
        'stone': """
;name Stone_{id}
;author ExplicitMutator1v1
;strategy Aggressive stone bomber
ORG start
start   MOV.I   $bomb, @ptr
        ADD.AB  #{step}, $ptr
        JMP.A   $start
bomb    DAT.F   #0, #0
ptr     DAT.F   #0, #{step}
END
""",

        # Dwarf bomber
        'dwarf': """
;name Dwarf_{id}
;author ExplicitMutator1v1
;strategy Classic DAT bomber
ORG start
bomb    DAT.F   #0, #0
start   ADD.AB  #{step}, $bomb
        MOV.AB  #0, @bomb
        JMP.A   $start
END
""",

        # Paper that also attacks (hybrid)
        'paper_attack': """
;name PaperAttack_{id}
;author ExplicitMutator1v1
;strategy Paper with bombing for 1v1
ORG boot
boot    SPL.A   $paper, $0
        JMP.A   $bomb
paper   SPL.A   $0, $0
        MOV.I   $-1, @pptr
        ADD.AB  #{pstep}, $pptr
        JMP.A   $-2
pptr    DAT.F   #0, #{pstep}
bomb    ADD.AB  #{bstep}, $bptr
        MOV.I   $dat, @bptr
        JMP.A   $bomb
dat     DAT.F   #0, #0
bptr    DAT.F   #0, #{bstep}
END
""",

        # Scanner-bomber - find then destroy
        'scanner': """
;name Scanner_{id}
;author ExplicitMutator1v1
;strategy Scan for enemy then bomb
ORG scan
scan    ADD.AB  #{scanstep}, $sptr
        SEQ.I   @sptr, $zero
        JMP.A   $found
        DJN.A   $scan, $scnt
        JMP.A   $bomb
found   MOV.I   $dat, @sptr
        SUB.AB  #{killstep}, $sptr
        DJN.F   $found, $kcnt
bomb    ADD.AB  #{step}, $bptr
        MOV.I   $dat, @bptr
        JMP.A   $bomb
dat     DAT.F   #0, #0
zero    DAT.F   #0, #0
sptr    DAT.F   #{scanstart}, #{scanstart}
scnt    DAT.F   #40, #40
bptr    DAT.F   #0, #{step}
kcnt    DAT.F   #20, #20
END
""",

        # Quickscanner - fast initial scan
        'quickscanner': """
;name QScan_{id}
;author ExplicitMutator1v1
;strategy Quick scan then focused attack
ORG qscan
qscan   SNE.I   ${q1}, ${q2}
        SEQ.I   ${q3}, ${q4}
        JMP.A   $qhit
        SNE.I   ${q5}, ${q6}
        SEQ.I   ${q7}, ${q8}
        JMP.A   $qhit2
        JMP.A   $bomb
qhit    MOV.I   $dat, @${q1}
        SUB.AB  #{qkill}, ${q1}
        DJN.F   $-2, $qcnt
        JMP.A   $bomb
qhit2   MOV.I   $dat, @${q5}
        SUB.AB  #{qkill}, ${q5}
        DJN.F   $-2, $qcnt
bomb    ADD.AB  #{step}, $ptr
        MOV.I   $dat, @ptr
        JMP.A   $bomb
dat     DAT.F   #0, #0
ptr     DAT.F   #0, #{step}
qcnt    DAT.F   #15, #15
END
""",

        # Core clear - systematic destruction (anti-tie)
        'clear': """
;name Clear_{id}
;author ExplicitMutator1v1
;strategy Systematic core destruction
ORG start
start   MOV.I   $bomb, >ptr
        DJN.F   $start, $count
        ADD.AB  #{step}, $ptr
        JMP.A   $start
bomb    DAT.F   #0, #0
ptr     DAT.F   #-1, #0
count   DAT.F   #{count}, #{count}
END
""",

        # Two-phase: scan then clear (for thorough kills)
        'two_phase': """
;name TwoPhase_{id}
;author ExplicitMutator1v1
;strategy Scan phase then clear phase
ORG scan
scan    ADD.AB  #{scanstep}, $sptr
        SEQ.I   @sptr, $zero
        JMP.A   $found
        DJN.A   $scan, $scnt
        JMP.A   $clear
found   MOV.I   $dat, @sptr
        ADD.AB  #{killstep}, $sptr
        DJN.F   $found, $kcnt
clear   MOV.I   $dat, >cptr
        DJN.F   $clear, $ccnt
        JMP.A   $scan
dat     DAT.F   #0, #0
zero    DAT.F   #0, #0
sptr    DAT.F   #1000, #1000
scnt    DAT.F   #30, #30
cptr    DAT.F   #-1, #0
ccnt    DAT.F   #150, #150
kcnt    DAT.F   #25, #25
END
""",

        # Vampire
        'vampire': """
;name Vampire_{id}
;author ExplicitMutator1v1
;strategy Convert enemy processes
ORG scan
pit     DAT.F   #0, #0
fang    JMP.A   $pit, <-{step}
scan    ADD.AB  #{step}, $ptr
        JMZ.F   $scan, @ptr
        MOV.I   $fang, @ptr
        JMP.A   $scan
ptr     DAT.F   #1000, #{step}
END
""",

        # Silk with bombing (for 1v1)
        'silk': """
;name Silk_{id}
;author ExplicitMutator1v1
;strategy Fast silk with bombing
ORG boot
boot    SPL.A   $silk, $0
        JMP.A   $bomb
silk    SPL.A   $2, $0
        SPL.A   $1, $0
        MOV.I   $-1, @sptr
        ADD.AB  #{sstep}, $sptr
        JMP.A   $-2
sptr    DAT.F   #0, #{sstep}
bomb    ADD.AB  #{bstep}, $bptr
        MOV.I   $dat, @bptr
        JMP.A   $bomb
dat     DAT.F   #0, #0
bptr    DAT.F   #0, #{bstep}
END
""",

        # OneShot scanner
        'oneshot': """
;name OneShot_{id}
;author ExplicitMutator1v1
;strategy One-shot scanner with bombing
ORG start
        SNE.I   ${q1}, ${q2}
        ADD.AB  #10, $ptr
attack  MOV.I   $bomb, @ptr
        ADD.AB  #{step}, $ptr
        DJN.F   $attack, $count
        JMP.A   $attack
bomb    DAT.F   #0, #0
ptr     DAT.F   #0, #{start}
count   DAT.F   #500, #500
END
""",

        # Imp gate stone (defense + attack)
        'gate_stone': """
;name GateStone_{id}
;author ExplicitMutator1v1
;strategy Stone with imp gate defense
ORG start
gate    SPL.A   $0, <gate-4
start   MOV.I   $bomb, @ptr
        ADD.AB  #{step}, $ptr
        DJN.F   $start, #0
bomb    DAT.F   #0, #0
ptr     DAT.F   #0, #{step}
END
""",
    }
    
    OPCODES = ['DAT', 'MOV', 'ADD', 'SUB', 'MUL', 'DIV', 'MOD',
               'JMP', 'JMZ', 'JMN', 'DJN', 'CMP', 'SEQ', 'SNE',
               'SLT', 'SPL', 'NOP']
    
    MODIFIERS = ['A', 'B', 'AB', 'BA', 'F', 'X', 'I']
    ADDR_MODES = ['#', '$', '*', '@', '{', '<', '}', '>']
    
    # Warrior type signatures
    TYPE_SIGNATURES = {
        'paper': ['SPL.A $0', 'SPL.A   $0'],
        'stone': ['MOV.I', '@ptr', '@0'],
        'scanner': ['SNE', 'SEQ', 'JMZ', 'sptr'],
        'bomber': ['ADD.AB', 'MOV.AB', 'bomb'],
        'clear': ['DJN', '>ptr', '>cptr'],
        'vampire': ['fang', 'pit', 'JMZ'],
        'hybrid': ['SPL', 'JMP.A $bomb'],
    }
    
    def __init__(self, environment=None, warmstart_mode=True):
        """
        Args:
            environment: Redcode parsing environment
            warmstart_mode: If True, use conservative mutations for refinement
        """
        self.environment = environment or {}
        self.all_generations = []
        self.warmstart_mode = warmstart_mode
        
        # Learned successful constants from archive
        self.learned_steps: Set[int] = set(self.GOOD_STEPS)
        self.learned_scan_positions: Set[int] = set()
        self.learned_loop_counts: Set[int] = set()
        
        # Mutation weights optimized for 1v1
        if warmstart_mode:
            # CONSERVATIVE: Focus on fine-tuning, preserve structure
            self.mutation_weights = {
                # Fine-tuning (HIGH WEIGHT - most important for warmstart)
                'fine_tune_step': 0.25,        # Critical: step sizes
                'fine_tune_constant': 0.15,    # Other constants
                'tune_loop_count': 0.08,       # Attack duration
                'tune_scan_position': 0.06,    # Scanner effectiveness
                
                # Attack enhancement (MEDIUM WEIGHT)
                'add_core_clear': 0.08,        # Anti-tie finishing move
                'increase_attack_range': 0.06, # More thorough destruction
                'add_kill_loop': 0.05,         # Ensure kills after scan hit
                
                # Conservative structure changes (LOW WEIGHT)
                'change_modifier': 0.05,
                'change_addr_mode': 0.04,
                'swap_adjacent': 0.03,         # Safer than random swap
                
                # Defensive (LOW WEIGHT)
                'add_imp_gate': 0.03,
                
                # Exploration (VERY LOW - warmstart = exploit)
                'change_opcode': 0.02,
                'good_step_replace': 0.04,     # Try known good values
                'archive_step': 0.04,          # Use learned values
                
                # Rarely used
                'insert_instruction': 0.01,
                'delete_instruction': 0.01,
            }
        else:
            # EXPLORATION: More diverse mutations
            self.mutation_weights = {
                'fine_tune_step': 0.12,
                'fine_tune_constant': 0.10,
                'good_step_replace': 0.10,
                'change_opcode': 0.06,
                'change_modifier': 0.06,
                'change_addr_mode': 0.08,
                'insert_instruction': 0.04,
                'delete_instruction': 0.03,
                'swap_adjacent': 0.03,
                'add_core_clear': 0.06,
                'add_kill_loop': 0.06,
                'increase_attack_range': 0.05,
                'tune_loop_count': 0.05,
                'add_imp_gate': 0.04,
                'inject_scanner': 0.04,
                'archive_step': 0.04,
                'tune_scan_position': 0.04,
            }
        
        self.mutation_stats = {k: {'tried': 0, 'improved': 0} for k in self.mutation_weights}
    
    # ==================== PARSING & VALIDATION ====================
    
    def _parse_warrior(self, code: str) -> Tuple[Optional[Warrior], Optional[str]]:
        """Parse Redcode string into Warrior object."""
        try:
            code_clean = re.sub(r"```.*", "", code)
            warrior = redcode.parse(code_clean.split("\n"), self.environment)
            return warrior, None
        except Exception as e:
            return None, str(e)
    
    def _validate_warrior(self, code: str) -> Tuple[bool, Optional[str]]:
        """Check if warrior code is valid."""
        warrior, error = self._parse_warrior(code)
        if warrior is None:
            return False, error
        if len(warrior.instructions) < 1:
            return False, "No instructions"
        if len(warrior.instructions) > 100:
            return False, "Too many instructions"
        return True, None
    
    def _to_explicit_warrior(self, code: str, parent_id: str = None) -> ExplicitWarrior:
        """Convert code string to ExplicitWarrior object."""
        warrior, error = self._parse_warrior(code)
        exp_warrior = ExplicitWarrior(
            code=code,
            warrior=warrior,
            error=error,
            id=hashlib.sha256(code.encode()).hexdigest(),
            parent_id=parent_id
        )
        return exp_warrior
    
    def _detect_warrior_type(self, code: str) -> str:
        """Detect warrior strategy type."""
        code_upper = code.upper()
        scores = {}
        for wtype, signatures in self.TYPE_SIGNATURES.items():
            score = sum(1 for sig in signatures if sig.upper() in code_upper)
            scores[wtype] = score
        
        if max(scores.values()) == 0:
            return 'unknown'
        return max(scores, key=scores.get)
    
    # ==================== ARCHIVE LEARNING ====================
    
    def learn_from_archive(self, archive: Dict):
        """
        Extract successful constants from archive warriors.
        Call this periodically with the current MapElites archive.
        """
        for bc, warrior in archive.items():
            if not hasattr(warrior, 'code') or not warrior.fitness or warrior.fitness <= 0:
                continue
            
            code = warrior.code
            
            # Extract step sizes from ADD instructions
            for match in re.finditer(r'ADD\.\w+\s+#(\d+)', code, re.IGNORECASE):
                step = int(match.group(1))
                if 10 < step < 7000:
                    self.learned_steps.add(step)
            
            # Extract loop counts
            for match in re.finditer(r'cnt\s+DAT\.F\s+#(\d+)', code, re.IGNORECASE):
                count = int(match.group(1))
                if 5 < count < 1000:
                    self.learned_loop_counts.add(count)
            
            # Extract scan positions
            for match in re.finditer(r'sptr\s+DAT\.F\s+#(\d+)', code, re.IGNORECASE):
                pos = int(match.group(1))
                if 100 < pos < 7000:
                    self.learned_scan_positions.add(pos)
    
    def _get_learned_step(self) -> int:
        """Get a step from learned values or good defaults."""
        if self.learned_steps and random.random() < 0.6:
            return random.choice(list(self.learned_steps))
        return random.choice(self.GOOD_STEPS)
    
    # ==================== NEW WARRIOR GENERATION ====================
    
    def _fill_template(self, template: str) -> str:
        """Fill template with random parameters."""
        unique_id = random.randint(1000, 9999)
        
        params = {
            'id': str(unique_id),
            'step': self._get_learned_step(),
            'scanstep': random.choice([s for s in self.GOOD_STEPS if s > 50]),
            'killstep': random.choice([3, 4, 5, 7, 11, 13]),
            'pstep': random.choice([s for s in self.GOOD_STEPS if s > 2000]),
            'bstep': random.choice([s for s in self.GOOD_STEPS if s > 2000]),
            'sstep': random.choice([s for s in self.GOOD_STEPS if s > 2000]),
            'qkill': random.choice([4, 5, 7, 11]),
            'count': random.randint(100, 400),
            'scanstart': random.randint(500, 2000),
            'start': random.choice(self.GOOD_STEPS),
        }
        
        # Quickscan positions (spread across core)
        base = random.randint(100, 2000)
        gap = random.choice([500, 800, 1000, 1200, 1500])
        for i in range(1, 9):
            params[f'q{i}'] = str((base + (i-1) * gap) % self.CORE_SIZE)
        
        code = template
        for key, value in params.items():
            code = code.replace('{' + key + '}', str(value))
        
        return code
    
    def _generate_new_warrior(self) -> str:
        """Generate a new warrior optimized for 1v1."""
        # Weight toward attack-focused strategies
        weights = {
            'stone': 0.18,
            'scanner': 0.15,
            'quickscanner': 0.12,
            'two_phase': 0.10,
            'paper_attack': 0.10,
            'silk': 0.08,
            'dwarf': 0.08,
            'clear': 0.06,
            'gate_stone': 0.05,
            'oneshot': 0.04,
            'vampire': 0.04,
        }
        
        template_name = random.choices(
            list(weights.keys()),
            weights=list(weights.values())
        )[0]
        
        template = self.TEMPLATES[template_name]
        return self._fill_template(template)
    
    async def new_warrior_async(self, n_warriors: int = 1, n_responses: int = 1) -> np.ndarray:
        """Generate new warriors."""
        exp_warriors = [[None for _ in range(n_responses)] for _ in range(n_warriors)]
        
        for i in range(n_warriors):
            for j in range(n_responses):
                for attempt in range(self.MAX_RETRIES):
                    code = self._generate_new_warrior()
                    valid, error = self._validate_warrior(code)
                    if valid:
                        break
                
                exp_warriors[i][j] = self._to_explicit_warrior(code)
        
        self.all_generations.append(("new_warrior", exp_warriors))
        return np.array(exp_warriors, dtype=object)
    
    # ==================== MUTATIONS ====================
    
    def _mutate_code(self, code: str) -> str:
        """Apply a weighted random mutation."""
        mutations = list(self.mutation_weights.keys())
        weights = list(self.mutation_weights.values())
        weights = np.array(weights) / sum(weights)
        mutation_name = np.random.choice(mutations, p=weights)
        
        self.mutation_stats[mutation_name]['tried'] += 1
        
        mutation_fn = getattr(self, f'_mut_{mutation_name}', None)
        if mutation_fn is None:
            return code
        
        return mutation_fn(code)
    
    # ----- FINE-TUNING MUTATIONS (warmstart focus) -----
    
    def _mut_fine_tune_step(self, code: str) -> str:
        """Fine-tune step size by very small delta (hill climbing)."""
        pattern = r'(ADD\.\w+\s+#)(\d+)'
        matches = list(re.finditer(pattern, code, re.IGNORECASE))
        if not matches:
            return code
        
        match = random.choice(matches)
        old_val = int(match.group(2))
        
        # Very small delta for precise tuning
        delta = random.choice(self.FINE_DELTAS)
        new_val = (old_val + delta) % self.CORE_SIZE
        
        # Ensure positive and reasonable
        if new_val < 3:
            new_val = 3
        
        return code[:match.start(2)] + str(new_val) + code[match.end(2):]
    
    def _mut_fine_tune_constant(self, code: str) -> str:
        """Fine-tune any numeric constant by small delta."""
        pattern = r'([#$@<>*{}\s])(\d+)'
        matches = list(re.finditer(pattern, code))
        if not matches:
            return code
        
        match = random.choice(matches)
        old_val = int(match.group(2))
        
        delta = random.choice(self.SMALL_DELTAS)
        new_val = (old_val + delta) % self.CORE_SIZE
        
        return code[:match.start(2)] + str(new_val) + code[match.end(2):]
    
    def _mut_tune_loop_count(self, code: str) -> str:
        """Tune loop counter for attack duration."""
        pattern = r'(cnt\s+DAT\.F\s+#)(\d+)(,\s*#)(\d+)'
        matches = list(re.finditer(pattern, code, re.IGNORECASE))
        if not matches:
            # Try simpler pattern
            pattern = r'(count\s+DAT\.F\s+#)(\d+)'
            matches = list(re.finditer(pattern, code, re.IGNORECASE))
            if not matches:
                return code
            
            match = random.choice(matches)
            old_val = int(match.group(2))
            delta = random.choice([-20, -10, 10, 20, 30, 50])
            new_val = max(10, min(1000, old_val + delta))
            return code[:match.start(2)] + str(new_val) + code[match.end(2):]
        
        match = random.choice(matches)
        old_val = int(match.group(2))
        delta = random.choice([-20, -10, 10, 20, 30, 50])
        new_val = max(10, min(1000, old_val + delta))
        
        # Update both A and B fields
        code = code[:match.start(2)] + str(new_val) + code[match.end(2):match.start(4)] + str(new_val) + code[match.end(4):]
        return code
    
    def _mut_tune_scan_position(self, code: str) -> str:
        """Tune scanner starting position."""
        pattern = r'(sptr\s+DAT\.F\s+#)(\d+)'
        match = re.search(pattern, code, re.IGNORECASE)
        if not match:
            return code
        
        old_val = int(match.group(2))
        delta = random.choice([-200, -100, -50, 50, 100, 200])
        new_val = (old_val + delta) % self.CORE_SIZE
        
        return code[:match.start(2)] + str(new_val) + code[match.end(2):]
    
    def _mut_good_step_replace(self, code: str) -> str:
        """Replace step with a known good value."""
        pattern = r'(ADD\.\w+\s+#)(\d+)'
        matches = list(re.finditer(pattern, code, re.IGNORECASE))
        if not matches:
            return code
        
        match = random.choice(matches)
        new_val = random.choice(self.GOOD_STEPS)
        
        return code[:match.start(2)] + str(new_val) + code[match.end(2):]
    
    def _mut_archive_step(self, code: str) -> str:
        """Replace step with a learned successful value."""
        if not self.learned_steps:
            return self._mut_good_step_replace(code)
        
        pattern = r'(ADD\.\w+\s+#)(\d+)'
        matches = list(re.finditer(pattern, code, re.IGNORECASE))
        if not matches:
            return code
        
        match = random.choice(matches)
        new_val = random.choice(list(self.learned_steps))
        
        return code[:match.start(2)] + str(new_val) + code[match.end(2):]
    
    # ----- ATTACK ENHANCEMENT MUTATIONS -----
    
    def _mut_add_core_clear(self, code: str) -> str:
        """Add core clear as finishing move (anti-tie)."""
        if 'clear' in code.lower() or 'ccnt' in code.lower() or '>cptr' in code.lower():
            return code  # Already has clear
        
        clear = """
        ; Core clear finishing move
clr     MOV.I   $cdat, >cptr
        DJN.F   $clr, $ccnt
        JMP.A   $start
cdat    DAT.F   #0, #0
cptr    DAT.F   #-1, #0
ccnt    DAT.F   #150, #150
"""
        code = re.sub(r'(\s*END\s*)', clear + r'\1', code, flags=re.IGNORECASE)
        return code
    
    def _mut_add_kill_loop(self, code: str) -> str:
        """Add kill loop after scanner hit."""
        if 'kill' in code.lower() or 'kcnt' in code.lower():
            return code
        
        # Look for scanner hit location
        if 'found' not in code.lower() and 'qhit' not in code.lower():
            return code  # Not a scanner
        
        kill = """
        ; Kill loop
        SUB.AB  #5, $kptr
        MOV.I   $kdat, @kptr
        DJN.F   $-2, $kcnt
kdat    DAT.F   #0, #0
kptr    DAT.F   #0, #0
kcnt    DAT.F   #20, #20
"""
        code = re.sub(r'(\s*END\s*)', kill + r'\1', code, flags=re.IGNORECASE)
        return code
    
    def _mut_increase_attack_range(self, code: str) -> str:
        """Increase attack range by boosting loop counts."""
        # Find all loop counts and increase one
        pattern = r'(#)(\d{2,3})(,\s*#)(\d{2,3})'
        matches = list(re.finditer(pattern, code))
        if not matches:
            return code
        
        match = random.choice(matches)
        old_val = int(match.group(2))
        increase = random.randint(20, 50)
        new_val = min(999, old_val + increase)
        
        code = code[:match.start(2)] + str(new_val) + code[match.end(2):match.start(4)] + str(new_val) + code[match.end(4):]
        return code
    
    # ----- DEFENSIVE MUTATIONS -----
    
    def _mut_add_imp_gate(self, code: str) -> str:
        """Add imp gate for defense."""
        if 'gate' in code.lower() or '<gate' in code.lower():
            return code
        
        gate = """gate    SPL.A   $0, <gate-4
"""
        code = re.sub(r'(ORG\s+\w+\s*\n)', r'\1' + gate, code, flags=re.IGNORECASE)
        return code
    
    def _mut_inject_scanner(self, code: str) -> str:
        """Add quick scanner to non-scanner warriors."""
        wtype = self._detect_warrior_type(code)
        if wtype == 'scanner':
            return code
        
        q1 = random.randint(800, 2500)
        q2 = q1 + random.randint(300, 800)
        
        scanner = f"""        ; Quick scan
        SNE.I   ${q1}, ${q2}
        JMP.A   $qhit
        JMP.A   $main
qhit    MOV.I   $qdat, @${q1}
qdat    DAT.F   #0, #0
main    ; continue
"""
        code = re.sub(r'(ORG\s+\w+\s*\n)', r'\1' + scanner, code, flags=re.IGNORECASE)
        return code
    
    # ----- STANDARD MUTATIONS -----
    
    def _mut_change_opcode(self, code: str) -> str:
        """Change an opcode."""
        lines = code.split('\n')
        instr_lines = [(i, l) for i, l in enumerate(lines)
                       if l.strip() and not l.strip().startswith(';')
                       and not l.strip().upper().startswith(('ORG', 'END', 'EQU'))]
        
        if not instr_lines:
            return code
        
        idx, line = random.choice(instr_lines)
        
        for op in self.OPCODES:
            pattern = rf'\b{op}\b'
            if re.search(pattern, line, re.IGNORECASE):
                new_op = random.choice(self.OPCODES)
                lines[idx] = re.sub(pattern, new_op, line, count=1, flags=re.IGNORECASE)
                break
        
        return '\n'.join(lines)
    
    def _mut_change_modifier(self, code: str) -> str:
        """Change instruction modifier."""
        pattern = r'\.([ABABFXI]{1,2})\b'
        matches = list(re.finditer(pattern, code, re.IGNORECASE))
        if not matches:
            return code
        
        match = random.choice(matches)
        new_mod = random.choice(self.MODIFIERS)
        
        return code[:match.start(1)] + new_mod + code[match.end(1):]
    
    def _mut_change_addr_mode(self, code: str) -> str:
        """Change addressing mode."""
        pattern = r'([#$*@{}<>])(-?\d+)'
        matches = list(re.finditer(pattern, code))
        if not matches:
            return code
        
        match = random.choice(matches)
        new_mode = random.choice(self.ADDR_MODES)
        
        return code[:match.start(1)] + new_mode + code[match.end(1):]
    
    def _mut_swap_adjacent(self, code: str) -> str:
        """Swap two adjacent instructions (safer than random swap)."""
        lines = code.split('\n')
        
        instr_indices = [i for i, l in enumerate(lines)
                        if l.strip() and not l.strip().startswith(';')
                        and not l.strip().upper().startswith(('ORG', 'END', 'EQU'))]
        
        if len(instr_indices) < 2:
            return code
        
        # Pick random adjacent pair
        idx = random.randint(0, len(instr_indices) - 2)
        idx1, idx2 = instr_indices[idx], instr_indices[idx + 1]
        lines[idx1], lines[idx2] = lines[idx2], lines[idx1]
        
        return '\n'.join(lines)
    
    def _mut_insert_instruction(self, code: str) -> str:
        """Insert a useful instruction."""
        lines = code.split('\n')
        
        instr_indices = [i for i, l in enumerate(lines)
                        if l.strip() and not l.strip().startswith(';')
                        and not l.strip().upper().startswith(('ORG', 'END', 'EQU'))]
        
        if not instr_indices:
            return code
        
        insert_idx = random.choice(instr_indices)
        
        # Bias toward useful instructions
        useful_instrs = [
            f"MOV.I   $dat, @ptr",
            f"ADD.AB  #{random.choice(self.GOOD_STEPS[:20])}, $ptr",
            f"DJN.F   $-1, $cnt",
        ]
        
        new_line = "        " + random.choice(useful_instrs)
        lines.insert(insert_idx, new_line)
        
        return '\n'.join(lines)
    
    def _mut_delete_instruction(self, code: str) -> str:
        """Delete a random instruction (careful - preserve structure)."""
        lines = code.split('\n')
        
        # Find deletable lines (not labels, not critical)
        deletable = []
        for i, l in enumerate(lines):
            stripped = l.strip()
            if not stripped or stripped.startswith(';'):
                continue
            if stripped.upper().startswith(('ORG', 'END', 'EQU')):
                continue
            # Don't delete lines with labels
            if re.match(r'^\w+\s+', stripped):
                continue
            deletable.append(i)
        
        if len(deletable) <= 1:
            return code
        
        del_idx = random.choice(deletable)
        del lines[del_idx]
        
        return '\n'.join(lines)
    
    # ==================== MUTATE WARRIOR ====================
    
    async def mutate_warrior_async(self, exp_warriors: List[ExplicitWarrior], n_responses: int = 1) -> np.ndarray:
        """
        Mutate warriors with conservative approach for warmstart.
        
        Key difference from multi-warrior mutator:
        - Usually only 1 mutation (warmstart = refine, not explore)
        - Bias toward fine-tuning mutations
        """
        result = [[None for _ in range(n_responses)] for _ in range(len(exp_warriors))]
        
        for i, parent in enumerate(exp_warriors):
            for j in range(n_responses):
                for attempt in range(self.MAX_RETRIES):
                    code = parent.code
                    
                    # In warmstart mode: usually just 1 mutation
                    # This preserves the good structure we're starting from
                    if self.warmstart_mode:
                        n_mutations = random.choices([1, 2], weights=[0.8, 0.2])[0]
                    else:
                        n_mutations = random.randint(1, 3)
                    
                    for _ in range(n_mutations):
                        code = self._mutate_code(code)
                    
                    valid, error = self._validate_warrior(code)
                    if valid:
                        break
                
                # Update name
                name_match = re.search(r';name\s+(.+)', code)
                if name_match:
                    old_name = name_match.group(1).strip()
                    if re.search(r'_v(\d+)$', old_name):
                        new_name = re.sub(r'_v(\d+)$', lambda m: f'_v{int(m.group(1))+1}', old_name)
                    else:
                        new_name = f"{old_name}_v{random.randint(2,99)}"
                    code = code.replace(f";name {old_name}", f";name {new_name}")
                
                exp_warrior = self._to_explicit_warrior(code, parent_id=parent.id)
                result[i][j] = exp_warrior
        
        self.all_generations.append(("mutate_warrior", result))
        return np.array(result, dtype=object)
