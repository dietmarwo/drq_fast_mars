"""
Improved Explicit Core War Mutator v2

Improvements over v1:
- More strategy templates (scanner, vampire, silk, clear, quickscan)
- Crossover operator for combining warriors
- Component-based generation (boot + attack + decoy)
- Warrior type detection for smarter mutations
- Validation with retry logic
- Boot patterns and decoy injection
- Adaptive mutation based on warrior structure
"""

import re
import random
import numpy as np
import hashlib
from dataclasses import dataclass
from typing import List, Tuple, Optional
from corewar import redcode, Warrior

from util import ExplicitWarrior


class ExplicitCoreMutator:
    """
    Improved explicit rule-based mutator for Core War warriors.
    """
    
    CORE_SIZE = 8000
    MAX_RETRIES = 5  # Validation retries
    
    # Proven effective step sizes (primes, co-primes with 8000)
    GOOD_STEPS = [
        # Small primes for tight bombing
        3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47,
        # Medium coverage
        53, 59, 61, 67, 71, 73, 79, 83, 89, 97,
        # Large steps for fast coverage
        2339, 2663, 2671, 3037, 3041, 3049, 3061, 3067, 3079, 3083,
        3089, 3109, 3119, 3121, 3137, 3163, 3167, 3169, 3181, 3187,
        3191, 3203, 3209, 3217, 3221, 3229, 3251, 3253, 3257, 3259,
        3271, 3299, 3301, 3307, 3313, 3319, 3323, 3329, 3331, 3343,
        3347, 3359, 3361, 3371, 3373, 3389, 3391, 3407, 3413, 3433,
        3449, 3457, 3461, 3463, 3467, 3469, 3491, 3499, 3511, 3517,
        3527, 3529, 3533, 3539, 3541, 3547, 3557, 3559, 3571,
        # Popular constants from KOTH
        2667, 5334, 1471, 1777, 3044, 4001, 4003, 4007,
    ]
    
    # Boot distances
    BOOT_DISTANCES = [100, 200, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000]
    
    # Scanner time delays
    SCAN_DELAYS = [4, 5, 6, 7, 8, 10, 12, 15]
    
    # ==================== TEMPLATES ====================
    
    TEMPLATES = {
        # Basic strategies
        'imp': """
;name Imp_{id}
;author ExplicitMutator
;strategy Simple imp - survives by copying itself forward
ORG start
start   MOV.I   $0, $1
END
""",

        'imp_spiral': """
;name ImpSpiral_{id}
;author ExplicitMutator
;strategy Imp spiral - multiple imp threads
ORG start
start   SPL.A   $1, $0
        MOV.I   $-1, $1
END
""",

        'dwarf': """
;name Dwarf_{id}
;author ExplicitMutator
;strategy Classic bomber - throws DAT bombs
ORG start
bomb    DAT.F   #0, #0
start   ADD.AB  #{step}, $bomb
        MOV.AB  #0, @bomb
        JMP.A   $start
END
""",

        'stone': """
;name Stone_{id}
;author ExplicitMutator
;strategy Stone - MOV.I bomber with decrement
ORG start
start   MOV.I   $2, @0
        ADD.AB  #{step}, $-1
        JMP.A   $-2
        DAT.F   #0, #{step}
END
""",

        'paper': """
;name Paper_{id}
;author ExplicitMutator
;strategy Paper - self-replicating SPL chain
ORG start
start   SPL.A   $0, $0
        MOV.I   $-1, @ptr
        ADD.AB  #{step}, $ptr
        JMP.A   $-2
ptr     DAT.F   #0, #{step}
END
""",

        'silk': """
;name Silk_{id}
;author ExplicitMutator
;strategy Silk - fast paper with 3-point attack
ORG start
start   SPL.A   $2, $0
        SPL.A   $1, $0
        MOV.I   $-1, @ptr
        ADD.AB  #{step}, $ptr
        JMP.A   $-2
ptr     DAT.F   #0, #{step}
END
""",

        'scissors': """
;name Scissors_{id}
;author ExplicitMutator
;strategy Scanner - searches then attacks
ORG start
scan    ADD.AB  #{step}, $ptr
        CMP.I   @ptr, $bomb
        JMP.A   $attack
        JMP.A   $scan
attack  MOV.I   $bomb, @ptr
        JMP.A   $scan
bomb    DAT.F   #0, #0
ptr     DAT.F   #0, #{step}
END
""",

        'vampire': """
;name Vampire_{id}
;author ExplicitMutator
;strategy Vampire - converts enemy to pit jumpers
ORG start
ptr     DAT.F   #0, #{step}
scan    ADD.AB  #{scanstep}, $ptr
        JMZ.F   $scan, @ptr
        MOV.I   $fang, @ptr
        JMP.A   $scan
fang    JMP.A   $pit
pit     DAT.F   #0, #0
END
""",

        'clear': """
;name Clear_{id}
;author ExplicitMutator
;strategy Clear - wipes memory systematically
ORG start
start   MOV.I   $bomb, >ptr
        DJN.F   $start, $count
        JMP.A   $start
bomb    DAT.F   #0, #0
ptr     DAT.F   #-1, #0
count   DAT.F   #{count}, #{count}
END
""",

        'quickscanner': """
;name QScan_{id}
;author ExplicitMutator
;strategy Quickscanner - fast initial scan then bomb
ORG start
        SNE.I   ${q1}, ${q2}
        SEQ.I   ${q3}, ${q4}
        JMP.A   $attack
        SNE.I   ${q5}, ${q6}
        SEQ.I   ${q7}, ${q8}
        JMP.A   $attack2
        JMP.A   $bomb
attack  MOV.I   $dat, @${q1}
        ADD.AB  #{qstep}, $-1
        JMP.A   $-2
attack2 MOV.I   $dat, @${q5}
        ADD.AB  #{qstep}, $-1
        JMP.A   $-2
bomb    ADD.AB  #{step}, $ptr
        MOV.I   $dat, @ptr
        JMP.A   $bomb
dat     DAT.F   #0, #0
ptr     DAT.F   #0, #{step}
END
""",

        'replicator': """
;name Replicator_{id}
;author ExplicitMutator
;strategy Self-replicating then executing
ORG start
start   MOV.I   $-1, @ptr
        MOV.I   @ptr, <dest
        CMP.I   $ptr, $dest
        JMP.A   $-2
        SPL.A   @dest, #0
        ADD.AB  #{dist}, $dest
        JMP.A   $start
ptr     DAT.F   #-5, #{size}
dest    DAT.F   #100, #{dist}
END
""",

        'oneshot': """
;name OneShot_{id}
;author ExplicitMutator
;strategy One-shot scanner with bombing fallback
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

        'paper_bomber': """
;name PaperBomb_{id}
;author ExplicitMutator  
;strategy Paper with integrated bomber
ORG boot
boot    SPL.A   $pstart, $0
        JMP.A   $bstart
pstart  SPL.A   $0, $0
        MOV.I   $-1, @pptr
        ADD.AB  #{pstep}, $pptr
        JMP.A   $-2
pptr    DAT.F   #0, #{pstep}
bstart  ADD.AB  #{bstep}, $bptr
        MOV.AB  #0, @bptr
        JMP.A   $bstart
bptr    DAT.F   #0, #{bstep}
END
""",

        'pale_shears': """
;name Pale Shears3
;author Matt Hastings
;strategy Pale Shears w/ Mintardjo's Anti-IMP code
ORG start
        MOV jb,3+156
        MOV sb,<-1
        ADD #156,-2
        DJN -3,#1599
sb      SPL 0,<0
        MOV 2,<-4
jb      JMP -1,-1
        DAT <-92,<-2
END
""",

        'ice_wall': """
;name Icewall
;author Matt Hastings
ORG start
inc     DAT #155,#-155
        MOV -2,@-2
jb      JMP 7,0
start   SPL 0,<2000-1368
b1      MOV <-2-5+155+2945,0+5-155-2945
        add inc,@t3
        JMP -2,<-3
        MOV @-9,<-6
        SPL -1,jb
        t3 JMP -2,b1
END
""",
    }
    
    # ==================== COMPONENTS ====================
    
    BOOT_COMPONENTS = {
        'simple': """
        ; Simple boot
        MOV.I   $code, ${bootdist}
        JMP.A   ${bootdist}
code    ; code follows
""",
        'spl_boot': """
        ; SPL boot - creates backup
        SPL.A   ${bootdist}, $0
        MOV.I   $0, ${bootdist}+1
        JMP.A   ${bootdist}
""",
        'multiboot': """
        ; Multi-point boot
        SPL.A   ${bootdist}, $0
        SPL.A   ${bootdist2}, $0
        JMP.A   $code
""",
    }
    
    DECOY_COMPONENTS = {
        'dat_field': """
        ; DAT field decoy
        DAT.F   #0, #0
        DAT.F   #0, #0
        DAT.F   #0, #0
        DAT.F   #0, #0
""",
        'imp_decoy': """
        ; Imp decoy
        MOV.I   $0, $1
        MOV.I   $0, $1
""",
        'jmp_decoy': """
        ; Jump decoy
        JMP.A   $0
        JMP.A   $-1
        DAT.F   #0, #0
""",
    }
    
    OPCODES = ['DAT', 'MOV', 'ADD', 'SUB', 'MUL', 'DIV', 'MOD',
               'JMP', 'JMZ', 'JMN', 'DJN', 'CMP', 'SEQ', 'SNE',
               'SLT', 'SPL', 'NOP']
    
    MODIFIERS = ['A', 'B', 'AB', 'BA', 'F', 'X', 'I']
    
    ADDR_MODES = ['#', '$', '*', '@', '{', '<', '}', '>']
    
    # Warrior type signatures (for detection)
    TYPE_SIGNATURES = {
        'paper': ['SPL', 'MOV.I', 'ADD.AB'],
        'stone': ['MOV.I', 'ADD.AB', 'JMP'],
        'scanner': ['SNE', 'SEQ', 'CMP', 'JMZ'],
        'bomber': ['ADD.AB', 'MOV.AB', 'JMP', 'DAT'],
        'imp': ['MOV.I $0, $1'],
        'vampire': ['JMZ', 'MOV.I', 'JMP'],
        'clear': ['MOV.I', 'DJN', '>'],
        'replicator': ['SPL', 'MOV.I', '<'],
    }
    
    def __init__(self, environment=None, mutation_weights=None):
        self.environment = environment or {}
        self.all_generations = []
        
        # Default mutation weights
        self.mutation_weights = mutation_weights or {
            # Constant mutations
            'change_constant_small': 0.12,
            'change_constant_good': 0.10,
            'prime_step': 0.08,
            # Instruction mutations  
            'change_opcode': 0.06,
            'change_modifier': 0.06,
            'change_addr_mode': 0.08,
            # Structural mutations
            'insert_instruction': 0.04,
            'delete_instruction': 0.03,
            'swap_instructions': 0.03,
            'duplicate_instruction': 0.03,
            # Pattern injection
            'insert_spl': 0.10,
            'inject_bomber_loop': 0.05,
            'inject_spl_fan': 0.06,
            'inject_quickscan': 0.04,
            'add_decoy': 0.04,
            'add_boot': 0.04,
            # Type-specific
            'optimize_paper': 0.02,
            'optimize_stone': 0.02,
        }
        
        # Track mutation success rates (for future adaptive weights)
        self.mutation_stats = {k: {'tried': 0, 'improved': 0} for k in self.mutation_weights}
    
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
        if len(warrior.instructions) > 100:  # Max length
            return False, "Too many instructions"
        return True, None
    
    def _to_explicit_warrior(self, code: str, parent_id: str = None) -> ExplicitWarrior:
        """Convert code string to GPTWarrior object."""
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
        """Detect the primary strategy type of a warrior."""
        code_upper = code.upper()
        
        scores = {}
        for wtype, signatures in self.TYPE_SIGNATURES.items():
            score = sum(1 for sig in signatures if sig.upper() in code_upper)
            scores[wtype] = score
        
        if max(scores.values()) == 0:
            return 'unknown'
        
        return max(scores, key=scores.get)
    
    # ==================== NEW WARRIOR GENERATION ====================
    
    def _fill_template(self, template: str) -> str:
        """Fill in template parameters with random values."""
        unique_id = random.randint(1000, 9999)
        
        # Basic parameters
        params = {
            'id': str(unique_id),
            'step': random.choice(self.GOOD_STEPS),
            'dist': random.choice(self.BOOT_DISTANCES),
            'size': random.randint(5, 20),
            'count': random.randint(100, 1000),
            'bootdist': random.choice(self.BOOT_DISTANCES),
            'bootdist2': random.choice(self.BOOT_DISTANCES),
            'scanstep': random.choice(self.GOOD_STEPS),
            'pstep': random.choice(self.GOOD_STEPS),
            'bstep': random.choice(self.GOOD_STEPS),
            'qstep': random.choice([3, 5, 7, 11, 13]),
            'start': random.choice(self.GOOD_STEPS),
        }
        
        # Quickscan positions (spread across core)
        base = random.randint(100, 2000)
        gap = random.randint(500, 2000)
        for i in range(1, 9):
            params[f'q{i}'] = str((base + (i-1) * gap) % self.CORE_SIZE)
        
        code = template
        for key, value in params.items():
            code = code.replace('{' + key + '}', str(value))
        
        return code
    
    def _generate_new_warrior(self) -> str:
        """Generate a new warrior from templates."""
        # Weighted template selection (favor more successful types)
        weights = {
            'paper': 0.15,
            'silk': 0.12,
            'stone': 0.12,
            'dwarf': 0.10,
            'scissors': 0.10,
            'quickscanner': 0.08,
            'replicator': 0.08,
            'oneshot': 0.06,
            'paper_bomber': 0.06,
            'vampire': 0.05,
            'clear': 0.04,
            'imp': 0.02,
            'imp_spiral': 0.02,
        }
        
        template_name = random.choices(
            list(weights.keys()),
            weights=list(weights.values())
        )[0]
        
        template = self.TEMPLATES[template_name]
        code = self._fill_template(template)
        
        return code
    
    def _generate_composite_warrior(self) -> str:
        """Generate a warrior by combining components."""
        # Pick a base strategy
        base_templates = ['stone', 'paper', 'dwarf']
        base = random.choice(base_templates)
        code = self._fill_template(self.TEMPLATES[base])
        
        # Maybe add boot
        if random.random() < 0.3:
            boot_type = random.choice(list(self.BOOT_COMPONENTS.keys()))
            boot = self._fill_template(self.BOOT_COMPONENTS[boot_type])
            # Insert boot before ORG
            code = boot + code
        
        # Maybe add decoy
        if random.random() < 0.2:
            decoy_type = random.choice(list(self.DECOY_COMPONENTS.keys()))
            decoy = self.DECOY_COMPONENTS[decoy_type]
            # Insert decoy before END
            code = re.sub(r'(\s*END\s*)', decoy + r'\1', code, flags=re.IGNORECASE)
        
        return code
    
    async def new_warrior_async(self, n_warriors: int = 1, n_responses: int = 1) -> np.ndarray:
        """Generate new warriors with validation."""
        exp_warriors = [[None for _ in range(n_responses)] for _ in range(n_warriors)]
        
        for i in range(n_warriors):
            for j in range(n_responses):
                # Try to generate valid warrior
                for attempt in range(self.MAX_RETRIES):
                    if random.random() < 0.2:
                        code = self._generate_composite_warrior()
                    else:
                        code = self._generate_new_warrior()
                    
                    valid, error = self._validate_warrior(code)
                    if valid:
                        break
                
                exp_warriors[i][j] = self._to_explicit_warrior(code)
        
        self.all_generations.append(("new_warrior", exp_warriors))
        return np.array(exp_warriors, dtype=object)
    
    # ==================== CROSSOVER ====================
    
    def _crossover(self, parent1_code: str, parent2_code: str) -> str:
        """Combine two warriors through crossover."""
        lines1 = [l for l in parent1_code.split('\n') 
                  if l.strip() and not l.strip().startswith(';')
                  and not l.strip().upper().startswith(('ORG', 'END', 'EQU'))]
        
        lines2 = [l for l in parent2_code.split('\n')
                  if l.strip() and not l.strip().startswith(';')
                  and not l.strip().upper().startswith(('ORG', 'END', 'EQU'))]
        
        if len(lines1) < 2 or len(lines2) < 2:
            return parent1_code  # Can't crossover
        
        # Single-point crossover
        cut1 = random.randint(1, len(lines1) - 1)
        cut2 = random.randint(1, len(lines2) - 1)
        
        new_lines = lines1[:cut1] + lines2[cut2:]
        
        # Reconstruct warrior
        unique_id = random.randint(1000, 9999)
        code = f";name Crossover_{unique_id}\n"
        code += ";author ExplicitMutator\n"
        code += ";strategy Crossover of two warriors\n"
        code += "ORG start\n"
        
        # Add label to first instruction
        if new_lines:
            new_lines[0] = "start   " + new_lines[0].lstrip()
        
        code += '\n'.join(new_lines) + '\nEND\n'
        
        return code
    
    # ==================== MUTATIONS ====================
    
    def _mutate_code(self, code: str) -> str:
        """Apply a random mutation to the code."""
        mutations = list(self.mutation_weights.keys())
        weights = list(self.mutation_weights.values())
        weights = np.array(weights) / sum(weights)
        mutation_name = np.random.choice(mutations, p=weights)
        
        self.mutation_stats[mutation_name]['tried'] += 1
        
        mutation_fn = getattr(self, f'_mut_{mutation_name}', None)
        if mutation_fn is None:
            return code
        
        return mutation_fn(code)
    
    def _mut_change_constant_small(self, code: str) -> str:
        """Change a numeric constant by a small delta."""
        pattern = r'([#$@<>*{}\s])(-?\d+)'
        matches = list(re.finditer(pattern, code))
        if not matches:
            return code
        
        match = random.choice(matches)
        old_val = int(match.group(2))
        delta = random.randint(-20, 20)
        new_val = (old_val + delta) % self.CORE_SIZE
        
        return code[:match.start(2)] + str(new_val) + code[match.end(2):]
    
    def _mut_change_constant_good(self, code: str) -> str:
        """Replace a constant with a known good value."""
        pattern = r'([#$@<>*{}\s])(\d+)'
        matches = list(re.finditer(pattern, code))
        if not matches:
            return code
        
        match = random.choice(matches)
        new_val = random.choice(self.GOOD_STEPS)
        
        return code[:match.start(2)] + str(new_val) + code[match.end(2):]
    
    def _mut_prime_step(self, code: str) -> str:
        """Replace step values with good primes."""
        pattern = r'#(\d{2,4})(?=\s*,|\s*$|\s*;)'
        matches = list(re.finditer(pattern, code))
        
        if not matches:
            return code
        
        match = random.choice(matches)
        new_val = random.choice(self.GOOD_STEPS)
        
        return code[:match.start(1)] + str(new_val) + code[match.end(1):]
    
    def _mut_change_opcode(self, code: str) -> str:
        """Change an opcode to a different one."""
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
        """Change an instruction modifier."""
        pattern = r'\.([ABABFXI]{1,2})\b'
        matches = list(re.finditer(pattern, code, re.IGNORECASE))
        if not matches:
            return code
        
        match = random.choice(matches)
        new_mod = random.choice(self.MODIFIERS)
        
        return code[:match.start(1)] + new_mod + code[match.end(1):]
    
    def _mut_change_addr_mode(self, code: str) -> str:
        """Change an addressing mode."""
        pattern = r'([#$*@{}<>])(-?\d+)'
        matches = list(re.finditer(pattern, code))
        if not matches:
            return code
        
        match = random.choice(matches)
        new_mode = random.choice(self.ADDR_MODES)
        
        return code[:match.start(1)] + new_mode + code[match.end(1):]
    
    def _mut_insert_spl(self, code: str) -> str:
        """Insert SPL instruction for better survival."""
        lines = code.split('\n')
        
        insert_idx = 0
        for i, line in enumerate(lines):
            if 'ORG' in line.upper():
                insert_idx = i + 1
                break
            if line.strip() and not line.strip().startswith(';'):
                insert_idx = i
                break
        
        spl_line = "        SPL.A   $0, $0"
        lines.insert(insert_idx, spl_line)
        
        return '\n'.join(lines)
    
    def _mut_insert_instruction(self, code: str) -> str:
        """Insert a random instruction."""
        lines = code.split('\n')
        
        instr_indices = [i for i, l in enumerate(lines)
                        if l.strip() and not l.strip().startswith(';')
                        and not l.strip().upper().startswith(('ORG', 'END', 'EQU'))]
        
        if not instr_indices:
            return code
        
        insert_idx = random.choice(instr_indices)
        
        op = random.choice(self.OPCODES)
        mod = random.choice(self.MODIFIERS)
        mode_a = random.choice(self.ADDR_MODES)
        mode_b = random.choice(self.ADDR_MODES)
        val_a = random.choice(self.GOOD_STEPS[:20] + list(range(-10, 11)))
        val_b = random.choice(self.GOOD_STEPS[:20] + list(range(-10, 11)))
        
        new_line = f"        {op}.{mod}   {mode_a}{val_a}, {mode_b}{val_b}"
        lines.insert(insert_idx, new_line)
        
        return '\n'.join(lines)
    
    def _mut_delete_instruction(self, code: str) -> str:
        """Delete a random instruction."""
        lines = code.split('\n')
        
        instr_indices = [i for i, l in enumerate(lines)
                        if l.strip() and not l.strip().startswith(';')
                        and not l.strip().upper().startswith(('ORG', 'END', 'EQU'))]
        
        if len(instr_indices) <= 2:
            return code
        
        del_idx = random.choice(instr_indices)
        del lines[del_idx]
        
        return '\n'.join(lines)
    
    def _mut_swap_instructions(self, code: str) -> str:
        """Swap two instructions."""
        lines = code.split('\n')
        
        instr_indices = [i for i, l in enumerate(lines)
                        if l.strip() and not l.strip().startswith(';')
                        and not l.strip().upper().startswith(('ORG', 'END', 'EQU'))]
        
        if len(instr_indices) < 2:
            return code
        
        idx1, idx2 = random.sample(instr_indices, 2)
        lines[idx1], lines[idx2] = lines[idx2], lines[idx1]
        
        return '\n'.join(lines)
    
    def _mut_duplicate_instruction(self, code: str) -> str:
        """Duplicate an instruction."""
        lines = code.split('\n')
        
        instr_indices = [i for i, l in enumerate(lines)
                        if l.strip() and not l.strip().startswith(';')
                        and not l.strip().upper().startswith(('ORG', 'END', 'EQU'))]
        
        if not instr_indices:
            return code
        
        dup_idx = random.choice(instr_indices)
        dup_line = re.sub(r'^\w+\s+', '        ', lines[dup_idx])
        lines.insert(dup_idx + 1, dup_line)
        
        return '\n'.join(lines)
    
    def _mut_inject_bomber_loop(self, code: str) -> str:
        """Inject a bomber loop pattern."""
        step = random.choice(self.GOOD_STEPS)
        bomber = f"""
bomb_inj DAT.F   #0, #0
bloop   ADD.AB  #{step}, $bomb_inj
        MOV.AB  #0, @bomb_inj
        JMP.A   $bloop
"""
        code = re.sub(r'(\s*END\s*)', bomber + r'\1', code, flags=re.IGNORECASE)
        return code
    
    def _mut_inject_spl_fan(self, code: str) -> str:
        """Inject SPL fan pattern."""
        fan = """
        SPL.A   $0, $0
        SPL.A   $0, $0
"""
        code = re.sub(r'(ORG\s+\w+\s*\n)', r'\1' + fan, code, flags=re.IGNORECASE)
        return code
    
    def _mut_inject_quickscan(self, code: str) -> str:
        """Inject a quickscan pattern."""
        q1 = random.randint(500, 3000)
        q2 = q1 + random.randint(200, 500)
        
        qscan = f"""
        SNE.I   ${q1}, ${q2}
        JMP.A   $qhit
        JMP.A   $qmiss
qhit    MOV.I   $qbomb, @${q1}
qmiss   ; continue
qbomb   DAT.F   #0, #0
"""
        code = re.sub(r'(ORG\s+\w+\s*\n)', r'\1' + qscan, code, flags=re.IGNORECASE)
        return code
    
    def _mut_add_decoy(self, code: str) -> str:
        """Add decoy instructions."""
        decoy_type = random.choice(list(self.DECOY_COMPONENTS.keys()))
        decoy = self.DECOY_COMPONENTS[decoy_type]
        
        code = re.sub(r'(\s*END\s*)', decoy + r'\1', code, flags=re.IGNORECASE)
        return code
    
    def _mut_add_boot(self, code: str) -> str:
        """Add boot pattern to move code away from origin."""
        if 'boot' in code.lower():
            return code  # Already has boot
        
        dist = random.choice(self.BOOT_DISTANCES)
        
        boot = f"""
boot    MOV.I   $code_start, ${dist}
        MOV.I   $code_start+1, ${dist}+1
        JMP.A   ${dist}
code_start
"""
        # Insert before ORG
        code = re.sub(r'(ORG\s+)', boot + r'\nORG boot\n; Original: ', code, flags=re.IGNORECASE)
        return code
    
    def _mut_optimize_paper(self, code: str) -> str:
        """Optimize paper-type warriors."""
        wtype = self._detect_warrior_type(code)
        if wtype not in ['paper', 'silk', 'replicator']:
            return code
        
        # Optimize step size for papers (2667 is popular)
        good_paper_steps = [2667, 5334, 1471, 2500, 3333]
        new_step = random.choice(good_paper_steps)
        
        # Find and replace step
        pattern = r'(ADD\.AB\s+#)(\d+)'
        code = re.sub(pattern, rf'\g<1>{new_step}', code, count=1, flags=re.IGNORECASE)
        
        return code
    
    def _mut_optimize_stone(self, code: str) -> str:
        """Optimize stone-type warriors."""
        wtype = self._detect_warrior_type(code)
        if wtype not in ['stone', 'bomber']:
            return code
        
        # Good stone steps
        good_stone_steps = [3044, 2365, 5363, 3039, 2377]
        new_step = random.choice(good_stone_steps)
        
        pattern = r'(ADD\.AB\s+#)(\d+)'
        code = re.sub(pattern, rf'\g<1>{new_step}', code, count=1, flags=re.IGNORECASE)
        
        return code
    
    # ==================== MUTATE WARRIOR ====================
    
    async def mutate_warrior_async(self, exp_warriors: List[ExplicitWarrior], n_responses: int = 1) -> np.ndarray:
        """Mutate warriors with validation and optional crossover."""
        result = [[None for _ in range(n_responses)] for _ in range(len(exp_warriors))]
        
        for i, parent in enumerate(exp_warriors):
            for j in range(n_responses):
                # Try to create valid mutant
                for attempt in range(self.MAX_RETRIES):
                    # Occasional crossover if we have multiple parents
                    if len(exp_warriors) > 1 and random.random() < 0.1:
                        other = random.choice([w for w in exp_warriors if w != parent])
                        code = self._crossover(parent.llm_response, other.llm_response)
                    else:
                        code = parent.code
                    
                    # Apply 1-3 mutations
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