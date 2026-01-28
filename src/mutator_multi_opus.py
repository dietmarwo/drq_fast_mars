"""
Explicit Core War Mutator v3 - Adaptive Learning

Based on explicit_mutator2_warmstart.py with four key improvements:
1. Mutation Success Tracking - Learn which mutations actually improve fitness
2. Type-Aware Mutation Selection - Apply warrior-type-specific mutation weights  
3. Directional Hill Climbing - Remember successful directions for constants
4. Protected Regions - Don't mutate critical code structures (boot, SPL gates, scan loops)

Use warmstart_mode=True when loading from a previous run's archive.
Use warmstart_mode=False when starting from scratch.
"""

import re
import random
import numpy as np
import hashlib
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Set, Dict
from collections import defaultdict
from corewar import redcode, Warrior

from util import ExplicitWarrior


# ============================================================================
# IMPROVEMENT 1: MUTATION SUCCESS TRACKING
# ============================================================================

class MutationTracker:
    """
    Track which mutations actually improve fitness.
    
    Usage:
        tracker = MutationTracker()
        
        # Before mutation - record what mutations were applied
        tracker.record_mutation(warrior_hash, "fine_tune_step")
        
        # After fitness evaluation - provide feedback
        tracker.record_fitness_result(warrior_hash, old_fitness, new_fitness)
        
        # Get adaptive weights (successful mutations get higher weight)
        weights = tracker.get_adaptive_weights(base_weights)
    """
    
    def __init__(self, window_size: int = 100):
        # mutation_name -> list of fitness deltas
        self.history: Dict[str, List[float]] = defaultdict(list)
        self.window_size = window_size
        
        # Track pending mutations: warrior_hash -> list of mutation names
        self.pending: Dict[str, List[str]] = {}
    
    def record_mutation(self, warrior_hash: str, mutation_name: str):
        """Record that a mutation was applied (before we know if it helped)."""
        if warrior_hash not in self.pending:
            self.pending[warrior_hash] = []
        self.pending[warrior_hash].append(mutation_name)
    
    def record_fitness_result(self, warrior_hash: str, old_fitness: float, new_fitness: float):
        """
        Record the fitness result after evaluation.
        Call this once fitness is known for a mutated warrior.
        """
        if warrior_hash not in self.pending:
            return
        
        mutations = self.pending.pop(warrior_hash)
        delta = new_fitness - old_fitness
        
        # Credit/blame all mutations applied
        for mutation_name in mutations:
            self.history[mutation_name].append(delta)
            
            # Keep window size bounded
            if len(self.history[mutation_name]) > self.window_size:
                self.history[mutation_name] = self.history[mutation_name][-self.window_size:]
    
    def get_success_rate(self, mutation_name: str) -> float:
        """Fraction of times this mutation improved fitness."""
        hist = self.history.get(mutation_name, [])
        if len(hist) < 5:
            return 0.5  # Unknown - neutral prior
        return sum(1 for d in hist if d > 0) / len(hist)
    
    def get_avg_improvement(self, mutation_name: str) -> float:
        """Average fitness change from this mutation."""
        hist = self.history.get(mutation_name, [])
        if not hist:
            return 0.0
        return sum(hist) / len(hist)
    
    def get_sample_count(self, mutation_name: str) -> int:
        """Number of samples for this mutation."""
        return len(self.history.get(mutation_name, []))
    
    def get_adaptive_weights(self, base_weights: Dict[str, float]) -> Dict[str, float]:
        """
        Adjust weights based on observed success rates.
        
        Mutations with higher success rates get boosted.
        Mutations with lower success rates get dampened.
        Range: 0.3x to 1.7x of base weight (never completely eliminated).
        """
        adapted = {}
        for name, base_weight in base_weights.items():
            success_rate = self.get_success_rate(name)
            # Linear scaling: 0% success -> 0.3x, 50% -> 1.0x, 100% -> 1.7x
            multiplier = 0.3 + success_rate * 1.4
            adapted[name] = base_weight * multiplier
        
        # Normalize to sum to 1
        total = sum(adapted.values())
        if total > 0:
            return {k: v / total for k, v in adapted.items()}
        return base_weights


# ============================================================================
# IMPROVEMENT 3: DIRECTIONAL HILL CLIMBING
# ============================================================================

class DirectionalTuner:
    """
    Remember which direction improved fitness for each constant location.
    
    When fine-tuning a constant:
    - If +1 improved fitness last time, bias toward +1/+2 next time
    - If -1 improved, bias toward -1/-2
    
    Uses (structure_hash, context_pattern) as key to generalize across similar warriors.
    """
    
    def __init__(self, max_entries: int = 1000):
        # (structure_hash, context) -> direction (+1 or -1)
        self.momentum: Dict[Tuple[str, str], int] = {}
        self.max_entries = max_entries
    
    def _get_structure_hash(self, code: str) -> str:
        """Hash the structure (opcodes only, not constants)."""
        opcodes = re.findall(
            r'\b(MOV|ADD|SUB|MUL|DIV|MOD|JMP|JMZ|JMN|DJN|CMP|SEQ|SNE|SLT|SPL|DAT|NOP)\b',
            code, re.IGNORECASE
        )
        return hashlib.md5(''.join(opcodes).upper().encode()).hexdigest()[:8]
    
    def _get_context(self, code: str, match_start: int) -> str:
        """Get context around a constant (opcode + field position)."""
        # Find the line containing this match
        line_start = code.rfind('\n', 0, match_start) + 1
        line_end = code.find('\n', match_start)
        if line_end == -1:
            line_end = len(code)
        line = code[line_start:line_end]
        
        # Extract opcode
        op_match = re.search(r'\b(MOV|ADD|SUB|JMP|JMZ|JMN|DJN|SPL|DAT|SNE|SEQ|CMP)\b', line, re.IGNORECASE)
        opcode = op_match.group(1).upper() if op_match else 'UNK'
        
        # Determine A-field or B-field based on position in line
        pos_in_line = match_start - line_start
        comma_pos = line.find(',')
        field = 'A' if comma_pos == -1 or pos_in_line < comma_pos else 'B'
        
        return f"{opcode}_{field}"
    
    def get_biased_delta(self, code: str, match_start: int, base_deltas: List[int]) -> int:
        """
        Get delta with directional bias based on previous success.
        
        70% chance to follow momentum if we have it.
        """
        struct_hash = self._get_structure_hash(code)
        context = self._get_context(code, match_start)
        key = (struct_hash, context)
        
        if key in self.momentum:
            direction = self.momentum[key]
            if random.random() < 0.7:
                # Bias toward previous successful direction
                biased = [d for d in base_deltas if d * direction > 0]
                if biased:
                    return random.choice(biased)
        
        return random.choice(base_deltas)
    
    def record_result(self, code: str, match_start: int, delta: int, improved: bool):
        """Record whether the delta improved fitness."""
        struct_hash = self._get_structure_hash(code)
        context = self._get_context(code, match_start)
        key = (struct_hash, context)
        
        if improved and delta != 0:
            self.momentum[key] = 1 if delta > 0 else -1
        elif not improved and delta != 0:
            # Try opposite direction next time
            self.momentum[key] = -1 if delta > 0 else 1
        
        # Limit size
        if len(self.momentum) > self.max_entries:
            # Remove oldest half
            items = list(self.momentum.items())
            self.momentum = dict(items[-(self.max_entries // 2):])


# ============================================================================
# MAIN MUTATOR CLASS
# ============================================================================

class ExplicitCoreMutator:
    """
    Explicit rule-based mutator with adaptive learning.
    
    Improvements over v2:
    1. Tracks mutation success and adapts weights accordingly
    2. Applies type-specific mutation weights (paper vs stone vs scanner etc.)
    3. Uses directional hill climbing for fine-tuning
    4. Protects critical code regions (boot, SPL gates, scan loops)
    
    When warmstart_mode=True:
    - Conservative: usually 1 mutation per step
    - Focus on fine-tuning constants
    - Learn from archive
    - Protect proven structure
    
    When warmstart_mode=False:
    - More exploratory: 1-3 mutations
    - Broader mutation selection
    - Still tracks success for learning
    """
    
    CORE_SIZE = 8000
    MAX_RETRIES = 5
    
    # Proven effective step sizes
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
    
    # Fine-tuning deltas for hill climbing
    FINE_DELTAS = [-2, -1, 1, 2]
    SMALL_DELTAS = [-5, -3, 3, 5]
    MEDIUM_DELTAS = [-20, -10, 10, 20]
    
    BOOT_DISTANCES = [100, 200, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000]
    SCAN_DELAYS = [4, 5, 6, 7, 8, 10, 12, 15]
    
    # ==================== TEMPLATES ====================
    
    TEMPLATES = {
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
    }
    
    DECOY_COMPONENTS = {
        'dat_field': """
        ; DAT field decoy
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
    
    # ==================== TYPE SIGNATURES ====================
    
    TYPE_SIGNATURES = {
        'paper': ['SPL', 'MOV.I', 'ADD.AB'],
        'stone': ['MOV.I', 'ADD.AB', 'JMP'],
        'scanner': ['SNE', 'SEQ', 'CMP', 'JMZ'],
        'bomber': ['ADD.AB', 'MOV.AB', 'JMP', 'DAT'],
        'imp': ['MOV.I $0, $1'],
        'vampire': ['JMZ', 'MOV.I', 'fang', 'pit'],
        'clear': ['MOV.I', 'DJN', '>'],
        'replicator': ['SPL', 'MOV.I', '<'],
    }
    
    # ==================== IMPROVEMENT 2: TYPE-AWARE WEIGHTS ====================
    
    TYPE_MUTATION_ADJUSTMENTS = {
        'paper': {
            'optimize_paper': 3.0,
            'inject_spl_fan': 2.0,
            'fine_tune_step': 1.5,
            'change_opcode': 0.3,
            'delete_instruction': 0.2,
            'inject_bomber_loop': 0.5,
        },
        'stone': {
            'optimize_stone': 3.0,
            'fine_tune_step': 2.5,
            'prime_step': 2.0,
            'inject_bomber_loop': 1.5,
            'insert_spl': 0.5,
            'add_decoy': 0.5,
        },
        'scanner': {
            'fine_tune_step': 2.5,
            'change_addr_mode': 1.5,
            'inject_quickscan': 2.0,
            'tune_loop_count': 1.5,
            'add_decoy': 0.3,
            'optimize_paper': 0.3,
        },
        'vampire': {
            'fine_tune_constant': 2.0,
            'fine_tune_step': 1.5,
            'change_addr_mode': 1.5,
            'delete_instruction': 0.2,
            'optimize_stone': 0.3,
        },
        'bomber': {
            'fine_tune_step': 2.0,
            'optimize_stone': 1.5,
            'prime_step': 1.5,
            'inject_bomber_loop': 1.5,
        },
        'imp': {
            'insert_spl': 2.5,
            'inject_spl_fan': 2.0,
            'change_constant_small': 0.3,
            'change_constant_good': 0.3,
            'fine_tune_step': 0.3,
        },
        'clear': {
            'fine_tune_step': 2.0,
            'tune_loop_count': 2.0,
            'optimize_stone': 1.5,
        },
    }
    
    def __init__(self, environment=None, warmstart_mode: bool = False, 
                 use_adaptive_weights: bool = True):
        """
        Args:
            environment: Redcode parsing environment
            warmstart_mode: If True, use conservative mutations for refinement
            use_adaptive_weights: If True, adapt weights based on observed success
        """
        self.environment = environment or {}
        self.warmstart_mode = warmstart_mode
        self.use_adaptive_weights = use_adaptive_weights
        self.all_generations = []
        
        # Learned successful constants from archive
        self.learned_steps: Set[int] = set()
        self.learned_counts: Set[int] = set()
        
        # IMPROVEMENT 1: Mutation success tracking
        self.mutation_tracker = MutationTracker(window_size=100)
        
        # IMPROVEMENT 3: Directional hill climbing
        self.directional_tuner = DirectionalTuner(max_entries=1000)
        
        # Track pending hill climbing deltas for feedback
        self._pending_deltas: Dict[str, List[Tuple[str, int, int]]] = {}
        
        # Select base mutation weights based on mode
        if warmstart_mode:
            self.base_mutation_weights = {
                'fine_tune_step': 0.22,
                'fine_tune_constant': 0.12,
                'tune_loop_count': 0.08,
                'archive_step': 0.06,
                'change_constant_good': 0.08,
                'prime_step': 0.06,
                'change_modifier': 0.06,
                'change_addr_mode': 0.06,
                'swap_adjacent': 0.04,
                'optimize_paper': 0.04,
                'optimize_stone': 0.04,
                'change_opcode': 0.03,
                'change_constant_small': 0.04,
                'insert_spl': 0.03,
                'inject_spl_fan': 0.02,
                'insert_instruction': 0.01,
                'delete_instruction': 0.01,
            }
        else:
            self.base_mutation_weights = {
                'change_constant_small': 0.12,
                'change_constant_good': 0.10,
                'prime_step': 0.08,
                'change_opcode': 0.06,
                'change_modifier': 0.06,
                'change_addr_mode': 0.08,
                'insert_instruction': 0.04,
                'delete_instruction': 0.03,
                'swap_instructions': 0.03,
                'duplicate_instruction': 0.03,
                'insert_spl': 0.10,
                'inject_bomber_loop': 0.05,
                'inject_spl_fan': 0.06,
                'inject_quickscan': 0.04,
                'add_decoy': 0.04,
                'add_boot': 0.04,
                'optimize_paper': 0.02,
                'optimize_stone': 0.02,
                'fine_tune_step': 0.05,
                'fine_tune_constant': 0.03,
            }
    
    # ==================== FEEDBACK API ====================
    
    def get_warrior_hash(self, code: str) -> str:
        """Get a hash for tracking a warrior."""
        return hashlib.sha256(code.encode()).hexdigest()[:16]
    
    def record_fitness_feedback(self, warrior_hash: str, old_fitness: float, new_fitness: float):
        """
        Call this after evaluating a mutated warrior to provide feedback.
        Enables adaptive mutation weights and directional hill climbing.
        """
        self.mutation_tracker.record_fitness_result(warrior_hash, old_fitness, new_fitness)
        
        improved = new_fitness > old_fitness
        if warrior_hash in self._pending_deltas:
            for code, match_start, delta in self._pending_deltas[warrior_hash]:
                self.directional_tuner.record_result(code, match_start, delta, improved)
            del self._pending_deltas[warrior_hash]
    
    # ==================== IMPROVEMENT 4: PROTECTED REGIONS ====================
    
    def _identify_protected_regions(self, code: str) -> List[Tuple[int, int]]:
        """Identify line ranges that should NOT be mutated."""
        protected = []
        lines = code.split('\n')
        
        # 1. Protect boot sequences
        for i, line in enumerate(lines):
            line_lower = line.lower().strip()
            if 'boot' in line_lower and not line_lower.startswith(';'):
                boot_start = i
                boot_end = i + 1
                for j in range(i, min(i + 8, len(lines))):
                    if 'JMP' in lines[j].upper():
                        boot_end = j + 1
                        break
                protected.append((boot_start, boot_end))
                break
        
        # 2. Protect SPL 0 gates
        for i, line in enumerate(lines):
            if re.search(r'\bSPL\.?\w*\s+\$?0\s*,', line, re.IGNORECASE):
                start = max(0, i - 1)
                end = min(len(lines), i + 2)
                protected.append((start, end))
        
        # 3. Protect scan loops
        in_scan = False
        scan_start = 0
        for i, line in enumerate(lines):
            if re.search(r'\b(JMZ|JMN|SNE|SEQ|CMP)\b', line, re.IGNORECASE) and not in_scan:
                in_scan = True
                scan_start = i
            elif in_scan and re.search(r'\bJMP\s+.*(-\d+|\$scan|\$-)', line, re.IGNORECASE):
                protected.append((scan_start, i + 1))
                in_scan = False
        
        # 4. Protect vampire pits
        for i, line in enumerate(lines):
            if re.search(r'^\s*(pit|fang)\s+', line, re.IGNORECASE):
                protected.append((i, min(len(lines), i + 2)))
        
        return protected
    
    def _is_line_protected(self, line_idx: int, protected: List[Tuple[int, int]]) -> bool:
        """Check if a line is in any protected region."""
        return any(start <= line_idx < end for start, end in protected)
    
    def _get_mutable_lines(self, code: str) -> List[int]:
        """Get indices of lines that CAN be mutated."""
        lines = code.split('\n')
        protected = self._identify_protected_regions(code)
        
        mutable = []
        for i, line in enumerate(lines):
            stripped = line.strip()
            if not stripped or stripped.startswith(';'):
                continue
            if stripped.upper().startswith(('ORG', 'END', 'EQU', ';NAME', ';AUTHOR', ';STRATEGY')):
                continue
            if self._is_line_protected(i, protected):
                continue
            mutable.append(i)
        
        return mutable
    
    # ==================== WEIGHT COMPUTATION ====================
    
    def _detect_warrior_type(self, code: str) -> str:
        """Detect the primary strategy type of a warrior."""
        code_upper = code.upper()
        scores = {}
        for wtype, signatures in self.TYPE_SIGNATURES.items():
            score = sum(1 for sig in signatures if sig.upper() in code_upper)
            scores[wtype] = score
        
        if not scores or max(scores.values()) == 0:
            return 'unknown'
        return max(scores, key=scores.get)
    
    def _get_effective_weights(self, code: str) -> Dict[str, float]:
        """Compute effective mutation weights (type-aware + adaptive)."""
        weights = self.base_mutation_weights.copy()
        
        # IMPROVEMENT 2: Apply type-specific adjustments
        warrior_type = self._detect_warrior_type(code)
        if warrior_type in self.TYPE_MUTATION_ADJUSTMENTS:
            adjustments = self.TYPE_MUTATION_ADJUSTMENTS[warrior_type]
            for mutation, multiplier in adjustments.items():
                if mutation in weights:
                    weights[mutation] *= multiplier
        
        # IMPROVEMENT 1: Apply adaptive weights
        if self.use_adaptive_weights:
            weights = self.mutation_tracker.get_adaptive_weights(weights)
        else:
            total = sum(weights.values())
            if total > 0:
                weights = {k: v / total for k, v in weights.items()}
        
        return weights
    
    # ==================== ARCHIVE LEARNING ====================
    
    def learn_from_archive(self, archive: Dict):
        """Extract successful constants from archive warriors."""
        for bc, warrior in archive.items():
            code = getattr(warrior, 'code', None) or getattr(warrior, 'llm_response', '')
            fitness = getattr(warrior, 'fitness', 0)
            
            if not code or not fitness or fitness <= 0:
                continue
            
            for match in re.finditer(r'ADD\.\w+\s+#(\d+)', code, re.IGNORECASE):
                step = int(match.group(1))
                if 10 < step < 7000:
                    self.learned_steps.add(step)
            
            for match in re.finditer(r'#(\d{2,3}),\s*#(\d{2,3})', code):
                count = int(match.group(1))
                if 10 < count < 1000:
                    self.learned_counts.add(count)
        
        if len(self.learned_steps) > 200:
            self.learned_steps = set(list(self.learned_steps)[-200:])
        if len(self.learned_counts) > 100:
            self.learned_counts = set(list(self.learned_counts)[-100:])
    
    def _get_learned_or_good_step(self) -> int:
        """Get step from learned values or known good values."""
        if self.learned_steps and random.random() < 0.5:
            return random.choice(list(self.learned_steps))
        return random.choice(self.GOOD_STEPS)
    
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
    
    # ==================== WARRIOR GENERATION ====================
    
    def _fill_template(self, template: str) -> str:
        """Fill in template parameters with random values."""
        unique_id = random.randint(1000, 9999)
        params = {
            'id': str(unique_id),
            'step': self._get_learned_or_good_step(),
            'dist': random.choice(self.BOOT_DISTANCES),
            'size': random.randint(5, 20),
            'count': random.randint(100, 1000),
            'bootdist': random.choice(self.BOOT_DISTANCES),
            'bootdist2': random.choice(self.BOOT_DISTANCES),
            'scanstep': self._get_learned_or_good_step(),
            'pstep': self._get_learned_or_good_step(),
            'bstep': self._get_learned_or_good_step(),
            'qstep': random.choice([3, 5, 7, 11, 13]),
            'start': self._get_learned_or_good_step(),
        }
        
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
        weights = {
            'paper': 0.15, 'silk': 0.12, 'stone': 0.12, 'dwarf': 0.10,
            'scissors': 0.10, 'quickscanner': 0.08, 'replicator': 0.08,
            'oneshot': 0.06, 'paper_bomber': 0.06, 'vampire': 0.05,
            'clear': 0.04, 'imp': 0.02, 'imp_spiral': 0.02,
        }
        template_name = random.choices(list(weights.keys()), weights=list(weights.values()))[0]
        return self._fill_template(self.TEMPLATES[template_name])
    
    async def new_warrior_async(self, n_warriors: int = 1, n_responses: int = 1) -> np.ndarray:
        """Generate new warriors with validation."""
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
            return parent1_code
        
        cut1 = random.randint(1, len(lines1) - 1)
        cut2 = random.randint(1, len(lines2) - 1)
        new_lines = lines1[:cut1] + lines2[cut2:]
        
        unique_id = random.randint(1000, 9999)
        code = f";name Crossover_{unique_id}\n;author ExplicitMutator\nORG start\n"
        if new_lines:
            new_lines[0] = "start   " + new_lines[0].lstrip()
        code += '\n'.join(new_lines) + '\nEND\n'
        return code
    
    # ==================== MUTATION DISPATCH ====================
    
    def _mutate_code(self, code: str, warrior_hash: str = None) -> str:
        """Apply a single mutation to the code."""
        weights = self._get_effective_weights(code)
        mutations = list(weights.keys())
        probs = list(weights.values())
        mutation_name = random.choices(mutations, weights=probs)[0]
        
        if warrior_hash:
            self.mutation_tracker.record_mutation(warrior_hash, mutation_name)
        
        method = getattr(self, f'_mut_{mutation_name}', None)
        if method is None:
            return code
        
        if mutation_name in ('fine_tune_step', 'fine_tune_constant'):
            return method(code, warrior_hash)
        else:
            return method(code)
    
    # ==================== FINE-TUNING MUTATIONS ====================
    
    def _mut_fine_tune_step(self, code: str, warrior_hash: str = None) -> str:
        """Fine-tune step size with directional hill climbing."""
        pattern = r'(ADD\.\w+\s+#)(\d+)'
        matches = list(re.finditer(pattern, code, re.IGNORECASE))
        if not matches:
            return code
        
        mutable_lines = set(self._get_mutable_lines(code))
        valid_matches = [m for m in matches if code[:m.start()].count('\n') in mutable_lines]
        if not valid_matches:
            return code
        
        match = random.choice(valid_matches)
        old_val = int(match.group(2))
        delta = self.directional_tuner.get_biased_delta(code, match.start(), self.FINE_DELTAS)
        new_val = (old_val + delta) % self.CORE_SIZE
        if new_val < 3:
            new_val = 3
        
        if warrior_hash:
            if warrior_hash not in self._pending_deltas:
                self._pending_deltas[warrior_hash] = []
            self._pending_deltas[warrior_hash].append((code, match.start(), delta))
        
        return code[:match.start(2)] + str(new_val) + code[match.end(2):]
    
    def _mut_fine_tune_constant(self, code: str, warrior_hash: str = None) -> str:
        """Fine-tune any numeric constant with directional hill climbing."""
        pattern = r'([#$@<>*{}\s])(\d+)'
        matches = list(re.finditer(pattern, code))
        if not matches:
            return code
        
        mutable_lines = set(self._get_mutable_lines(code))
        valid_matches = [m for m in matches if code[:m.start()].count('\n') in mutable_lines]
        if not valid_matches:
            return code
        
        match = random.choice(valid_matches)
        old_val = int(match.group(2))
        delta = self.directional_tuner.get_biased_delta(code, match.start(), self.SMALL_DELTAS)
        new_val = (old_val + delta) % self.CORE_SIZE
        
        if warrior_hash:
            if warrior_hash not in self._pending_deltas:
                self._pending_deltas[warrior_hash] = []
            self._pending_deltas[warrior_hash].append((code, match.start(), delta))
        
        return code[:match.start(2)] + str(new_val) + code[match.end(2):]
    
    def _mut_tune_loop_count(self, code: str) -> str:
        """Tune loop counter."""
        pattern = r'(#)(\d{2,3})(,\s*#)(\d{2,3})'
        matches = list(re.finditer(pattern, code))
        if not matches:
            return code
        
        mutable_lines = set(self._get_mutable_lines(code))
        valid_matches = [m for m in matches if code[:m.start()].count('\n') in mutable_lines]
        if not valid_matches:
            return code
        
        match = random.choice(valid_matches)
        old_val = int(match.group(2))
        delta = random.choice([-30, -20, -10, 10, 20, 30])
        new_val = max(10, min(999, old_val + delta))
        
        return code[:match.start(2)] + str(new_val) + code[match.end(2):match.start(4)] + str(new_val) + code[match.end(4):]
    
    def _mut_archive_step(self, code: str) -> str:
        """Replace step with a learned successful value."""
        if not self.learned_steps:
            return self._mut_change_constant_good(code)
        
        pattern = r'(ADD\.\w+\s+#)(\d+)'
        matches = list(re.finditer(pattern, code, re.IGNORECASE))
        if not matches:
            return code
        
        mutable_lines = set(self._get_mutable_lines(code))
        valid_matches = [m for m in matches if code[:m.start()].count('\n') in mutable_lines]
        if not valid_matches:
            return code
        
        match = random.choice(valid_matches)
        new_val = random.choice(list(self.learned_steps))
        return code[:match.start(2)] + str(new_val) + code[match.end(2):]
    
    def _mut_swap_adjacent(self, code: str) -> str:
        """Swap two adjacent instructions (safer)."""
        mutable = self._get_mutable_lines(code)
        if len(mutable) < 2:
            return code
        
        adjacent_pairs = [(mutable[i], mutable[i+1]) 
                         for i in range(len(mutable) - 1)
                         if mutable[i+1] == mutable[i] + 1]
        if not adjacent_pairs:
            return code
        
        idx1, idx2 = random.choice(adjacent_pairs)
        lines = code.split('\n')
        lines[idx1], lines[idx2] = lines[idx2], lines[idx1]
        return '\n'.join(lines)
    
    # ==================== STANDARD MUTATIONS ====================
    
    def _mut_change_constant_small(self, code: str) -> str:
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
        pattern = r'([#$@<>*{}\s])(\d+)'
        matches = list(re.finditer(pattern, code))
        if not matches:
            return code
        match = random.choice(matches)
        new_val = self._get_learned_or_good_step()
        return code[:match.start(2)] + str(new_val) + code[match.end(2):]
    
    def _mut_prime_step(self, code: str) -> str:
        pattern = r'#(\d{2,4})(?=\s*,|\s*$|\s*;)'
        matches = list(re.finditer(pattern, code))
        if not matches:
            return code
        match = random.choice(matches)
        new_val = self._get_learned_or_good_step()
        return code[:match.start(1)] + str(new_val) + code[match.end(1):]
    
    def _mut_change_opcode(self, code: str) -> str:
        mutable = self._get_mutable_lines(code)
        if not mutable:
            return code
        lines = code.split('\n')
        idx = random.choice(mutable)
        line = lines[idx]
        for op in self.OPCODES:
            pattern = rf'\b{op}\b'
            if re.search(pattern, line, re.IGNORECASE):
                new_op = random.choice(self.OPCODES)
                lines[idx] = re.sub(pattern, new_op, line, count=1, flags=re.IGNORECASE)
                break
        return '\n'.join(lines)
    
    def _mut_change_modifier(self, code: str) -> str:
        pattern = r'\.([ABABFXI]{1,2})\b'
        matches = list(re.finditer(pattern, code, re.IGNORECASE))
        if not matches:
            return code
        match = random.choice(matches)
        new_mod = random.choice(self.MODIFIERS)
        return code[:match.start(1)] + new_mod + code[match.end(1):]
    
    def _mut_change_addr_mode(self, code: str) -> str:
        pattern = r'([#$*@{}<>])(-?\d+)'
        matches = list(re.finditer(pattern, code))
        if not matches:
            return code
        match = random.choice(matches)
        new_mode = random.choice(self.ADDR_MODES)
        return code[:match.start(1)] + new_mode + code[match.end(1):]
    
    def _mut_insert_spl(self, code: str) -> str:
        mutable = self._get_mutable_lines(code)
        if not mutable:
            return code
        lines = code.split('\n')
        insert_idx = mutable[0]
        lines.insert(insert_idx, "        SPL.A   $0, $0")
        return '\n'.join(lines)
    
    def _mut_insert_instruction(self, code: str) -> str:
        mutable = self._get_mutable_lines(code)
        if not mutable:
            return code
        lines = code.split('\n')
        insert_idx = random.choice(mutable)
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
        mutable = self._get_mutable_lines(code)
        if len(mutable) <= 2:
            return code
        lines = code.split('\n')
        del_idx = random.choice(mutable)
        del lines[del_idx]
        return '\n'.join(lines)
    
    def _mut_swap_instructions(self, code: str) -> str:
        mutable = self._get_mutable_lines(code)
        if len(mutable) < 2:
            return code
        lines = code.split('\n')
        idx1, idx2 = random.sample(mutable, 2)
        lines[idx1], lines[idx2] = lines[idx2], lines[idx1]
        return '\n'.join(lines)
    
    def _mut_duplicate_instruction(self, code: str) -> str:
        mutable = self._get_mutable_lines(code)
        if not mutable:
            return code
        lines = code.split('\n')
        dup_idx = random.choice(mutable)
        dup_line = re.sub(r'^\w+\s+', '        ', lines[dup_idx])
        lines.insert(dup_idx + 1, dup_line)
        return '\n'.join(lines)
    
    def _mut_inject_bomber_loop(self, code: str) -> str:
        step = self._get_learned_or_good_step()
        bomber = f"\nbomb_inj DAT.F   #0, #0\nbloop   ADD.AB  #{step}, $bomb_inj\n        MOV.AB  #0, @bomb_inj\n        JMP.A   $bloop\n"
        return re.sub(r'(\s*END\s*)', bomber + r'\1', code, flags=re.IGNORECASE)
    
    def _mut_inject_spl_fan(self, code: str) -> str:
        fan = "\n        SPL.A   $0, $0\n        SPL.A   $0, $0\n"
        return re.sub(r'(ORG\s+\w+\s*\n)', r'\1' + fan, code, flags=re.IGNORECASE)
    
    def _mut_inject_quickscan(self, code: str) -> str:
        q1 = random.randint(500, 3000)
        q2 = q1 + random.randint(200, 500)
        qscan = f"\n        SNE.I   ${q1}, ${q2}\n        JMP.A   $qhit\n        JMP.A   $qmiss\nqhit    MOV.I   $qbomb, @${q1}\nqmiss   ; continue\nqbomb   DAT.F   #0, #0\n"
        return re.sub(r'(ORG\s+\w+\s*\n)', r'\1' + qscan, code, flags=re.IGNORECASE)
    
    def _mut_add_decoy(self, code: str) -> str:
        decoy_type = random.choice(list(self.DECOY_COMPONENTS.keys()))
        decoy = self.DECOY_COMPONENTS[decoy_type]
        return re.sub(r'(\s*END\s*)', decoy + r'\1', code, flags=re.IGNORECASE)
    
    def _mut_add_boot(self, code: str) -> str:
        if 'boot' in code.lower():
            return code
        dist = random.choice(self.BOOT_DISTANCES)
        boot = f"\nboot    MOV.I   $code_start, ${dist}\n        MOV.I   $code_start+1, ${dist}+1\n        JMP.A   ${dist}\ncode_start\n"
        return re.sub(r'(ORG\s+)', boot + r'\nORG boot\n; Original: ', code, flags=re.IGNORECASE)
    
    def _mut_optimize_paper(self, code: str) -> str:
        wtype = self._detect_warrior_type(code)
        if wtype not in ['paper', 'silk', 'replicator']:
            return code
        good_paper_steps = [2667, 5334, 1471, 2500, 3333]
        if self.learned_steps:
            good_paper_steps.extend(list(self.learned_steps)[:10])
        new_step = random.choice(good_paper_steps)
        return re.sub(r'(ADD\.AB\s+#)(\d+)', rf'\g<1>{new_step}', code, count=1, flags=re.IGNORECASE)
    
    def _mut_optimize_stone(self, code: str) -> str:
        wtype = self._detect_warrior_type(code)
        if wtype not in ['stone', 'bomber']:
            return code
        good_stone_steps = [3044, 2365, 5363, 3039, 2377]
        if self.learned_steps:
            good_stone_steps.extend(list(self.learned_steps)[:10])
        new_step = random.choice(good_stone_steps)
        return re.sub(r'(ADD\.AB\s+#)(\d+)', rf'\g<1>{new_step}', code, count=1, flags=re.IGNORECASE)
    
    # ==================== MAIN MUTATION INTERFACE ====================
    
    async def mutate_warrior_async(self, exp_warriors: List[ExplicitWarrior], n_responses: int = 1) -> np.ndarray:
        """Mutate warriors with validation and optional crossover."""
        result = [[None for _ in range(n_responses)] for _ in range(len(exp_warriors))]
        
        for i, parent in enumerate(exp_warriors):
            for j in range(n_responses):
                parent_code = getattr(parent, 'code', '') or getattr(parent, 'llm_response', '')
                
                for attempt in range(self.MAX_RETRIES):
                    crossover_prob = 0.05 if self.warmstart_mode else 0.10
                    if len(exp_warriors) > 1 and random.random() < crossover_prob:
                        other = random.choice([w for w in exp_warriors if w != parent])
                        other_code = getattr(other, 'code', '') or getattr(other, 'llm_response', '')
                        code = self._crossover(parent_code, other_code)
                    else:
                        code = parent_code
                    
                    warrior_hash = self.get_warrior_hash(code)
                    
                    if self.warmstart_mode:
                        n_mutations = random.choices([1, 2], weights=[0.8, 0.2])[0]
                    else:
                        n_mutations = random.randint(1, 3)
                    
                    for _ in range(n_mutations):
                        code = self._mutate_code(code, warrior_hash=warrior_hash)
                    
                    valid, error = self._validate_warrior(code)
                    if valid:
                        break
                
                name_match = re.search(r';name\s+(.+)', code)
                if name_match:
                    old_name = name_match.group(1).strip()
                    if re.search(r'_v(\d+)$', old_name):
                        new_name = re.sub(r'_v(\d+)$', lambda m: f'_v{int(m.group(1))+1}', old_name)
                    else:
                        new_name = f"{old_name}_v{random.randint(2,99)}"
                    code = code.replace(f";name {old_name}", f";name {new_name}")
                
                exp_warrior = self._to_explicit_warrior(code, parent_id=parent.id if hasattr(parent, 'id') else None)
                result[i][j] = exp_warrior
        
        self.all_generations.append(("mutate_warrior", result))
        return np.array(result, dtype=object)
    
    # ==================== DIAGNOSTICS ====================
    
    def get_mutation_stats(self) -> Dict[str, Dict]:
        """Get statistics about mutation effectiveness."""
        stats = {}
        for mutation_name in self.base_mutation_weights.keys():
            stats[mutation_name] = {
                'base_weight': self.base_mutation_weights.get(mutation_name, 0),
                'success_rate': self.mutation_tracker.get_success_rate(mutation_name),
                'avg_improvement': self.mutation_tracker.get_avg_improvement(mutation_name),
                'sample_count': self.mutation_tracker.get_sample_count(mutation_name),
            }
        return stats
    
    def print_mutation_stats(self):
        """Print mutation effectiveness statistics."""
        stats = self.get_mutation_stats()
        print("\n" + "=" * 75)
        print("MUTATION EFFECTIVENESS STATISTICS")
        print("=" * 75)
        print(f"{'Mutation':<28} {'Base%':>7} {'Success%':>9} {'AvgΔFit':>10} {'Samples':>8}")
        print("-" * 75)
        
        sorted_stats = sorted(stats.items(), key=lambda x: (x[1]['sample_count'] > 0, x[1]['success_rate']), reverse=True)
        
        for name, s in sorted_stats:
            base_pct = s['base_weight'] * 100
            success_pct = s['success_rate'] * 100
            if s['sample_count'] > 0:
                print(f"{name:<28} {base_pct:>6.1f}% {success_pct:>8.1f}% {s['avg_improvement']:>+10.4f} {s['sample_count']:>8}")
            else:
                print(f"{name:<28} {base_pct:>6.1f}%      N/A        N/A        0")
        
        print("=" * 75)
        total_samples = sum(s['sample_count'] for s in stats.values())
        tracked = sum(1 for s in stats.values() if s['sample_count'] > 0)
        print(f"Total mutations tracked: {total_samples}")
        print(f"Mutations with data: {tracked}/{len(stats)}")
        print(f"Directional tuner entries: {len(self.directional_tuner.momentum)}")
        print(f"Learned steps: {len(self.learned_steps)}")