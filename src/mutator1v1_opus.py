"""
Explicit Core War Mutator v2 - 1v1 Optimized with Adaptive Learning

Improvements over v1:
1. Mutation Success Tracking - Learn which mutations actually improve fitness
2. Type-Aware Mutation Selection - Apply warrior-type-specific mutation weights  
3. Directional Hill Climbing - Remember successful directions for constants
4. Protected Regions - Don't mutate critical code structures

Design Philosophy for 1v1:
=========================
Multi-warrior rewards SURVIVAL - papers with many SPL processes excel.
1v1 rewards KILLING - ties give only 1 point vs 3 for wins.

In warmstart, we have already-good warriors. We need REFINEMENT, not exploration.
This means: fewer mutations, smaller changes, preserve what works.

Key 1v1 Features:
================
1. KILL-FOCUSED: Add finishing moves, core clears, kill loops
2. ANTI-TIE: Ensure opponent is fully destroyed, not just damaged
3. FINE-TUNING: Small constant changes (+/-1 to +/-5) for hill climbing  
4. CONSERVATIVE: Usually 1 mutation, preserve proven structure
5. ARCHIVE LEARNING: Extract successful constants from good warriors

1v1 Scoring: Win=3, Tie=1, Loss=0
A tie is only 33% as valuable as a win - we must actively KILL!
"""

import re
import random
import numpy as np
import hashlib
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Set
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
        tracker.record_mutation(warrior_hash, "fine_tune_step")
        # After fitness evaluation:
        tracker.record_fitness_result(warrior_hash, old_fitness, new_fitness)
        # Get adaptive weights:
        weights = tracker.get_adaptive_weights(base_weights)
    """
    
    def __init__(self, window_size: int = 100):
        self.history: Dict[str, List[float]] = defaultdict(list)
        self.window_size = window_size
        self.pending: Dict[str, List[str]] = {}
    
    def record_mutation(self, warrior_hash: str, mutation_name: str):
        """Record that a mutation was applied."""
        if warrior_hash not in self.pending:
            self.pending[warrior_hash] = []
        self.pending[warrior_hash].append(mutation_name)
    
    def record_fitness_result(self, warrior_hash: str, old_fitness: float, new_fitness: float):
        """Record fitness result after evaluation."""
        if warrior_hash not in self.pending:
            return
        
        mutations = self.pending.pop(warrior_hash)
        delta = new_fitness - old_fitness
        
        for mutation_name in mutations:
            self.history[mutation_name].append(delta)
            if len(self.history[mutation_name]) > self.window_size:
                self.history[mutation_name] = self.history[mutation_name][-self.window_size:]
    
    def get_success_rate(self, mutation_name: str) -> float:
        """Fraction of times this mutation improved fitness."""
        hist = self.history.get(mutation_name, [])
        if len(hist) < 5:
            return 0.5
        return sum(1 for d in hist if d > 0) / len(hist)
    
    def get_avg_improvement(self, mutation_name: str) -> float:
        """Average fitness change from this mutation."""
        hist = self.history.get(mutation_name, [])
        return sum(hist) / len(hist) if hist else 0.0
    
    def get_sample_count(self, mutation_name: str) -> int:
        return len(self.history.get(mutation_name, []))
    
    def get_adaptive_weights(self, base_weights: Dict[str, float]) -> Dict[str, float]:
        """Adjust weights based on observed success rates (0.3x to 1.7x)."""
        adapted = {}
        for name, base_weight in base_weights.items():
            success_rate = self.get_success_rate(name)
            multiplier = 0.3 + success_rate * 1.4
            adapted[name] = base_weight * multiplier
        
        total = sum(adapted.values())
        return {k: v / total for k, v in adapted.items()} if total > 0 else base_weights


# ============================================================================
# IMPROVEMENT 3: DIRECTIONAL HILL CLIMBING
# ============================================================================

class DirectionalTuner:
    """
    Remember which direction improved fitness for each constant location.
    If +1 improved last time, bias toward +1/+2 next time.
    """
    
    def __init__(self, max_entries: int = 1000):
        self.momentum: Dict[Tuple[str, str], int] = {}
        self.max_entries = max_entries
    
    def _get_structure_hash(self, code: str) -> str:
        opcodes = re.findall(
            r'\b(MOV|ADD|SUB|MUL|DIV|MOD|JMP|JMZ|JMN|DJN|CMP|SEQ|SNE|SLT|SPL|DAT|NOP)\b',
            code, re.IGNORECASE
        )
        return hashlib.md5(''.join(opcodes).upper().encode()).hexdigest()[:8]
    
    def _get_context(self, code: str, match_start: int) -> str:
        line_start = code.rfind('\n', 0, match_start) + 1
        line_end = code.find('\n', match_start)
        if line_end == -1:
            line_end = len(code)
        line = code[line_start:line_end]
        
        op_match = re.search(r'\b(MOV|ADD|SUB|JMP|JMZ|JMN|DJN|SPL|DAT|SNE|SEQ|CMP)\b', line, re.IGNORECASE)
        opcode = op_match.group(1).upper() if op_match else 'UNK'
        
        pos_in_line = match_start - line_start
        comma_pos = line.find(',')
        field = 'A' if comma_pos == -1 or pos_in_line < comma_pos else 'B'
        
        return f"{opcode}_{field}"
    
    def get_biased_delta(self, code: str, match_start: int, base_deltas: List[int]) -> int:
        """Get delta with directional bias (70% follow momentum)."""
        struct_hash = self._get_structure_hash(code)
        context = self._get_context(code, match_start)
        key = (struct_hash, context)
        
        if key in self.momentum and random.random() < 0.7:
            direction = self.momentum[key]
            biased = [d for d in base_deltas if d * direction > 0]
            if biased:
                return random.choice(biased)
        
        return random.choice(base_deltas)
    
    def record_result(self, code: str, match_start: int, delta: int, improved: bool):
        """Record whether the delta improved fitness."""
        struct_hash = self._get_structure_hash(code)
        context = self._get_context(code, match_start)
        key = (struct_hash, context)
        
        if delta != 0:
            self.momentum[key] = (1 if delta > 0 else -1) if improved else (-1 if delta > 0 else 1)
        
        if len(self.momentum) > self.max_entries:
            items = list(self.momentum.items())
            self.momentum = dict(items[-(self.max_entries // 2):])


# ============================================================================
# MAIN MUTATOR CLASS
# ============================================================================

class ExplicitCoreMutator:
    """
    Mutator optimized for 1v1 hill scoring with adaptive learning.
    
    Improvements over v1:
    1. Tracks mutation success and adapts weights
    2. Type-specific mutation weights (stone vs scanner vs hybrid)
    3. Directional hill climbing for fine-tuning
    4. Protects critical code regions
    
    Key 1v1 features:
    - Kill-focused mutations (core clear, finishing moves)
    - Anti-tie strategies
    - Fine-grained constant tuning
    - Conservative mutations for warmstart
    """
    
    CORE_SIZE = 8000
    MAX_RETRIES = 5
    
    # Proven step sizes
    GOOD_STEPS = [
        3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47,
        53, 59, 61, 67, 71, 73, 79, 83, 89, 97,
        2339, 2663, 2671, 3037, 3041, 3049, 3061, 3067, 3079, 3083,
        3089, 3109, 3119, 3121, 3137, 3163, 3167, 3169, 3181, 3187,
        3191, 3203, 3209, 3217, 3221, 3229, 3251, 3253, 3257, 3259,
        3271, 3299, 3301, 3307, 3313, 3319, 3323, 3329, 3331, 3343,
        3347, 3359, 3361, 3371, 3373, 3389, 3391, 3407, 3413, 3433,
        3449, 3457, 3461, 3463, 3467, 3469, 3491, 3499, 3511, 3517,
        3527, 3529, 3533, 3539, 3541, 3547, 3557, 3559, 3571,
        2667, 5334, 1471, 1777, 3044, 4001, 4003, 4007,
        2000, 2666, 4000, 1333, 1600, 800, 400,
    ]
    
    FINE_DELTAS = [-2, -1, 1, 2]
    SMALL_DELTAS = [-5, -3, 3, 5]
    MEDIUM_DELTAS = [-20, -10, 10, 20]
    
    BOOT_DISTANCES = [100, 200, 500, 800, 1000, 1500, 2000, 2500, 3000, 3500, 4000]
    
    # ==================== 1v1 OPTIMIZED TEMPLATES ====================
    
    TEMPLATES = {
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
    
    # ==================== TYPE SIGNATURES ====================
    
    TYPE_SIGNATURES = {
        'paper': ['SPL.A $0', 'SPL.A   $0'],
        'stone': ['MOV.I', '@ptr', '@0', 'bomb'],
        'scanner': ['SNE', 'SEQ', 'JMZ', 'sptr', 'scan'],
        'bomber': ['ADD.AB', 'MOV.AB', 'bomb'],
        'clear': ['DJN', '>ptr', '>cptr', 'ccnt'],
        'vampire': ['fang', 'pit', 'JMZ'],
        'hybrid': ['SPL', 'JMP.A $bomb', 'boot'],
    }
    
    # ==================== IMPROVEMENT 2: TYPE-AWARE WEIGHTS ====================
    
    TYPE_MUTATION_ADJUSTMENTS = {
        'stone': {
            'fine_tune_step': 2.5,       # Step is critical for stones
            'good_step_replace': 2.0,
            'add_core_clear': 1.5,       # Helps kill
            'insert_instruction': 0.3,   # Don't mess with tight loop
            'delete_instruction': 0.2,
        },
        'scanner': {
            'fine_tune_step': 2.0,
            'tune_scan_position': 2.5,   # Critical for scanners
            'tune_loop_count': 2.0,      # Scan duration matters
            'add_kill_loop': 2.0,        # Ensure kills after hit
            'increase_attack_range': 1.5,
            'inject_scanner': 0.3,       # Already a scanner
        },
        'paper': {
            'fine_tune_step': 1.5,
            'add_core_clear': 2.0,       # Papers need killing power
            'increase_attack_range': 1.5,
            'delete_instruction': 0.2,   # Preserve replication
        },
        'bomber': {
            'fine_tune_step': 2.5,
            'good_step_replace': 2.0,
            'add_core_clear': 1.5,
            'increase_attack_range': 1.5,
        },
        'clear': {
            'tune_loop_count': 2.5,      # Clear count is critical
            'fine_tune_step': 2.0,
            'increase_attack_range': 2.0,
            'add_core_clear': 0.5,       # Already a clear
        },
        'vampire': {
            'fine_tune_step': 2.0,
            'fine_tune_constant': 1.5,
            'add_kill_loop': 0.5,        # Vampires convert, not kill
            'delete_instruction': 0.2,
        },
        'hybrid': {
            'fine_tune_step': 2.0,
            'add_core_clear': 1.5,
            'tune_loop_count': 1.5,
            'increase_attack_range': 1.5,
        },
    }
    
    def __init__(self, environment=None, warmstart_mode: bool = True,
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
        self.learned_steps: Set[int] = set(self.GOOD_STEPS)
        self.learned_scan_positions: Set[int] = set()
        self.learned_loop_counts: Set[int] = set()
        
        # IMPROVEMENT 1: Mutation success tracking
        self.mutation_tracker = MutationTracker(window_size=100)
        
        # IMPROVEMENT 3: Directional hill climbing
        self.directional_tuner = DirectionalTuner(max_entries=1000)
        
        # Track pending deltas for feedback
        self._pending_deltas: Dict[str, List[Tuple[str, int, int]]] = {}
        
        # Base mutation weights
        if warmstart_mode:
            self.base_mutation_weights = {
                # Fine-tuning (HIGH WEIGHT)
                'fine_tune_step': 0.25,
                'fine_tune_constant': 0.15,
                'tune_loop_count': 0.08,
                'tune_scan_position': 0.06,
                
                # Attack enhancement (MEDIUM WEIGHT)
                'add_core_clear': 0.08,
                'increase_attack_range': 0.06,
                'add_kill_loop': 0.05,
                
                # Conservative structure changes (LOW WEIGHT)
                'change_modifier': 0.05,
                'change_addr_mode': 0.04,
                'swap_adjacent': 0.03,
                
                # Defensive
                'add_imp_gate': 0.03,
                
                # Exploration (VERY LOW)
                'change_opcode': 0.02,
                'good_step_replace': 0.04,
                'archive_step': 0.04,
                
                # Rarely used
                'insert_instruction': 0.01,
                'delete_instruction': 0.01,
            }
        else:
            self.base_mutation_weights = {
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
        
        # 4. Protect vampire pits and kill loops
        for i, line in enumerate(lines):
            if re.search(r'^\s*(pit|fang|kcnt|kill)\s+', line, re.IGNORECASE):
                protected.append((i, min(len(lines), i + 2)))
        
        # 5. Protect core clear counters (critical for anti-tie)
        for i, line in enumerate(lines):
            if re.search(r'^\s*(ccnt|cptr)\s+', line, re.IGNORECASE):
                protected.append((i, min(len(lines), i + 1)))
        
        return protected
    
    def _is_line_protected(self, line_idx: int, protected: List[Tuple[int, int]]) -> bool:
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
        """Detect warrior strategy type."""
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
            
            for match in re.finditer(r'cnt\s+DAT\.F\s+#(\d+)', code, re.IGNORECASE):
                count = int(match.group(1))
                if 5 < count < 1000:
                    self.learned_loop_counts.add(count)
            
            for match in re.finditer(r'sptr\s+DAT\.F\s+#(\d+)', code, re.IGNORECASE):
                pos = int(match.group(1))
                if 100 < pos < 7000:
                    self.learned_scan_positions.add(pos)
    
    def _get_learned_step(self) -> int:
        if self.learned_steps and random.random() < 0.6:
            return random.choice(list(self.learned_steps))
        return random.choice(self.GOOD_STEPS)
    
    # ==================== PARSING & VALIDATION ====================
    
    def _parse_warrior(self, code: str) -> Tuple[Optional[Warrior], Optional[str]]:
        try:
            code_clean = re.sub(r"```.*", "", code)
            warrior = redcode.parse(code_clean.split("\n"), self.environment)
            return warrior, None
        except Exception as e:
            return None, str(e)
    
    def _validate_warrior(self, code: str) -> Tuple[bool, Optional[str]]:
        warrior, error = self._parse_warrior(code)
        if warrior is None:
            return False, error
        if len(warrior.instructions) < 1:
            return False, "No instructions"
        if len(warrior.instructions) > 100:
            return False, "Too many instructions"
        return True, None
    
    def _to_explicit_warrior(self, code: str, parent_id: str = None) -> ExplicitWarrior:
        warrior, error = self._parse_warrior(code)
        return ExplicitWarrior(
            code=code,
            warrior=warrior,
            error=error,
            id=hashlib.sha256(code.encode()).hexdigest(),
            parent_id=parent_id
        )
    
    # ==================== NEW WARRIOR GENERATION ====================
    
    def _fill_template(self, template: str) -> str:
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
        
        base = random.randint(100, 2000)
        gap = random.choice([500, 800, 1000, 1200, 1500])
        for i in range(1, 9):
            params[f'q{i}'] = str((base + (i-1) * gap) % self.CORE_SIZE)
        
        code = template
        for key, value in params.items():
            code = code.replace('{' + key + '}', str(value))
        return code
    
    def _generate_new_warrior(self) -> str:
        weights = {
            'stone': 0.18, 'scanner': 0.15, 'quickscanner': 0.12,
            'two_phase': 0.10, 'paper_attack': 0.10, 'silk': 0.08,
            'dwarf': 0.08, 'clear': 0.06, 'gate_stone': 0.05,
            'oneshot': 0.04, 'vampire': 0.04,
        }
        template_name = random.choices(list(weights.keys()), weights=list(weights.values()))[0]
        return self._fill_template(self.TEMPLATES[template_name])
    
    async def new_warrior_async(self, n_warriors: int = 1, n_responses: int = 1) -> np.ndarray:
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
    
    # ==================== MUTATION DISPATCH ====================
    
    def _mutate_code(self, code: str, warrior_hash: str = None) -> str:
        """Apply a mutation based on effective weights."""
        weights = self._get_effective_weights(code)
        mutations = list(weights.keys())
        probs = list(weights.values())
        mutation_name = random.choices(mutations, weights=probs)[0]
        
        if warrior_hash:
            self.mutation_tracker.record_mutation(warrior_hash, mutation_name)
        
        method = getattr(self, f'_mut_{mutation_name}', None)
        if method is None:
            return code
        
        # Some mutations need warrior_hash for directional tracking
        if mutation_name in ('fine_tune_step', 'fine_tune_constant'):
            return method(code, warrior_hash)
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
            valid_matches = matches  # Fallback
        
        match = random.choice(valid_matches)
        old_val = int(match.group(2))
        
        # IMPROVEMENT 3: Get directionally-biased delta
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
            valid_matches = matches
        
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
        """Tune loop counter for attack duration."""
        pattern = r'(cnt\s+DAT\.F\s+#)(\d+)(,\s*#)(\d+)'
        matches = list(re.finditer(pattern, code, re.IGNORECASE))
        if not matches:
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
        
        return code[:match.start(2)] + str(new_val) + code[match.end(2):match.start(4)] + str(new_val) + code[match.end(4):]
    
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
    
    # ==================== ATTACK ENHANCEMENT MUTATIONS ====================
    
    def _mut_add_core_clear(self, code: str) -> str:
        """Add core clear as finishing move (anti-tie)."""
        if 'clear' in code.lower() or 'ccnt' in code.lower() or '>cptr' in code.lower():
            return code
        
        clear = """
        ; Core clear finishing move
clr     MOV.I   $cdat, >cptr
        DJN.F   $clr, $ccnt
        JMP.A   $start
cdat    DAT.F   #0, #0
cptr    DAT.F   #-1, #0
ccnt    DAT.F   #150, #150
"""
        return re.sub(r'(\s*END\s*)', clear + r'\1', code, flags=re.IGNORECASE)
    
    def _mut_add_kill_loop(self, code: str) -> str:
        """Add kill loop after scanner hit."""
        if 'kill' in code.lower() or 'kcnt' in code.lower():
            return code
        if 'found' not in code.lower() and 'qhit' not in code.lower():
            return code
        
        kill = """
        ; Kill loop
        SUB.AB  #5, $kptr
        MOV.I   $kdat, @kptr
        DJN.F   $-2, $kcnt
kdat    DAT.F   #0, #0
kptr    DAT.F   #0, #0
kcnt    DAT.F   #20, #20
"""
        return re.sub(r'(\s*END\s*)', kill + r'\1', code, flags=re.IGNORECASE)
    
    def _mut_increase_attack_range(self, code: str) -> str:
        """Increase attack range by boosting loop counts."""
        pattern = r'(#)(\d{2,3})(,\s*#)(\d{2,3})'
        matches = list(re.finditer(pattern, code))
        if not matches:
            return code
        
        match = random.choice(matches)
        old_val = int(match.group(2))
        increase = random.randint(20, 50)
        new_val = min(999, old_val + increase)
        
        return code[:match.start(2)] + str(new_val) + code[match.end(2):match.start(4)] + str(new_val) + code[match.end(4):]
    
    # ==================== DEFENSIVE MUTATIONS ====================
    
    def _mut_add_imp_gate(self, code: str) -> str:
        """Add imp gate for defense."""
        if 'gate' in code.lower() or '<gate' in code.lower():
            return code
        gate = "gate    SPL.A   $0, <gate-4\n"
        return re.sub(r'(ORG\s+\w+\s*\n)', r'\1' + gate, code, flags=re.IGNORECASE)
    
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
        return re.sub(r'(ORG\s+\w+\s*\n)', r'\1' + scanner, code, flags=re.IGNORECASE)
    
    # ==================== STANDARD MUTATIONS ====================
    
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
    
    def _mut_insert_instruction(self, code: str) -> str:
        mutable = self._get_mutable_lines(code)
        if not mutable:
            return code
        lines = code.split('\n')
        insert_idx = random.choice(mutable)
        
        useful_instrs = [
            "MOV.I   $dat, @ptr",
            f"ADD.AB  #{random.choice(self.GOOD_STEPS[:20])}, $ptr",
            "DJN.F   $-1, $cnt",
        ]
        new_line = "        " + random.choice(useful_instrs)
        lines.insert(insert_idx, new_line)
        return '\n'.join(lines)
    
    def _mut_delete_instruction(self, code: str) -> str:
        """Delete a random instruction (careful - preserve structure)."""
        lines = code.split('\n')
        deletable = []
        for i, l in enumerate(lines):
            stripped = l.strip()
            if not stripped or stripped.startswith(';'):
                continue
            if stripped.upper().startswith(('ORG', 'END', 'EQU')):
                continue
            if re.match(r'^\w+\s+', stripped):  # Has label
                continue
            deletable.append(i)
        
        if len(deletable) <= 1:
            return code
        
        del_idx = random.choice(deletable)
        del lines[del_idx]
        return '\n'.join(lines)
    
    # ==================== MAIN MUTATION INTERFACE ====================
    
    async def mutate_warrior_async(self, exp_warriors: List[ExplicitWarrior], n_responses: int = 1) -> np.ndarray:
        """Mutate warriors with conservative approach for warmstart."""
        result = [[None for _ in range(n_responses)] for _ in range(len(exp_warriors))]
        
        for i, parent in enumerate(exp_warriors):
            for j in range(n_responses):
                parent_code = getattr(parent, 'code', '') or getattr(parent, 'llm_response', '')
                
                for attempt in range(self.MAX_RETRIES):
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
        print("1v1 MUTATION EFFECTIVENESS STATISTICS")
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