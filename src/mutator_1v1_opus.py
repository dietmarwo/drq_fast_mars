"""
Explicit Core War Mutator v2 - 1v1 Optimized with Full Adaptive Learning

All 9 improvements implemented:
1. Mutation Success Tracking - Learn which mutations actually improve fitness
2. Type-Aware Mutation Selection - Apply warrior-type-specific mutation weights
3. Directional Hill Climbing - Remember successful directions for constants
4. Protected Regions - Don't mutate critical code structures
5. Contextual Pattern Learning - Learn successful patterns, not just constants
6. Adaptive Mutation Count - Adjust mutations based on fitness progress
7. Component-based Crossover - Respect component boundaries in crossover
8. New Tactical Mutations - Additional strategic mutations (kill-focused)
9. Number-Theoretic Steps - Generate mathematically optimal step sizes

1v1 Design Philosophy:
=====================
Multi-warrior rewards SURVIVAL - papers with many SPL processes excel.
1v1 rewards KILLING - ties give only 1 point vs 3 for wins.

Key 1v1 Features:
- KILL-FOCUSED: Add finishing moves, core clears, kill loops
- ANTI-TIE: Ensure opponent is fully destroyed
- FINE-TUNING: Small constant changes for hill climbing
- CONSERVATIVE: Preserve proven structure in warmstart

1v1 Scoring: Win=3, Tie=1, Loss=0
"""

import re
import random
import numpy as np
import hashlib
import math
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Set
from collections import defaultdict
from corewar import redcode, Warrior

from util import ExplicitWarrior


# ============================================================================
# IMPROVEMENT 1: MUTATION SUCCESS TRACKING
# ============================================================================

class MutationTracker:
    """Track which mutations actually improve fitness."""

    def __init__(self, window_size: int = 100):
        self.history: Dict[str, List[float]] = defaultdict(list)
        self.window_size = window_size
        self.pending: Dict[str, List[str]] = {}

    def record_mutation(self, warrior_hash: str, mutation_name: str):
        if warrior_hash not in self.pending:
            self.pending[warrior_hash] = []
        self.pending[warrior_hash].append(mutation_name)

    def record_fitness_result(self, warrior_hash: str, old_fitness: float, new_fitness: float):
        if warrior_hash not in self.pending:
            return
        mutations = self.pending.pop(warrior_hash)
        delta = new_fitness - old_fitness
        for mutation_name in mutations:
            self.history[mutation_name].append(delta)
            if len(self.history[mutation_name]) > self.window_size:
                self.history[mutation_name] = self.history[mutation_name][-self.window_size:]

    def get_success_rate(self, mutation_name: str) -> float:
        hist = self.history.get(mutation_name, [])
        if len(hist) < 5:
            return 0.5
        return sum(1 for d in hist if d > 0) / len(hist)

    def get_avg_improvement(self, mutation_name: str) -> float:
        hist = self.history.get(mutation_name, [])
        return sum(hist) / len(hist) if hist else 0.0

    def get_sample_count(self, mutation_name: str) -> int:
        return len(self.history.get(mutation_name, []))

    def get_adaptive_weights(self, base_weights: Dict[str, float]) -> Dict[str, float]:
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
    """Remember which direction improved fitness for each constant location."""

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
        struct_hash = self._get_structure_hash(code)
        context = self._get_context(code, match_start)
        key = (struct_hash, context)
        if delta != 0:
            self.momentum[key] = (1 if delta > 0 else -1) if improved else (-1 if delta > 0 else 1)
        if len(self.momentum) > self.max_entries:
            items = list(self.momentum.items())
            self.momentum = dict(items[-(self.max_entries // 2):])


# ============================================================================
# IMPROVEMENT 5: CONTEXTUAL PATTERN LEARNING
# ============================================================================

class PatternLibrary:
    """Learn successful instruction patterns for 1v1 combat."""

    def __init__(self, max_patterns: int = 100):
        self.max_patterns = max_patterns
        self.attack_patterns: Dict[str, Tuple[int, float]] = {}
        self.kill_patterns: Dict[str, Tuple[int, float]] = {}  # 1v1 specific
        self.step_count_pairs: Dict[Tuple[int, int], Tuple[int, float]] = {}
        self.steps_by_type: Dict[str, List[Tuple[int, float]]] = defaultdict(list)

    def _normalize_pattern(self, pattern: str) -> str:
        normalized = re.sub(r'#\d+', '#N', pattern)
        normalized = re.sub(r'\$-?\d+', '$N', normalized)
        normalized = re.sub(r'@-?\d+', '@N', normalized)
        return normalized.strip()

    def _update_pattern(self, patterns: Dict, pattern: str, fitness: float):
        if pattern in patterns:
            count, total = patterns[pattern]
            patterns[pattern] = (count + 1, total + fitness)
        else:
            patterns[pattern] = (1, fitness)
        if len(patterns) > self.max_patterns:
            sorted_patterns = sorted(patterns.items(), key=lambda x: x[1][1]/x[1][0])
            patterns.clear()
            for k, v in sorted_patterns[-(self.max_patterns//2):]:
                patterns[k] = v

    def extract_from_warrior(self, code: str, fitness: float, warrior_type: str):
        """Extract patterns from a successful warrior."""
        if fitness <= 0:
            return

        # Extract kill loop patterns (1v1 specific)
        kill_match = re.search(
            r'((?:MOV|SUB)[^\n]+@[^\n]+\n\s*(?:ADD|SUB)[^\n]+\n\s*DJN[^\n]+)',
            code, re.IGNORECASE
        )
        if kill_match:
            pattern = self._normalize_pattern(kill_match.group(1))
            self._update_pattern(self.kill_patterns, pattern, fitness)

        # Extract attack patterns
        attack_match = re.search(
            r'((?:MOV|ADD)[^\n]+\n\s*(?:ADD|MOV)[^\n]+\n\s*(?:JMP|DJN)[^\n]+)',
            code, re.IGNORECASE
        )
        if attack_match:
            pattern = self._normalize_pattern(attack_match.group(1))
            self._update_pattern(self.attack_patterns, pattern, fitness)

        # Extract step + count combinations
        step_match = re.search(r'ADD\.\w+\s+#(\d+)', code, re.IGNORECASE)
        count_match = re.search(r'cnt\s+DAT\.F\s+#(\d+)', code, re.IGNORECASE)
        if step_match:
            step = int(step_match.group(1))
            count = int(count_match.group(1)) if count_match else 100
            key = (step, count)
            if key in self.step_count_pairs:
                c, t = self.step_count_pairs[key]
                self.step_count_pairs[key] = (c + 1, t + fitness)
            else:
                self.step_count_pairs[key] = (1, fitness)

            self.steps_by_type[warrior_type].append((step, fitness))
            if len(self.steps_by_type[warrior_type]) > 50:
                self.steps_by_type[warrior_type].sort(key=lambda x: x[1], reverse=True)
                self.steps_by_type[warrior_type] = self.steps_by_type[warrior_type][:30]

    def get_best_step_for_type(self, warrior_type: str) -> Optional[int]:
        """Get a proven step size for this warrior type."""
        steps = self.steps_by_type.get(warrior_type, [])
        if steps:
            total_fitness = sum(f for _, f in steps)
            if total_fitness > 0:
                r = random.random() * total_fitness
                cumsum = 0
                for step, fitness in steps:
                    cumsum += fitness
                    if cumsum >= r:
                        return step
            return random.choice(steps)[0]
        return None

    def get_compatible_count(self, step: int) -> Optional[int]:
        """Get a loop count that worked well with this step."""
        compatible = [(k[1], v[1]/v[0]) for k, v in self.step_count_pairs.items()
                      if k[0] == step and v[0] >= 2]
        if compatible:
            counts, scores = zip(*compatible)
            return random.choices(counts, weights=scores)[0]
        return None


# ============================================================================
# IMPROVEMENT 9: NUMBER-THEORETIC STEP GENERATION
# ============================================================================

class StepOptimizer:
    """Generate step sizes with good mathematical properties for 1v1."""

    CORE_SIZE = 8000

    # Pre-computed good primes for fast coverage
    GOOD_PRIMES = [
        2339, 2341, 2347, 2351, 2357, 2371, 2377, 2381, 2383, 2389,
        2663, 2671, 2677, 2683, 2687, 2689, 2693, 2699, 2707, 2711,
        3037, 3041, 3049, 3061, 3067, 3079, 3083, 3089, 3109, 3119,
        3121, 3137, 3163, 3167, 3169, 3181, 3187, 3191, 3203, 3209,
    ]

    # Small primes for tight kill loops
    KILL_PRIMES = [3, 5, 7, 11, 13, 17, 19, 23, 29, 31]

    @staticmethod
    def gcd(a: int, b: int) -> int:
        while b:
            a, b = b, a % b
        return a

    @classmethod
    def is_coprime(cls, a: int, b: int = None) -> bool:
        if b is None:
            b = cls.CORE_SIZE
        return cls.gcd(a, b) == 1

    @classmethod
    def generate_optimal_step(cls, strategy: str = 'balanced') -> int:
        """
        Generate step size for 1v1 combat.

        Strategies:
        - 'fast': Large steps for quick initial coverage
        - 'kill': Small primes for thorough kill loops
        - 'scan': Medium steps for effective scanning
        - 'balanced': Mix of strategies
        """
        if strategy == 'fast':
            base = cls.CORE_SIZE // 3
            candidates = [base + d for d in range(-50, 51) if cls.is_coprime(base + d)]
        elif strategy == 'kill':
            return random.choice(cls.KILL_PRIMES)
        elif strategy == 'scan':
            base = cls.CORE_SIZE // 5
            candidates = [base + d for d in range(-30, 31) if cls.is_coprime(base + d)]
        else:
            if random.random() < 0.5:
                return cls.generate_optimal_step('fast')
            else:
                return random.choice(cls.GOOD_PRIMES)

        return random.choice(candidates) if candidates else 2667

    @classmethod
    def optimize_existing_step(cls, step: int, delta_range: int = 10) -> int:
        """Find the nearest coprime value."""
        if cls.is_coprime(step):
            return step
        for delta in range(1, delta_range + 1):
            if cls.is_coprime(step + delta):
                return step + delta
            if cls.is_coprime(step - delta):
                return step - delta
        return step


# ============================================================================
# MAIN MUTATOR CLASS
# ============================================================================

class ExplicitCoreMutator:
    """
    Mutator optimized for 1v1 combat with full adaptive learning (9 improvements).
    """

    CORE_SIZE = 8000
    MAX_RETRIES = 5

    GOOD_STEPS = [
        3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47,
        53, 59, 61, 67, 71, 73, 79, 83, 89, 97,
        2339, 2663, 2671, 3037, 3041, 3049, 3061, 3067, 3079, 3083,
        3089, 3109, 3119, 3121, 3137, 3163, 3167, 3169, 3181, 3187,
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
               'JMP', 'JMZ', 'JMN', 'DJN', 'CMP', 'SEQ', 'SNE', 'SLT', 'SPL', 'NOP']
    MODIFIERS = ['A', 'B', 'AB', 'BA', 'F', 'X', 'I']
    ADDR_MODES = ['#', '$', '*', '@', '{', '<', '}', '>']

    TYPE_SIGNATURES = {
        'paper': ['SPL.A $0', 'SPL.A   $0'],
        'stone': ['MOV.I', '@ptr', '@0', 'bomb'],
        'scanner': ['SNE', 'SEQ', 'JMZ', 'sptr', 'scan'],
        'bomber': ['ADD.AB', 'MOV.AB', 'bomb'],
        'clear': ['DJN', '>ptr', '>cptr', 'ccnt'],
        'vampire': ['fang', 'pit', 'JMZ'],
        'hybrid': ['SPL', 'JMP.A $bomb', 'boot'],
    }

    # 1v1-specific type adjustments (focus on killing)
    TYPE_MUTATION_ADJUSTMENTS = {
        'stone': {
            'fine_tune_step': 2.5, 'optimal_step': 2.0, 'add_core_clear': 1.5,
            'insert_instruction': 0.3, 'delete_instruction': 0.2,
        },
        'scanner': {
            'fine_tune_step': 2.0, 'tune_scan_position': 2.5, 'tune_loop_count': 2.0,
            'add_kill_loop': 2.0, 'increase_attack_range': 1.5, 'inject_scanner': 0.3,
        },
        'paper': {
            'fine_tune_step': 1.5, 'add_core_clear': 2.0, 'increase_attack_range': 1.5,
            'delete_instruction': 0.2,
        },
        'bomber': {
            'fine_tune_step': 2.5, 'optimal_step': 2.0, 'add_core_clear': 1.5,
            'unroll_loop': 1.5,
        },
        'clear': {
            'tune_loop_count': 2.5, 'fine_tune_step': 2.0, 'increase_attack_range': 2.0,
            'add_core_clear': 0.5,
        },
        'vampire': {
            'fine_tune_step': 2.0, 'fine_tune_constant': 1.5, 'add_kill_loop': 0.5,
            'delete_instruction': 0.2,
        },
        'hybrid': {
            'fine_tune_step': 2.0, 'add_core_clear': 1.5, 'tune_loop_count': 1.5,
        },
    }

    def __init__(self, environment=None, warmstart_mode: bool = True, use_adaptive_weights: bool = True):
        self.environment = environment or {}
        self.warmstart_mode = warmstart_mode
        self.use_adaptive_weights = use_adaptive_weights
        self.all_generations = []

        # Learned constants
        self.learned_steps: Set[int] = set(self.GOOD_STEPS)
        self.learned_scan_positions: Set[int] = set()
        self.learned_loop_counts: Set[int] = set()

        # Improvement 1: Mutation tracking
        self.mutation_tracker = MutationTracker(window_size=100)

        # Improvement 3: Directional hill climbing
        self.directional_tuner = DirectionalTuner(max_entries=1000)
        self._pending_deltas: Dict[str, List[Tuple[str, int, int]]] = {}

        # Improvement 5: Pattern learning
        self.pattern_library = PatternLibrary(max_patterns=100)

        # Improvement 6: Track fitness for adaptive mutation count
        self.max_fitness_seen: float = 0.0
        self.generation_count: int = 0

        # Base mutation weights for 1v1
        if warmstart_mode:
            self.base_mutation_weights = {
                # Fine-tuning (HIGH WEIGHT)
                'fine_tune_step': 0.20, 'fine_tune_constant': 0.12,
                'tune_loop_count': 0.06, 'tune_scan_position': 0.05,
                'pattern_step': 0.04, 'optimal_step': 0.04,

                # Attack enhancement (MEDIUM - 1v1 KILL FOCUS)
                'add_core_clear': 0.08, 'add_kill_loop': 0.06,
                'increase_attack_range': 0.05, 'add_finishing_move': 0.04,

                # Conservative changes
                'change_modifier': 0.04, 'change_addr_mode': 0.03, 'swap_adjacent': 0.02,
                'good_step_replace': 0.04, 'archive_step': 0.03,

                # Defensive
                'add_imp_gate': 0.03,

                # Tactical (NEW)
                'add_qscan_prefix': 0.02, 'unroll_loop': 0.02, 'optimize_scan': 0.02,

                # Exploration (LOW)
                'change_opcode': 0.02,
                'insert_instruction': 0.01, 'delete_instruction': 0.01,
            }
        else:
            self.base_mutation_weights = {
                'fine_tune_step': 0.10, 'fine_tune_constant': 0.08,
                'good_step_replace': 0.08, 'optimal_step': 0.04,
                'change_opcode': 0.05, 'change_modifier': 0.05, 'change_addr_mode': 0.06,
                'insert_instruction': 0.03, 'delete_instruction': 0.02, 'swap_adjacent': 0.02,
                'add_core_clear': 0.06, 'add_kill_loop': 0.06,
                'increase_attack_range': 0.05, 'add_finishing_move': 0.04,
                'tune_loop_count': 0.04, 'tune_scan_position': 0.04,
                'add_imp_gate': 0.04, 'inject_scanner': 0.04,
                'archive_step': 0.03, 'pattern_step': 0.03,
                'add_qscan_prefix': 0.03, 'unroll_loop': 0.02, 'optimize_scan': 0.02,
            }

    # ==================== FEEDBACK API ====================

    def get_warrior_hash(self, code: str) -> str:
        return hashlib.sha256(code.encode()).hexdigest()[:16]

    def record_fitness_feedback(self, warrior_hash: str, old_fitness: float, new_fitness: float):
        """Call after evaluation to enable adaptive learning."""
        self.mutation_tracker.record_fitness_result(warrior_hash, old_fitness, new_fitness)
        self.max_fitness_seen = max(self.max_fitness_seen, new_fitness)

        improved = new_fitness > old_fitness
        if warrior_hash in self._pending_deltas:
            for code, match_start, delta in self._pending_deltas[warrior_hash]:
                self.directional_tuner.record_result(code, match_start, delta, improved)
            del self._pending_deltas[warrior_hash]

    # ==================== IMPROVEMENT 4: PROTECTED REGIONS ====================

    def _identify_protected_regions(self, code: str) -> List[Tuple[int, int]]:
        protected = []
        lines = code.split('\n')

        # Boot sequences
        for i, line in enumerate(lines):
            if 'boot' in line.lower() and not line.strip().startswith(';'):
                boot_end = i + 1
                for j in range(i, min(i + 8, len(lines))):
                    if 'JMP' in lines[j].upper():
                        boot_end = j + 1
                        break
                protected.append((i, boot_end))
                break

        # SPL 0 gates
        for i, line in enumerate(lines):
            if re.search(r'\bSPL\.?\w*\s+\$?0\s*,', line, re.IGNORECASE):
                protected.append((max(0, i-1), min(len(lines), i+2)))

        # Scan loops
        in_scan = False
        scan_start = 0
        for i, line in enumerate(lines):
            if re.search(r'\b(JMZ|JMN|SNE|SEQ|CMP)\b', line, re.IGNORECASE) and not in_scan:
                in_scan = True
                scan_start = i
            elif in_scan and re.search(r'\bJMP\s+.*(-\d+|\$scan)', line, re.IGNORECASE):
                protected.append((scan_start, i + 1))
                in_scan = False

        # Vampire pits & kill loops
        for i, line in enumerate(lines):
            if re.search(r'^\s*(pit|fang|kcnt|kill|ccnt|cptr)\s+', line, re.IGNORECASE):
                protected.append((i, min(len(lines), i + 2)))

        return protected

    def _get_mutable_lines(self, code: str) -> List[int]:
        lines = code.split('\n')
        protected = self._identify_protected_regions(code)
        mutable = []
        for i, line in enumerate(lines):
            stripped = line.strip()
            if not stripped or stripped.startswith(';'):
                continue
            if stripped.upper().startswith(('ORG', 'END', 'EQU')):
                continue
            if any(start <= i < end for start, end in protected):
                continue
            mutable.append(i)
        return mutable

    # ==================== WEIGHT COMPUTATION ====================

    def _detect_warrior_type(self, code: str) -> str:
        code_upper = code.upper()
        scores = {wtype: sum(1 for sig in sigs if sig.upper() in code_upper)
                  for wtype, sigs in self.TYPE_SIGNATURES.items()}
        return max(scores, key=scores.get) if scores and max(scores.values()) > 0 else 'unknown'

    def _get_effective_weights(self, code: str) -> Dict[str, float]:
        weights = self.base_mutation_weights.copy()
        warrior_type = self._detect_warrior_type(code)
        if warrior_type in self.TYPE_MUTATION_ADJUSTMENTS:
            for mutation, mult in self.TYPE_MUTATION_ADJUSTMENTS[warrior_type].items():
                if mutation in weights:
                    weights[mutation] *= mult
        if self.use_adaptive_weights:
            weights = self.mutation_tracker.get_adaptive_weights(weights)
        else:
            total = sum(weights.values())
            weights = {k: v / total for k, v in weights.items()} if total > 0 else weights
        return weights

    # ==================== IMPROVEMENT 6: ADAPTIVE MUTATION COUNT ====================

    def _get_adaptive_mutation_count(self, fitness: float) -> int:
        """Adjust mutation count based on fitness - more conservative for high performers."""
        if self.warmstart_mode:
            if self.max_fitness_seen > 0:
                ratio = fitness / self.max_fitness_seen
                if ratio > 0.95:
                    return random.choices([1, 2], weights=[0.9, 0.1])[0]
                elif ratio > 0.8:
                    return random.choices([1, 2], weights=[0.7, 0.3])[0]
                else:
                    return random.choices([1, 2, 3], weights=[0.5, 0.3, 0.2])[0]
            return random.choices([1, 2], weights=[0.8, 0.2])[0]
        else:
            return random.randint(1, 3)

    # ==================== ARCHIVE & PATTERN LEARNING ====================

    def learn_from_archive(self, archive: Dict):
        """Extract successful patterns from archive."""
        for bc, warrior in archive.items():
            code = getattr(warrior, 'code', None) or getattr(warrior, 'llm_response', '')
            fitness = getattr(warrior, 'fitness', 0)
            if not code or not fitness or fitness <= 0:
                continue

            warrior_type = self._detect_warrior_type(code)
            self.pattern_library.extract_from_warrior(code, fitness, warrior_type)

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
            code=code, warrior=warrior, error=error,
            id=hashlib.sha256(code.encode()).hexdigest(), parent_id=parent_id
        )

    # ==================== WARRIOR GENERATION ====================

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
        }
        for i in range(1, 9):
            params[f'q{i}'] = str(random.randint(500, 7500))
        code = template
        for key, value in params.items():
            code = code.replace('{' + key + '}', str(value))
        return code

    def _generate_new_warrior(self) -> str:
        weights = {
            'stone': 0.18, 'scanner': 0.15, 'quickscanner': 0.12,
            'two_phase': 0.10, 'paper_attack': 0.10, 'silk': 0.08,
            'dwarf': 0.08, 'clear': 0.06, 'gate_stone': 0.05, 'vampire': 0.04,
        }
        template_name = random.choices(list(weights.keys()), weights=list(weights.values()))[0]
        return self._fill_template(self.TEMPLATES.get(template_name, self.TEMPLATES['stone']))

    async def new_warrior_async(self, n_warriors: int = 1, n_responses: int = 1) -> np.ndarray:
        exp_warriors = [[None for _ in range(n_responses)] for _ in range(n_warriors)]
        for i in range(n_warriors):
            for j in range(n_responses):
                for _ in range(self.MAX_RETRIES):
                    code = self._generate_new_warrior()
                    if self._validate_warrior(code)[0]:
                        break
                exp_warriors[i][j] = self._to_explicit_warrior(code)
        self.all_generations.append(("new_warrior", exp_warriors))
        return np.array(exp_warriors, dtype=object)

    # ==================== IMPROVEMENT 7: COMPONENT-BASED CROSSOVER ====================

    def _extract_components(self, code: str) -> Dict[str, str]:
        """Extract boot, attack, kill, decoy components."""
        components = {'boot': '', 'attack': '', 'kill': '', 'decoy': '', 'other': ''}
        lines = code.split('\n')
        current = 'other'

        for line in lines:
            line_lower = line.lower()
            if 'boot' in line_lower and not line_lower.strip().startswith(';'):
                current = 'boot'
            elif any(x in line_lower for x in ['bomb', 'attack', 'stone', 'start']):
                current = 'attack'
            elif any(x in line_lower for x in ['kill', 'found', 'clear', 'ccnt']):
                current = 'kill'
            elif 'decoy' in line_lower:
                current = 'decoy'
            elif 'ORG' in line.upper() or 'END' in line.upper():
                current = 'other'
            components[current] += line + '\n'

        return components

    def _component_crossover(self, code1: str, code2: str) -> str:
        """Crossover respecting component boundaries."""
        comp1 = self._extract_components(code1)
        comp2 = self._extract_components(code2)

        result = {}
        for name in ['boot', 'attack', 'kill', 'decoy']:
            if random.random() < 0.5 and comp1[name].strip():
                result[name] = comp1[name]
            elif comp2[name].strip():
                result[name] = comp2[name]
            else:
                result[name] = comp1[name]

        unique_id = random.randint(1000, 9999)
        code = f";name ComponentCross_{unique_id}\n;author ExplicitMutator1v1\nORG start\n"
        for comp in ['boot', 'attack', 'kill', 'decoy']:
            code += result.get(comp, '')
        if 'END' not in code.upper():
            code += '\nEND\n'
        return code

    def _crossover(self, parent1_code: str, parent2_code: str) -> str:
        """50% component-based, 50% single-point crossover."""
        if random.random() < 0.5:
            return self._component_crossover(parent1_code, parent2_code)

        lines1 = [l for l in parent1_code.split('\n') if l.strip() and not l.strip().startswith(';')
                  and not l.strip().upper().startswith(('ORG', 'END'))]
        lines2 = [l for l in parent2_code.split('\n') if l.strip() and not l.strip().startswith(';')
                  and not l.strip().upper().startswith(('ORG', 'END'))]

        if len(lines1) < 2 or len(lines2) < 2:
            return parent1_code

        cut1 = random.randint(1, len(lines1) - 1)
        cut2 = random.randint(1, len(lines2) - 1)
        new_lines = lines1[:cut1] + lines2[cut2:]

        unique_id = random.randint(1000, 9999)
        code = f";name Cross_{unique_id}\n;author ExplicitMutator1v1\nORG start\n"
        if new_lines:
            new_lines[0] = "start   " + new_lines[0].lstrip()
        code += '\n'.join(new_lines) + '\nEND\n'
        return code

    # ==================== MUTATION DISPATCH ====================

    def _mutate_code(self, code: str, warrior_hash: str = None) -> str:
        weights = self._get_effective_weights(code)
        mutation_name = random.choices(list(weights.keys()), weights=list(weights.values()))[0]

        if warrior_hash:
            self.mutation_tracker.record_mutation(warrior_hash, mutation_name)

        method = getattr(self, f'_mut_{mutation_name}', None)
        if method is None:
            return code

        if mutation_name in ('fine_tune_step', 'fine_tune_constant'):
            return method(code, warrior_hash)
        return method(code)

    # ==================== FINE-TUNING MUTATIONS ====================

    def _mut_fine_tune_step(self, code: str, warrior_hash: str = None) -> str:
        pattern = r'(ADD\.\w+\s+#)(\d+)'
        matches = list(re.finditer(pattern, code, re.IGNORECASE))
        if not matches:
            return code
        mutable_lines = set(self._get_mutable_lines(code))
        valid_matches = [m for m in matches if code[:m.start()].count('\n') in mutable_lines] or matches
        match = random.choice(valid_matches)
        old_val = int(match.group(2))
        delta = self.directional_tuner.get_biased_delta(code, match.start(), self.FINE_DELTAS)
        new_val = max(3, (old_val + delta) % self.CORE_SIZE)
        if warrior_hash:
            if warrior_hash not in self._pending_deltas:
                self._pending_deltas[warrior_hash] = []
            self._pending_deltas[warrior_hash].append((code, match.start(), delta))
        return code[:match.start(2)] + str(new_val) + code[match.end(2):]

    def _mut_fine_tune_constant(self, code: str, warrior_hash: str = None) -> str:
        pattern = r'([#$@<>*{}\s])(\d+)'
        matches = list(re.finditer(pattern, code))
        if not matches:
            return code
        mutable_lines = set(self._get_mutable_lines(code))
        valid_matches = [m for m in matches if code[:m.start()].count('\n') in mutable_lines] or matches
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
        pattern = r'(cnt\s+DAT\.F\s+#)(\d+)(,\s*#)(\d+)'
        matches = list(re.finditer(pattern, code, re.IGNORECASE))
        if not matches:
            pattern = r'(count\s+DAT\.F\s+#)(\d+)'
            matches = list(re.finditer(pattern, code, re.IGNORECASE))
            if matches:
                match = random.choice(matches)
                old_val = int(match.group(2))
                new_val = max(10, min(1000, old_val + random.choice([-20, -10, 10, 20, 30])))
                return code[:match.start(2)] + str(new_val) + code[match.end(2):]
            return code
        match = random.choice(matches)
        old_val = int(match.group(2))
        new_val = max(10, min(1000, old_val + random.choice([-20, -10, 10, 20, 30, 50])))
        return code[:match.start(2)] + str(new_val) + code[match.end(2):match.start(4)] + str(new_val) + code[match.end(4):]

    def _mut_tune_scan_position(self, code: str) -> str:
        match = re.search(r'(sptr\s+DAT\.F\s+#)(\d+)', code, re.IGNORECASE)
        if not match:
            return code
        old_val = int(match.group(2))
        new_val = (old_val + random.choice([-200, -100, -50, 50, 100, 200])) % self.CORE_SIZE
        return code[:match.start(2)] + str(new_val) + code[match.end(2):]

    def _mut_good_step_replace(self, code: str) -> str:
        pattern = r'(ADD\.\w+\s+#)(\d+)'
        matches = list(re.finditer(pattern, code, re.IGNORECASE))
        if not matches:
            return code
        match = random.choice(matches)
        return code[:match.start(2)] + str(random.choice(self.GOOD_STEPS)) + code[match.end(2):]

    def _mut_archive_step(self, code: str) -> str:
        if not self.learned_steps:
            return self._mut_good_step_replace(code)
        pattern = r'(ADD\.\w+\s+#)(\d+)'
        matches = list(re.finditer(pattern, code, re.IGNORECASE))
        if not matches:
            return code
        match = random.choice(matches)
        return code[:match.start(2)] + str(random.choice(list(self.learned_steps))) + code[match.end(2):]

    # IMPROVEMENT 5: Pattern-based step selection
    def _mut_pattern_step(self, code: str) -> str:
        warrior_type = self._detect_warrior_type(code)
        step = self.pattern_library.get_best_step_for_type(warrior_type)
        if step is None:
            return self._mut_archive_step(code)
        pattern = r'(ADD\.\w+\s+#)(\d+)'
        matches = list(re.finditer(pattern, code, re.IGNORECASE))
        if not matches:
            return code
        match = random.choice(matches)
        return code[:match.start(2)] + str(step) + code[match.end(2):]

    # IMPROVEMENT 9: Number-theoretic step
    def _mut_optimal_step(self, code: str) -> str:
        warrior_type = self._detect_warrior_type(code)
        strategy = 'fast' if warrior_type in ('stone', 'bomber') else 'balanced'
        new_step = StepOptimizer.generate_optimal_step(strategy)
        pattern = r'(ADD\.\w+\s+#)(\d+)'
        matches = list(re.finditer(pattern, code, re.IGNORECASE))
        if not matches:
            return code
        match = random.choice(matches)
        return code[:match.start(2)] + str(new_step) + code[match.end(2):]

    # ==================== 1v1 ATTACK ENHANCEMENT MUTATIONS ====================

    def _mut_add_core_clear(self, code: str) -> str:
        """Add core clear as finishing move (anti-tie)."""
        if any(x in code.lower() for x in ['clear', 'ccnt', '>cptr']):
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
        if any(x in code.lower() for x in ['kill', 'kcnt']):
            return code
        if not any(x in code.lower() for x in ['found', 'qhit']):
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

    def _mut_add_finishing_move(self, code: str) -> str:
        """Add a finishing move to ensure kills."""
        if 'finish' in code.lower():
            return code
        finish = """
        ; Finishing move
finish  MOV.I   $fdat, >fptr
        DJN.F   $finish, $fcnt
        JMP.A   $start
fdat    DAT.F   #0, #0
fptr    DAT.F   #0, #0
fcnt    DAT.F   #100, #100
"""
        return re.sub(r'(\s*END\s*)', finish + r'\1', code, flags=re.IGNORECASE)

    def _mut_increase_attack_range(self, code: str) -> str:
        """Increase loop counts for more thorough attacks."""
        pattern = r'(#)(\d{2,3})(,\s*#)(\d{2,3})'
        matches = list(re.finditer(pattern, code))
        if not matches:
            return code
        match = random.choice(matches)
        old_val = int(match.group(2))
        new_val = min(999, old_val + random.randint(20, 50))
        return code[:match.start(2)] + str(new_val) + code[match.end(2):match.start(4)] + str(new_val) + code[match.end(4):]

    # ==================== IMPROVEMENT 8: NEW TACTICAL MUTATIONS ====================

    def _mut_add_imp_gate(self, code: str) -> str:
        """Add imp-catching gate."""
        if 'gate' in code.lower():
            return code
        gate = "gate    SPL.A   $0, <gate-4\n"
        return re.sub(r'(ORG\s+\w+\s*\n)', r'\1' + gate, code, flags=re.IGNORECASE)

    def _mut_inject_scanner(self, code: str) -> str:
        """Add quick scanner to non-scanner warriors."""
        if self._detect_warrior_type(code) == 'scanner':
            return code
        q1, q2 = random.randint(800, 2500), random.randint(3000, 5000)
        scanner = f"""        ; Quick scan
        SNE.I   ${q1}, ${q2}
        JMP.A   $qhit
        JMP.A   $main
qhit    MOV.I   $qdat, @${q1}
qdat    DAT.F   #0, #0
main    """
        return re.sub(r'(ORG\s+\w+\s*\n)', r'\1' + scanner, code, flags=re.IGNORECASE)

    def _mut_add_qscan_prefix(self, code: str) -> str:
        """Add quickscan prefix to catch large/slow warriors."""
        if any(x in code.lower() for x in ['qscan', 'quick']):
            return code
        q1, q2 = random.randint(1000, 3000), random.randint(4000, 6000)
        qscan = f""";quickscan prefix
        SNE.I   ${q1}, ${q2}
        JMP.A   $qattack
        JMP.A   $main
qattack MOV.I   $qdat, @${q1}
qdat    DAT.F   #0, #0
main    """
        return re.sub(r'(ORG\s+\w+\s*\n)', r'\1' + qscan, code, flags=re.IGNORECASE)

    def _mut_unroll_loop(self, code: str) -> str:
        """Partially unroll bombing loops for speed."""
        pattern = r'((\w+)\s+MOV[^\n]+\n)(\s*ADD[^\n]+\n)(\s*JMP\s+\$\2)'
        match = re.search(pattern, code, re.IGNORECASE)
        if not match:
            return code
        mov_line = match.group(1).strip()
        add_line = match.group(3).strip()
        jmp_match = match.group(4).strip()
        label = match.group(2)
        unrolled = f"{mov_line}\n        {add_line}\n        {mov_line.replace(label, '        ')}\n        {add_line}\n        {jmp_match}"
        return code[:match.start()] + unrolled + code[match.end():]

    def _mut_optimize_scan(self, code: str) -> str:
        """Optimize scan instruction sequence."""
        if 'SNE' not in code.upper():
            return code
        code = re.sub(r'SEQ\.I\s+@(\w+),\s*\$zero', r'JMZ.F $scan, @\1', code, flags=re.IGNORECASE)
        return code

    # ==================== STANDARD MUTATIONS ====================

    def _mut_change_opcode(self, code: str) -> str:
        mutable = self._get_mutable_lines(code)
        if not mutable:
            return code
        lines = code.split('\n')
        idx = random.choice(mutable)
        for op in self.OPCODES:
            if re.search(rf'\b{op}\b', lines[idx], re.IGNORECASE):
                lines[idx] = re.sub(rf'\b{op}\b', random.choice(self.OPCODES), lines[idx], count=1, flags=re.IGNORECASE)
                break
        return '\n'.join(lines)

    def _mut_change_modifier(self, code: str) -> str:
        matches = list(re.finditer(r'\.([ABABFXI]{1,2})\b', code, re.IGNORECASE))
        if not matches:
            return code
        match = random.choice(matches)
        return code[:match.start(1)] + random.choice(self.MODIFIERS) + code[match.end(1):]

    def _mut_change_addr_mode(self, code: str) -> str:
        matches = list(re.finditer(r'([#$*@{}<>])(-?\d+)', code))
        if not matches:
            return code
        match = random.choice(matches)
        return code[:match.start(1)] + random.choice(self.ADDR_MODES) + code[match.end(1):]

    def _mut_swap_adjacent(self, code: str) -> str:
        mutable = self._get_mutable_lines(code)
        if len(mutable) < 2:
            return code
        adjacent = [(mutable[i], mutable[i+1]) for i in range(len(mutable)-1) if mutable[i+1] == mutable[i]+1]
        if not adjacent:
            return code
        idx1, idx2 = random.choice(adjacent)
        lines = code.split('\n')
        lines[idx1], lines[idx2] = lines[idx2], lines[idx1]
        return '\n'.join(lines)

    def _mut_insert_instruction(self, code: str) -> str:
        mutable = self._get_mutable_lines(code)
        if not mutable:
            return code
        lines = code.split('\n')
        useful = [
            "MOV.I   $dat, @ptr",
            f"ADD.AB  #{random.choice(self.GOOD_STEPS[:20])}, $ptr",
            "DJN.F   $-1, $cnt",
        ]
        lines.insert(random.choice(mutable), "        " + random.choice(useful))
        return '\n'.join(lines)

    def _mut_delete_instruction(self, code: str) -> str:
        lines = code.split('\n')
        deletable = [i for i, l in enumerate(lines)
                     if l.strip() and not l.strip().startswith(';')
                     and not l.strip().upper().startswith(('ORG', 'END'))
                     and not re.match(r'^\w+\s+', l.strip())]
        if len(deletable) <= 1:
            return code
        del lines[random.choice(deletable)]
        return '\n'.join(lines)

    # ==================== MAIN MUTATION INTERFACE ====================

    async def mutate_warrior_async(self, exp_warriors: List[ExplicitWarrior], n_responses: int = 1) -> np.ndarray:
        result = [[None for _ in range(n_responses)] for _ in range(len(exp_warriors))]
        self.generation_count += 1

        for i, parent in enumerate(exp_warriors):
            for j in range(n_responses):
                parent_code = getattr(parent, 'code', '') or getattr(parent, 'llm_response', '')
                parent_fitness = getattr(parent, 'fitness', 0) or 0

                for attempt in range(self.MAX_RETRIES):
                    # Crossover (IMPROVEMENT 7)
                    crossover_prob = 0.05 if self.warmstart_mode else 0.10
                    if len(exp_warriors) > 1 and random.random() < crossover_prob:
                        other = random.choice([w for w in exp_warriors if w != parent])
                        other_code = getattr(other, 'code', '') or getattr(other, 'llm_response', '')
                        code = self._crossover(parent_code, other_code)
                    else:
                        code = parent_code

                    warrior_hash = self.get_warrior_hash(code)

                    # IMPROVEMENT 6: Adaptive mutation count
                    n_mutations = self._get_adaptive_mutation_count(parent_fitness)

                    for _ in range(n_mutations):
                        code = self._mutate_code(code, warrior_hash=warrior_hash)

                    if self._validate_warrior(code)[0]:
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

                result[i][j] = self._to_explicit_warrior(code, parent_id=parent.id if hasattr(parent, 'id') else None)

        self.all_generations.append(("mutate_warrior", result))
        return np.array(result, dtype=object)

    # ==================== DIAGNOSTICS ====================

    def get_mutation_stats(self) -> Dict[str, Dict]:
        stats = {}
        for name in self.base_mutation_weights:
            stats[name] = {
                'base_weight': self.base_mutation_weights.get(name, 0),
                'success_rate': self.mutation_tracker.get_success_rate(name),
                'avg_improvement': self.mutation_tracker.get_avg_improvement(name),
                'sample_count': self.mutation_tracker.get_sample_count(name),
            }
        return stats

    def print_mutation_stats(self):
        stats = self.get_mutation_stats()
        print("\n" + "=" * 80)
        print("1v1 MUTATION EFFECTIVENESS STATISTICS")
        print("=" * 80)
        print(f"{'Mutation':<28} {'Base%':>7} {'Success%':>9} {'AvgΔFit':>10} {'Samples':>8}")
        print("-" * 80)

        sorted_stats = sorted(stats.items(), key=lambda x: (x[1]['sample_count'] > 0, x[1]['success_rate']), reverse=True)
        for name, s in sorted_stats:
            if s['sample_count'] > 0:
                print(f"{name:<28} {s['base_weight']*100:>6.1f}% {s['success_rate']*100:>8.1f}% {s['avg_improvement']:>+10.4f} {s['sample_count']:>8}")
            else:
                print(f"{name:<28} {s['base_weight']*100:>6.1f}%      N/A        N/A        0")

        print("=" * 80)
        print(f"Generations: {self.generation_count} | Max fitness: {self.max_fitness_seen:.4f}")
        print(f"Directional tuner: {len(self.directional_tuner.momentum)} | Learned steps: {len(self.learned_steps)}")
        print(f"Kill patterns: {len(self.pattern_library.kill_patterns)} | Attack patterns: {len(self.pattern_library.attack_patterns)}")