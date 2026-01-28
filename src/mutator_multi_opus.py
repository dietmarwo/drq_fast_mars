"""
Explicit Core War Mutator v3 - Full Adaptive Learning

All 9 improvements implemented:
1. Mutation Success Tracking - Learn which mutations actually improve fitness
2. Type-Aware Mutation Selection - Apply warrior-type-specific mutation weights
3. Directional Hill Climbing - Remember successful directions for constants
4. Protected Regions - Don't mutate critical code structures
5. Contextual Pattern Learning - Learn successful patterns, not just constants
6. Adaptive Mutation Count - Adjust mutations based on fitness progress
7. Component-based Crossover - Respect component boundaries in crossover
8. New Tactical Mutations - Additional strategic mutations
9. Number-Theoretic Steps - Generate mathematically optimal step sizes

Use warmstart_mode=True when loading from a previous run's archive.
Use warmstart_mode=False when starting from scratch.
"""

import re
import random
import numpy as np
import hashlib
import math
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Set, Dict
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
    """Learn successful instruction patterns, not just constants."""

    def __init__(self, max_patterns: int = 100):
        self.max_patterns = max_patterns
        # Pattern -> (count, total_fitness)
        self.attack_patterns: Dict[str, Tuple[int, float]] = {}
        self.boot_patterns: Dict[str, Tuple[int, float]] = {}
        # (step, count) -> (count, total_fitness) for step+count combinations
        self.step_count_pairs: Dict[Tuple[int, int], Tuple[int, float]] = {}
        # Warrior type -> list of successful steps
        self.steps_by_type: Dict[str, List[Tuple[int, float]]] = defaultdict(list)

    def _normalize_pattern(self, pattern: str) -> str:
        """Normalize a pattern by removing specific constants."""
        # Replace numbers with placeholders
        normalized = re.sub(r'#\d+', '#N', pattern)
        normalized = re.sub(r'\$-?\d+', '$N', normalized)
        normalized = re.sub(r'@-?\d+', '@N', normalized)
        return normalized.strip()

    def _update_pattern(self, patterns: Dict, pattern: str, fitness: float):
        """Update pattern statistics."""
        if pattern in patterns:
            count, total = patterns[pattern]
            patterns[pattern] = (count + 1, total + fitness)
        else:
            patterns[pattern] = (1, fitness)
        # Limit size
        if len(patterns) > self.max_patterns:
            # Remove lowest-scoring patterns
            sorted_patterns = sorted(patterns.items(), key=lambda x: x[1][1]/x[1][0])
            patterns.clear()
            for k, v in sorted_patterns[-(self.max_patterns//2):]:
                patterns[k] = v

    def extract_from_warrior(self, code: str, fitness: float, warrior_type: str):
        """Extract and score patterns from a successful warrior."""
        if fitness <= 0:
            return

        # Extract bombing loop pattern
        bomb_match = re.search(
            r'((?:MOV|ADD)[^\n]+\n\s*(?:ADD|MOV)[^\n]+\n\s*(?:JMP|DJN)[^\n]+)',
            code, re.IGNORECASE
        )
        if bomb_match:
            pattern = self._normalize_pattern(bomb_match.group(1))
            self._update_pattern(self.attack_patterns, pattern, fitness)

        # Extract boot pattern
        boot_match = re.search(r'(boot[^\n]*\n(?:[^\n]+\n){0,3}JMP[^\n]+)', code, re.IGNORECASE)
        if boot_match:
            pattern = self._normalize_pattern(boot_match.group(1))
            self._update_pattern(self.boot_patterns, pattern, fitness)

        # Extract step + count combination
        step_match = re.search(r'ADD\.\w+\s+#(\d+)', code, re.IGNORECASE)
        count_match = re.search(r'DJN[^,]+,\s*(?:\$\w+|#(\d+))', code, re.IGNORECASE)
        if step_match:
            step = int(step_match.group(1))
            count = int(count_match.group(1)) if count_match and count_match.group(1) else 100
            key = (step, count)
            if key in self.step_count_pairs:
                c, t = self.step_count_pairs[key]
                self.step_count_pairs[key] = (c + 1, t + fitness)
            else:
                self.step_count_pairs[key] = (1, fitness)

            # Track by warrior type
            self.steps_by_type[warrior_type].append((step, fitness))
            # Keep only top performers per type
            if len(self.steps_by_type[warrior_type]) > 50:
                self.steps_by_type[warrior_type].sort(key=lambda x: x[1], reverse=True)
                self.steps_by_type[warrior_type] = self.steps_by_type[warrior_type][:30]

    def get_best_step_for_type(self, warrior_type: str) -> Optional[int]:
        """Get a proven step size for this warrior type."""
        steps = self.steps_by_type.get(warrior_type, [])
        if steps:
            # Weight by fitness
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
    """Generate step sizes with good mathematical properties."""

    CORE_SIZE = 8000

    # Pre-computed good primes
    GOOD_PRIMES = [
        2339, 2341, 2347, 2351, 2357, 2371, 2377, 2381, 2383, 2389,
        2393, 2399, 2411, 2417, 2423, 2437, 2441, 2447, 2459, 2467,
        2473, 2477, 2503, 2521, 2531, 2539, 2543, 2549, 2551, 2557,
        2579, 2591, 2593, 2609, 2617, 2621, 2633, 2647, 2657, 2659,
        2663, 2671, 2677, 2683, 2687, 2689, 2693, 2699, 2707, 2711,
        2713, 2719, 2729, 2731, 2741, 2749, 2753, 2767, 2777, 2789,
        2791, 2797, 2801, 2803, 2819, 2833, 2837, 2843, 2851, 2857,
        2861, 2879, 2887, 2897, 2903, 2909, 2917, 2927, 2939, 2953,
        2957, 2963, 2969, 2971, 2999, 3001, 3011, 3019, 3023, 3037,
        3041, 3049, 3061, 3067, 3079, 3083, 3089, 3109, 3119, 3121,
    ]

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
        Generate step size with good mathematical properties.

        Strategies:
        - 'fast': Large steps for quick coverage (~core/3)
        - 'thorough': Smaller steps for thorough coverage (~core/5)
        - 'prime': Pure prime in good range
        - 'balanced': Mix of strategies
        """
        if strategy == 'fast':
            # Around core_size / 3 (good for 3-point coverage)
            base = cls.CORE_SIZE // 3
            candidates = [base + d for d in range(-50, 51) if cls.is_coprime(base + d)]
        elif strategy == 'thorough':
            # Around core_size / 5
            base = cls.CORE_SIZE // 5
            candidates = [base + d for d in range(-30, 31) if cls.is_coprime(base + d)]
        elif strategy == 'prime':
            candidates = cls.GOOD_PRIMES
        else:  # balanced
            if random.random() < 0.4:
                return cls.generate_optimal_step('fast')
            elif random.random() < 0.6:
                return cls.generate_optimal_step('prime')
            else:
                return cls.generate_optimal_step('thorough')

        return random.choice(candidates) if candidates else 2667

    @classmethod
    def optimize_existing_step(cls, step: int, delta_range: int = 10) -> int:
        """Find the nearest coprime value to an existing step."""
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
    Explicit rule-based mutator with full adaptive learning (9 improvements).
    """

    CORE_SIZE = 8000
    MAX_RETRIES = 5

    GOOD_STEPS = [
        3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47,
        53, 59, 61, 67, 71, 73, 79, 83, 89, 97,
        2339, 2663, 2671, 3037, 3041, 3049, 3061, 3067, 3079, 3083,
        3089, 3109, 3119, 3121, 3137, 3163, 3167, 3169, 3181, 3187,
        3191, 3203, 3209, 3217, 3221, 3229, 3251, 3253, 3257, 3259,
        3271, 3299, 3301, 3307, 3313, 3319, 3323, 3329, 3331, 3343,
        2667, 5334, 1471, 1777, 3044, 4001, 4003, 4007,
    ]

    FINE_DELTAS = [-2, -1, 1, 2]
    SMALL_DELTAS = [-5, -3, 3, 5]
    MEDIUM_DELTAS = [-20, -10, 10, 20]

    BOOT_DISTANCES = [100, 200, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000]

    # ==================== TEMPLATES ====================

    TEMPLATES = {
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
        'scanner': """
;name Scanner_{id}
;author ExplicitMutator
;strategy Scanner - searches then attacks
ORG scan
scan    ADD.AB  #{step}, $ptr
        SEQ.I   @ptr, $zero
        JMP.A   $found
        JMP.A   $scan
found   MOV.I   $bomb, @ptr
        JMP.A   $scan
bomb    DAT.F   #0, #0
zero    DAT.F   #0, #0
ptr     DAT.F   #0, #{step}
END
""",
        'quickscanner': """
;name QScan_{id}
;author ExplicitMutator
;strategy Quickscanner - fast initial scan then bomb
ORG qscan
qscan   SNE.I   ${q1}, ${q2}
        SEQ.I   ${q3}, ${q4}
        JMP.A   $attack
        JMP.A   $bomb
attack  MOV.I   $dat, @${q1}
        ADD.AB  #{qstep}, $-1
        JMP.A   $-2
bomb    ADD.AB  #{step}, $ptr
        MOV.I   $dat, @ptr
        JMP.A   $bomb
dat     DAT.F   #0, #0
ptr     DAT.F   #0, #{step}
END
""",
        'vampire': """
;name Vampire_{id}
;author ExplicitMutator
;strategy Vampire - converts enemy to pit jumpers
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
    }

    DECOY_COMPONENTS = {
        'dat_field': "\n        DAT.F   #0, #0\n        DAT.F   #0, #0\n",
        'imp_decoy': "\n        MOV.I   $0, $1\n        MOV.I   $0, $1\n",
        'jmp_decoy': "\n        JMP.A   $0\n        DAT.F   #0, #0\n",
    }

    OPCODES = ['DAT', 'MOV', 'ADD', 'SUB', 'MUL', 'DIV', 'MOD',
               'JMP', 'JMZ', 'JMN', 'DJN', 'CMP', 'SEQ', 'SNE', 'SLT', 'SPL', 'NOP']
    MODIFIERS = ['A', 'B', 'AB', 'BA', 'F', 'X', 'I']
    ADDR_MODES = ['#', '$', '*', '@', '{', '<', '}', '>']

    TYPE_SIGNATURES = {
        'paper': ['SPL', 'MOV.I', 'ADD.AB'],
        'stone': ['MOV.I', 'ADD.AB', 'JMP'],
        'scanner': ['SNE', 'SEQ', 'CMP', 'JMZ'],
        'bomber': ['ADD.AB', 'MOV.AB', 'JMP', 'DAT'],
        'vampire': ['JMZ', 'MOV.I', 'fang', 'pit'],
        'clear': ['MOV.I', 'DJN', '>'],
    }

    TYPE_MUTATION_ADJUSTMENTS = {
        'paper': {
            'optimize_paper': 3.0, 'inject_spl_fan': 2.0, 'fine_tune_step': 1.5,
            'change_opcode': 0.3, 'delete_instruction': 0.2, 'inject_bomber_loop': 0.5,
        },
        'stone': {
            'optimize_stone': 3.0, 'fine_tune_step': 2.5, 'prime_step': 2.0,
            'optimal_step': 2.0, 'inject_bomber_loop': 1.5, 'insert_spl': 0.5,
        },
        'scanner': {
            'fine_tune_step': 2.5, 'change_addr_mode': 1.5, 'inject_quickscan': 2.0,
            'add_qscan_prefix': 1.5, 'tune_loop_count': 1.5, 'add_decoy': 0.3,
        },
        'vampire': {
            'fine_tune_constant': 2.0, 'fine_tune_step': 1.5, 'change_addr_mode': 1.5,
            'delete_instruction': 0.2,
        },
        'bomber': {
            'fine_tune_step': 2.0, 'optimize_stone': 1.5, 'prime_step': 1.5,
            'optimal_step': 1.5, 'unroll_loop': 1.5,
        },
        'clear': {
            'fine_tune_step': 2.0, 'tune_loop_count': 2.5, 'increase_attack_range': 2.0,
        },
    }

    def __init__(self, environment=None, warmstart_mode: bool = False, use_adaptive_weights: bool = True):
        self.environment = environment or {}
        self.warmstart_mode = warmstart_mode
        self.use_adaptive_weights = use_adaptive_weights
        self.all_generations = []

        # Learned constants
        self.learned_steps: Set[int] = set()
        self.learned_counts: Set[int] = set()

        # Improvement 1: Mutation tracking
        self.mutation_tracker = MutationTracker(window_size=100)

        # Improvement 3: Directional hill climbing
        self.directional_tuner = DirectionalTuner(max_entries=1000)
        self._pending_deltas: Dict[str, List[Tuple[str, int, int]]] = {}

        # Improvement 5: Pattern learning
        self.pattern_library = PatternLibrary(max_patterns=100)

        # Improvement 6: Track max fitness for adaptive mutation count
        self.max_fitness_seen: float = 0.0
        self.generation_count: int = 0

        # Base mutation weights
        if warmstart_mode:
            self.base_mutation_weights = {
                'fine_tune_step': 0.20, 'fine_tune_constant': 0.10, 'tune_loop_count': 0.06,
                'archive_step': 0.05, 'pattern_step': 0.05, 'optimal_step': 0.04,
                'change_constant_good': 0.06, 'prime_step': 0.04,
                'change_modifier': 0.05, 'change_addr_mode': 0.05, 'swap_adjacent': 0.03,
                'optimize_paper': 0.04, 'optimize_stone': 0.04,
                'change_opcode': 0.02, 'change_constant_small': 0.03,
                'insert_spl': 0.02, 'inject_spl_fan': 0.02,
                'insert_instruction': 0.01, 'delete_instruction': 0.01,
                'add_imp_gate': 0.02, 'add_qscan_prefix': 0.02, 'unroll_loop': 0.02,
                'optimize_scan': 0.02,
            }
        else:
            self.base_mutation_weights = {
                'change_constant_small': 0.10, 'change_constant_good': 0.08,
                'prime_step': 0.06, 'optimal_step': 0.04,
                'change_opcode': 0.05, 'change_modifier': 0.05, 'change_addr_mode': 0.06,
                'insert_instruction': 0.03, 'delete_instruction': 0.02,
                'swap_instructions': 0.02, 'duplicate_instruction': 0.02,
                'insert_spl': 0.08, 'inject_bomber_loop': 0.04, 'inject_spl_fan': 0.05,
                'inject_quickscan': 0.03, 'add_decoy': 0.03, 'add_boot': 0.03,
                'optimize_paper': 0.02, 'optimize_stone': 0.02,
                'fine_tune_step': 0.04, 'fine_tune_constant': 0.02,
                'add_imp_gate': 0.03, 'add_qscan_prefix': 0.03, 'unroll_loop': 0.02,
                'optimize_scan': 0.02,
            }

    # ==================== FEEDBACK API ====================

    def get_warrior_hash(self, code: str) -> str:
        return hashlib.sha256(code.encode()).hexdigest()[:16]

    def record_fitness_feedback(self, warrior_hash: str, old_fitness: float, new_fitness: float):
        """Call after evaluation to enable adaptive learning."""
        self.mutation_tracker.record_fitness_result(warrior_hash, old_fitness, new_fitness)

        # Update max fitness for adaptive mutation count
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

        # Vampire pits
        for i, line in enumerate(lines):
            if re.search(r'^\s*(pit|fang)\s+', line, re.IGNORECASE):
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
        """Adjust mutation count based on fitness relative to best seen."""
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
        """Extract successful patterns and constants from archive."""
        for bc, warrior in archive.items():
            code = getattr(warrior, 'code', None) or getattr(warrior, 'llm_response', '')
            fitness = getattr(warrior, 'fitness', 0)
            if not code or not fitness or fitness <= 0:
                continue

            warrior_type = self._detect_warrior_type(code)

            # Pattern learning (Improvement 5)
            self.pattern_library.extract_from_warrior(code, fitness, warrior_type)

            # Extract constants
            for match in re.finditer(r'ADD\.\w+\s+#(\d+)', code, re.IGNORECASE):
                step = int(match.group(1))
                if 10 < step < 7000:
                    self.learned_steps.add(step)
            for match in re.finditer(r'#(\d{2,3}),\s*#(\d{2,3})', code):
                count = int(match.group(1))
                if 10 < count < 1000:
                    self.learned_counts.add(count)

        # Limit size
        if len(self.learned_steps) > 200:
            self.learned_steps = set(list(self.learned_steps)[-200:])
        if len(self.learned_counts) > 100:
            self.learned_counts = set(list(self.learned_counts)[-100:])

    def _get_learned_or_good_step(self) -> int:
        if self.learned_steps and random.random() < 0.5:
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
            'step': self._get_learned_or_good_step(),
            'count': random.randint(100, 500),
            'qstep': random.choice([5, 7, 11, 13, 17]),
        }
        for i in range(1, 9):
            params[f'q{i}'] = str(random.randint(500, 7500))
        code = template
        for key, value in params.items():
            code = code.replace('{' + key + '}', str(value))
        return code

    def _generate_new_warrior(self) -> str:
        weights = {'paper': 0.18, 'silk': 0.15, 'stone': 0.15, 'scanner': 0.12,
                   'quickscanner': 0.10, 'dwarf': 0.10, 'clear': 0.08, 'vampire': 0.06}
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
        """Extract boot, attack, decoy components from warrior."""
        components = {'boot': '', 'attack': '', 'decoy': '', 'other': ''}
        lines = code.split('\n')
        current = 'other'

        for line in lines:
            line_lower = line.lower()
            if 'boot' in line_lower and not line_lower.strip().startswith(';'):
                current = 'boot'
            elif any(x in line_lower for x in ['bomb', 'attack', 'stone', 'start']):
                current = 'attack'
            elif 'decoy' in line_lower or re.match(r'\s*DAT\s+[<>]', line):
                current = 'decoy'
            elif 'ORG' in line.upper() or 'END' in line.upper():
                current = 'other'
            components[current] += line + '\n'

        return components

    def _component_crossover(self, code1: str, code2: str) -> str:
        """Crossover that respects component boundaries."""
        comp1 = self._extract_components(code1)
        comp2 = self._extract_components(code2)

        # Randomly select each component from either parent
        result_components = {}
        for comp_name in ['boot', 'attack', 'decoy']:
            if random.random() < 0.5 and comp1[comp_name].strip():
                result_components[comp_name] = comp1[comp_name]
            elif comp2[comp_name].strip():
                result_components[comp_name] = comp2[comp_name]
            else:
                result_components[comp_name] = comp1[comp_name]

        # Reconstruct
        unique_id = random.randint(1000, 9999)
        code = f";name ComponentCross_{unique_id}\n;author ExplicitMutator\nORG start\n"
        code += result_components.get('boot', '')
        code += result_components.get('attack', '')
        code += result_components.get('decoy', '')
        if 'END' not in code.upper():
            code += '\nEND\n'
        return code

    def _crossover(self, parent1_code: str, parent2_code: str) -> str:
        """Crossover - 50% component-based, 50% single-point."""
        if random.random() < 0.5:
            return self._component_crossover(parent1_code, parent2_code)

        # Single-point crossover
        lines1 = [l for l in parent1_code.split('\n') if l.strip() and not l.strip().startswith(';')
                  and not l.strip().upper().startswith(('ORG', 'END', 'EQU'))]
        lines2 = [l for l in parent2_code.split('\n') if l.strip() and not l.strip().startswith(';')
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
        valid_matches = [m for m in matches if code[:m.start()].count('\n') in mutable_lines]
        if not valid_matches:
            valid_matches = matches
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
        pattern = r'(#)(\d{2,3})(,\s*#)(\d{2,3})'
        matches = list(re.finditer(pattern, code))
        if not matches:
            return code
        match = random.choice(matches)
        old_val = int(match.group(2))
        delta = random.choice([-30, -20, -10, 10, 20, 30])
        new_val = max(10, min(999, old_val + delta))
        return code[:match.start(2)] + str(new_val) + code[match.end(2):match.start(4)] + str(new_val) + code[match.end(4):]

    def _mut_archive_step(self, code: str) -> str:
        if not self.learned_steps:
            return self._mut_change_constant_good(code)
        pattern = r'(ADD\.\w+\s+#)(\d+)'
        matches = list(re.finditer(pattern, code, re.IGNORECASE))
        if not matches:
            return code
        match = random.choice(matches)
        new_val = random.choice(list(self.learned_steps))
        return code[:match.start(2)] + str(new_val) + code[match.end(2):]

    # IMPROVEMENT 5: Pattern-based step selection
    def _mut_pattern_step(self, code: str) -> str:
        """Use pattern library to select step for this warrior type."""
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

    # IMPROVEMENT 9: Number-theoretic step optimization
    def _mut_optimal_step(self, code: str) -> str:
        """Replace step with mathematically optimal value."""
        warrior_type = self._detect_warrior_type(code)
        strategy = 'fast' if warrior_type in ('stone', 'bomber') else 'balanced'
        new_step = StepOptimizer.generate_optimal_step(strategy)

        pattern = r'(ADD\.\w+\s+#)(\d+)'
        matches = list(re.finditer(pattern, code, re.IGNORECASE))
        if not matches:
            return code
        match = random.choice(matches)
        return code[:match.start(2)] + str(new_step) + code[match.end(2):]

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

    # ==================== STANDARD MUTATIONS ====================

    def _mut_change_constant_small(self, code: str) -> str:
        pattern = r'([#$@<>*{}\s])(-?\d+)'
        matches = list(re.finditer(pattern, code))
        if not matches:
            return code
        match = random.choice(matches)
        old_val = int(match.group(2))
        new_val = (old_val + random.randint(-20, 20)) % self.CORE_SIZE
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
        new_val = random.choice(StepOptimizer.GOOD_PRIMES)
        return code[:match.start(1)] + str(new_val) + code[match.end(1):]

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

    def _mut_insert_spl(self, code: str) -> str:
        mutable = self._get_mutable_lines(code)
        if not mutable:
            return code
        lines = code.split('\n')
        lines.insert(mutable[0], "        SPL.A   $0, $0")
        return '\n'.join(lines)

    def _mut_insert_instruction(self, code: str) -> str:
        mutable = self._get_mutable_lines(code)
        if not mutable:
            return code
        lines = code.split('\n')
        op = random.choice(self.OPCODES)
        mod = random.choice(self.MODIFIERS)
        new_line = f"        {op}.{mod}   {random.choice(self.ADDR_MODES)}{random.randint(-10,10)}, {random.choice(self.ADDR_MODES)}{random.randint(-10,10)}"
        lines.insert(random.choice(mutable), new_line)
        return '\n'.join(lines)

    def _mut_delete_instruction(self, code: str) -> str:
        mutable = self._get_mutable_lines(code)
        if len(mutable) <= 2:
            return code
        lines = code.split('\n')
        del lines[random.choice(mutable)]
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
        idx = random.choice(mutable)
        dup = re.sub(r'^\w+\s+', '        ', lines[idx])
        lines.insert(idx + 1, dup)
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
        decoy = random.choice(list(self.DECOY_COMPONENTS.values()))
        return re.sub(r'(\s*END\s*)', decoy + r'\1', code, flags=re.IGNORECASE)

    def _mut_add_boot(self, code: str) -> str:
        if 'boot' in code.lower():
            return code
        dist = random.choice(self.BOOT_DISTANCES)
        boot = f"\nboot    MOV.I   $code_start, ${dist}\n        JMP.A   ${dist}\ncode_start\n"
        return re.sub(r'(ORG\s+)', boot + r'\nORG boot\n', code, flags=re.IGNORECASE)

    def _mut_optimize_paper(self, code: str) -> str:
        if self._detect_warrior_type(code) not in ['paper', 'silk']:
            return code
        good_steps = [2667, 5334, 1471, 2500, 3333] + list(self.learned_steps)[:10]
        new_step = random.choice(good_steps) if good_steps else 2667
        return re.sub(r'(ADD\.AB\s+#)(\d+)', rf'\g<1>{new_step}', code, count=1, flags=re.IGNORECASE)

    def _mut_optimize_stone(self, code: str) -> str:
        if self._detect_warrior_type(code) not in ['stone', 'bomber']:
            return code
        good_steps = [3044, 2365, 5363, 3039, 2377] + list(self.learned_steps)[:10]
        new_step = random.choice(good_steps) if good_steps else 3044
        return re.sub(r'(ADD\.AB\s+#)(\d+)', rf'\g<1>{new_step}', code, count=1, flags=re.IGNORECASE)

    # ==================== IMPROVEMENT 8: NEW TACTICAL MUTATIONS ====================

    def _mut_add_imp_gate(self, code: str) -> str:
        """Add imp-catching gate for defense."""
        if 'gate' in code.lower():
            return code
        gate = "gate    SPL.A   $0, <gate-4\n"
        return re.sub(r'(ORG\s+\w+\s*\n)', r'\1' + gate, code, flags=re.IGNORECASE)

    def _mut_add_qscan_prefix(self, code: str) -> str:
        """Add quickscan prefix to catch large/slow warriors early."""
        if 'qscan' in code.lower() or 'quick' in code.lower():
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
        # Find MOV+ADD+JMP pattern
        pattern = r'((\w+)\s+MOV[^\n]+\n)(\s*ADD[^\n]+\n)(\s*JMP\s+\$\2)'
        match = re.search(pattern, code, re.IGNORECASE)
        if not match:
            return code

        mov_line = match.group(1).strip()
        add_line = match.group(3).strip()
        jmp_match = match.group(4).strip()
        label = match.group(2)

        # Unroll 2x
        unrolled = f"{mov_line}\n        {add_line}\n        {mov_line.replace(label, '        ')}\n        {add_line}\n        {jmp_match}"
        return code[:match.start()] + unrolled + code[match.end():]

    def _mut_optimize_scan(self, code: str) -> str:
        """Optimize scan instruction sequence."""
        # Replace SNE with JMZ equivalent where beneficial (JMZ is sometimes faster)
        if 'SNE' not in code.upper():
            return code
        # Simple optimization: ensure scan uses efficient addressing
        code = re.sub(r'SEQ\.I\s+@(\w+),\s*\$zero', r'JMZ.F $scan, @\1', code, flags=re.IGNORECASE)
        return code

    def _mut_increase_attack_range(self, code: str) -> str:
        """Increase attack range by boosting loop counts."""
        pattern = r'(#)(\d{2,3})(,\s*#)(\d{2,3})'
        matches = list(re.finditer(pattern, code))
        if not matches:
            return code
        match = random.choice(matches)
        old_val = int(match.group(2))
        new_val = min(999, old_val + random.randint(20, 50))
        return code[:match.start(2)] + str(new_val) + code[match.end(2):match.start(4)] + str(new_val) + code[match.end(4):]

    # ==================== MAIN MUTATION INTERFACE ====================

    async def mutate_warrior_async(self, exp_warriors: List[ExplicitWarrior], n_responses: int = 1) -> np.ndarray:
        result = [[None for _ in range(n_responses)] for _ in range(len(exp_warriors))]
        self.generation_count += 1

        for i, parent in enumerate(exp_warriors):
            for j in range(n_responses):
                parent_code = getattr(parent, 'code', '') or getattr(parent, 'llm_response', '')
                parent_fitness = getattr(parent, 'fitness', 0) or 0

                for attempt in range(self.MAX_RETRIES):
                    # Crossover
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
        print("MUTATION EFFECTIVENESS STATISTICS (Melee)")
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
        print(f"Generations: {self.generation_count} | Max fitness seen: {self.max_fitness_seen:.4f}")
        print(f"Directional tuner entries: {len(self.directional_tuner.momentum)}")
        print(f"Learned steps: {len(self.learned_steps)} | Pattern library attack patterns: {len(self.pattern_library.attack_patterns)}")