# Optimized CoreWar MARS Simulator

A performance-optimized fork of the CoreWar MARS (Memory Array Redcode Simulator) from [SakanaAI/drq](https://github.com/SakanaAI/drq).

The initial optimization work was created with Claude Opus 4.5. Optimizing MARS is also a useful LLM benchmark: in practice, I encountered recurring correctness and performance pitfalls when attempting similar changes with other models. I tried ChatGPT 5.2 extended thinking and Gemini 3.0 Pro. 

The corresponding publication, [`drq publication`](https://pub.sakana.ai/drq/) can also serve as an additional benchmark for LLM-based critique tasks. 
For example: “Review the paper and identify weaknesses.” An example output (Claude Opus 4.5) 
is shown here: [`drq review`](https://github.com/dietmarwo/drq_fast_mars/blob/master/drq_analysis.md).

The main criticism is that the approach involves little or no learning in the usual sense. 
The core mechanism is the well-established MAP-Elites optimization algorithm, paired with a fixed (non-adaptive) LLM-based mutator. 
That mutator is itself pretrained on human-produced data and therefore implicitly leverages human knowledge—potentially including 
Core War tutorials, strategy discussions, academic papers, and archives of successful warriors.

That said, using LLMs as mutation operators within evolutionary optimization is a strong and promising idea.

Although the review of the drq publication could be fully automated, we switch to "AI-assisted"-mode for a more detailed analysis:
[`fixed mutator analysis`](https://github.com/dietmarwo/drq_fast_mars/blob/master/fixed_mutator.md) . In this context we extend 
the original repository by new python scripts: 
- [`drq_explicit.py`](https://github.com/dietmarwo/drq_fast_mars/blob/master/src/drq_explicit.py)
- [`drq_cached.py`](https://github.com/dietmarwo/drq_fast_mars/blob/master/src/drq_cached.py)
- [`evaluate_1v1.py`](https://github.com/dietmarwo/drq_fast_mars/blob/master/src/evaluate_1v1.py)
- [`evaluate_multi.py`](https://github.com/dietmarwo/drq_fast_mars/blob/master/src/evaluate_multi.py)
- [`explicit_mutator2_warmstart.py`](https://github.com/dietmarwo/drq_fast_mars/blob/master/src/explicit_mutator2_warmstart.py)
- [`explicit_mutator2.py`](https://github.com/dietmarwo/drq_fast_mars/blob/master/src/explicit_mutator2.py)
- [`explicit_mutator_1v1.py`](https://github.com/dietmarwo/drq_fast_mars/blob/master/src/explicit_mutator_1v1.py)

Some files were extended for 1v1 scoring:
[`corewar_util.py`](https://github.com/dietmarwo/drq_fast_mars/blob/master/src/corewar_util.py)

## Performance Summary

Benchmark configuration: **480 rounds**, **5 warriors**, **24 parallel worker processes**, running on an **AMD 9950X (16 cores)**. End-to-end runtime is **~13.9 seconds** for this configuration.

| Optimization | Speedup | Cumulative |
|--------------|---------|------------|
| Baseline (original) | 1x | 1x |
| Deque for task queues | ~1% | ~1% |
| Guarded debug assertions | ~23% | ~24% |
| Numba JIT compilation | ~3.0x | **~3.7x** |
| Reduced parallelization overhead | ~2–3% | **~3.85x** |

**Total speedup:** approximately **3.85x** compared to the original implementation.

## Usage

- The optimized MARS engine is implemented in [`mars.py`](https://github.com/dietmarwo/drq_fast_mars/blob/master/src/corewar/mars.py). Do not install the `corewar` library separately for this fork; when developing the engine, keep it inside the `src` directory.
- The `MARS` constructor supports two additional parameters:
  - `debug=False`: set to `True` to enable debug/trace output and debug assertions (disabled by default).
  - `track_coverage=True`: set to `False` to disable memory coverage tracking when it is not needed (negligible impact in benchmarks).
- [`corewar_util.py`](https://github.com/dietmarwo/drq_fast_mars/blob/master/src/corewar_util.py) is adapted accordingly.
  - `run_multiple_rounds`: when `timeout=None`, an optimized parallelization strategy is used.
- [`drqtest.py`](https://github.com/dietmarwo/drq_fast_mars/blob/master/src/drqtest.py) provides a simple performance test harness.
  - Run `time python src/drqtest.py` to collect timing information.

## Changes

### 1. Code Refactoring

**Removed the `MyMARS` subclass**: functionality is merged directly into the base `MARS` class in `mars.py`, reducing inheritance overhead and simplifying the codebase.

### 2. Deque for Task Queues (~1% speedup)

**Problem:** Task queues used Python lists with `pop(0)`, which is O(n).

**Solution:** Replaced lists with `collections.deque` and `popleft()`, which is O(1).

```python
from collections import deque
warrior.task_queue = deque([start_addr])
pc = warrior.task_queue.popleft()  # O(1) instead of O(n)
```

### 3. Guarded Debug Assertions (~23% speedup)

**Problem:** `core_event()` performed expensive type checks and assertions on every memory access (millions of calls per simulation).

**Solution:** Guard assertions behind `if self.debug:` and keep `debug=False` by default.

```python
def core_event(self, warrior, address, event_type):
    if self.debug:
        # assertions only run when debug=True
        ...
```

**Usage:**
```python
simulation = MARS(warriors=warriors, debug=False)  # production
simulation = MARS(warriors=warriors, debug=True)   # debugging
```

### 4. Optional Coverage Tracking

**Problem:** Memory coverage tracking (`warrior_cov`) added overhead even when not required.

**Solution:** Added a `track_coverage` parameter to enable or disable coverage tracking.

```python
simulation = MARS(warriors=warriors, track_coverage=False)  # faster
simulation = MARS(warriors=warriors, track_coverage=True)   # when coverage is needed
```

### 5. Numba JIT Compilation (major speedup in the hot loop)

**Problem:** `step()` is a tight loop with integer arithmetic and branching. In pure Python, interpreter overhead dominates.

**Solution:** Rewrote the hot path using Numba with NumPy arrays:

- **Core memory:** `(coresize, 6)` `int32` NumPy array instead of Python objects
- **Task queues:** NumPy circular buffers instead of Python deques
- **Coverage tracking:** handled inside JIT-compiled code
- **Reduced Python overhead:** the hot loop runs in compiled machine code

**Key implementation details:**

```python
@njit(cache=True)
def _step_numba(core, num_warriors, task_queues, tq_heads, tq_tails,
                tq_sizes, coresize, max_processes, debug=False, coverage=None):
    # Entire simulation step in compiled code
    ...
```

**Array layout for core:**
```python
# Index constants
_OPCODE = 0
_MODIFIER = 1
_A_MODE = 2
_B_MODE = 3
_A_NUMBER = 4
_B_NUMBER = 5

# Core is (coresize, 6) array
core[address, _OPCODE]  # access opcode at address
```

**Task queue as a circular buffer:**
```python
task_queues: (num_warriors, max_processes)  # pre-allocated
tq_heads: (num_warriors,)  # read pointer
tq_tails: (num_warriors,)  # write pointer
tq_sizes: (num_warriors,)  # current size
```

**Direct NumPy access from `corewar_util.py`:**
```python
# Instead of: nprocs = np.array([len(w.task_queue) for w in simulation.warriors])
nprocs = simulation._tq_sizes  # direct array access, no Python loop
```

### 6. Reduced Parallelization Overhead (~2–3% speedup)

**Problem:** `Pool.map_async` introduces chunking and task distribution overhead.

**Solution:** Use a worker-loop pattern with a shared counter to enable finer-grained work stealing.

```python
def run_multiple_rounds_fast(simargs, warriors, n_processes=1):
    manager = Manager()
    outputs = manager.dict()
    mutex = mp.Lock()
    seed = mp.RawValue(ct.c_int, 0)

    def loop():
        while True:
            with mutex:
                if seed.value >= simargs.rounds:
                    return
                seed_val = seed.value
                seed.value += 1
            outputs[seed_val] = run_single_round_fn(seed_val)
```

**Usage:**
```python
# With timeout (uses Pool.map_async)
results = run_multiple_rounds(simargs, warriors, n_processes=12, timeout=900)

# Without timeout (uses fast worker-loop pattern)
results = run_multiple_rounds(simargs, warriors, n_processes=12, timeout=None)
```

## Requirements

```text
numpy
numba
tqdm
```

## API Compatibility

The optimized `MARS` class maintains API compatibility with the original implementation:

```python
from corewar import MARS, redcode

warriors = [redcode.parse(code, environment) for code in warrior_codes]
simulation = MARS(
    warriors=warriors,
    minimum_separation=100,
    max_processes=8000,
    randomize=True,
    debug=False,          # disable for production
    track_coverage=True   # enable if coverage data is needed
)

for cycle in range(80000):
    simulation.step()

    # Access queue sizes (fast, direct NumPy access)
    sizes = simulation._tq_sizes

    # Or via method
    sizes = simulation.get_queue_sizes()

# Get coverage sums
coverage = simulation.get_coverage_sums()
```

## Notes

- The first run is slower due to Numba JIT compilation. With `cache=True`, compiled code is persisted across runs.
- Delete `__pycache__` if you modify Numba-decorated functions and want to force recompilation.
- The implementation assumes default core settings (`read_limit == write_limit == coresize`). Non-standard limits may require additional changes.

## License

Same as the original repository.