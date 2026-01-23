#! /usr/bin/env python
# coding: utf-8
"""
Numba-optimized MARS implementation.
Core memory and task queues use numpy arrays for JIT compilation.
"""

from collections import deque
from copy import copy
import operator
from random import randint

import numpy as np
from numba import njit, int32, int8, boolean
from numba.typed import List as NumbaList

from .core import Core, DEFAULT_INITIAL_INSTRUCTION
from .redcode import *

__all__ = ['MARS', 'EVENT_EXECUTED', 'EVENT_I_WRITE', 'EVENT_I_READ',
           'EVENT_A_DEC', 'EVENT_A_INC', 'EVENT_B_DEC', 'EVENT_B_INC',
           'EVENT_A_READ', 'EVENT_A_WRITE', 'EVENT_B_READ', 'EVENT_B_WRITE',
           'EVENT_A_ARITH', 'EVENT_B_ARITH']

# Event types
EVENT_EXECUTED = 0
EVENT_I_WRITE  = 1
EVENT_I_READ   = 2
EVENT_A_DEC    = 3
EVENT_A_INC    = 4
EVENT_B_DEC    = 5
EVENT_B_INC    = 6
EVENT_A_READ   = 7
EVENT_A_WRITE  = 8
EVENT_B_READ   = 9
EVENT_B_WRITE  = 10
EVENT_A_ARITH  = 11
EVENT_B_ARITH  = 12

# Redcode constants (duplicated here for numba)
# Opcodes
_DAT = 0
_MOV = 1
_ADD = 2
_SUB = 3
_MUL = 4
_DIV = 5
_MOD = 6
_JMP = 7
_JMZ = 8
_JMN = 9
_DJN = 10
_SPL = 11
_SLT = 12
_CMP = 13
_SEQ = 14
_SNE = 15
_NOP = 16

# Modifiers
_M_A  = 0
_M_B  = 1
_M_AB = 2
_M_BA = 3
_M_F  = 4
_M_X  = 5
_M_I  = 6

# Address modes - must match redcode.py values
_IMMEDIATE  = 0
_DIRECT     = 1
_INDIRECT_B = 2
_PREDEC_B   = 3
_POSTINC_B  = 4
_INDIRECT_A = 5
_PREDEC_A   = 6
_POSTINC_A  = 7

# Core array indices
_OPCODE = 0
_MODIFIER = 1
_A_MODE = 2
_B_MODE = 3
_A_NUMBER = 4
_B_NUMBER = 5


@njit(cache=True)
def _trim(value, coresize):
    """Normalize address to core bounds."""
    return value % coresize


@njit(cache=True)
def _enqueue(task_queues, tq_heads, tq_tails, tq_sizes, warrior_idx, address, max_processes, coresize, coverage=None):
    """Add a process to warrior's task queue (circular buffer)."""
    if tq_sizes[warrior_idx] < max_processes:
        trimmed = address % coresize
        task_queues[warrior_idx, tq_tails[warrior_idx]] = trimmed
        tq_tails[warrior_idx] = (tq_tails[warrior_idx] + 1) % max_processes
        tq_sizes[warrior_idx] += 1
        # Track coverage if array provided
        if coverage is not None:
            coverage[warrior_idx, trimmed] = True
        return trimmed
    return -1


@njit(cache=True)
def _dequeue(task_queues, tq_heads, tq_tails, tq_sizes, warrior_idx, max_processes):
    """Remove and return next process from warrior's task queue."""
    if tq_sizes[warrior_idx] > 0:
        pc = task_queues[warrior_idx, tq_heads[warrior_idx]]
        tq_heads[warrior_idx] = (tq_heads[warrior_idx] + 1) % max_processes
        tq_sizes[warrior_idx] -= 1
        return pc
    return -1


@njit(cache=True)
def _step_numba(core, num_warriors, task_queues, tq_heads, tq_tails, tq_sizes, 
                coresize, max_processes, debug=False, coverage=None):
    """
    Execute one simulation step for all warriors.
    
    core: (coresize, 6) array - [opcode, modifier, a_mode, b_mode, a_number, b_number]
    task_queues: (num_warriors, max_processes) array - circular buffers
    tq_heads, tq_tails, tq_sizes: (num_warriors,) arrays - queue state
    coverage: optional (num_warriors, coresize) bool array for tracking visited addresses
    """
    
    for w_idx in range(num_warriors):
        if tq_sizes[w_idx] == 0:
            continue
            
        # Dequeue next instruction address
        pc = _dequeue(task_queues, tq_heads, tq_tails, tq_sizes, w_idx, max_processes)
        
        # Load instruction register
        ir_opcode = core[pc, _OPCODE]
        ir_modifier = core[pc, _MODIFIER]
        ir_a_mode = core[pc, _A_MODE]
        ir_b_mode = core[pc, _B_MODE]
        ir_a_number = core[pc, _A_NUMBER]
        ir_b_number = core[pc, _B_NUMBER]
        
        if debug:
            print("W", w_idx, "PC", pc, "OP", ir_opcode, "MOD", ir_modifier, 
                  "AM", ir_a_mode, "BM", ir_b_mode, "A", ir_a_number, "B", ir_b_number)
        
        # Evaluate A-operand
        pip_a = pc  # Default value, only used if indirect mode
        if ir_a_mode == _IMMEDIATE:
            rpa = 0
            wpa = 0
        else:
            rpa = _trim(ir_a_number, coresize)
            wpa = _trim(ir_a_number, coresize)
            
            if ir_a_mode != _DIRECT:
                pip_a = _trim(pc + wpa, coresize)
                
                # Pre-decrement
                if ir_a_mode == _PREDEC_A:
                    core[pip_a, _A_NUMBER] -= 1
                elif ir_a_mode == _PREDEC_B:
                    core[pip_a, _B_NUMBER] -= 1
                
                # Calculate indirect address
                if ir_a_mode in (_PREDEC_A, _INDIRECT_A, _POSTINC_A):
                    indirect_val = core[_trim(pc + rpa, coresize), _A_NUMBER]
                else:
                    indirect_val = core[_trim(pc + rpa, coresize), _B_NUMBER]
                
                rpa = _trim(rpa + indirect_val, coresize)
                wpa = _trim(wpa + indirect_val, coresize)
        
        # Load instruction at A pointer
        ira_opcode = core[_trim(pc + rpa, coresize), _OPCODE]
        ira_modifier = core[_trim(pc + rpa, coresize), _MODIFIER]
        ira_a_mode = core[_trim(pc + rpa, coresize), _A_MODE]
        ira_b_mode = core[_trim(pc + rpa, coresize), _B_MODE]
        ira_a_number = core[_trim(pc + rpa, coresize), _A_NUMBER]
        ira_b_number = core[_trim(pc + rpa, coresize), _B_NUMBER]
        
        # Post-increment for A
        if ir_a_mode == _POSTINC_A:
            core[pip_a, _A_NUMBER] += 1
        elif ir_a_mode == _POSTINC_B:
            core[pip_a, _B_NUMBER] += 1
        
        # Evaluate B-operand
        pip_b = pc  # Default value, only used if indirect mode
        if ir_b_mode == _IMMEDIATE:
            rpb = 0
            wpb = 0
        else:
            rpb = _trim(ir_b_number, coresize)
            wpb = _trim(ir_b_number, coresize)
            
            if ir_b_mode != _DIRECT:
                pip_b = _trim(pc + wpb, coresize)
                
                # Pre-decrement
                if ir_b_mode == _PREDEC_A:
                    core[pip_b, _A_NUMBER] -= 1
                elif ir_b_mode == _PREDEC_B:
                    core[pip_b, _B_NUMBER] -= 1
                
                # Calculate indirect address
                if ir_b_mode in (_PREDEC_A, _INDIRECT_A, _POSTINC_A):
                    indirect_val = core[_trim(pc + rpb, coresize), _A_NUMBER]
                else:
                    indirect_val = core[_trim(pc + rpb, coresize), _B_NUMBER]
                
                rpb = _trim(rpb + indirect_val, coresize)
                wpb = _trim(wpb + indirect_val, coresize)
        
        # Load instruction at B pointer
        irb_a_number = core[_trim(pc + rpb, coresize), _A_NUMBER]
        irb_b_number = core[_trim(pc + rpb, coresize), _B_NUMBER]
        
        # Post-increment for B
        if ir_b_mode == _POSTINC_A:
            core[pip_b, _A_NUMBER] += 1
        elif ir_b_mode == _POSTINC_B:
            core[pip_b, _B_NUMBER] += 1
        
        # Absolute write address
        wpb_abs = _trim(pc + wpb, coresize)
        rpa_abs = _trim(pc + rpa, coresize)
        
        if debug:
            print("  rpa", rpa, "wpa", wpa, "rpb", rpb, "wpb", wpb, "wpb_abs", wpb_abs)
            print("  ira_a", ira_a_number, "ira_b", ira_b_number, "irb_a", irb_a_number, "irb_b", irb_b_number)
        
        # Execute instruction
        if ir_opcode == _DAT:
            # Kill process - don't enqueue
            pass
            
        elif ir_opcode == _MOV:
            if ir_modifier == _M_A:
                core[wpb_abs, _A_NUMBER] = ira_a_number
            elif ir_modifier == _M_B:
                core[wpb_abs, _B_NUMBER] = ira_b_number
            elif ir_modifier == _M_AB:
                core[wpb_abs, _B_NUMBER] = ira_a_number
            elif ir_modifier == _M_BA:
                core[wpb_abs, _A_NUMBER] = ira_b_number
            elif ir_modifier == _M_F:
                core[wpb_abs, _A_NUMBER] = ira_a_number
                core[wpb_abs, _B_NUMBER] = ira_b_number
            elif ir_modifier == _M_X:
                core[wpb_abs, _A_NUMBER] = ira_b_number
                core[wpb_abs, _B_NUMBER] = ira_a_number
            elif ir_modifier == _M_I:
                core[wpb_abs, _OPCODE] = ira_opcode
                core[wpb_abs, _MODIFIER] = ira_modifier
                core[wpb_abs, _A_MODE] = ira_a_mode
                core[wpb_abs, _B_MODE] = ira_b_mode
                core[wpb_abs, _A_NUMBER] = ira_a_number
                core[wpb_abs, _B_NUMBER] = ira_b_number
            _enqueue(task_queues, tq_heads, tq_tails, tq_sizes, w_idx, pc + 1, max_processes, coresize, coverage)
            
        elif ir_opcode == _ADD:
            if ir_modifier == _M_A:
                core[wpb_abs, _A_NUMBER] = irb_a_number + ira_a_number
            elif ir_modifier == _M_B:
                core[wpb_abs, _B_NUMBER] = irb_b_number + ira_b_number
            elif ir_modifier == _M_AB:
                core[wpb_abs, _B_NUMBER] = irb_b_number + ira_a_number
            elif ir_modifier == _M_BA:
                core[wpb_abs, _A_NUMBER] = irb_a_number + ira_b_number
            elif ir_modifier == _M_F or ir_modifier == _M_I:
                core[wpb_abs, _A_NUMBER] = irb_a_number + ira_a_number
                core[wpb_abs, _B_NUMBER] = irb_b_number + ira_b_number
            elif ir_modifier == _M_X:
                core[wpb_abs, _A_NUMBER] = irb_a_number + ira_b_number
                core[wpb_abs, _B_NUMBER] = irb_b_number + ira_a_number
            _enqueue(task_queues, tq_heads, tq_tails, tq_sizes, w_idx, pc + 1, max_processes, coresize, coverage)
            
        elif ir_opcode == _SUB:
            if ir_modifier == _M_A:
                core[wpb_abs, _A_NUMBER] = irb_a_number - ira_a_number
            elif ir_modifier == _M_B:
                core[wpb_abs, _B_NUMBER] = irb_b_number - ira_b_number
            elif ir_modifier == _M_AB:
                core[wpb_abs, _B_NUMBER] = irb_b_number - ira_a_number
            elif ir_modifier == _M_BA:
                core[wpb_abs, _A_NUMBER] = irb_a_number - ira_b_number
            elif ir_modifier == _M_F or ir_modifier == _M_I:
                core[wpb_abs, _A_NUMBER] = irb_a_number - ira_a_number
                core[wpb_abs, _B_NUMBER] = irb_b_number - ira_b_number
            elif ir_modifier == _M_X:
                core[wpb_abs, _A_NUMBER] = irb_a_number - ira_b_number
                core[wpb_abs, _B_NUMBER] = irb_b_number - ira_a_number
            _enqueue(task_queues, tq_heads, tq_tails, tq_sizes, w_idx, pc + 1, max_processes, coresize, coverage)
            
        elif ir_opcode == _MUL:
            if ir_modifier == _M_A:
                core[wpb_abs, _A_NUMBER] = irb_a_number * ira_a_number
            elif ir_modifier == _M_B:
                core[wpb_abs, _B_NUMBER] = irb_b_number * ira_b_number
            elif ir_modifier == _M_AB:
                core[wpb_abs, _B_NUMBER] = irb_b_number * ira_a_number
            elif ir_modifier == _M_BA:
                core[wpb_abs, _A_NUMBER] = irb_a_number * ira_b_number
            elif ir_modifier == _M_F or ir_modifier == _M_I:
                core[wpb_abs, _A_NUMBER] = irb_a_number * ira_a_number
                core[wpb_abs, _B_NUMBER] = irb_b_number * ira_b_number
            elif ir_modifier == _M_X:
                core[wpb_abs, _A_NUMBER] = irb_a_number * ira_b_number
                core[wpb_abs, _B_NUMBER] = irb_b_number * ira_a_number
            _enqueue(task_queues, tq_heads, tq_tails, tq_sizes, w_idx, pc + 1, max_processes, coresize, coverage)
            
        elif ir_opcode == _DIV:
            # Division by zero kills the process
            div_ok = True
            if ir_modifier == _M_A:
                if ira_a_number != 0:
                    core[wpb_abs, _A_NUMBER] = irb_a_number // ira_a_number
                else:
                    div_ok = False
            elif ir_modifier == _M_B:
                if ira_b_number != 0:
                    core[wpb_abs, _B_NUMBER] = irb_b_number // ira_b_number
                else:
                    div_ok = False
            elif ir_modifier == _M_AB:
                if ira_a_number != 0:
                    core[wpb_abs, _B_NUMBER] = irb_b_number // ira_a_number
                else:
                    div_ok = False
            elif ir_modifier == _M_BA:
                if ira_b_number != 0:
                    core[wpb_abs, _A_NUMBER] = irb_a_number // ira_b_number
                else:
                    div_ok = False
            elif ir_modifier == _M_F or ir_modifier == _M_I:
                if ira_a_number != 0 and ira_b_number != 0:
                    core[wpb_abs, _A_NUMBER] = irb_a_number // ira_a_number
                    core[wpb_abs, _B_NUMBER] = irb_b_number // ira_b_number
                else:
                    div_ok = False
            elif ir_modifier == _M_X:
                if ira_a_number != 0 and ira_b_number != 0:
                    core[wpb_abs, _A_NUMBER] = irb_a_number // ira_b_number
                    core[wpb_abs, _B_NUMBER] = irb_b_number // ira_a_number
                else:
                    div_ok = False
            if div_ok:
                _enqueue(task_queues, tq_heads, tq_tails, tq_sizes, w_idx, pc + 1, max_processes, coresize, coverage)
                
        elif ir_opcode == _MOD:
            # Mod by zero kills the process
            mod_ok = True
            if ir_modifier == _M_A:
                if ira_a_number != 0:
                    core[wpb_abs, _A_NUMBER] = irb_a_number % ira_a_number
                else:
                    mod_ok = False
            elif ir_modifier == _M_B:
                if ira_b_number != 0:
                    core[wpb_abs, _B_NUMBER] = irb_b_number % ira_b_number
                else:
                    mod_ok = False
            elif ir_modifier == _M_AB:
                if ira_a_number != 0:
                    core[wpb_abs, _B_NUMBER] = irb_b_number % ira_a_number
                else:
                    mod_ok = False
            elif ir_modifier == _M_BA:
                if ira_b_number != 0:
                    core[wpb_abs, _A_NUMBER] = irb_a_number % ira_b_number
                else:
                    mod_ok = False
            elif ir_modifier == _M_F or ir_modifier == _M_I:
                if ira_a_number != 0 and ira_b_number != 0:
                    core[wpb_abs, _A_NUMBER] = irb_a_number % ira_a_number
                    core[wpb_abs, _B_NUMBER] = irb_b_number % ira_b_number
                else:
                    mod_ok = False
            elif ir_modifier == _M_X:
                if ira_a_number != 0 and ira_b_number != 0:
                    core[wpb_abs, _A_NUMBER] = irb_a_number % ira_b_number
                    core[wpb_abs, _B_NUMBER] = irb_b_number % ira_a_number
                else:
                    mod_ok = False
            if mod_ok:
                _enqueue(task_queues, tq_heads, tq_tails, tq_sizes, w_idx, pc + 1, max_processes, coresize, coverage)
                
        elif ir_opcode == _JMP:
            _enqueue(task_queues, tq_heads, tq_tails, tq_sizes, w_idx, pc + rpa, max_processes, coresize, coverage)
            
        elif ir_opcode == _JMZ:
            if ir_modifier == _M_A or ir_modifier == _M_BA:
                if irb_a_number == 0:
                    _enqueue(task_queues, tq_heads, tq_tails, tq_sizes, w_idx, pc + rpa, max_processes, coresize, coverage)
                else:
                    _enqueue(task_queues, tq_heads, tq_tails, tq_sizes, w_idx, pc + 1, max_processes, coresize, coverage)
            elif ir_modifier == _M_B or ir_modifier == _M_AB:
                if irb_b_number == 0:
                    _enqueue(task_queues, tq_heads, tq_tails, tq_sizes, w_idx, pc + rpa, max_processes, coresize, coverage)
                else:
                    _enqueue(task_queues, tq_heads, tq_tails, tq_sizes, w_idx, pc + 1, max_processes, coresize, coverage)
            else:  # M_F, M_X, M_I
                if irb_a_number == 0 and irb_b_number == 0:
                    _enqueue(task_queues, tq_heads, tq_tails, tq_sizes, w_idx, pc + rpa, max_processes, coresize, coverage)
                else:
                    _enqueue(task_queues, tq_heads, tq_tails, tq_sizes, w_idx, pc + 1, max_processes, coresize, coverage)
                    
        elif ir_opcode == _JMN:
            if ir_modifier == _M_A or ir_modifier == _M_BA:
                if irb_a_number != 0:
                    _enqueue(task_queues, tq_heads, tq_tails, tq_sizes, w_idx, pc + rpa, max_processes, coresize, coverage)
                else:
                    _enqueue(task_queues, tq_heads, tq_tails, tq_sizes, w_idx, pc + 1, max_processes, coresize, coverage)
            elif ir_modifier == _M_B or ir_modifier == _M_AB:
                if irb_b_number != 0:
                    _enqueue(task_queues, tq_heads, tq_tails, tq_sizes, w_idx, pc + rpa, max_processes, coresize, coverage)
                else:
                    _enqueue(task_queues, tq_heads, tq_tails, tq_sizes, w_idx, pc + 1, max_processes, coresize, coverage)
            else:  # M_F, M_X, M_I
                if irb_a_number != 0 or irb_b_number != 0:
                    _enqueue(task_queues, tq_heads, tq_tails, tq_sizes, w_idx, pc + rpa, max_processes, coresize, coverage)
                else:
                    _enqueue(task_queues, tq_heads, tq_tails, tq_sizes, w_idx, pc + 1, max_processes, coresize, coverage)
                    
        elif ir_opcode == _DJN:
            if ir_modifier == _M_A or ir_modifier == _M_BA:
                core[wpb_abs, _A_NUMBER] -= 1
                if core[wpb_abs, _A_NUMBER] != 0:
                    _enqueue(task_queues, tq_heads, tq_tails, tq_sizes, w_idx, pc + rpa, max_processes, coresize, coverage)
                else:
                    _enqueue(task_queues, tq_heads, tq_tails, tq_sizes, w_idx, pc + 1, max_processes, coresize, coverage)
            elif ir_modifier == _M_B or ir_modifier == _M_AB:
                core[wpb_abs, _B_NUMBER] -= 1
                if core[wpb_abs, _B_NUMBER] != 0:
                    _enqueue(task_queues, tq_heads, tq_tails, tq_sizes, w_idx, pc + rpa, max_processes, coresize, coverage)
                else:
                    _enqueue(task_queues, tq_heads, tq_tails, tq_sizes, w_idx, pc + 1, max_processes, coresize, coverage)
            else:  # M_F, M_X, M_I
                core[wpb_abs, _A_NUMBER] -= 1
                core[wpb_abs, _B_NUMBER] -= 1
                if core[wpb_abs, _A_NUMBER] != 0 or core[wpb_abs, _B_NUMBER] != 0:
                    _enqueue(task_queues, tq_heads, tq_tails, tq_sizes, w_idx, pc + rpa, max_processes, coresize, coverage)
                else:
                    _enqueue(task_queues, tq_heads, tq_tails, tq_sizes, w_idx, pc + 1, max_processes, coresize, coverage)
                    
        elif ir_opcode == _SPL:
            _enqueue(task_queues, tq_heads, tq_tails, tq_sizes, w_idx, pc + 1, max_processes, coresize, coverage)
            _enqueue(task_queues, tq_heads, tq_tails, tq_sizes, w_idx, pc + rpa, max_processes, coresize, coverage)
            
        elif ir_opcode == _SLT:
            skip = False
            if ir_modifier == _M_A:
                skip = ira_a_number < irb_a_number
            elif ir_modifier == _M_B:
                skip = ira_b_number < irb_b_number
            elif ir_modifier == _M_AB:
                skip = ira_a_number < irb_b_number
            elif ir_modifier == _M_BA:
                skip = ira_b_number < irb_a_number
            elif ir_modifier == _M_F:
                skip = ira_a_number < irb_a_number and ira_b_number < irb_b_number
            elif ir_modifier == _M_X:
                skip = ira_a_number < irb_b_number and ira_b_number < irb_a_number
            elif ir_modifier == _M_I:
                skip = (ira_opcode == core[_trim(pc + rpb, coresize), _OPCODE] and
                        ira_modifier == core[_trim(pc + rpb, coresize), _MODIFIER] and
                        ira_a_mode == core[_trim(pc + rpb, coresize), _A_MODE] and
                        ira_b_mode == core[_trim(pc + rpb, coresize), _B_MODE] and
                        ira_a_number < irb_a_number and ira_b_number < irb_b_number)
            _enqueue(task_queues, tq_heads, tq_tails, tq_sizes, w_idx, pc + (2 if skip else 1), max_processes, coresize, coverage)
            
        elif ir_opcode == _CMP or ir_opcode == _SEQ:
            skip = False
            if ir_modifier == _M_A:
                skip = ira_a_number == irb_a_number
            elif ir_modifier == _M_B:
                skip = ira_b_number == irb_b_number
            elif ir_modifier == _M_AB:
                skip = ira_a_number == irb_b_number
            elif ir_modifier == _M_BA:
                skip = ira_b_number == irb_a_number
            elif ir_modifier == _M_F:
                skip = ira_a_number == irb_a_number and ira_b_number == irb_b_number
            elif ir_modifier == _M_X:
                skip = ira_a_number == irb_b_number and ira_b_number == irb_a_number
            elif ir_modifier == _M_I:
                irb_full = core[_trim(pc + rpb, coresize)]
                skip = (ira_opcode == irb_full[_OPCODE] and
                        ira_modifier == irb_full[_MODIFIER] and
                        ira_a_mode == irb_full[_A_MODE] and
                        ira_b_mode == irb_full[_B_MODE] and
                        ira_a_number == irb_full[_A_NUMBER] and
                        ira_b_number == irb_full[_B_NUMBER])
            _enqueue(task_queues, tq_heads, tq_tails, tq_sizes, w_idx, pc + (2 if skip else 1), max_processes, coresize, coverage)
            
        elif ir_opcode == _SNE:
            skip = False
            if ir_modifier == _M_A:
                skip = ira_a_number != irb_a_number
            elif ir_modifier == _M_B:
                skip = ira_b_number != irb_b_number
            elif ir_modifier == _M_AB:
                skip = ira_a_number != irb_b_number
            elif ir_modifier == _M_BA:
                skip = ira_b_number != irb_a_number
            elif ir_modifier == _M_F:
                skip = ira_a_number != irb_a_number or ira_b_number != irb_b_number
            elif ir_modifier == _M_X:
                skip = ira_a_number != irb_b_number or ira_b_number != irb_a_number
            elif ir_modifier == _M_I:
                irb_full = core[_trim(pc + rpb, coresize)]
                skip = not (ira_opcode == irb_full[_OPCODE] and
                           ira_modifier == irb_full[_MODIFIER] and
                           ira_a_mode == irb_full[_A_MODE] and
                           ira_b_mode == irb_full[_B_MODE] and
                           ira_a_number == irb_full[_A_NUMBER] and
                           ira_b_number == irb_full[_B_NUMBER])
            _enqueue(task_queues, tq_heads, tq_tails, tq_sizes, w_idx, pc + (2 if skip else 1), max_processes, coresize, coverage)
            
        elif ir_opcode == _NOP:
            _enqueue(task_queues, tq_heads, tq_tails, tq_sizes, w_idx, pc + 1, max_processes, coresize, coverage)


class MARS(object):
    """The MARS. Encapsulates a simulation with Numba-optimized step function."""

    def __init__(self, core=None, warriors=None, minimum_separation=100,
                 randomize=True, max_processes=None, debug=False, track_coverage=False):
        self.minimum_separation = minimum_separation
        self.warriors = warriors if warriors else []
        self.debug = debug
        self.cycle = 0
        self.track_coverage = track_coverage
        self.warrior_cov = {}
        
        # Determine core size
        if core:
            self._legacy_core = core
            self.coresize = len(core)
        else:
            self._legacy_core = Core()
            self.coresize = len(self._legacy_core)
        
        self.max_processes = max_processes if max_processes else self.coresize
        
        # Initialize numpy arrays for core: (coresize, 6)
        # [opcode, modifier, a_mode, b_mode, a_number, b_number]
        self._core = np.zeros((self.coresize, 6), dtype=np.int32)
        
        # Initialize task queue arrays
        num_warriors = len(self.warriors) if self.warriors else 1
        self._task_queues = np.zeros((num_warriors, self.max_processes), dtype=np.int32)
        self._tq_heads = np.zeros(num_warriors, dtype=np.int32)
        self._tq_tails = np.zeros(num_warriors, dtype=np.int32)
        self._tq_sizes = np.zeros(num_warriors, dtype=np.int32)
        
        # Coverage tracking array (num_warriors, coresize) - only allocated if needed
        self._coverage = np.zeros((num_warriors, self.coresize), dtype=np.bool_) if track_coverage else None

        if self.warriors:
            self.load_warriors(randomize)

    def core_event(self, warrior, address, event_type):
        """For compatibility - no-op in numba version."""
        pass

    def reset(self, clear_instruction=DEFAULT_INITIAL_INSTRUCTION):
        "Clears core and re-loads warriors."
        self._core.fill(0)
        self.load_warriors()

    def load_warriors(self, randomize=True):
        "Loads warriors to memory with starting task queues"
        
        space = self.coresize // len(self.warriors)

        for n, warrior in enumerate(self.warriors):
            warrior_position = n * space

            if randomize:
                warrior_position += randint(0, max(0, space - len(warrior) - self.minimum_separation))

            # Set up task queue for this warrior
            start_addr = (warrior_position + warrior.start) % self.coresize
            self._task_queues[n, 0] = start_addr
            self._tq_heads[n] = 0
            self._tq_tails[n] = 1
            self._tq_sizes[n] = 1
            
            # Mark initial position in coverage
            if self._coverage is not None:
                self._coverage[n, start_addr] = True
            
            # Also set legacy task_queue for compatibility with corewar_util
            warrior.task_queue = deque([start_addr])

            # Copy warrior's instructions to the numpy core
            for i, instruction in enumerate(warrior.instructions):
                addr = (warrior_position + i) % self.coresize
                self._core[addr, _OPCODE] = instruction.opcode
                self._core[addr, _MODIFIER] = instruction.modifier
                self._core[addr, _A_MODE] = instruction.a_mode
                self._core[addr, _B_MODE] = instruction.b_mode
                self._core[addr, _A_NUMBER] = instruction.a_number
                self._core[addr, _B_NUMBER] = instruction.b_number

    def step(self):
        """Run one simulation step using numba-optimized function."""
        self.cycle += 1
        
        # Call the numba JIT-compiled step function
        _step_numba(
            self._core,
            len(self.warriors),
            self._task_queues,
            self._tq_heads,
            self._tq_tails,
            self._tq_sizes,
            self.coresize,
            self.max_processes,
            self.debug,
            self._coverage
        )
    
    def get_queue_sizes(self):
        """Return array of task queue sizes for each warrior."""
        return self._tq_sizes.copy()
    
    def get_coverage_sums(self):
        """Return array of coverage sums for each warrior."""
        if self._coverage is not None:
            return self._coverage.sum(axis=1)
        return np.zeros(len(self.warriors), dtype=int)
    
    def sync_queues(self):
        """Sync numpy task queues back to warrior.task_queue deques (for compatibility)."""
        for i, warrior in enumerate(self.warriors):
            warrior.task_queue.clear()
            if self._tq_sizes[i] > 0:
                head = self._tq_heads[i]
                for j in range(self._tq_sizes[i]):
                    idx = (head + j) % self.max_processes
                    addr = self._task_queues[i, idx]
                    warrior.task_queue.append(addr)
            # Populate legacy warrior_cov dict from numpy array
            if self._coverage is not None:
                self.warrior_cov[warrior] = self._coverage[i]

    def __iter__(self):
        return iter(self._legacy_core)

    def __len__(self):
        return self.coresize

    def __getitem__(self, address):
        return self._legacy_core[address]

    # Property to access core for compatibility
    @property
    def core(self):
        return self._legacy_core