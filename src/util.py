import os
import json
import pickle

from dataclasses import dataclass, field
from corewar import redcode, Warrior
import numpy as np
import hashlib

@dataclass
class ExplicitWarrior:
    """Warrior container for explicit mutation (no LLM)."""
    warrior: Warrior | None = None
    code: str = ""
    fitness: float | None = None  # Hill score (wins*3 + ties*1)
    bc: tuple | None = None
    outputs: dict = field(default_factory=dict)
    id: str = ""
    error: str = ""
    parent_id: str = ""
    
    # Hill scoring stats
    wins: int = 0
    losses: int = 0
    ties: int = 0
    total_rounds: int = 0
    hill_score_pct: float = 0.0  # As percentage (0-100)

    def __post_init__(self):
        if self.id == "" and self.warrior is not None:
            self.id = self._compute_id()

    def _compute_id(self) -> str:
        """Compute unique ID from warrior instructions."""
        if self.warrior is None:
            return hashlib.sha256(self.code.encode()).hexdigest()[:16]

        instr_str = ""
        for instr in self.warrior.instructions:
            instr_str += f"{instr.opcode}:{instr.modifier}:{instr.a_mode}:{instr.a_number}:{instr.b_mode}:{instr.a_number}|"
        return hashlib.sha256(instr_str.encode()).hexdigest()[:16]

    @property
    def name(self) -> str:
        if self.warrior is not None and self.warrior.name:
            return self.warrior.name
        return f"warrior_{self.id[:8]}"
    
    @property
    def win_pct(self) -> float:
        return self.wins / self.total_rounds * 100 if self.total_rounds > 0 else 0
    
    @property
    def tie_pct(self) -> float:
        return self.ties / self.total_rounds * 100 if self.total_rounds > 0 else 0
    
    @property
    def loss_pct(self) -> float:
        return self.losses / self.total_rounds * 100 if self.total_rounds > 0 else 0
    
def save_json(save_dir, name, item):
    if save_dir is not None:
        os.makedirs(f"{save_dir}/", exist_ok=True)
        with open(f"{save_dir}/{name}.json", "w") as f:
            json.dump(item, f)
            
def load_json(load_dir, name):
    if load_dir is not None:
        with open(f"{load_dir}/{name}.json", "r") as f:
            return json.load(f)
    else:
        return None

def save_pkl(save_dir, name, item):
    if save_dir is not None:
        os.makedirs(f"{save_dir}/", exist_ok=True)
        with open(f"{save_dir}/{name}.pkl", "wb") as f:
            pickle.dump(item, f)


def load_pkl(load_dir, name):
    if load_dir is not None:
        with open(f"{load_dir}/{name}.pkl", "rb") as f:
            return pickle.load(f)
    else:
        return None