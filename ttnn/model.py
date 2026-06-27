"""
Top-level Orchestrator uniting State, Memory, Routing, and Reasoning modules.
"""

from __future__ import annotations
import pickle
import numpy as np
from typing import List
from nanograd import Tensor
from nanograd import Module
from ttnn.state import MachineState
from ttnn.memory import TNMemory
from ttnn.routing import ContractionRouter
from ttnn.reasoning import TopologyEvolver


class TensorNetworkStateMachine(Module):
    """
    An adaptive, self-modulating causal text sequencing model.
    """
    def __init__(self, vocab_size: int, initial_bond_dim: int = 4) -> None:
        self.vocab_size = vocab_size
        self.memory = TNMemory(vocab_size, initial_bond_dim)
        self.state = MachineState(initial_bond_dim)
        self.evolver = TopologyEvolver()

    def forward(self, token_sequence: List[int]) -> List[Tensor]:
        """
        Processes a sequence of inputs token-by-token.
        Updates internal bond states and generates next-token predictions.
        """
        self.state.reset()
        logits_history = []
        
        for token_idx in token_sequence:
            # 1. Route token index into a square transformation operator
            transition_matrix = ContractionRouter.route_token(
                token_idx, self.vocab_size, self.memory.W_core, self.memory.bond_dim
            )
            
            # 2. Update state history via local matrix contraction
            # (1, bond_dim) @ (bond_dim, bond_dim) -> (1, bond_dim)
            new_h = self.state.h.matmul(transition_matrix)
            self.state.update(new_h)
            
            # 3. Map the active hidden state to vocabulary prediction logits
            # (1, bond_dim) @ (bond_dim, vocab_size) -> (1, vocab_size)
            logits = self.state.h.matmul(self.memory.W_head)
            logits_history.append(logits)
            
        return logits_history

    def evolve_topology(self) -> bool:
        """
        Invokes the reasoning engine to rewrite operational shape states.
        Synchronizes internal trackers if dimensions shift.
        """
        mutated = self.evolver.evolve_memory(self.memory)
        if mutated:
            self.state.bond_dim = self.memory.bond_dim
        return mutated

    def parameters(self) -> List[Tensor]:
        return self.memory.parameters()

    def get_state_dict(self) -> dict:
        """Extracts clean structural metadata and raw numerical weight matrices."""
        return {
            "vocab_size": self.vocab_size,
            "bond_dim": self.memory.bond_dim,
            "W_core_data": self.memory.W_core.data.copy(),
            "W_head_data": self.memory.W_head.data.copy(),
        }

    def load_state_dict(self, state_dict: dict) -> None:
        """Restores structural dimensions and weight parameters cleanly."""
        self.vocab_size = state_dict["vocab_size"]
        self.memory.bond_dim = state_dict["bond_dim"]
        self.state.bond_dim = state_dict["bond_dim"]
        
        # Instantiate fresh, unlinked Tensors to clear old autograd pointers
        self.memory.W_core = Tensor(state_dict["W_core_data"], label="W_core")
        self.memory.W_head = Tensor(state_dict["W_head_data"], label="W_head")


# =====================================================================
#  INFERENCE & SERIALIZATION UTILITIES
# =====================================================================

def generate_text(
    model: TensorNetworkStateMachine, 
    seed_text: str, 
    length: int, 
    char_to_idx: dict, 
    idx_to_char: dict, 
    temperature: float = 0.8
) -> str:
    """
    Generates text autoregressively by shifting the internal memory state wheel forward.
    """
    model.state.reset()
    generated = seed_text
    
    # 1. Warm up the state machine memory with the seed text prompt
    for char in seed_text:
        if char in char_to_idx:
            idx = char_to_idx[char]
            transition = ContractionRouter.route_token(
                idx, model.vocab_size, model.memory.W_core, model.memory.bond_dim
            )
            model.state.update(model.state.h.matmul(transition))
            
    # 2. Infinite horizon rolling generation loop
    for _ in range(length):
        # Decode current hidden state to vocabulary space
        logits = model.state.h.matmul(model.memory.W_head).data.flatten()
        
        # Apply temperature scaling to control creativity
        logits = logits / max(temperature, 1e-5)
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / np.sum(exp_logits)
        
        # Sample next token from the structural distribution
        next_idx = np.random.choice(len(probs), p=probs)
        next_char = idx_to_char[next_idx]
        generated += next_char
        
        # Cycle the generated token back into the memory matrix to advance the state
        transition = ContractionRouter.route_token(
            next_idx, model.vocab_size, model.memory.W_core, model.memory.bond_dim
        )
        model.state.update(model.state.h.matmul(transition))
        
    return generated

def save_model(model: TensorNetworkStateMachine, filepath: str) -> None:
    """Serializes a clean dictionary state to disk bypassing autograd nodes."""
    state = model.get_state_dict()
    with open(filepath, 'wb') as f:
        pickle.dump(state, f)
    print(f"[System] Model topology and weights successfully saved to {filepath}")


def load_model(filepath: str) -> TensorNetworkStateMachine:
    """Instantiates a state machine using saved parameter layouts."""
    with open(filepath, 'rb') as f:
        state_dict = pickle.load(f)
        
    model = TensorNetworkStateMachine(
        vocab_size=state_dict["vocab_size"], 
        initial_bond_dim=state_dict["bond_dim"]
    )
    model.load_state_dict(state_dict)
    return model
