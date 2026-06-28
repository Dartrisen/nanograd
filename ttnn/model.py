"""
Top-level Orchestrator uniting State, Memory, Routing, and Reasoning modules.
"""

from __future__ import annotations
import pickle
import numpy as np
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

    def forward(self, token_sequence: list[int]) -> list[Tensor]:
        """
        Processes a sequence of inputs token-by-token.
        Updates internal bond states and generates next-token predictions.
        """
        self.state.reset()
        logits_history = []

        for token_idx in token_sequence:
            # 1. Embed the token and route it into a transition operator.
            token_one_hot = np.zeros((1, self.vocab_size), dtype=np.float32)
            token_one_hot[0, token_idx] = 1.0
            token_embed = Tensor(token_one_hot, label=f"token_embed_{token_idx}").matmul(self.memory.W_embed)

            transition_matrix = ContractionRouter.route_token(
                token_idx, self.vocab_size, self.memory.W_core, self.memory.bond_dim
            )

            # 2. Update the hidden state with a residual-style transition and a nonlinear activation.
            state_transition = self.state.h.matmul(transition_matrix)
            combined_state = state_transition + token_embed + 0.5 * self.state.h
            activated_state = combined_state.tanh()
            self.state.update(activated_state)

            # 3. Project the hidden state and map it to vocabulary prediction logits.
            projected_state = self.state.h.matmul(self.memory.W_proj)
            logits = projected_state.matmul(self.memory.W_head)
            logits_history.append(logits)

        return logits_history

    def evolve_topology(self) -> bool:
        """
        Invokes the reasoning engine to rewrite operational shape states.
        Synchronizes internal trackers and slices auxiliary weights if dimensions shift.
        """
        mutated = self.evolver.evolve_memory(self.memory)
        if mutated:
            new_dim = self.memory.bond_dim
            self.state.bond_dim = new_dim
            
            print(f"[System] Syncing auxiliary matrices to new bond dimension: {new_dim}")
            # Slice embedding matrix: Keep rows, truncate columns to new_dim
            self.memory.W_embed = Tensor(self.memory.W_embed.data[:, :new_dim], label="W_embed")
            # Slice projection matrix: Truncate both rows and columns to new_dim
            self.memory.W_proj = Tensor(self.memory.W_proj.data[:new_dim, :new_dim], label="W_proj")
            # Slice prediction head matrix: Truncate rows, keep vocabulary columns
            self.memory.W_head = Tensor(self.memory.W_head.data[:new_dim, :], label="W_head")
            
        return mutated

    def parameters(self) -> list[Tensor]:
        return self.memory.parameters()

    def get_state_dict(self) -> dict:
        """Extracts clean structural metadata and raw numerical weight matrices."""
        return {
            "vocab_size": self.vocab_size,
            "bond_dim": self.memory.bond_dim,
            "W_embed_data": self.memory.W_embed.data.copy(),
            "W_core_data": self.memory.W_core.data.copy(),
            "W_proj_data": self.memory.W_proj.data.copy(),
            "W_head_data": self.memory.W_head.data.copy(),
        }

    def load_state_dict(self, state_dict: dict) -> None:
        """Restores structural dimensions and weight parameters cleanly."""
        self.vocab_size = state_dict["vocab_size"]
        self.memory.bond_dim = state_dict["bond_dim"]
        self.state.bond_dim = state_dict["bond_dim"]

        # Instantiate fresh, unlinked Tensors to clear old autograd pointers.
        embed_shape = (self.vocab_size, self.memory.bond_dim)
        embed_data = state_dict.get("W_embed_data")
        if embed_data is None or embed_data.shape != embed_shape:
            embed_data = np.random.randn(*embed_shape) * 0.05
        self.memory.W_embed = Tensor(embed_data, label="W_embed")
        self.memory.W_core = Tensor(state_dict["W_core_data"], label="W_core")
        self.memory.W_proj = Tensor(state_dict.get("W_proj_data", np.random.randn(self.memory.bond_dim, self.memory.bond_dim) * 0.05), label="W_proj")
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
    # model.state.reset()
    generated = seed_text

    # 1. Warm up the state machine memory with the seed text prompt
    for char in seed_text:
        if char in char_to_idx:
            idx = char_to_idx[char]
            token_one_hot = np.zeros((1, model.vocab_size), dtype=np.float32)
            token_one_hot[0, idx] = 1.0
            token_embed = Tensor(token_one_hot, label=f"seed_embed_{idx}").matmul(model.memory.W_embed)
            transition = ContractionRouter.route_token(
                idx, model.vocab_size, model.memory.W_core, model.memory.bond_dim
            )
            combined_state = model.state.h.matmul(transition) + token_embed + 0.5 * model.state.h
            model.state.update(combined_state.tanh())

    # 2. Infinite horizon rolling generation loop
    for _ in range(length):
        # Decode current hidden state to vocabulary space
        projected_state = model.state.h.matmul(model.memory.W_proj)
        logits = projected_state.matmul(model.memory.W_head).data.flatten()

        # Apply temperature scaling to control creativity
        logits = logits / max(temperature, 1e-5)
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / np.sum(exp_logits)

        # Sample next token from the structural distribution
        next_idx = np.random.choice(len(probs), p=probs)
        next_char = idx_to_char[next_idx]
        generated += next_char

        # Cycle the generated token back into the memory matrix to advance the state.
        token_one_hot = np.zeros((1, model.vocab_size), dtype=np.float32)
        token_one_hot[0, next_idx] = 1.0
        token_embed = Tensor(token_one_hot, label=f"gen_embed_{next_idx}").matmul(model.memory.W_embed)
        transition = ContractionRouter.route_token(
            next_idx, model.vocab_size, model.memory.W_core, model.memory.bond_dim
        )
        combined_state = model.state.h.matmul(transition) + token_embed + 0.5 * model.state.h
        model.state.update(combined_state.tanh())

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
