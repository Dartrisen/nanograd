"""
Top-level Orchestrator uniting State, Memory, Routing, and Reasoning modules.
"""
from __future__ import annotations
import pickle
import numpy as np
from nanograd import Tensor, Module
from ttnn.state import MachineState
from ttnn.memory import TNMemory
from ttnn.routing import ContractionRouter
from ttnn.reasoning import TopologyEvolver


class TensorNetworkStateMachine(Module):
    """
    Full Tensor Train state machine with Bilinear TT-Gate recurrence for long-context sequences.
    """
    def __init__(self, vocab_size: int, initial_bond_dim: int = 16, tt_rank: int = 16) -> None:
        self.vocab_size = vocab_size
        self.memory = TNMemory(vocab_size, initial_bond_dim, tt_rank=tt_rank)
        self.state = MachineState(initial_bond_dim)
        self.evolver = TopologyEvolver()

    def forward(self, token_sequence: list[int]) -> list[Tensor]:
        self.state.reset()
        logits_history = []

        for token_idx in token_sequence:
            token_one_hot = np.zeros((1, self.vocab_size), dtype=np.float32)
            token_one_hot[0, token_idx] = 1.0
            token_tensor = Tensor(token_one_hot, label=f"token_{token_idx}")

            token_embed = self.memory.W_embed.forward(token_tensor)

            # Compute continuous state retention gate in (0, 1) using bounded activation
            raw_gate = self.memory.W_gate.forward(token_tensor)
            gate = 0.5 * (1.0 + raw_gate.tanh())

            # Route token through 2-core transition operator
            transition_matrix = ContractionRouter.route_token(
                token_idx, self.vocab_size, self.memory.G1_core, self.memory.G2_core, self.memory.bond_dim
            )

            # Bilinear gated linear update: h_t = gate * (h_{t-1} @ T_t) + (1 - gate) * embed_t
            state_transition = self.state.h.matmul(transition_matrix)
            updated_state = gate * state_transition + (1.0 - gate) * token_embed
            self.state.update(updated_state)

            # Project state to vocabulary logits
            projected_state = self.memory.W_proj.forward(self.state.h)
            logits = self.memory.W_head.forward(projected_state)
            logits_history.append(logits)

        return logits_history

    def parameters(self) -> list[Tensor]:
        return self.memory.parameters()

    def get_state_dict(self) -> dict:
        return {
            "vocab_size": self.vocab_size,
            "bond_dim": self.memory.bond_dim,
            "tt_rank": self.memory.tt_rank,
            "G1_core_data": self.memory.G1_core.data.copy(),
            "G2_core_data": self.memory.G2_core.data.copy(),
            "W_embed_cores": [c.data.copy() for c in self.memory.W_embed.cores],
            "W_gate_cores": [c.data.copy() for c in self.memory.W_gate.cores],
            "W_proj_cores": [c.data.copy() for c in self.memory.W_proj.cores],
            "W_head_cores": [c.data.copy() for c in self.memory.W_head.cores],
        }

    def load_state_dict(self, state_dict: dict) -> None:
        self.vocab_size = state_dict["vocab_size"]
        self.memory.bond_dim = state_dict["bond_dim"]
        self.memory.tt_rank = state_dict["tt_rank"]
        self.state.bond_dim = state_dict["bond_dim"]

        self.memory.G1_core = Tensor(state_dict["G1_core_data"], label="G1_core")
        self.memory.G2_core = Tensor(state_dict["G2_core_data"], label="G2_core")

        for core, data in zip(self.memory.W_embed.cores, state_dict["W_embed_cores"]):
            core.data = data.copy()
        for core, data in zip(self.memory.W_gate.cores, state_dict["W_gate_cores"]):
            core.data = data.copy()
        for core, data in zip(self.memory.W_proj.cores, state_dict["W_proj_cores"]):
            core.data = data.copy()
        for core, data in zip(self.memory.W_head.cores, state_dict["W_head_cores"]):
            core.data = data.copy()

# =====================================================================
#  INFERENCE & SERIALIZATION UTILITIES
# =====================================================================

def generate_text(
    model: TensorNetworkStateMachine,
    seed_text: str,
    length: int,
    char_to_idx: dict,
    idx_to_char: dict,
    temperature: float = 0.7
) -> str:
    generated = seed_text
    model.state.reset()

    for char in seed_text:
        if char in char_to_idx:
            idx = char_to_idx[char]
            token_one_hot = np.zeros((1, model.vocab_size), dtype=np.float32)
            token_one_hot[0, idx] = 1.0
            token_tensor = Tensor(token_one_hot, label=f"seed_{idx}")

            token_embed = model.memory.W_embed.forward(token_tensor)
            raw_gate = model.memory.W_gate.forward(token_tensor)
            gate = 0.5 * (1.0 + raw_gate.tanh())

            transition = ContractionRouter.route_token(
                idx, model.vocab_size, model.memory.G1_core, model.memory.G2_core, model.memory.bond_dim
            )
            state_transition = model.state.h.matmul(transition)
            updated_state = gate * state_transition + (1.0 - gate) * token_embed
            model.state.update(updated_state)

    for _ in range(length):
        projected_state = model.memory.W_proj.forward(model.state.h)
        logits = model.memory.W_head.forward(projected_state).data.flatten()

        logits = logits / max(temperature, 1e-5)
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / np.sum(exp_logits)

        next_idx = np.random.choice(len(probs), p=probs)
        next_char = idx_to_char[next_idx]
        generated += next_char

        token_one_hot = np.zeros((1, model.vocab_size), dtype=np.float32)
        token_one_hot[0, next_idx] = 1.0
        token_tensor = Tensor(token_one_hot, label=f"gen_{next_idx}")

        token_embed = model.memory.W_embed.forward(token_tensor)
        raw_gate = model.memory.W_gate.forward(token_tensor)
        gate = 0.5 * (1.0 + raw_gate.tanh())

        transition = ContractionRouter.route_token(
            next_idx, model.vocab_size, model.memory.G1_core, model.memory.G2_core, model.memory.bond_dim
        )
        state_transition = model.state.h.matmul(transition)
        updated_state = gate * state_transition + (1.0 - gate) * token_embed
        model.state.update(updated_state)

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
