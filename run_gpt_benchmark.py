"""
GPT-style training, checkpointing, and $O(1)$ inference execution track.
"""

import os
import urllib.request
import numpy as np
from nanograd import Tensor, cross_entropy_loss
from nanograd.optimiser import Adam
from ttnn import TensorNetworkStateMachine, save_model, load_model, generate_text


def main():
    np.random.seed(1337)
    checkpoint_file = "tnsm_shakespeare.pkl"

    # --- PHASE 1: DATA ACQUISITION & PROCESSING ---
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    print("[Data] Syncing with Tiny Shakespeare repository...")
    try:
        with urllib.request.urlopen(url) as response:
            raw_data = response.read().decode('utf-8')
        # Isolate a clean text slice appropriate for micro-framework tracking speeds
        corpus = raw_data[:25000]
        print(f"[Data] Successfully loaded {len(corpus)} characters.")
    except Exception as e:
        print(f"[Warning] Online fetch failed ({e}). Falling back to internal text asset.")
        corpus = "To be, or not to be, that is the question: Whether 'tis nobler in the mind to suffer the slings and arrows of outrageous fortune." * 200

    chars = sorted(list(set(corpus)))
    vocab_size = len(chars)
    char_to_idx = {ch: i for i, ch in enumerate(chars)}
    idx_to_char = {i: ch for i, ch in enumerate(chars)}
    tokenized = [char_to_idx[ch] for ch in corpus]

    # --- PHASE 2: INITIALIZATION OR RESUMPTION ---
    resume_training = True
    if resume_training and os.path.exists(checkpoint_file):
        print("\n[System] Found existing checkpoint. Loading model...")
        model = load_model(checkpoint_file)
    else:
        print("\n[System] Initializing new Tensor Network State Machine...")
        # Start with a conservative rank footprint
        model = TensorNetworkStateMachine(vocab_size=vocab_size, initial_bond_dim=32)
    # Force the model to allow a larger rank ceiling for potential topology evolution
    model.evolver.max_rank = 32

    # --- PHASE 3: CAUSAL NEXT-TOKEN TRAINING LOOP ---
    epochs = 30
    seq_len = 32
    learning_rate = 0.02

    print(f"\nTraining configuration: {epochs} Epochs | Sequence Window: {seq_len} | Vocabulary: {vocab_size}")
    print(f"Starting weights grid dimensions: {model.memory.W_core.data.shape}")

    optimizer = Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        total_loss = 0.0
        steps = 0

        # Step through data using a mix of overlapping windows and random start points
        window_starts = list(range(0, len(tokenized) - seq_len - 1, seq_len))
        np.random.shuffle(window_starts)
        if not window_starts:
            window_starts = [0]

        for offset in window_starts:
            inputs = tokenized[offset : offset + seq_len]
            targets = tokenized[offset + 1 : offset + seq_len + 1]

            model.zero_grad()
            logits_history = model.forward(inputs)

            # Accumulate loss over the execution sequence steps
            loss_node = Tensor(np.array([[0.0]]))
            for t in range(seq_len):
                step_loss = cross_entropy_loss(logits_history[t], targets[t])
                loss_node = loss_node + step_loss

            total_loss += loss_node.data[0][0]
            steps += 1

            # Compute gradients via backpropagation
            loss_node.backward()

            max_norm = 5.0
            current_lr = learning_rate / (1.0 + 0.005 * epoch)
            optimizer.lr = current_lr
            optimizer.step(max_norm=max_norm)

        # --- PHASE 4: MID-FLIGHT TOPOLOGY EVOLUTION ---
        avg_loss = total_loss / steps
        print(f"Epoch {epoch+1:02d}/{epochs:02d} | Causal Loss: {avg_loss:.4f}")

        # Every epoch, let the network evaluate its own singular value allocations
        structure_changed = model.evolve_topology()
        if structure_changed:
            print(f"  -> Adapted configuration tracking shape: {model.memory.W_core.data.shape}")
            optimizer.parameters = model.parameters()

    # Save final model state
    save_model(model, checkpoint_file)

    # --- PHASE 5: AUTOREGRESSIVE GENERATION INFERENCE ---
    print("\n" + "="*52)
    print("INFERENCE RUN: GENERATING FROM TRAINED STATE MACHINE")
    print("="*52)

    prompts = ["ROMEO: ", "The king ", "To suffer "]
    for seed in prompts:
        output_generation = generate_text(
            model=model,
            seed_text=seed,
            length=80,
            char_to_idx=char_to_idx,
            idx_to_char=idx_to_char,
            temperature=0.7
        )
        print(f"\nPrompt Input: '{seed}'\nGenerated Extension:\n{output_generation}")
        print("-" * 40)

if __name__ == "__main__":
    main()
