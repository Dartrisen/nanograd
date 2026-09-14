"""
GPT-style training and $O(1)$ inference execution track.
Configured for higher state capacity (bond_dim=16, tt_rank=16) and normalized per-token loss.
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
    cache_file = "tinyshakespeare.txt"

    # --- DATA ACQUISITION & CACHING ---
    if os.path.exists(cache_file):
        print(f"[Data] Loading dataset from local cache: '{cache_file}'...")
        with open(cache_file, "r", encoding="utf-8") as f:
            raw_data = f.read()
        print(f"[Data] Successfully loaded {len(raw_data)} characters from disk.")
    else:
        url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        print("[Data] Cache not found. Fetching online dataset...")
        try:
            with urllib.request.urlopen(url) as response:
                raw_data = response.read().decode('utf-8')
            with open(cache_file, "w", encoding="utf-8") as f:
                f.write(raw_data)
            print(f"[Data] Saved {len(raw_data)} characters locally to '{cache_file}'.")
        except Exception as e:
            print(f"[Warning] Online fetch failed ({e}). Using local fallback string.")
            raw_data = "To be, or not to be, that is the question: Whether 'tis nobler in the mind to suffer." * 200

    corpus = raw_data[:25000]
    chars = sorted(list(set(corpus)))
    vocab_size = len(chars)
    char_to_idx = {ch: i for i, ch in enumerate(chars)}
    idx_to_char = {i: ch for i, ch in enumerate(chars)}
    tokenized = [char_to_idx[ch] for ch in corpus]

    # --- INITIALIZATION (Bond Dim = 16, TT Rank = 16) ---
    resume_training = False  # Set to False to train the new high-capacity state machine from scratch
    if resume_training and os.path.exists(checkpoint_file):
        print(f"\n[System] Found existing checkpoint. Loading model...")
        model = load_model(checkpoint_file)
    else:
        print(f"\n[System] Initializing new High-Capacity Tensor Network State Machine...")
        model = TensorNetworkStateMachine(vocab_size=vocab_size, initial_bond_dim=16, tt_rank=16)

    # --- TRAINING CONFIGURATION ---
    epochs = 30
    seq_len = 12
    learning_rate = 0.01

    print(f"\nTraining configuration: {epochs} Epochs | Window: {seq_len} | Vocab: {vocab_size}")
    print(f"Bond Dimension: {model.memory.bond_dim} | TT Rank: {model.memory.tt_rank}")
    print(f"TT Cores: G1={model.memory.G1_core.data.shape}, G2={model.memory.G2_core.data.shape}")

    optimizer = Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        total_loss = 0.0
        steps = 0

        window_starts = list(range(0, len(tokenized) - seq_len - 1, seq_len))
        np.random.shuffle(window_starts)
        if not window_starts:
            window_starts = [0]

        for offset in window_starts:
            inputs = tokenized[offset : offset + seq_len]
            targets = tokenized[offset + 1 : offset + seq_len + 1]

            model.zero_grad()
            logits_history = model.forward(inputs)

            # Compute sequence loss
            loss_node = Tensor(np.array([[0.0]]))
            for t in range(seq_len):
                step_loss = cross_entropy_loss(logits_history[t], targets[t])
                loss_node = loss_node + step_loss

            # Normalize loss per token step (Cross Entropy Per Token)
            normalized_loss = loss_node * (1.0 / seq_len)
            total_loss += normalized_loss.data[0][0]
            steps += 1

            normalized_loss.backward()

            optimizer.lr = learning_rate / (1.0 + 0.005 * epoch)
            optimizer.step(max_norm=5.0)

        avg_token_loss = total_loss / steps
        print(f"Epoch {epoch+1:02d}/{epochs:02d} | Per-Token Loss: {avg_token_loss:.4f}")

    save_model(model, checkpoint_file)

    # --- INFERENCE RUN ---
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
