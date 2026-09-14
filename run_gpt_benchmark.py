"""
GPT-style long-context training benchmark with Bilinear TT-Gates over L=256 sequence windows.
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

    if os.path.exists(cache_file):
        print(f"[Data] Loading dataset from local cache: '{cache_file}'...")
        with open(cache_file, "r", encoding="utf-8") as f:
            raw_data = f.read()
    else:
        url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        print("[Data] Syncing with Tiny Shakespeare repository...")
        with urllib.request.urlopen(url) as response:
            raw_data = response.read().decode('utf-8')
        with open(cache_file, "w", encoding="utf-8") as f:
            f.write(raw_data)

    corpus = raw_data[:50000]
    chars = sorted(set(corpus))
    vocab_size = len(chars)
    char_to_idx = {ch: i for i, ch in enumerate(chars)}
    idx_to_char = {i: ch for i, ch in enumerate(chars)}
    tokenized = [char_to_idx[ch] for ch in corpus]

    print("\n[System] Initializing Bilinear TT-Gated State Machine...")
    model = TensorNetworkStateMachine(vocab_size=vocab_size, initial_bond_dim=32, tt_rank=32)

    # Long-Context Configuration
    epochs = 20
    seq_len = 256
    learning_rate = 0.015

    print(f"Training parameters: {epochs} Epochs | Context Window L={seq_len} | Vocab={vocab_size}")
    optimizer = Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        total_loss = 0.0
        steps = 0

        window_starts = list(range(0, len(tokenized) - seq_len - 1, seq_len))
        np.random.shuffle(window_starts)

        for offset in window_starts:
            inputs = tokenized[offset : offset + seq_len]
            targets = tokenized[offset + 1 : offset + seq_len + 1]

            model.zero_grad()
            logits_history = model.forward(inputs)

            loss_node = Tensor(np.array([[0.0]]))
            for t in range(seq_len):
                step_loss = cross_entropy_loss(logits_history[t], targets[t])
                loss_node = loss_node + step_loss

            normalized_loss = loss_node * (1.0 / seq_len)
            total_loss += normalized_loss.data[0][0]
            steps += 1

            normalized_loss.backward()
            optimizer.lr = learning_rate / (1.0 + 0.01 * epoch)
            optimizer.step(max_norm=5.0)

        avg_loss = total_loss / max(steps, 1)
        print(f"Epoch {epoch+1:02d}/{epochs:02d} | Long-Context Loss (L={seq_len}): {avg_loss:.4f}")

    save_model(model, checkpoint_file)

    print("\n" + "="*52)
    print("INFERENCE GENERATION (L=256 Bilinear TT-Gate Model)")
    print("="*52)

    for seed in ["ROMEO: ", "The king "]:
        output = generate_text(model, seed, 100, char_to_idx, idx_to_char)
        print(f"\nPrompt: '{seed}'\nGenerated:\n{output}\n" + "-"*40)


if __name__ == "__main__":
    main()
