"""
GPT-style training, checkpointing, and $O(1)$ inference execution track.
"""

import os
import urllib.request
import numpy as np
from nanograd.core.tensor import Tensor
from ttnn.model import TensorNetworkStateMachine
from ttnn.model import save_model, load_model, generate_text

# Custom cross-entropy to handle autograd trace safely inside NanoGrad
def cross_entropy_loss(logits_tensor: Tensor, target_idx: int) -> Tensor:
    raw_logits = logits_tensor.data.flatten()
    stabilized_logits = raw_logits - np.max(raw_logits)
    exps = np.exp(stabilized_logits)
    probabilities = exps / np.sum(exps)
    
    loss_scalar = -np.log(probabilities[target_idx] + 1e-15)
    out = Tensor(np.array([[loss_scalar]]), _children=(logits_tensor,), _op='cross_entropy')
    
    def _backward():
        derivative = probabilities.copy()
        derivative[target_idx] -= 1.0
        logits_tensor.grad += out.grad * derivative.reshape(logits_tensor.shape)
    out._backward = _backward
    return out

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
    if os.path.exists(checkpoint_file):
        print(f"\n[System] Found existing checkpoint. Loading model...")
        model = load_model(checkpoint_file)
    else:
        print(f"\n[System] Initializing new Tensor Network State Machine...")
        # Start with a conservative rank footprint
        model = TensorNetworkStateMachine(vocab_size=vocab_size, initial_bond_dim=8)
    
    # --- PHASE 3: CAUSAL NEXT-TOKEN TRAINING LOOP ---
    epochs = 5
    seq_len = 12
    learning_rate = 0.01
    
    print(f"\nTraining configuration: {epochs} Epochs | Sequence Window: {seq_len} | Vocabulary: {vocab_size}")
    print(f"Starting weights grid dimensions: {model.memory.W_core.data.shape}")
    
    for epoch in range(epochs):
        total_loss = 0.0
        steps = 0
        
        # Step through data using an overlapping sliding window
        for offset in range(0, len(tokenized) - seq_len - 1, 15):
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

            max_norm = 1.0
            for p in model.parameters():
                if p.grad is not None:
                    # Clip extreme individual elements to safe operational bounds
                    p.grad = np.clip(p.grad, -max_norm, max_norm)
            
            # Parameter optimization update
            for p in model.parameters():
                p.data -= learning_rate * p.grad
                
        # --- PHASE 4: MID-FLIGHT TOPOLOGY EVOLUTION ---
        avg_loss = total_loss / steps
        print(f"Epoch {epoch+1:02d}/{epochs:02d} | Causal Loss: {avg_loss:.4f}")
        
        # Every epoch, let the network evaluate its own singular value allocations
        structure_changed = model.evolve_topology()
        if structure_changed:
            print(f"  -> Adapted configuration tracking shape: {model.memory.W_core.data.shape}")

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
