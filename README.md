# nanograd

nanograd is a compact, from-scratch machine learning library that combines a tiny autograd engine with a simple tensor-train (TT) decomposition toolkit. The project is intentionally educational: it focuses on clarity over production polish, making it a good reference for anyone who wants to understand how neural networks, automatic differentiation, and tensor compression work under the hood.

## What is inside

- A lightweight autograd implementation with `Value` and `Tensor` objects
- Small neural building blocks such as `Module`, `Neuron`, `Layer`, and `MLP`
- A tensor-train representation with `TTCore` and `TensorTrain`
- TT-SVD based decomposition helpers via `from_tensor`
- A few small examples and tests that exercise the core ideas

## Why this project exists

This repository is a useful bridge between two related ideas:

1. Building neural networks by hand with a minimal autograd engine
2. Compressing high-order tensors with tensor-train decomposition

That makes it a good sandbox for experiments in model training, low-rank structure, and representation learning without relying on larger frameworks.

## Quick start

Install the package locally:

```bash
uv pip install -e .
```

Then import the pieces you need:

```python
from nanograd import Tensor, Module, MLP
from tt import TTCore, TensorTrain, from_tensor
```

### Minimal autograd example

```python
from nanograd import Tensor

x = Tensor([[2.0]])
y = x * x + 3.0
print(y.data)
```

### A slightly more advanced training loop

```python
import numpy as np
from nanograd import Tensor

# A tiny regression problem: fit y = 2x + 1
xs = [Tensor([[x]]) for x in np.linspace(0.0, 1.0, 5)]
ys = [Tensor([[2.0 * x + 1.0]]) for x in np.linspace(0.0, 1.0, 5)]

w = Tensor([[0.0]])
b = Tensor([[0.0]])

for _ in range(200):
    loss = Tensor([[0.0]])
    for x, y_true in zip(xs, ys):
        pred = w * x + b
        loss = loss + (pred - y_true) ** 2

    loss.backward()
    w.data -= 0.01 * w.grad
    b.data -= 0.01 * b.grad
    w.grad = 0.0
    b.grad = 0.0

print("learned slope:", w.data)
print("learned bias:", b.data)
```

### Tensor train example

```python
import numpy as np
from tt import from_tensor

x = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
tt = from_tensor(x)
reconstructed = tt.to_tensor()
print(tt)
print(reconstructed.shape)
```

### Tensor train with rank control

```python
import numpy as np
from tt import from_tensor

# Build a higher-order tensor and compress it with a strict bond limit.
tensor = np.random.randn(4, 4, 4, 4)
tt = from_tensor(tensor, max_rank=2)
print(tt.ranks)
print(tt.to_tensor().shape)
```

## Project layout

- `nanograd/` – core autograd and neural network primitives
- `tt/` – tensor-train data structures and decomposition logic
- `ttnn/` – experimental tensor-network state-machine ideas
- `tests/` – regression tests for the package and TT framework

## Testing

The project uses Python’s built-in `unittest` runner:

```bash
uv run python -m unittest discover -s tests -v
```

## Notes

This is a research-oriented codebase rather than a polished production library. The goal is to keep the implementation understandable and approachable while still demonstrating meaningful concepts in automatic differentiation and tensor networks.
