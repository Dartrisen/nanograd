from __future__ import annotations


class Value:
    """Value class which stores a single scalar and it's gradient.

    """

    def __init__(self, data: int | float, _children: tuple = (), _op: str = "") -> None:
        self.data = data
        self._prev = set(_children)

    def __repr__(self) -> str:
        return f"Value(data={self.data})"

    def __add__(self, other: Value) -> Value:
        out = Value(self.data + other.data, (self, other), "+")
        return out

    def __mul__(self, other: Value) -> Value:
        out = Value(self.data * other.data, (self, other), "*")
        return out


if __name__ == '__main__':
    a = Value(2.0)
    b = Value(-3.0)
    c = Value(10.0)

    d = a * b + c
    print(d)
