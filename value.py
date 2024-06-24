from __future__ import annotations
from plot_graph import draw_dot


class Value:
    """Value class which stores a single scalar and it's gradient.

    """

    def __init__(self, data: int | float, _children: tuple = (), _op: str = "", label: str = "") -> None:
        self.data = data
        self._prev = set(_children)
        self._op = _op
        self.label = label

    def __repr__(self) -> str:
        return f"Value(data={self.data})"

    def __add__(self, other: Value) -> Value:
        out = Value(self.data + other.data, (self, other), "+")
        return out

    def __mul__(self, other: Value) -> Value:
        out = Value(self.data * other.data, (self, other), "*")
        return out


if __name__ == '__main__':
    a = Value(2.0, label="a")
    b = Value(-3.0, label="b")
    c = Value(10.0, label="c")

    e = a * b
    e.label = "e"
    d = e + c
    d.label = "d"
    dot = draw_dot(d)
    dot.render('graph', format="jpg", view=True)
