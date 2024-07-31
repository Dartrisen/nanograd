from nanograd.mlp import MLP


def main() -> None:
    xs = [
        [2.0, 3.0, -1.0],
        [3.0, -1.0, 0.5],
        [0.5, 1.0, 1.0],
        [1.0, 1.0, -1.0]
    ]
    ys = [1.0, -1.0, -1.0, 1.0]

    model = MLP(3, [4, 4, 1])

    num_epochs = 20
    lr = 0.05
    for k in range(num_epochs):
        # forward
        y_pred = [model(x) for x in xs]
        loss = sum((y_out[0] - y_gt) ** 2 for y_gt, y_out in zip(ys, y_pred))

        # backward
        model.zero_grad()
        loss.backward()

        for p in model.parameters():
            p.data += - lr * p.grad

        print(k, loss.data)


if __name__ == '__main__':
    main()
