import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

import torch
from torch.optim import Adam

from pathlib import Path

from src.extrapolator import Extrapolator


def rosenbrock(tensor):
    x, y = tensor
    return (1 - x) ** 2 + 100 * (y - x ** 2) ** 2


def beale(tensor):
    x, y = tensor
    return (1.5 - x + x*y)**2 + (2.25 - x + x*y**2)**2 + (2.625 - x + x*y**3)**2


def goldstein(tensor):
    x, y = tensor
    return (1 + ((x + y + 1.0)**2)*(19 - 14*x + 3*x**2 - 14*y + 6*x*y + 3*y**2))\
           *(30 + ((2*x - 3*y)**2)*(18 - 32*x + 12*x**2 + 48*y - 36*x*y + 27*y**2))


def train(lr, experiments, extrapolation=False, dt=0.0):

    optimizer = experiments["optimization"]["optimizer"]
    n_iterations = experiments["optimization"]["n_iterations"]
    function = experiments["landscape"]["function"]
    start = experiments["landscape"]["start"]
    minimum = experiments["landscape"]["minimum"]

    x = torch.tensor(start).requires_grad_(True)

    optimizer = optimizer(params=[x], lr=lr)

    if extrapolation:
        extrapolator = Extrapolator(params=[x], eta=lr, h=lr, dt=dt)

    position = np.zeros(shape=(n_iterations + 1, len(start)))
    position[0, :] = np.array(start)
    loss = list()

    for i in range(1, n_iterations + 1):
        optimizer.zero_grad()
        f = function(x)
        f.backward(create_graph=True, retain_graph=True)
        optimizer.step()
        if extrapolation:
            extrapolator.step()
        position[i, :] = x.detach().numpy()
        loss.append((minimum[0] - position[i][0]) ** 2 + (minimum[1] - position[i][1]) ** 2)
    return position, loss


def run_experiment(lr, experiments):

    position = list()
    loss = list()

    use_extrapolation = [False, True]
    for mode in use_extrapolation:
        if mode:
            dt = experiments["optimization"]["dt"]
        else:
            dt = 0.0
        position_, loss_ = train(lr, experiments, extrapolation=mode, dt=dt)
        position.append(position_)
        loss.append(loss_)

    return position, loss


def plot_path(position, experiments):
    """Plots gradient descent path"""
    x_1, y_1 = position[0][:, 0], position[0][:, 1]
    x_2, y_2 = position[1][:, 0], position[1][:, 1]

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 5))

    plot_surface(ax, experiments)

    ax.plot(x_1, y_1, color="r", marker=".", markersize=4, linewidth=0.1, label="GD")
    ax.plot(x_2, y_2, color="lime", marker=".", markersize=4, linewidth=0.1, label="GD+")
    ax.legend()
    plt.tight_layout()
    plt.savefig("./results/" + experiments["landscape"]["name"] + "_path.png", dpi=120)
    plt.close()


def plot_surface(ax, experiments):
    step_size = 0.05  # resolution
    levels = 64

    x_min = experiments["landscape"]["domain"]["x_min"]
    x_max = experiments["landscape"]["domain"]["x_max"]
    y_min = experiments["landscape"]["domain"]["y_min"]
    y_max = experiments["landscape"]["domain"]["y_max"]
    function = experiments["landscape"]["function"]
    x_pos_min, y_pos_min = experiments["landscape"]["minimum"]

    style_dict = dict(cmap="inferno", alpha=0.5, linewidths=0.8)
    x = np.arange(x_min, x_max, step_size)
    y = np.arange(y_min, y_max, step_size)
    xx, yy = np.meshgrid(x, y)

    tensor = (torch.tensor(xx), torch.tensor(yy))
    z = function(tensor)
    z = z.numpy()

    ax.contour(x, y, z, levels=np.logspace(0.0, 4.0, levels), norm=LogNorm(), **style_dict)
    ax.plot(x_pos_min, y_pos_min, "rx")


def plot_loss(loss, experiments):
    """Plots loss"""

    loss_1 = loss[0]
    loss_2 = loss[1]

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 5))

    ax.plot(loss_1, color="r", marker=".", markersize=4, linewidth=0.1, label="GD")
    ax.plot(loss_2, color="lime", marker=".", markersize=4, linewidth=0.1, label="GD+")

    ax.legend()
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss")

    ax.set_yscale("log")
    ax.grid("True", linestyle="--", linewidth=0.5)

    first_n_points = 100
    ax2 = ax.inset_axes([0.07, 0.02, 0.42, 0.42])
    ax2.plot(loss_1[:first_n_points], color="r", marker=".", markersize=4, linewidth=0.1, label="GD")
    ax2.plot(loss_2[:first_n_points], color="lime", marker=".", markersize=4, linewidth=0.1, label="GD+")
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.set_facecolor((1.0, 1.0, 1.0, 0.5))
    ax2.grid("True", linestyle="--", linewidth=0.5)

    ax.indicate_inset_zoom(ax2, linewidth=1)

    plt.tight_layout()
    plt.savefig("./results/" + experiments["landscape"]["name"] + "_loss.png", dpi=120)
    plt.close()


def learning_rate_finder(experiments):

    lr_min = experiments["optimization"]["lr_min"]
    lr_max = experiments["optimization"]["lr_max"]
    n_tests = experiments["optimization"]["n_tests"]

    lrs = np.geomspace(lr_min, lr_max, num=n_tests)
    loss = list()
    for lr in lrs:
        _, current_loss = train(lr, experiments)
        loss.append(current_loss[-1])
    idx = np.argmin(loss)

    return lrs[idx]


if __name__ == "__main__":

    Path("./results/").mkdir(parents=True, exist_ok=True)
    optimizer = Adam
    n_iterations = 40
    n_tests = 10
    dt = 1.0e-05

    experiments = dict(
        rosenbrock=
        dict(
            landscape=
            dict(
                name="rosenbrock",
                function=rosenbrock,
                domain=dict(x_min=-1.75, x_max=1.25, y_min=-1.0, y_max=2.0),
                start=[-1.5, 1.0],
                minimum=[1.0, 1.0]
            ),
            optimization=
            dict(
                optimizer=optimizer,
                n_iterations=n_iterations,
                n_tests=n_tests,
                lr_min=1.0e-04,
                lr_max=1.0e-01,
                dt=dt
            )
        ),
        beale=
        dict(
            landscape=
            dict(
                name="beale",
                function=beale,
                domain=dict(x_min=-2.75, x_max=3.25, y_min=-2.0, y_max=2.0),
                start=[-2.5, -1.0],
                minimum=[3.0, 0.5]
            ),
            optimization=
            dict(
                optimizer=optimizer,
                n_iterations=n_iterations,
                n_tests=n_tests,
                lr_min=1.0e-04,
                lr_max=1.0e-02,
                dt=dt
            )
        ),
        goldstein =
        dict(
            landscape=
            dict(
                name="goldstein",
                function=goldstein,
                domain=dict(x_min=-1.0, x_max=1.0, y_min=-1.5, y_max=0.5),
                start=[0.5, 0.25],
                minimum=[0.0, -1.0]
            ),
            optimization=
            dict(
                optimizer=optimizer,
                n_iterations=n_iterations,
                n_tests=n_tests,
                lr_min=1.0e-04,
                lr_max=1.0e-03,
                dt=dt
            )
        )
    )

    landscapes = ["rosenbrock", "beale", "goldstein"]

    for landscape in landscapes:
        # Compute optimal learning rate
        lr = learning_rate_finder(experiments[landscape])
        print("lr=", lr)

        # Perform gradient descent with optimal learning rate
        position, loss = run_experiment(lr, experiments[landscape])

        # Plot results
        plot_path(position, experiments[landscape])
        plot_loss(loss, experiments[landscape])
