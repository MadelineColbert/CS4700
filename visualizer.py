import csv
import os
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.colors import to_rgb

FILE = "results.csv"


def load_data():
    if not os.path.exists(FILE):
        return []

    data = []

    with open(FILE, "r") as f:
        reader = csv.DictReader(f)

        for row in reader:
            row["step_size"] = float(row["step_size"])
            row["internal_size"] = int(row["internal_size"])
            row["accuracy"] = float(row["accuracy"])
            row["time"] = float(row["time"])
            row["nnzs"] = int(row["nnzs"])
            row["test_time"] = float(row["test_time"])
            row["threshold"] = float(row["threshold"])
            row["decay"] = float(row["decay"])

            data.append(row)

    return data


def total_weights(h):
    return 2 * h * h + 794 * h


def nnz_percentage(d):
    return (d["nnzs"] / total_weights(d["internal_size"])) * 100.0


def config_key(d):
    return (
        d["mode"],
        d["threshold"] > 0,
        d["decay"] > 0
    )


def shade(base, factor):
    r, g, b = to_rgb(base)
    return (r * factor, g * factor, b * factor)


def shade_factor(pruned, decay):
    if pruned and decay:
        return 0.55
    if pruned and not decay:
        return 0.75
    if not pruned and decay:
        return 0.90
    return 1.00


def make_colors(data):
    sparse_base = "tab:blue"
    dense_base = "tab:red"

    colors = []
    legend_map = {}

    for d in data:
        mode, pruned, decay = config_key(d)
        factor = shade_factor(pruned, decay)

        if mode == "GATES":
            base = sparse_base
            mode_label = "Gated"
        else:
            base = dense_base
            mode_label = "Dense"

        color = shade(base, factor)
        colors.append(color)

        key = (mode_label, pruned, decay)
        if key not in legend_map:
            legend_map[key] = color

    legend = []

    for (mode_label, pruned, decay), color in legend_map.items():
        parts = [mode_label]
        parts.append("Pruned" if pruned else "NoPrune")
        parts.append("Decay" if decay else "NoDecay")

        legend.append(Patch(color=color, label=" | ".join(parts)))

    legend.append(Patch(color="gray", label="Darker = more pruning/decay"))

    return colors, legend


def make_positions(data, gap=2):
    positions = []
    x = 0

    for h in sorted(set(d["internal_size"] for d in data)):
        group = [d for d in data if d["internal_size"] == h]

        for _ in group:
            positions.append(x)
            x += 1

        x += gap

    return positions


def draw_groups(data, positions):
    for h in sorted(set(d["internal_size"] for d in data)):
        idx = [i for i, d in enumerate(data) if d["internal_size"] == h]
        xs = [positions[i] for i in idx]

        plt.text(
            sum(xs) / len(xs),
            -0.05,
            f"{h} Hidden",
            ha="center",
            va="top",
            transform=plt.gca().get_xaxis_transform(),
            fontweight="bold",
            fontsize=20
        )


def plot_metric(data, values, positions, colors, legend, title, ylabel, filename):
    plt.figure(figsize=(22, 8))

    plt.bar(positions, values, color=colors)

    plt.title(title, fontsize=40)
    plt.ylabel(ylabel, fontsize=40)

    plt.xticks([])
    plt.yticks(fontsize=20)

    # plt.legend(
    #     handles=legend,
    #     loc="upper left",
    #     bbox_to_anchor=(1.02, 1),
    #     borderaxespad=0
    # )

    draw_groups(data, positions)

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()


def plot(data):
    if not data:
        return

    data = sorted(data, key=lambda d: (d["internal_size"], d["mode"]))

    colors, legend = make_colors(data)
    positions = make_positions(data)

    acc = [d["accuracy"] for d in data]
    t = [d["time"] for d in data]
    tt = [d["test_time"] for d in data]
    nnz = [nnz_percentage(d) for d in data]

    plot_metric(data, acc, positions, colors, legend,
                "Accuracy", "Acc (%)", "accuracy.png")

    plot_metric(data, tt, positions, colors, legend,
                "Inference Time", "ms", "inference_time.png")

    plot_metric(data, t, positions, colors, legend,
                "Training Time", "ms", "training_time.png")

    plot_metric(data, nnz, positions, colors, legend,
                "Sparsity", "NNZ %", "sparsity.png")


if __name__ == "__main__":
    plot(load_data())