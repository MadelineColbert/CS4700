# visualizer.py
import csv
import os
import matplotlib.pyplot as plt

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
            data.append(row)
    return data


def plot(data):
    if not data:
        print("No data yet.")
        return

    labels = [
        f"({d['step_size']},{d['internal_size']}) {d['mode']}"
        for d in data
    ]

    accuracies = [d["accuracy"] for d in data]
    times = [d["time"] for d in data]

    x = range(len(data))

    plt.figure()
    plt.title("Accuracy per Configuration")
    plt.bar(x, accuracies)
    plt.xticks(x, labels, rotation=90)
    plt.tight_layout()

    plt.figure()
    plt.title("Time per Configuration (ms)")
    plt.bar(x, times)
    plt.xticks(x, labels, rotation=90)
    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    plot(load_data())