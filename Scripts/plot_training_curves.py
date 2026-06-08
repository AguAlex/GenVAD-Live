"""
Generează curbe de antrenare (loss + Micro/Macro AUC) pentru licență.

Suportă:
  - log v2/v1 combinat (train_loss + test_micro/test_macro pe linii separate)
  - log_test.txt + log_train.txt separate (Experiments/RUN03)
  - log v4/v4_convlstm (doar train_loss; AUC dacă există test_micro)

Exemple:
  # MAE v2 UCSD Ped2 — RUN01 (un singur log.txt cu train_loss + test AUC)
  python Scripts/plot_training_curves.py \\
      --log Experiments/RUN01/Checkpoints/log.txt \\
      --title "MAE v2 — UCSD Ped2" \\
      --output Licenta/figuri/curba_mae_v2_ucsd.png

  # MAE v2 Avenue — RUN03 (log_test + log_train)
  python Scripts/plot_training_curves.py \\
      --log Experiments/RUN03/Checkpoints/log_test.txt \\
      --log-train Experiments/RUN03/Checkpoints/log_train.txt \\
      --title "MAE v2 — Avenue" \\
      --output Licenta/figuri/curba_mae_v2_avenue.png

  python Scripts/plot_training_curves.py \\
      --log Experiments/RUN07/log.txt \\
      --title "Future Frame U-Net — UCSD Ped2" \\
      --output Licenta/figuri/curba_v4_ucsd.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def _parse_log(path: Path) -> dict[int, dict]:
    epochs: dict[int, dict] = {}
    if not path.is_file():
        return epochs
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            ep = entry.get("epoch")
            if ep is None:
                continue
            if ep not in epochs:
                epochs[ep] = {}
            for key in ("train_loss", "test_micro", "test_macro"):
                if key in entry:
                    epochs[ep][key] = entry[key]
    return epochs


def merge_logs(*paths: Path) -> dict[int, dict]:
    merged: dict[int, dict] = {}
    for path in paths:
        for ep, data in _parse_log(path).items():
            merged.setdefault(ep, {}).update(data)
    return merged


def plot_curves(
    epochs_data: dict[int, dict],
    title: str,
    output: Path,
    dpi: int = 300,
) -> None:
    sorted_epochs = sorted(epochs_data.keys())
    epochs, train_loss, test_micro, test_macro = [], [], [], []

    for ep in sorted_epochs:
        d = epochs_data[ep]
        if "train_loss" in d:
            epochs.append(ep)
            train_loss.append(d["train_loss"])
            test_micro.append(d.get("test_micro"))
            test_macro.append(d.get("test_macro"))

    if not epochs:
        raise SystemExit("Nu s-au găsit epoci cu train_loss în log.")

    has_auc = any(v is not None for v in test_micro) or any(
        v is not None for v in test_macro
    )

    if has_auc:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(12, 4))
        ax2 = None

    fig.suptitle(title, fontsize=14, fontweight="bold")

    ax1.plot(epochs, train_loss, color="tab:blue", linewidth=2, marker="o", markersize=3)
    ax1.set_ylabel("Train loss (MSE)")
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.set_title("Eroare de antrenare")

    if ax2 is not None:
        micro_ep = [(e, v) for e, v in zip(epochs, test_micro) if v is not None]
        macro_ep = [(e, v) for e, v in zip(epochs, test_macro) if v is not None]
        if micro_ep:
            ex, vy = zip(*micro_ep)
            ax2.plot(ex, vy, color="tab:green", linewidth=2, marker="s", markersize=3, label="Micro AUC")
            best_i = max(range(len(vy)), key=lambda i: vy[i])
            ax2.annotate(
                f"Best {vy[best_i]:.3f} (ep {ex[best_i]})",
                xy=(ex[best_i], vy[best_i]),
                xytext=(10, -15),
                textcoords="offset points",
                fontsize=9,
                color="darkgreen",
            )
        if macro_ep:
            ex, vy = zip(*macro_ep)
            ax2.plot(ex, vy, color="tab:red", linewidth=2, marker="^", markersize=3, label="Macro AUC")
            best_i = max(range(len(vy)), key=lambda i: vy[i])
            ax2.annotate(
                f"Best {vy[best_i]:.3f} (ep {ex[best_i]})",
                xy=(ex[best_i], vy[best_i]),
                xytext=(10, 10),
                textcoords="offset points",
                fontsize=9,
                color="darkred",
            )
        ax2.set_xlabel("Epocă")
        ax2.set_ylabel("AUC")
        ax2.set_ylim(0, 1.02)
        ax2.grid(True, linestyle="--", alpha=0.6)
        ax2.legend(loc="lower right")
        ax2.set_title("Evaluare pe setul de test")
    else:
        ax1.set_xlabel("Epocă")
        fig.text(
            0.5,
            0.02,
            "Notă: AUC calculat doar la finalul antrenării.",
            ha="center",
            fontsize=9,
            style="italic",
        )

    plt.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"Salvat: {output}")


def plot_comparison_bars(results_paths: list[Path], output: Path, dataset: str) -> None:
    """Bar chart Micro/Macro din fișiere results.json."""
    labels, micros, macros = [], [], []
    for p in results_paths:
        if not p.is_file():
            continue
        with open(p, encoding="utf-8") as f:
            r = json.load(f)
        method = r.get("method", p.parent.name)
        labels.append(method.replace("FutureFrame", "FF ").replace("Prediction", ""))
        micros.append(r.get("micro", 0))
        macros.append(r.get("macro", 0))

    if not labels:
        raise SystemExit("Niciun results.json valid.")

    x = range(len(labels))
    w = 0.35
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar([i - w / 2 for i in x], micros, w, label="Micro AUC", color="tab:green")
    ax.bar([i + w / 2 for i in x], macros, w, label="Macro AUC", color="tab:red")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("AUC")
    ax.set_title(f"Comparație metode — {dataset}")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Salvat: {output}")


def main():
    p = argparse.ArgumentParser(description="Curbe antrenare pentru licență")
    p.add_argument("--log", type=str, required=True, help="log.txt sau log_test.txt")
    p.add_argument("--log-train", type=str, default="", help="log_train.txt opțional")
    p.add_argument("--title", type=str, default="Evoluția antrenării")
    p.add_argument("--output", type=str, required=True)
    p.add_argument("--compare-json", nargs="*", default=[], help="results.json pentru bar chart")
    p.add_argument("--compare-dataset", type=str, default="UCSD Ped2")
    p.add_argument("--compare-output", type=str, default="")
    args = p.parse_args()

    paths = [Path(args.log)]
    if args.log_train:
        paths.append(Path(args.log_train))
    data = merge_logs(*paths)
    plot_curves(data, args.title, Path(args.output))

    if args.compare_json and args.compare_output:
        plot_comparison_bars(
            [Path(x) for x in args.compare_json],
            Path(args.compare_output),
            args.compare_dataset,
        )


if __name__ == "__main__":
    main()
