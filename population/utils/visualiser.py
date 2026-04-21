"""
visualiser.py
-------------
Plotting utilities for training progress and game analysis.

Call plot_training_curves() after training to see how your agents
improved across generations.
"""

import json
import os
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np


def plot_training_curves(metrics_path: str, save_path: Optional[str] = None) -> None:
    """
    Plot ELO progression, win rates, and loss curves from a saved metrics file.

    Args:
        metrics_path: Path to metrics.json saved by MetricsTracker.
        save_path:    If provided, saves the figure here instead of showing it.
    """
    with open(metrics_path, "r") as f:
        metrics = json.load(f)

    generations = metrics["generation"]

    fig = plt.figure(figsize=(16, 10))
    fig.suptitle("Chess RL Engine — Training Progress", fontsize=16, fontweight="bold")

    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

    # ── Plot 1: ELO over generations ──────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.plot(generations, metrics["best_elo"], "b-o", label="Best Agent ELO", linewidth=2)
    ax1.plot(generations, metrics["mean_elo"], "b--", label="Mean Population ELO", alpha=0.6)
    ax1.fill_between(generations, metrics["mean_elo"], metrics["best_elo"], alpha=0.1)
    ax1.set_title("ELO Rating Over Generations")
    ax1.set_xlabel("Generation")
    ax1.set_ylabel("ELO Rating")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Starting ELO annotation
    ax1.axhline(y=1200, color="gray", linestyle=":", alpha=0.5, label="Starting ELO (1200)")

    # ── Plot 2: Win/Draw/Loss rates ───────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 2])
    win_rates = metrics["win_rate"]
    draw_rates = metrics["draw_rate"]
    loss_rates = metrics["loss_rate"]

    ax2.stackplot(
        generations,
        win_rates, draw_rates, loss_rates,
        labels=["Win", "Draw", "Loss"],
        colors=["#2ecc71", "#f39c12", "#e74c3c"],
        alpha=0.8
    )
    ax2.set_title("Win / Draw / Loss Rates")
    ax2.set_xlabel("Generation")
    ax2.set_ylabel("Rate")
    ax2.legend(loc="upper left", fontsize=8)
    ax2.set_ylim(0, 1)
    ax2.grid(True, alpha=0.3)

    # ── Plot 3: Policy loss ───────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(generations, metrics["policy_loss"], "r-", linewidth=2)
    ax3.set_title("Policy Loss")
    ax3.set_xlabel("Generation")
    ax3.set_ylabel("Loss")
    ax3.grid(True, alpha=0.3)

    # ── Plot 4: Value loss ────────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(generations, metrics["value_loss"], "g-", linewidth=2)
    ax4.set_title("Value Loss")
    ax4.set_xlabel("Generation")
    ax4.set_ylabel("Loss")
    ax4.grid(True, alpha=0.3)

    # ── Plot 5: Entropy (exploration) ─────────────────────────────────────
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.plot(generations, metrics["entropy"], "purple", linewidth=2)
    ax5.set_title("Policy Entropy (Exploration)")
    ax5.set_xlabel("Generation")
    ax5.set_ylabel("Entropy")
    # Lower entropy = more confident/deterministic policy
    ax5.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Figure saved to {save_path}")
    else:
        plt.show()

    plt.close()


def plot_elo_distribution(elo_ratings: list, generation: int, save_path: Optional[str] = None) -> None:
    """
    Plot the distribution of ELO ratings in the current population.

    Useful for seeing how spread out or converged the population is.

    Args:
        elo_ratings: List of ELO ratings for each agent in the population.
        generation:  Current generation number (for the title).
        save_path:   Optional path to save the figure.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.hist(elo_ratings, bins=10, color="#3498db", edgecolor="white", alpha=0.8)
    ax.axvline(np.mean(elo_ratings), color="red", linestyle="--", label=f"Mean: {np.mean(elo_ratings):.0f}")
    ax.axvline(max(elo_ratings), color="green", linestyle="--", label=f"Best: {max(elo_ratings):.0f}")

    ax.set_title(f"ELO Distribution — Generation {generation}")
    ax.set_xlabel("ELO Rating")
    ax.set_ylabel("Number of Agents")
    ax.legend()
    ax.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    else:
        plt.show()

    plt.close()
