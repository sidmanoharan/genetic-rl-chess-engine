"""
logger.py
---------
Logging and metrics tracking for the training run.

Keeps a history of ELO ratings, win rates, and loss values so you
can visualise training progress and debug issues.
"""

import os
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional


def setup_logger(log_dir: str, name: str = "chess_rl") -> logging.Logger:
    """
    Create a logger that writes to both the console and a log file.

    Args:
        log_dir: Directory where the log file will be saved.
        name:    Logger name.

    Returns:
        Configured Python logger.
    """
    os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Avoid adding duplicate handlers if called multiple times
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console handler — prints to terminal
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler — saves to disk
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"{name}_{timestamp}.log")
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


class MetricsTracker:
    """
    Tracks training metrics across generations and saves them to disk.

    After training, you can load the metrics JSON and plot it in
    notebooks/analysis.ipynb to see how the agents improved over time.
    """

    def __init__(self, log_dir: str):
        """
        Args:
            log_dir: Directory where metrics.json will be saved.
        """
        os.makedirs(log_dir, exist_ok=True)
        self.log_dir = log_dir
        self.metrics: Dict[str, List] = {
            "generation": [],
            "best_elo": [],
            "mean_elo": [],
            "win_rate": [],
            "draw_rate": [],
            "loss_rate": [],
            "policy_loss": [],
            "value_loss": [],
            "entropy": [],
            "mean_game_length": [],
        }

    def record_generation(
        self,
        generation: int,
        best_elo: float,
        mean_elo: float,
        win_rate: float,
        draw_rate: float,
        loss_rate: float,
        policy_loss: float,
        value_loss: float,
        entropy: float,
        mean_game_length: float,
    ) -> None:
        """Record metrics for one completed generation."""
        self.metrics["generation"].append(generation)
        self.metrics["best_elo"].append(best_elo)
        self.metrics["mean_elo"].append(mean_elo)
        self.metrics["win_rate"].append(win_rate)
        self.metrics["draw_rate"].append(draw_rate)
        self.metrics["loss_rate"].append(loss_rate)
        self.metrics["policy_loss"].append(policy_loss)
        self.metrics["value_loss"].append(value_loss)
        self.metrics["entropy"].append(entropy)
        self.metrics["mean_game_length"].append(mean_game_length)

    def save(self, filename: str = "metrics.json") -> None:
        """Save metrics to a JSON file."""
        path = os.path.join(self.log_dir, filename)
        with open(path, "w") as f:
            json.dump(self.metrics, f, indent=2)

    def load(self, filename: str = "metrics.json") -> Dict:
        """Load previously saved metrics."""
        path = os.path.join(self.log_dir, filename)
        with open(path, "r") as f:
            self.metrics = json.load(f)
        return self.metrics

    def summary(self) -> str:
        """Return a human-readable summary of the latest generation."""
        if not self.metrics["generation"]:
            return "No metrics recorded yet."

        gen = self.metrics["generation"][-1]
        best_elo = self.metrics["best_elo"][-1]
        win = self.metrics["win_rate"][-1]
        draw = self.metrics["draw_rate"][-1]

        return (
            f"Gen {gen:3d} | Best ELO: {best_elo:.0f} | "
            f"W/D/L: {win:.1%}/{draw:.1%}/{1-win-draw:.1%}"
        )
