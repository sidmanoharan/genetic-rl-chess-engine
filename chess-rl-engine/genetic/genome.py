"""
genome.py
---------
The "genome" of a chess agent: a set of hyperparameters that can be
evolved by the genetic algorithm.

In biological evolution, a genome encodes the blueprint for an organism.
Here, the genome encodes the hyperparameters that define an agent:
    - How deep is its neural network?
    - How aggressively does it learn?
    - How much does it explore vs exploit?

The genetic algorithm evolves these genomes across generations,
keeping the hyperparameters that produce the strongest agents.
"""

import numpy as np
import copy
from dataclasses import dataclass, field
from typing import Dict, Any


@dataclass
class Genome:
    """
    Hyperparameter genome for one chess agent.

    Each gene is a float that the genetic algorithm can mutate and cross
    over between parent genomes. After each generation, the genes of the
    fittest agents are combined to produce the next generation.

    Note: integer genes (num_residual_blocks) are rounded when used.
    """

    # ── Network architecture genes ────────────────────────────────────────
    num_residual_blocks: float = 4.0   # Depth of the residual tower (rounded to int)
    num_filters: float = 128.0         # Width of conv layers (rounded to int)

    # ── PPO optimisation genes ────────────────────────────────────────────
    learning_rate: float = 3e-4        # Adam learning rate
    gamma: float = 0.99                # Reward discount factor
    clip_epsilon: float = 0.2          # PPO clipping range
    entropy_coef: float = 0.01         # Exploration bonus strength
    gae_lambda: float = 0.95           # GAE advantage smoothing

    # ── Valid ranges for each gene (used during mutation) ─────────────────
    RANGES: Dict[str, tuple] = field(default_factory=lambda: {
        "num_residual_blocks": (2.0, 8.0),
        "num_filters": (64.0, 256.0),
        "learning_rate": (1e-5, 1e-2),
        "gamma": (0.90, 0.999),
        "clip_epsilon": (0.1, 0.4),
        "entropy_coef": (0.001, 0.1),
        "gae_lambda": (0.80, 0.99),
    })

    def to_dict(self) -> Dict[str, Any]:
        """Convert genome to a dictionary of agent hyperparameters."""
        return {
            "num_residual_blocks": max(2, int(round(self.num_residual_blocks))),
            "num_filters": max(64, int(round(self.num_filters))),
            "learning_rate": float(np.clip(self.learning_rate, 1e-5, 1e-2)),
            "gamma": float(np.clip(self.gamma, 0.90, 0.999)),
            "clip_epsilon": float(np.clip(self.clip_epsilon, 0.1, 0.4)),
            "entropy_coef": float(np.clip(self.entropy_coef, 0.001, 0.1)),
            "gae_lambda": float(np.clip(self.gae_lambda, 0.80, 0.99)),
        }

    def genes(self) -> Dict[str, float]:
        """Return all evolvable genes as a flat dictionary."""
        return {
            "num_residual_blocks": self.num_residual_blocks,
            "num_filters": self.num_filters,
            "learning_rate": self.learning_rate,
            "gamma": self.gamma,
            "clip_epsilon": self.clip_epsilon,
            "entropy_coef": self.entropy_coef,
            "gae_lambda": self.gae_lambda,
        }

    def set_genes(self, genes: Dict[str, float]) -> None:
        """Update gene values from a dictionary."""
        for name, value in genes.items():
            if hasattr(self, name):
                setattr(self, name, value)

    def __repr__(self) -> str:
        d = self.to_dict()
        return (
            f"Genome(blocks={d['num_residual_blocks']}, filters={d['num_filters']}, "
            f"lr={d['learning_rate']:.2e}, γ={d['gamma']:.3f}, "
            f"ε={d['clip_epsilon']:.2f}, H={d['entropy_coef']:.3f})"
        )


def random_genome() -> Genome:
    """
    Create a genome with randomly sampled gene values.

    Used to initialise the first generation of agents.
    Each gene is sampled uniformly from its valid range.
    """
    g = Genome()
    for gene_name, (low, high) in g.RANGES.items():
        # Use log-uniform sampling for the learning rate
        # (avoids almost always sampling near the high end)
        if gene_name == "learning_rate":
            log_val = np.random.uniform(np.log10(low), np.log10(high))
            value = 10 ** log_val
        else:
            value = np.random.uniform(low, high)
        setattr(g, gene_name, float(value))
    return g


def default_genome() -> Genome:
    """Return a genome with sensible default hyperparameters."""
    return Genome()
