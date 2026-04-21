"""
config.py
---------
Central configuration file for the Chess RL Engine.

All hyperparameters live here so you never have to hunt through
multiple files to change a setting. This is a best practice for
any serious ML project.
"""

from dataclasses import dataclass, field
from typing import List


# ---------------------------------------------------------------------------
# Board / Game settings
# ---------------------------------------------------------------------------

@dataclass
class GameConfig:
    """Settings that define how a single chess game is played."""

    # Maximum moves before declaring a draw (prevents infinite games)
    max_moves: int = 200

    # Rewards given to the agent at the end of a game.
    # The losing agent receives the negative of the winning agent's reward.
    win_reward: float = 1.0
    draw_reward: float = 0.0
    loss_reward: float = -1.0

    # Small penalty per move to encourage decisive play and shorter games.
    # Without this, agents may learn to stall indefinitely.
    move_penalty: float = -0.001


# ---------------------------------------------------------------------------
# Neural Network settings
# ---------------------------------------------------------------------------

@dataclass
class NetworkConfig:
    """Architecture settings for the policy/value neural network."""

    # Input: 8x8 board with 17 feature planes (see board.py for details)
    input_channels: int = 17

    # Number of residual blocks in the network tower.
    # More blocks = stronger but slower to train.
    num_residual_blocks: int = 4

    # Number of filters (channels) in each convolutional layer.
    num_filters: int = 128

    # Total number of possible moves in chess (UCI encoding).
    # This is the size of the policy head output.
    num_actions: int = 4672


# ---------------------------------------------------------------------------
# PPO (Proximal Policy Optimisation) settings
# ---------------------------------------------------------------------------

@dataclass
class PPOConfig:
    """Hyperparameters for the PPO reinforcement learning algorithm."""

    # Learning rate for the Adam optimiser.
    learning_rate: float = 3e-4

    # Discount factor: how much the agent values future rewards vs immediate ones.
    # 0 = only immediate rewards, 1 = infinite future horizon.
    gamma: float = 0.99

    # PPO clip epsilon: prevents the new policy from deviating too far from
    # the old policy in a single update. Key stability mechanism in PPO.
    clip_epsilon: float = 0.2

    # Entropy bonus coefficient: encourages the agent to explore rather than
    # collapsing to always picking the same move.
    entropy_coef: float = 0.01

    # Value function loss coefficient: balances policy vs value head training.
    value_coef: float = 0.5

    # Number of PPO update epochs per batch of collected experience.
    ppo_epochs: int = 4

    # Number of games to collect before each PPO update.
    games_per_update: int = 20

    # Mini-batch size for PPO updates.
    mini_batch_size: int = 256

    # GAE lambda: controls bias/variance tradeoff in advantage estimation.
    # Higher = less bias, more variance.
    gae_lambda: float = 0.95

    # Maximum gradient norm for gradient clipping (prevents exploding gradients).
    max_grad_norm: float = 0.5


# ---------------------------------------------------------------------------
# Genetic Algorithm settings
# ---------------------------------------------------------------------------

@dataclass
class GeneticConfig:
    """Settings for the genetic algorithm that evolves agent hyperparameters."""

    # Number of agents in the population.
    population_size: int = 16

    # Number of generations to run.
    num_generations: int = 30

    # Number of games each agent plays to determine its fitness (ELO rating).
    games_per_evaluation: int = 10

    # Fraction of the population that survives to reproduce each generation.
    # e.g., 0.25 = top 25% survive.
    survival_rate: float = 0.25

    # Probability that any single gene mutates during reproduction.
    mutation_rate: float = 0.15

    # How much a mutated gene changes (as fraction of its valid range).
    mutation_strength: float = 0.2

    # If True, the best agent from the previous generation is always kept.
    # This prevents the best solution from being lost (elitism).
    elitism: bool = True

    # Hyperparameter search ranges for the genetic algorithm.
    # Format: (min_value, max_value)
    gene_ranges: dict = field(default_factory=lambda: {
        "learning_rate": (1e-5, 1e-2),
        "hidden_filters": (64, 256),
        "num_residual_blocks": (2, 8),
        "gamma": (0.90, 0.999),
        "entropy_coef": (0.001, 0.1),
        "clip_epsilon": (0.1, 0.4),
        "gae_lambda": (0.8, 0.99),
    })


# ---------------------------------------------------------------------------
# Training settings
# ---------------------------------------------------------------------------

@dataclass
class TrainingConfig:
    """Top-level settings that control the overall training run."""

    # Directory to save agent checkpoints.
    checkpoint_dir: str = "checkpoints"

    # Directory to save logs and metrics.
    log_dir: str = "logs"

    # Save a checkpoint every N generations.
    checkpoint_every: int = 5

    # Use GPU if available. Set to False to force CPU.
    use_gpu: bool = True

    # Random seed for reproducibility.
    seed: int = 42

    # Number of parallel game workers (for faster self-play data collection).
    num_workers: int = 4


# ---------------------------------------------------------------------------
# Master config: bundles all sub-configs together
# ---------------------------------------------------------------------------

@dataclass
class Config:
    """
    Master configuration object.

    Usage:
        from utils.config import Config
        cfg = Config()
        print(cfg.ppo.learning_rate)
    """
    game: GameConfig = field(default_factory=GameConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    genetic: GeneticConfig = field(default_factory=GeneticConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
