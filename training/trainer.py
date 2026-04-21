"""
training/trainer.py
-------------------
The RL training loop: collects self-play experience and runs PPO updates.

The trainer orchestrates the inner loop of training for a single generation:
    1. Collect self-play experience (agents play games, store transitions)
    2. Run PPO updates on each agent using its collected experience
    3. Return training metrics (losses, entropy) for logging

This is separate from the evolution loop (evolution.py), which handles
the outer genetic algorithm loop across generations.
"""

import logging
from typing import List, Dict
from tqdm import tqdm

from rl_agent.agent import ChessAgent
from training.self_play import collect_training_data

logger = logging.getLogger("chess_rl")


class Trainer:
    """
    Manages the RL training loop for a population of agents.

    Wraps self-play data collection and PPO updates into a clean interface
    that the evolution loop can call once per generation.
    """

    def __init__(
        self,
        games_per_update: int = 20,
        ppo_epochs: int = 4,
        max_moves: int = 200,
        win_reward: float = 1.0,
        draw_reward: float = 0.0,
        loss_reward: float = -1.0,
        move_penalty: float = -0.001,
    ):
        """
        Args:
            games_per_update: Number of self-play games per agent before PPO update.
            ppo_epochs:       Number of gradient update passes per data batch.
            max_moves:        Maximum moves per game before declaring draw.
            win_reward:       Reward for winning a game.
            draw_reward:      Reward for a draw.
            loss_reward:      Reward for losing.
            move_penalty:     Per-move negative reward.
        """
        self.games_per_update = games_per_update
        self.ppo_epochs = ppo_epochs
        self.max_moves = max_moves
        self.win_reward = win_reward
        self.draw_reward = draw_reward
        self.loss_reward = loss_reward
        self.move_penalty = move_penalty

    def train_generation(self, agents: List[ChessAgent]) -> Dict[str, float]:
        """
        Run one full training cycle for a generation of agents.

        Steps:
            1. Self-play: agents play games and store experience
            2. PPO updates: each agent learns from its experience
            3. Return averaged training metrics

        Args:
            agents: List of agents to train.

        Returns:
            Dictionary with averaged training metrics across all agents.
        """
        logger.info(f"Starting training cycle for {len(agents)} agents...")

        # ── Step 1: Collect self-play experience ──────────────────────────
        game_stats = collect_training_data(
            agents=agents,
            games_per_update=self.games_per_update,
            max_moves=self.max_moves,
            win_reward=self.win_reward,
            draw_reward=self.draw_reward,
            loss_reward=self.loss_reward,
            move_penalty=self.move_penalty,
        )

        logger.info(
            f"Self-play complete: {game_stats['total_games']} games | "
            f"W/D/L: {game_stats['win_rate']:.1%}/{game_stats['draw_rate']:.1%}/{game_stats['loss_rate']:.1%} | "
            f"Avg length: {game_stats['mean_game_length']:.1f} moves"
        )

        # ── Step 2: PPO updates for each agent ────────────────────────────
        all_policy_losses = []
        all_value_losses = []
        all_entropies = []

        logger.info("Running PPO updates...")
        for i, agent in enumerate(tqdm(agents, desc="PPO updates", unit="agent")):
            if len(agent.buffer) == 0:
                logger.warning(f"Agent {i} has no experience in buffer — skipping update.")
                continue

            metrics = agent.update()

            if metrics:
                all_policy_losses.append(metrics.get("policy_loss", 0.0))
                all_value_losses.append(metrics.get("value_loss", 0.0))
                all_entropies.append(metrics.get("entropy", 0.0))

        # ── Step 3: Aggregate metrics ─────────────────────────────────────
        def safe_mean(lst):
            return sum(lst) / len(lst) if lst else 0.0

        training_metrics = {
            "policy_loss": safe_mean(all_policy_losses),
            "value_loss": safe_mean(all_value_losses),
            "entropy": safe_mean(all_entropies),
            **game_stats,
        }

        logger.info(
            f"PPO update complete. "
            f"Policy loss: {training_metrics['policy_loss']:.4f} | "
            f"Value loss: {training_metrics['value_loss']:.4f} | "
            f"Entropy: {training_metrics['entropy']:.4f}"
        )

        return training_metrics
