"""
training/evolution.py
---------------------
The outer evolution loop: runs the genetic algorithm across generations.

This is the top-level training orchestrator. It ties together:
    - The population of agents (genetic/population.py)
    - The self-play trainer (training/trainer.py)
    - Fitness evaluation (training/self_play.py)
    - Checkpointing and logging

One generation looks like:
    ┌─────────────────────────────────────────────────────────┐
    │  1. Train: agents play self-play games → PPO updates    │
    │  2. Evaluate: agents play each other → ELO ratings      │
    │  3. Evolve: GA selects/breeds/mutates → next generation │
    │  4. Log metrics and save checkpoint                      │
    └─────────────────────────────────────────────────────────┘
"""

import os
import torch
import logging
import numpy as np
from typing import Optional

from genetic.population import Population
from training.trainer import Trainer
from training.self_play import evaluate_population
from utils.logger import MetricsTracker, setup_logger
from utils.config import Config

logger = logging.getLogger("chess_rl")


class EvolutionLoop:
    """
    Runs the full genetic + RL training loop across multiple generations.

    This is the main class you interact with when starting a training run.

    Usage:
        from utils.config import Config
        from training.evolution import EvolutionLoop

        cfg = Config()
        loop = EvolutionLoop(cfg)
        loop.run()
    """

    def __init__(self, config: Config):
        """
        Args:
            config: Master Config object with all hyperparameters.
        """
        self.cfg = config

        # Set up logging
        setup_logger(config.training.log_dir)

        # Reproducibility
        torch.manual_seed(config.training.seed)
        np.random.seed(config.training.seed)

        # Determine device
        if config.training.use_gpu and torch.cuda.is_available():
            self.device = torch.device("cuda")
            logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            self.device = torch.device("cpu")
            logger.info("Using CPU (GPU not available or disabled)")

        # Initialise population
        self.population = Population(
            size=config.genetic.population_size,
            survival_rate=config.genetic.survival_rate,
            mutation_rate=config.genetic.mutation_rate,
            mutation_strength=config.genetic.mutation_strength,
            elitism=config.genetic.elitism,
            device=self.device,
        )

        # Trainer for the PPO inner loop
        self.trainer = Trainer(
            games_per_update=config.ppo.games_per_update,
            ppo_epochs=config.ppo.ppo_epochs,
            max_moves=config.game.max_moves,
            win_reward=config.game.win_reward,
            draw_reward=config.game.draw_reward,
            loss_reward=config.game.loss_reward,
            move_penalty=config.game.move_penalty,
        )

        # Metrics tracking
        self.metrics = MetricsTracker(config.training.log_dir)

        # Create checkpoint directory
        os.makedirs(config.training.checkpoint_dir, exist_ok=True)

    def run(self, start_generation: int = 0) -> None:
        """
        Run the full evolution loop.

        Args:
            start_generation: Resume from this generation (0 = start fresh).
        """
        cfg = self.cfg

        # Initialise the population (create agents)
        self.population.initialise()

        logger.info("=" * 60)
        logger.info("Chess RL Engine — Training Started")
        logger.info(f"Generations: {cfg.genetic.num_generations}")
        logger.info(f"Population: {cfg.genetic.population_size}")
        logger.info(f"Device: {self.device}")
        logger.info("=" * 60)

        for generation in range(start_generation, cfg.genetic.num_generations):
            logger.info(f"\n{'='*60}")
            logger.info(f"GENERATION {generation + 1} / {cfg.genetic.num_generations}")
            logger.info(f"{'='*60}")

            agents = self.population.agents

            # ── Phase 1: Train (self-play + PPO) ─────────────────────────
            logger.info("Phase 1: Self-play training...")
            training_metrics = self.trainer.train_generation(agents)

            # ── Phase 2: Evaluate (determine fitness via ELO) ─────────────
            logger.info("Phase 2: Fitness evaluation...")
            fitness_scores = evaluate_population(
                agents=agents,
                games_per_agent=cfg.genetic.games_per_evaluation,
                max_moves=cfg.game.max_moves,
            )

            # ── Log metrics ───────────────────────────────────────────────
            elo_ratings = fitness_scores
            self.metrics.record_generation(
                generation=generation + 1,
                best_elo=max(elo_ratings),
                mean_elo=float(np.mean(elo_ratings)),
                win_rate=training_metrics["win_rate"],
                draw_rate=training_metrics["draw_rate"],
                loss_rate=training_metrics["loss_rate"],
                policy_loss=training_metrics["policy_loss"],
                value_loss=training_metrics["value_loss"],
                entropy=training_metrics["entropy"],
                mean_game_length=training_metrics["mean_game_length"],
            )
            self.metrics.save()

            logger.info(self.metrics.summary())

            # ── Save checkpoint ───────────────────────────────────────────
            if (generation + 1) % cfg.training.checkpoint_every == 0:
                self._save_checkpoint(generation + 1)

            # ── Phase 3: Evolve (create next generation) ──────────────────
            # Skip evolution on the last generation (no point creating unused agents)
            if generation < cfg.genetic.num_generations - 1:
                logger.info("Phase 3: Genetic evolution...")
                self.population.evolve(fitness_scores)

        # ── Training complete ─────────────────────────────────────────────
        logger.info("\n" + "=" * 60)
        logger.info("Training complete!")

        # Save the final best agent
        best = self.population.best_agent()
        best_path = os.path.join(cfg.training.checkpoint_dir, "best_agent.pt")
        best.save(best_path)
        logger.info(f"Best agent saved to {best_path} (ELO: {best.elo:.0f})")

        # Save final metrics
        self.metrics.save("metrics_final.json")
        logger.info("Final metrics saved.")

    def _save_checkpoint(self, generation: int) -> None:
        """Save the best agent as a checkpoint for this generation."""
        best = self.population.best_agent()
        path = os.path.join(
            self.cfg.training.checkpoint_dir,
            f"gen_{generation:04d}.pt"
        )
        best.save(path)
        logger.info(f"Checkpoint saved: {path} (ELO: {best.elo:.0f})")
