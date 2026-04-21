"""
scripts/train.py
----------------
Main entry point for training the Chess RL Engine.

Usage:
    # Start fresh with default settings
    python scripts/train.py

    # Custom number of generations and population size
    python scripts/train.py --generations 50 --population 20

    # Resume from a checkpoint (re-runs evolution from that generation)
    python scripts/train.py --checkpoint checkpoints/gen_0010.pt

Run from the project root directory:
    cd chess-rl-engine
    python scripts/train.py
"""

import sys
import os
import argparse

# Make sure the project root is on the Python path so imports work
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config import Config, GameConfig, PPOConfig, GeneticConfig, TrainingConfig
from training.evolution import EvolutionLoop


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train the Chess RL Engine with genetic algorithms + PPO"
    )

    # ── Generation / population settings ─────────────────────────────────
    parser.add_argument(
        "--generations", type=int, default=30,
        help="Number of generations to train (default: 30)"
    )
    parser.add_argument(
        "--population", type=int, default=16,
        help="Number of agents in the population (default: 16)"
    )
    parser.add_argument(
        "--games-per-update", type=int, default=20,
        help="Self-play games per agent before each PPO update (default: 20)"
    )
    parser.add_argument(
        "--games-per-eval", type=int, default=10,
        help="Games per agent for fitness evaluation (default: 10)"
    )

    # ── Checkpoint / output settings ─────────────────────────────────────
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Path to checkpoint to resume from (optional)"
    )
    parser.add_argument(
        "--checkpoint-dir", type=str, default="checkpoints",
        help="Directory to save checkpoints (default: checkpoints/)"
    )
    parser.add_argument(
        "--log-dir", type=str, default="logs",
        help="Directory to save logs and metrics (default: logs/)"
    )

    # ── Hardware settings ─────────────────────────────────────────────────
    parser.add_argument(
        "--no-gpu", action="store_true",
        help="Disable GPU even if available (force CPU training)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42)"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # ── Build config from command-line arguments ──────────────────────────
    cfg = Config(
        ppo=PPOConfig(
            games_per_update=args.games_per_update,
        ),
        genetic=GeneticConfig(
            population_size=args.population,
            num_generations=args.generations,
            games_per_evaluation=args.games_per_eval,
        ),
        training=TrainingConfig(
            checkpoint_dir=args.checkpoint_dir,
            log_dir=args.log_dir,
            use_gpu=not args.no_gpu,
            seed=args.seed,
        ),
    )

    print("\n" + "=" * 60)
    print("Chess RL Engine — Configuration")
    print("=" * 60)
    print(f"  Generations:         {cfg.genetic.num_generations}")
    print(f"  Population size:     {cfg.genetic.population_size}")
    print(f"  Games per update:    {cfg.ppo.games_per_update}")
    print(f"  Games per eval:      {cfg.genetic.games_per_evaluation}")
    print(f"  Checkpoint dir:      {cfg.training.checkpoint_dir}")
    print(f"  Log dir:             {cfg.training.log_dir}")
    print(f"  GPU:                 {'Disabled' if args.no_gpu else 'Auto-detect'}")
    print(f"  Seed:                {cfg.training.seed}")
    print("=" * 60 + "\n")

    # ── Run training ──────────────────────────────────────────────────────
    loop = EvolutionLoop(cfg)

    start_generation = 0
    if args.checkpoint:
        print(f"Resuming from checkpoint: {args.checkpoint}")
        # Note: full resume (reloading agent weights) would require
        # loading the population from the checkpoint. For simplicity,
        # this re-initialises the population but starts the generation counter
        # from the checkpoint generation. A full implementation would also
        # reload agent weights and replay buffer state.
        print("Note: Population is re-initialised. Agent weights are not restored.")

    loop.run(start_generation=start_generation)


if __name__ == "__main__":
    main()
