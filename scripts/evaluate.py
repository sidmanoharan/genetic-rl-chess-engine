"""
scripts/evaluate.py
-------------------
Evaluate a saved agent: play games and report performance statistics.

Usage:
    # Evaluate against a random agent (baseline)
    python scripts/evaluate.py --agent checkpoints/best_agent.pt

    # Evaluate two saved agents against each other
    python scripts/evaluate.py --agent checkpoints/gen_0030.pt --opponent checkpoints/gen_0010.pt

    # Play more games for more accurate statistics
    python scripts/evaluate.py --agent checkpoints/best_agent.pt --games 200

Run from the project root:
    cd chess-rl-engine
    python scripts/evaluate.py --agent checkpoints/best_agent.pt
"""

import sys
import os
import argparse
import torch
import numpy as np
import chess

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rl_agent.agent import ChessAgent
from chess_engine.game import play_game
from chess_engine.board import board_to_tensor, get_legal_move_mask


class RandomAgent:
    """
    A baseline agent that picks moves uniformly at random.

    Used as a sanity-check opponent — a trained agent should beat
    a random agent almost every time after sufficient training.
    """

    def select_action(self, state: np.ndarray, legal_mask: np.ndarray):
        """Pick a random legal move."""
        legal_indices = np.where(legal_mask)[0]
        action = np.random.choice(legal_indices)
        return action, 0.0, 0.0  # action, log_prob, value


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a saved Chess RL agent")

    parser.add_argument(
        "--agent", type=str, required=True,
        help="Path to the saved agent checkpoint (.pt file)"
    )
    parser.add_argument(
        "--opponent", type=str, default=None,
        help="Path to opponent agent checkpoint. If not set, plays against a random agent."
    )
    parser.add_argument(
        "--games", type=int, default=100,
        help="Number of games to play (default: 100)"
    )
    parser.add_argument(
        "--max-moves", type=int, default=200,
        help="Max moves per game (default: 200)"
    )
    parser.add_argument(
        "--no-gpu", action="store_true",
        help="Force CPU evaluation"
    )

    return parser.parse_args()


def evaluate(agent, opponent, num_games: int, max_moves: int) -> dict:
    """
    Play games between agent and opponent, report statistics.

    The agent plays as white in half the games and black in the other half
    to ensure results are not biased by colour assignment.

    Args:
        agent:     The agent being evaluated.
        opponent:  The opponent agent.
        num_games: Total number of games to play.
        max_moves: Maximum moves per game.

    Returns:
        Dictionary with win/draw/loss rates and other statistics.
    """
    wins = draws = losses = 0
    game_lengths = []

    print(f"\nPlaying {num_games} games...")
    print("-" * 40)

    for game_idx in range(num_games):
        # Alternate colours every game
        if game_idx % 2 == 0:
            white_agent, black_agent = agent, opponent
            agent_is_white = True
        else:
            white_agent, black_agent = opponent, agent
            agent_is_white = False

        result = play_game(
            white_agent=white_agent,
            black_agent=black_agent,
            max_moves=max_moves,
        )

        game_lengths.append(result.num_moves)

        if result.winner is None:
            draws += 1
            outcome = "Draw"
        elif (result.winner == chess.WHITE) == agent_is_white:
            wins += 1
            outcome = "Win"
        else:
            losses += 1
            outcome = "Loss"

        # Print progress every 10 games
        if (game_idx + 1) % 10 == 0:
            print(
                f"Game {game_idx + 1:4d}/{num_games} | "
                f"W/D/L: {wins}/{draws}/{losses} | "
                f"Last: {outcome} ({result.termination}, {result.num_moves} moves)"
            )

    total = wins + draws + losses
    return {
        "games": total,
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "win_rate": wins / total,
        "draw_rate": draws / total,
        "loss_rate": losses / total,
        "mean_game_length": np.mean(game_lengths),
        "median_game_length": np.median(game_lengths),
    }


def main():
    args = parse_args()

    device = torch.device("cpu" if args.no_gpu else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device: {device}")

    # ── Load the agent being evaluated ───────────────────────────────────
    print(f"\nLoading agent from: {args.agent}")
    agent = ChessAgent(device=device)
    agent.load(args.agent)
    print(f"Agent ELO (from training): {agent.elo:.0f}")

    # ── Load or create opponent ───────────────────────────────────────────
    if args.opponent:
        print(f"Loading opponent from: {args.opponent}")
        opponent = ChessAgent(device=device)
        opponent.load(args.opponent)
        print(f"Opponent ELO (from training): {opponent.elo:.0f}")
        opponent_name = f"Saved agent (ELO: {opponent.elo:.0f})"
    else:
        print("Opponent: Random agent (baseline)")
        opponent = RandomAgent()
        opponent_name = "Random agent"

    # ── Run evaluation ────────────────────────────────────────────────────
    print(f"\nEvaluating: Trained agent vs {opponent_name}")
    stats = evaluate(agent, opponent, num_games=args.games, max_moves=args.max_moves)

    # ── Print results ─────────────────────────────────────────────────────
    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    print(f"  Games played:     {stats['games']}")
    print(f"  Wins:             {stats['wins']} ({stats['win_rate']:.1%})")
    print(f"  Draws:            {stats['draws']} ({stats['draw_rate']:.1%})")
    print(f"  Losses:           {stats['losses']} ({stats['loss_rate']:.1%})")
    print(f"  Mean game length: {stats['mean_game_length']:.1f} moves")
    print(f"  Median game len:  {stats['median_game_length']:.0f} moves")
    print("=" * 50)

    # Interpret results
    if stats["win_rate"] > 0.8:
        print("\n✓ Strong performance — agent dominates the opponent.")
    elif stats["win_rate"] > 0.5:
        print("\n~ Solid performance — agent wins more than it loses.")
    elif stats["win_rate"] > 0.3:
        print("\n△ Moderate performance — agent is competitive but not dominant.")
    else:
        print("\n✗ Weak performance — more training recommended.")


if __name__ == "__main__":
    main()
