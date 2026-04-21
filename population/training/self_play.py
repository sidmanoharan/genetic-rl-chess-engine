"""
training/self_play.py
---------------------
Self-play: agents improve by playing against each other.

This is the core insight from AlphaZero — instead of training against
a fixed opponent or human games, agents generate their own training data
by playing against themselves or each other.

Why self-play works:
    - The opponent always matches the agent's current skill level
    - As the agent improves, it faces stronger opposition automatically
    - No need for human-labelled data or hand-crafted positions
    - The agent discovers strategies humans may never have considered

The self-play loop:
    1. Pick two agents from the population (e.g. current vs random opponent)
    2. Play N games between them
    3. Collect (state, action, reward) experience for both agents
    4. Use this experience to update both agents via PPO
    5. Update ELO ratings based on game results
"""

import numpy as np
import logging
from typing import List, Tuple, Optional
from tqdm import tqdm

from chess_engine.game import play_game, GameResult
from rl_agent.agent import ChessAgent

logger = logging.getLogger("chess_rl")


def run_self_play_games(
    agent_white: ChessAgent,
    agent_black: ChessAgent,
    num_games: int,
    max_moves: int = 200,
    win_reward: float = 1.0,
    draw_reward: float = 0.0,
    loss_reward: float = -1.0,
    move_penalty: float = -0.001,
    collect_experience: bool = True,
    update_elo: bool = True,
) -> List[GameResult]:
    """
    Play multiple games between two agents and collect training experience.

    After each game, the experience (states, actions, rewards) is stored
    in each agent's replay buffer. Call agent.update() afterwards to
    run PPO on the collected experience.

    Args:
        agent_white:         Agent playing as white.
        agent_black:         Agent playing as black.
        num_games:           How many games to play.
        max_moves:           Max half-moves per game before declaring draw.
        win_reward:          Reward for winning.
        draw_reward:         Reward for a draw.
        loss_reward:         Reward for losing.
        move_penalty:        Small per-move negative reward.
        collect_experience:  If True, store experience in agents' buffers.
        update_elo:          If True, update ELO ratings after each game.

    Returns:
        List of GameResult objects (one per game).
    """
    results: List[GameResult] = []

    for game_idx in range(num_games):
        result = play_game(
            white_agent=agent_white,
            black_agent=agent_black,
            max_moves=max_moves,
            win_reward=win_reward,
            draw_reward=draw_reward,
            loss_reward=loss_reward,
            move_penalty=move_penalty,
        )
        results.append(result)

        # ── Store experience in replay buffers ────────────────────────────
        if collect_experience:
            agent_white.buffer.add_game_steps(result.white_steps)
            agent_black.buffer.add_game_steps(result.black_steps)

        # ── Update ELO ratings ────────────────────────────────────────────
        if update_elo:
            if result.winner is None:
                # Draw: both get 0.5
                agent_white.update_elo(agent_black.elo, 0.5)
                agent_black.update_elo(agent_white.elo, 0.5)
            else:
                import chess
                if result.winner == chess.WHITE:
                    agent_white.update_elo(agent_black.elo, 1.0)
                    agent_black.update_elo(agent_white.elo, 0.0)
                else:
                    agent_white.update_elo(agent_black.elo, 0.0)
                    agent_black.update_elo(agent_white.elo, 1.0)

    return results


def evaluate_population(
    agents: List[ChessAgent],
    games_per_agent: int = 10,
    max_moves: int = 200,
) -> List[float]:
    """
    Evaluate the fitness of every agent by having them play round-robin games.

    Each agent plays `games_per_agent` games against randomly selected
    opponents from the population. ELO is updated after each game.

    Args:
        agents:          List of agents to evaluate.
        games_per_agent: How many games each agent plays in total.
        max_moves:       Max moves per game.

    Returns:
        List of ELO ratings, one per agent (in the same order as `agents`).
    """
    n = len(agents)
    logger.info(f"Evaluating {n} agents ({games_per_agent} games each)...")

    total_games = n * games_per_agent // 2  # Each game involves 2 agents
    pbar = tqdm(total=total_games, desc="Evaluating", unit="game")

    games_played = [0] * n

    # Round-robin: pair up agents that haven't played enough games yet
    agent_indices = list(range(n))

    for _ in range(total_games):
        # Pick two different agents at random, preferring those with fewer games
        weights = [1.0 / (games_played[i] + 1) for i in range(n)]
        weights = np.array(weights) / sum(weights)

        idx_a = np.random.choice(n, p=weights)
        # Pick idx_b different from idx_a
        remaining = [i for i in range(n) if i != idx_a]
        idx_b = np.random.choice(remaining)

        # Randomly assign colours (important for fairness)
        if np.random.random() < 0.5:
            white_idx, black_idx = idx_a, idx_b
        else:
            white_idx, black_idx = idx_b, idx_a

        run_self_play_games(
            agent_white=agents[white_idx],
            agent_black=agents[black_idx],
            num_games=1,
            max_moves=max_moves,
            collect_experience=False,  # Evaluation only, don't collect training data
            update_elo=True,
        )

        games_played[white_idx] += 1
        games_played[black_idx] += 1
        pbar.update(1)

    pbar.close()

    fitness = [agent.elo for agent in agents]
    logger.info(
        f"Evaluation complete. Best ELO: {max(fitness):.0f}, Mean: {np.mean(fitness):.0f}"
    )
    return fitness


def collect_training_data(
    agents: List[ChessAgent],
    games_per_update: int = 20,
    max_moves: int = 200,
    win_reward: float = 1.0,
    draw_reward: float = 0.0,
    loss_reward: float = -1.0,
    move_penalty: float = -0.001,
) -> dict:
    """
    Have agents play self-play games and collect training experience.

    Each agent plays against randomly selected opponents. Experience is
    stored in each agent's replay buffer for the PPO update.

    Args:
        agents:          List of agents to train.
        games_per_update: Number of games to collect before training.
        max_moves:       Max moves per game.

    Returns:
        Dictionary with game statistics (win/draw/loss rates, game lengths).
    """
    n = len(agents)
    wins = draws = losses = 0
    total_moves = 0
    total_games = 0

    import chess
    pbar = tqdm(total=n * games_per_update, desc="Self-play", unit="game")

    for agent_idx, agent in enumerate(agents):
        for _ in range(games_per_update):
            # Pick a random opponent (could be the same agent — self-play!)
            opponent_idx = np.random.randint(0, n)
            opponent = agents[opponent_idx]

            # Randomly assign colours
            if np.random.random() < 0.5:
                white_agent, black_agent = agent, opponent
                agent_is_white = True
            else:
                white_agent, black_agent = opponent, agent
                agent_is_white = False

            result = play_game(
                white_agent=white_agent,
                black_agent=black_agent,
                max_moves=max_moves,
                win_reward=win_reward,
                draw_reward=draw_reward,
                loss_reward=loss_reward,
                move_penalty=move_penalty,
            )

            # Store experience for both agents
            agent.buffer.add_game_steps(result.white_steps if agent_is_white else result.black_steps)
            opponent.buffer.add_game_steps(result.black_steps if agent_is_white else result.white_steps)

            # Track statistics
            total_moves += result.num_moves
            total_games += 1

            if result.winner is None:
                draws += 1
            elif (result.winner == chess.WHITE) == agent_is_white:
                wins += 1
            else:
                losses += 1

            pbar.update(1)

    pbar.close()

    return {
        "win_rate": wins / max(total_games, 1),
        "draw_rate": draws / max(total_games, 1),
        "loss_rate": losses / max(total_games, 1),
        "mean_game_length": total_moves / max(total_games, 1),
        "total_games": total_games,
    }
