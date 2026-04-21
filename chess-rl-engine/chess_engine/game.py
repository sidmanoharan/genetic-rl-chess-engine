"""
game.py
-------
Game loop: runs a full chess game between two agents and returns the result.

This is the core of the self-play system. Each "episode" in RL terms is
one complete chess game. The game loop:
    1. Resets the board to the starting position
    2. Alternates between the two agents
    3. Collects (state, action, reward) tuples for training
    4. Terminates when checkmate, stalemate, or draw conditions are met
"""

import chess
import numpy as np
from typing import List, Tuple, Optional, TYPE_CHECKING
from dataclasses import dataclass, field

from chess_engine.board import board_to_tensor, get_legal_move_mask, move_to_index

if TYPE_CHECKING:
    from rl_agent.agent import ChessAgent


@dataclass
class GameStep:
    """
    One step (half-move / ply) in a chess game.

    This is the data structure stored in the experience buffer.
    After the game ends, we go back and assign the final reward
    to each step taken by the winning agent.
    """
    state: np.ndarray          # Board tensor at this step: shape (17, 8, 8)
    action: int                # Move chosen (integer index)
    action_log_prob: float     # Log probability of this action (for PPO)
    value: float               # Agent's estimated value of this state
    reward: float              # Reward received (filled in after game ends)
    done: bool                 # Whether this was the final step
    legal_mask: np.ndarray     # Boolean mask of legal moves at this state


@dataclass
class GameResult:
    """
    The complete result of one chess game.

    Contains the full trajectory of steps for both agents, plus
    metadata about how the game ended.
    """
    winner: Optional[chess.Color]   # chess.WHITE, chess.BLACK, or None (draw)
    termination: str                 # "checkmate", "stalemate", "draw", "max_moves"
    num_moves: int                   # Total number of half-moves played
    white_steps: List[GameStep]      # Steps taken by the white agent
    black_steps: List[GameStep]      # Steps taken by the black agent


def play_game(
    white_agent: "ChessAgent",
    black_agent: "ChessAgent",
    max_moves: int = 200,
    win_reward: float = 1.0,
    draw_reward: float = 0.0,
    loss_reward: float = -1.0,
    move_penalty: float = -0.001,
) -> GameResult:
    """
    Run one complete chess game between two agents.

    Args:
        white_agent:  Agent playing as white.
        black_agent:  Agent playing as black.
        max_moves:    Maximum half-moves before declaring a draw.
        win_reward:   Reward for winning.
        draw_reward:  Reward for drawing.
        loss_reward:  Reward for losing.
        move_penalty: Per-move penalty to encourage decisive play.

    Returns:
        GameResult with full trajectories and outcome metadata.
    """
    board = chess.Board()

    white_steps: List[GameStep] = []
    black_steps: List[GameStep] = []

    for move_number in range(max_moves):
        # Determine whose turn it is
        current_agent = white_agent if board.turn == chess.WHITE else black_agent
        current_steps = white_steps if board.turn == chess.WHITE else black_steps

        # ── Get board state ───────────────────────────────────────────────
        state = board_to_tensor(board)                    # (17, 8, 8)
        legal_mask = get_legal_move_mask(board)           # (NUM_ACTIONS,)

        # ── Agent selects a move ──────────────────────────────────────────
        action, log_prob, value = current_agent.select_action(state, legal_mask)

        # Convert integer index to chess.Move and apply it
        move_uci = None
        for move in board.legal_moves:
            if move_to_index(move) == action:
                move_uci = move
                break

        # Fallback: pick a random legal move if index doesn't match
        # (can happen in early training with random policies)
        if move_uci is None:
            legal_moves = list(board.legal_moves)
            move_uci = np.random.choice(legal_moves)
            action = move_to_index(move_uci)

        # ── Apply move to board ───────────────────────────────────────────
        board.push(move_uci)

        # Small per-move penalty to encourage decisive play
        step_reward = move_penalty

        step = GameStep(
            state=state,
            action=action,
            action_log_prob=log_prob,
            value=value,
            reward=step_reward,   # Will be updated with final reward below
            done=False,
            legal_mask=legal_mask,
        )
        current_steps.append(step)

        # ── Check for game over ───────────────────────────────────────────
        if board.is_game_over():
            break

    # ── Determine outcome and assign final rewards ─────────────────────────
    outcome = board.outcome()

    if outcome is None:
        # Max moves reached with no winner
        winner = None
        termination = "max_moves"
    elif outcome.winner is None:
        # Stalemate or other draw
        winner = None
        termination = str(outcome.termination.name).lower()
    else:
        winner = outcome.winner
        termination = str(outcome.termination.name).lower()

    # Assign final rewards to the last step of each agent
    if winner is None:
        # Draw: both agents get draw reward
        _assign_final_reward(white_steps, draw_reward)
        _assign_final_reward(black_steps, draw_reward)
    elif winner == chess.WHITE:
        _assign_final_reward(white_steps, win_reward)
        _assign_final_reward(black_steps, loss_reward)
    else:
        _assign_final_reward(white_steps, loss_reward)
        _assign_final_reward(black_steps, win_reward)

    # Mark the last step as terminal
    if white_steps:
        white_steps[-1].done = True
    if black_steps:
        black_steps[-1].done = True

    return GameResult(
        winner=winner,
        termination=termination,
        num_moves=len(white_steps) + len(black_steps),
        white_steps=white_steps,
        black_steps=black_steps,
    )


def _assign_final_reward(steps: List[GameStep], final_reward: float) -> None:
    """
    Add the final reward to the last step in a trajectory.

    We don't overwrite the per-move penalty; we add the final outcome
    reward on top of it.
    """
    if steps:
        steps[-1].reward += final_reward
        steps[-1].done = True
