"""
rl_agent/mcts.py
----------------
Monte Carlo Tree Search (MCTS) — applied at inference time only.
No retraining needed. Plug onto any existing checkpoint for ~200-300 ELO boost.

Without MCTS: network looks at board once, picks highest-probability move.
With MCTS:    simulates N future positions, guided by the network, then picks
              the move with the most visits (most consistently good).

UCB formula used at each selection step:
    UCB = Q(s,a) + c_puct * P(s,a) * sqrt(N_parent) / (1 + N(s,a))
    Q     = average value of simulations through this move
    P     = prior probability from the neural network policy head
    N     = visit counts (how many times we've explored this branch)
    c_puct = exploration constant (higher = more exploration)
"""

import math
import chess
import numpy as np
import torch
from typing import Optional, Dict

from chess_engine.board import board_to_tensor, get_legal_move_mask, move_to_index, NUM_ACTIONS
from rl_agent.network import ChessNet


class MCTSNode:
    """One node in the search tree — represents a board position."""

    def __init__(self, board: chess.Board, parent: Optional["MCTSNode"] = None, prior: float = 0.0):
        self.board = board.copy()
        self.parent = parent
        self.prior = prior          # P(s,a): network's prior probability for this move

        self.children: Dict[int, "MCTSNode"] = {}  # action_idx -> child node
        self.visit_count = 0        # N(s,a)
        self.value_sum = 0.0        # sum of all backup values (for computing Q)

    @property
    def q_value(self) -> float:
        """Average value of all simulations through this node."""
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def ucb_score(self, c_puct: float = 1.5) -> float:
        """
        Upper Confidence Bound score — balances exploitation (Q) vs exploration (prior/visits).
        Nodes with high Q *and* high prior but few visits score highest.
        """
        if self.parent is None:
            return 0.0
        exploration = c_puct * self.prior * math.sqrt(self.parent.visit_count) / (1 + self.visit_count)
        return self.q_value + exploration


class MCTS:
    """
    Monte Carlo Tree Search wrapper around a trained ChessNet.

    Usage:
        mcts = MCTS(agent.network, num_simulations=200)
        move  = mcts.select_action(board)
    """

    def __init__(self, network: ChessNet, num_simulations: int = 200, c_puct: float = 1.5, device=None):
        """
        Args:
            network:         Trained ChessNet (policy + value heads).
            num_simulations: Rollouts per move. 200 = good for interactive play.
            c_puct:          Exploration constant.
            device:          Torch device.
        """
        self.network = network
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.network.eval()

    def select_action(self, board: chess.Board, temperature: float = 0.0) -> chess.Move:
        """
        Run MCTS and return the best move for the current position.

        Args:
            board:       Current board position.
            temperature: 0.0 = pick move with most visits (use for play).
                         1.0 = sample proportional to visits (use during training).

        Returns:
            Best chess.Move according to MCTS.
        """
        root = MCTSNode(board)
        self._expand(root)

        for _ in range(self.num_simulations):
            node = self._select(root)
            if not node.board.is_game_over() and node.is_leaf() and node != root:
                self._expand(node)
            value = self._evaluate(node)
            self._backup(node, value)

        if not root.children:
            return list(board.legal_moves)[0]

        action_visits = {a: c.visit_count for a, c in root.children.items()}

        if temperature == 0.0:
            best_action = max(action_visits, key=action_visits.get)
        else:
            visits = np.array(list(action_visits.values()), dtype=np.float32)
            visits = visits ** (1.0 / temperature)
            probs = visits / visits.sum()
            best_action = np.random.choice(list(action_visits.keys()), p=probs)

        for move in board.legal_moves:
            if move_to_index(move) == best_action:
                return move

        return list(board.legal_moves)[0]

    def _select(self, node: MCTSNode) -> MCTSNode:
        """Walk down the tree, always picking the child with highest UCB."""
        while not node.is_leaf() and not node.board.is_game_over():
            node = max(node.children.values(), key=lambda c: c.ucb_score(self.c_puct))
        return node

    def _expand(self, node: MCTSNode) -> None:
        """Ask the network for move priors, create one child node per legal move."""
        if node.board.is_game_over():
            return

        state = torch.tensor(
            board_to_tensor(node.board), dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        mask = torch.tensor(
            get_legal_move_mask(node.board), dtype=torch.bool, device=self.device
        )

        with torch.no_grad():
            logits, _ = self.network(state)
            logits = logits.squeeze(0).masked_fill(~mask, float("-inf"))
            priors = torch.softmax(logits, dim=-1).cpu().numpy()

        for move in node.board.legal_moves:
            action_idx = move_to_index(move)
            child_board = node.board.copy()
            child_board.push(move)
            node.children[action_idx] = MCTSNode(
                board=child_board, parent=node, prior=float(priors[action_idx])
            )

    def _evaluate(self, node: MCTSNode) -> float:
        """Get value head estimate for this position. Returns value in [-1, 1]."""
        if node.board.is_game_over():
            outcome = node.board.outcome()
            if outcome is None or outcome.winner is None:
                return 0.0
            just_moved = not node.board.turn
            return 1.0 if outcome.winner == just_moved else -1.0

        state = torch.tensor(
            board_to_tensor(node.board), dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        with torch.no_grad():
            _, value = self.network(state)
        return value.item()

    def _backup(self, node: MCTSNode, value: float) -> None:
        """Propagate value up the tree, flipping sign at each level."""
        while node is not None:
            node.visit_count += 1
            node.value_sum += value
            value = -value  # Flip: good for one side = bad for the other
            node = node.parent
