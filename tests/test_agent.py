"""
tests/test_agent.py
-------------------
Unit tests for the neural network and PPO agent.

Run with:
    pytest tests/test_agent.py -v
"""

import pytest
import torch
import numpy as np
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rl_agent.network import ChessNet
from rl_agent.agent import ChessAgent
from chess_engine.board import NUM_ACTIONS, board_to_tensor, get_legal_move_mask
import chess


class TestChessNet:
    """Tests for the neural network architecture."""

    def setup_method(self):
        self.device = torch.device("cpu")
        self.net = ChessNet(num_residual_blocks=2, num_filters=32).to(self.device)

    def test_forward_output_shapes(self):
        """Network outputs must have the correct shapes."""
        batch_size = 4
        x = torch.randn(batch_size, 17, 8, 8)
        policy_logits, value = self.net(x)

        assert policy_logits.shape == (batch_size, NUM_ACTIONS), (
            f"Policy logits: expected ({batch_size}, {NUM_ACTIONS}), got {policy_logits.shape}"
        )
        assert value.shape == (batch_size, 1), (
            f"Value: expected ({batch_size}, 1), got {value.shape}"
        )

    def test_value_in_range(self):
        """Value head should output values in [-1, 1] (tanh activation)."""
        x = torch.randn(10, 17, 8, 8)
        _, value = self.net(x)
        assert (value >= -1).all() and (value <= 1).all(), (
            "Value head outputs should be in [-1, 1]"
        )

    def test_get_action_probs_sums_to_one(self):
        """get_action_probs should return a valid probability distribution."""
        board = chess.Board()
        state = torch.tensor(board_to_tensor(board), dtype=torch.float32).unsqueeze(0)
        mask = torch.tensor(get_legal_move_mask(board), dtype=torch.bool)

        probs = self.net.get_action_probs(state, mask)

        # Should sum to approximately 1.0
        assert abs(probs.sum().item() - 1.0) < 1e-4, (
            f"Probabilities sum to {probs.sum().item()}, expected ~1.0"
        )

    def test_illegal_moves_have_zero_prob(self):
        """Illegal moves should have exactly zero probability after masking."""
        board = chess.Board()
        state = torch.tensor(board_to_tensor(board), dtype=torch.float32).unsqueeze(0)
        mask = torch.tensor(get_legal_move_mask(board), dtype=torch.bool)

        probs = self.net.get_action_probs(state, mask)

        # All illegal moves should have zero probability
        illegal_probs = probs[~mask]
        assert (illegal_probs == 0).all(), "Illegal moves should have zero probability"

    def test_no_nan_in_outputs(self):
        """Network should not produce NaN values."""
        x = torch.randn(4, 17, 8, 8)
        policy_logits, value = self.net(x)
        assert not torch.isnan(policy_logits).any(), "NaN in policy logits"
        assert not torch.isnan(value).any(), "NaN in value"

    def test_gradient_flows(self):
        """Gradients should flow through the entire network."""
        x = torch.randn(2, 17, 8, 8, requires_grad=True)
        policy_logits, value = self.net(x)
        loss = policy_logits.mean() + value.mean()
        loss.backward()
        assert x.grad is not None, "No gradient flowed to input"


class TestChessAgent:
    """Tests for the PPO agent."""

    def setup_method(self):
        self.device = torch.device("cpu")
        self.agent = ChessAgent(
            num_residual_blocks=2,
            num_filters=32,
            device=self.device,
        )

    def test_select_action_returns_valid_action(self):
        """select_action should return a legal move index."""
        board = chess.Board()
        state = board_to_tensor(board)
        mask = get_legal_move_mask(board)

        action, log_prob, value = self.agent.select_action(state, mask)

        assert isinstance(action, int), "Action should be an integer"
        assert 0 <= action < NUM_ACTIONS, f"Action {action} out of range [0, {NUM_ACTIONS})"
        assert mask[action], f"Action {action} is illegal (mask is False)"
        assert isinstance(log_prob, float), "log_prob should be a float"
        assert isinstance(value, float), "value should be a float"

    def test_initial_elo(self):
        """New agent should start at the initial ELO."""
        assert self.agent.elo == 1200, f"Expected initial ELO 1200, got {self.agent.elo}"

    def test_elo_update_win(self):
        """Winning against an equal opponent should increase ELO."""
        initial_elo = self.agent.elo
        self.agent.update_elo(opponent_elo=initial_elo, result=1.0)  # Win
        assert self.agent.elo > initial_elo, "ELO should increase after a win"

    def test_elo_update_loss(self):
        """Losing against an equal opponent should decrease ELO."""
        initial_elo = self.agent.elo
        self.agent.update_elo(opponent_elo=initial_elo, result=0.0)  # Loss
        assert self.agent.elo < initial_elo, "ELO should decrease after a loss"

    def test_elo_update_draw(self):
        """Drawing against an equal opponent should leave ELO unchanged."""
        initial_elo = self.agent.elo
        self.agent.update_elo(opponent_elo=initial_elo, result=0.5)  # Draw
        assert self.agent.elo == initial_elo, "ELO should not change after a draw vs equal opponent"

    def test_save_and_load(self, tmp_path):
        """Agent should be saveable and loadable."""
        save_path = str(tmp_path / "agent.pt")
        original_elo = self.agent.elo

        self.agent.save(save_path)

        new_agent = ChessAgent(num_residual_blocks=2, num_filters=32, device=self.device)
        new_agent.load(save_path)

        assert new_agent.elo == original_elo, "ELO should be preserved after save/load"
