"""
agent.py
--------
The PPO chess agent: wraps the neural network and implements the
Proximal Policy Optimisation algorithm.

PPO in plain English:
─────────────────────
1. Collect experience by playing games (self-play)
2. Compute how much better/worse each move was than expected (advantage)
3. Update the network to make good moves more likely
4. Crucially: don't update too much at once — the "clip" in PPO prevents
   the new policy from straying too far from the old one, which keeps
   training stable.
5. Repeat.

The "proximal" in PPO refers to keeping the new policy close (proximal)
to the old policy. This is the key stability trick.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

from rl_agent.network import ChessNet
from rl_agent.memory import ReplayBuffer, Batch
from chess_engine.board import NUM_ACTIONS


# Starting ELO for new agents (standard chess rating baseline)
INITIAL_ELO = 1200


class ChessAgent:
    """
    A chess-playing agent that learns via PPO.

    Each agent has:
        - A neural network (policy + value heads)
        - An optimiser (Adam)
        - An ELO rating (for genetic fitness evaluation)
        - A genome (hyperparameters from the genetic algorithm)
    """

    def __init__(
        self,
        num_residual_blocks: int = 4,
        num_filters: int = 128,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        clip_epsilon: float = 0.2,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        ppo_epochs: int = 4,
        mini_batch_size: int = 256,
        gae_lambda: float = 0.95,
        max_grad_norm: float = 0.5,
        device: torch.device = None,
    ):
        """
        Args:
            num_residual_blocks: Depth of the network tower.
            num_filters:         Width of convolutional layers.
            learning_rate:       Adam learning rate.
            gamma:               Reward discount factor.
            clip_epsilon:        PPO clipping range.
            entropy_coef:        Weight of entropy bonus (encourages exploration).
            value_coef:          Weight of value loss vs policy loss.
            ppo_epochs:          Number of gradient updates per batch.
            mini_batch_size:     Samples per gradient update.
            gae_lambda:          GAE smoothing parameter.
            max_grad_norm:       Gradient clipping threshold.
            device:              Torch device. Auto-detects GPU if None.
        """
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Neural network
        self.network = ChessNet(
            num_residual_blocks=num_residual_blocks,
            num_filters=num_filters,
        ).to(self.device)

        self.optimiser = torch.optim.Adam(
            self.network.parameters(),
            lr=learning_rate,
            eps=1e-5,  # Small epsilon for numerical stability
        )

        # PPO hyperparameters
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.ppo_epochs = ppo_epochs
        self.mini_batch_size = mini_batch_size
        self.max_grad_norm = max_grad_norm

        # Experience buffer (cleared after each PPO update)
        self.buffer = ReplayBuffer(gamma=gamma, gae_lambda=gae_lambda)

        # ELO rating (used as fitness score in the genetic algorithm)
        self.elo = INITIAL_ELO

        # Training statistics (for logging)
        self.total_policy_loss = 0.0
        self.total_value_loss = 0.0
        self.total_entropy = 0.0
        self.update_count = 0

    # ── Action selection ──────────────────────────────────────────────────

    def select_action(
        self,
        state: np.ndarray,
        legal_mask: np.ndarray,
    ) -> Tuple[int, float, float]:
        """
        Choose a move given the current board state.

        During training, we sample from the policy distribution to
        encourage exploration. The legal mask ensures we never pick
        an illegal move.

        Args:
            state:      Board tensor of shape (17, 8, 8).
            legal_mask: Boolean array of shape (NUM_ACTIONS,).

        Returns:
            action:   Integer index of the chosen move.
            log_prob: Log probability of this action (needed for PPO).
            value:    Estimated value of this state (needed for GAE).
        """
        self.network.eval()
        with torch.no_grad():
            state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            mask_t = torch.tensor(legal_mask, dtype=torch.bool, device=self.device)

            logits, value = self.network(state_t)
            logits = logits.squeeze(0)

            # Mask illegal moves
            logits = logits.masked_fill(~mask_t, float("-inf"))

            # Sample from the distribution (stochastic during training)
            probs = F.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()

            log_prob = dist.log_prob(action)

        return (
            action.item(),
            log_prob.item(),
            value.item(),
        )

    def select_best_action(
        self,
        state: np.ndarray,
        legal_mask: np.ndarray,
    ) -> int:
        """
        Choose the move with the highest probability (greedy / deterministic).

        Use this for evaluation, not training — during training we need
        stochastic actions for exploration.

        Args:
            state:      Board tensor of shape (17, 8, 8).
            legal_mask: Boolean array of shape (NUM_ACTIONS,).

        Returns:
            Integer index of the best move.
        """
        self.network.eval()
        with torch.no_grad():
            state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            mask_t = torch.tensor(legal_mask, dtype=torch.bool, device=self.device)

            logits, _ = self.network(state_t)
            logits = logits.squeeze(0)
            logits = logits.masked_fill(~mask_t, float("-inf"))

            return logits.argmax().item()

    # ── PPO Training ──────────────────────────────────────────────────────

    def update(self) -> dict:
        """
        Run PPO updates on the experience stored in the buffer.

        Call this after collecting enough self-play games.

        Returns:
            Dictionary of training metrics (for logging).
        """
        if len(self.buffer) == 0:
            return {}

        self.network.train()

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        num_updates = 0

        # Multiple epochs over the same batch (PPO key feature)
        for _ in range(self.ppo_epochs):
            for batch in self.buffer.get_batches(self.mini_batch_size, self.device):
                policy_loss, value_loss, entropy = self._ppo_update(batch)
                total_policy_loss += policy_loss
                total_value_loss += value_loss
                total_entropy += entropy
                num_updates += 1

        # Clear buffer after update (on-policy requirement)
        self.buffer.clear()

        avg = lambda x: x / max(num_updates, 1)
        return {
            "policy_loss": avg(total_policy_loss),
            "value_loss": avg(total_value_loss),
            "entropy": avg(total_entropy),
        }

    def _ppo_update(self, batch: Batch) -> Tuple[float, float, float]:
        """
        One PPO gradient update on a mini-batch.

        The PPO loss has three components:
            1. Policy loss: make good actions more likely, bad actions less likely.
               The clip prevents too large an update.
            2. Value loss: make the value head more accurate.
            3. Entropy bonus: discourages collapsing to a single move (exploration).

        Args:
            batch: Mini-batch of experience.

        Returns:
            (policy_loss, value_loss, entropy) as Python floats.
        """
        # Forward pass with current network
        logits, values = self.network(batch.states)

        # Mask illegal moves
        logits = logits.masked_fill(~batch.legal_masks, float("-inf"))
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)

        # Log prob of the actions that were actually taken
        new_log_probs = dist.log_prob(batch.actions)
        entropy = dist.entropy().mean()

        # ── Policy loss (PPO-clip) ────────────────────────────────────────
        # Ratio = new_prob / old_prob (in log space for numerical stability)
        ratio = torch.exp(new_log_probs - batch.old_log_probs)

        # Unclipped surrogate objective
        surrogate1 = ratio * batch.advantages

        # Clipped surrogate objective: prevents too large an update
        surrogate2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch.advantages

        # We minimise the negative of the minimum (we want to maximise reward)
        policy_loss = -torch.min(surrogate1, surrogate2).mean()

        # ── Value loss ────────────────────────────────────────────────────
        # MSE between predicted value and actual discounted return
        values = values.squeeze(-1)
        value_loss = F.mse_loss(values, batch.returns)

        # ── Total loss ────────────────────────────────────────────────────
        loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

        # Gradient update
        self.optimiser.zero_grad()
        loss.backward()

        # Gradient clipping prevents exploding gradients
        nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)

        self.optimiser.step()

        return policy_loss.item(), value_loss.item(), entropy.item()

    # ── ELO rating ────────────────────────────────────────────────────────

    def update_elo(self, opponent_elo: float, result: float) -> None:
        """
        Update this agent's ELO rating after a game.

        ELO formula: new_rating = old_rating + K * (actual - expected)
        where expected = 1 / (1 + 10^((opponent_elo - my_elo) / 400))

        Args:
            opponent_elo: ELO rating of the opponent.
            result:       1.0 for win, 0.5 for draw, 0.0 for loss.
        """
        K = 32  # K-factor: controls how quickly ratings change
        expected = 1 / (1 + 10 ** ((opponent_elo - self.elo) / 400))
        self.elo += K * (result - expected)

    # ── Serialisation ─────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        """Save agent weights and state to disk."""
        torch.save({
            "network_state": self.network.state_dict(),
            "optimiser_state": self.optimiser.state_dict(),
            "elo": self.elo,
        }, path)

    def load(self, path: str) -> None:
        """Load agent weights and state from disk."""
        checkpoint = torch.load(path, map_location=self.device)
        self.network.load_state_dict(checkpoint["network_state"])
        self.optimiser.load_state_dict(checkpoint["optimiser_state"])
        self.elo = checkpoint["elo"]
