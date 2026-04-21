"""
memory.py
---------
Experience replay buffer for PPO training.

PPO (unlike DQN) is an on-policy algorithm, meaning it trains on
freshly collected experience rather than a random sample from the past.

After each batch of self-play games, we:
    1. Collect all (state, action, reward, ...) steps into this buffer
    2. Compute advantages using GAE (Generalised Advantage Estimation)
    3. Train the network with PPO on mini-batches from this buffer
    4. Clear the buffer and start fresh (on-policy requirement)
"""

import numpy as np
import torch
from typing import List, Iterator, Tuple
from dataclasses import dataclass

from chess_engine.game import GameStep


@dataclass
class Batch:
    """A mini-batch of experience ready for a PPO gradient update."""
    states: torch.Tensor          # (batch, 17, 8, 8)
    actions: torch.Tensor         # (batch,) int64
    old_log_probs: torch.Tensor   # (batch,) float32 — log probs when collected
    advantages: torch.Tensor      # (batch,) float32 — GAE advantages
    returns: torch.Tensor         # (batch,) float32 — discounted returns
    legal_masks: torch.Tensor     # (batch, NUM_ACTIONS) bool


class ReplayBuffer:
    """
    Stores experience from self-play games and provides mini-batches for PPO.

    Usage:
        buffer = ReplayBuffer(gamma=0.99, gae_lambda=0.95)
        buffer.add_game_steps(steps, values_bootstrap=0.0)
        for batch in buffer.get_batches(mini_batch_size=256):
            # run PPO update on batch
        buffer.clear()
    """

    def __init__(self, gamma: float = 0.99, gae_lambda: float = 0.95):
        """
        Args:
            gamma:       Discount factor for future rewards.
            gae_lambda:  GAE lambda. Controls bias/variance of advantage estimates.
                         Higher lambda = less bias, more variance.
        """
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clear()

    def clear(self) -> None:
        """Empty the buffer. Call this after each PPO update."""
        self.states: List[np.ndarray] = []
        self.actions: List[int] = []
        self.old_log_probs: List[float] = []
        self.values: List[float] = []
        self.rewards: List[float] = []
        self.dones: List[bool] = []
        self.legal_masks: List[np.ndarray] = []

    def add_game_steps(self, steps: List[GameStep]) -> None:
        """
        Add all steps from one side (white or black) of a completed game.

        Args:
            steps: List of GameStep objects from play_game().
        """
        for step in steps:
            self.states.append(step.state)
            self.actions.append(step.action)
            self.old_log_probs.append(step.action_log_prob)
            self.values.append(step.value)
            self.rewards.append(step.reward)
            self.dones.append(step.done)
            self.legal_masks.append(step.legal_mask)

    def compute_advantages(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute GAE advantages and discounted returns.

        GAE (Generalised Advantage Estimation) is a technique that
        reduces variance in the advantage estimate by looking ahead
        multiple steps rather than just one.

        Returns:
            advantages: GAE advantage estimates.
            returns:    Discounted cumulative rewards (targets for value head).
        """
        n = len(self.rewards)
        advantages = np.zeros(n, dtype=np.float32)
        returns = np.zeros(n, dtype=np.float32)

        # Bootstrap from the value of the last state.
        # If the game ended, next value is 0 (no future rewards).
        last_advantage = 0.0

        for t in reversed(range(n)):
            # If this step ended the game, there's no next value
            next_value = 0.0 if self.dones[t] else self.values[t + 1] if t + 1 < n else 0.0
            next_non_terminal = 0.0 if self.dones[t] else 1.0

            # TD error (1-step advantage estimate)
            delta = self.rewards[t] + self.gamma * next_value * next_non_terminal - self.values[t]

            # GAE: exponentially-weighted sum of future TD errors
            last_advantage = delta + self.gamma * self.gae_lambda * next_non_terminal * last_advantage
            advantages[t] = last_advantage

        # Returns = advantages + baseline (value estimates)
        returns = advantages + np.array(self.values, dtype=np.float32)

        # Normalise advantages (reduces training instability)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return advantages, returns

    def get_batches(
        self,
        mini_batch_size: int,
        device: torch.device,
    ) -> Iterator[Batch]:
        """
        Yield shuffled mini-batches for PPO updates.

        Args:
            mini_batch_size: Size of each mini-batch.
            device:          Torch device (cpu or cuda).

        Yields:
            Batch objects ready for the PPO loss computation.
        """
        advantages, returns = self.compute_advantages()
        n = len(self.states)

        # Convert everything to tensors once
        states_t = torch.tensor(np.array(self.states), dtype=torch.float32, device=device)
        actions_t = torch.tensor(self.actions, dtype=torch.int64, device=device)
        log_probs_t = torch.tensor(self.old_log_probs, dtype=torch.float32, device=device)
        advantages_t = torch.tensor(advantages, dtype=torch.float32, device=device)
        returns_t = torch.tensor(returns, dtype=torch.float32, device=device)
        masks_t = torch.tensor(np.array(self.legal_masks), dtype=torch.bool, device=device)

        # Shuffle indices for stochastic mini-batches
        indices = np.random.permutation(n)

        for start in range(0, n, mini_batch_size):
            end = min(start + mini_batch_size, n)
            idx = torch.tensor(indices[start:end], device=device)

            yield Batch(
                states=states_t[idx],
                actions=actions_t[idx],
                old_log_probs=log_probs_t[idx],
                advantages=advantages_t[idx],
                returns=returns_t[idx],
                legal_masks=masks_t[idx],
            )

    def __len__(self) -> int:
        return len(self.states)
