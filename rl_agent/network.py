"""
network.py
----------
Neural network architecture for the chess policy and value heads.

Architecture: Convolutional residual tower + two heads
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Input: (batch, 17, 8, 8) tensor
  │
  ▼
Conv stem: 17 → num_filters, 3×3 conv, BatchNorm, ReLU
  │
  ▼
Residual tower: N × ResidualBlock(num_filters)
  │
  ├─── Policy head ────────────────────────────────────
  │    Conv 2→2 → Flatten → Linear → softmax(NUM_ACTIONS)
  │    Outputs: probability over all possible moves
  │
  └─── Value head ─────────────────────────────────────
       Conv 1→1 → Flatten → Linear(256) → Linear(1) → tanh
       Outputs: scalar in [-1, 1] estimating who is winning

This is directly inspired by AlphaZero's architecture. The residual
blocks are the key insight — they let gradients flow through many layers
without vanishing (the skip connection adds the input back to the output).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from chess_engine.board import NUM_ACTIONS


class ResidualBlock(nn.Module):
    """
    A single residual block.

    Structure:
        input ─► Conv ─► BN ─► ReLU ─► Conv ─► BN ─► + ─► ReLU ─► output
          └────────────────────────────────────────────┘
          (skip connection: adds input directly to output)

    The skip connection is the key innovation: without it, gradients
    vanish in deep networks and training fails.
    """

    def __init__(self, num_filters: int):
        super().__init__()
        self.conv1 = nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_filters)
        self.conv2 = nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_filters)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x  # Save input for skip connection

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        out = out + residual  # Skip connection: add original input back
        return F.relu(out)


class ChessNet(nn.Module):
    """
    Combined policy + value network for chess.

    Takes a board state tensor and outputs:
        - policy_logits: raw scores for each of the 4672 possible moves
        - value:         scalar estimate of position quality in [-1, 1]

    During play, policy_logits are masked (illegal moves zeroed) and
    passed through softmax to get move probabilities.
    During training, both heads are trained simultaneously via PPO loss.
    """

    def __init__(self, num_residual_blocks: int = 4, num_filters: int = 128):
        """
        Args:
            num_residual_blocks: Depth of the network tower. More blocks = stronger.
            num_filters:         Width of each convolutional layer.
        """
        super().__init__()

        # ── Input stem: project 17 input planes into num_filters ──────────
        self.stem = nn.Sequential(
            nn.Conv2d(17, num_filters, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_filters),
            nn.ReLU(),
        )

        # ── Residual tower ────────────────────────────────────────────────
        self.tower = nn.Sequential(
            *[ResidualBlock(num_filters) for _ in range(num_residual_blocks)]
        )

        # ── Policy head: predicts move probabilities ───────────────────────
        # Reduces to 2 channels, flattens, then outputs logits for each move
        self.policy_conv = nn.Conv2d(num_filters, 2, kernel_size=1, bias=False)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * 8 * 8, NUM_ACTIONS)

        # ── Value head: predicts game outcome ─────────────────────────────
        # Reduces to 1 channel, flattens, then outputs a scalar
        self.value_conv = nn.Conv2d(num_filters, 1, kernel_size=1, bias=False)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(1 * 8 * 8, 256)
        self.value_fc2 = nn.Linear(256, 1)

        # Initialise weights with small values for stable early training
        self._init_weights()

    def _init_weights(self):
        """Xavier initialisation for convolutional and linear layers."""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.xavier_uniform_(module.weight)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the network.

        Args:
            x: Board tensor of shape (batch_size, 17, 8, 8).

        Returns:
            policy_logits: Shape (batch_size, NUM_ACTIONS). Raw (unmasked) move scores.
            value:         Shape (batch_size, 1). Position evaluation in [-1, 1].
        """
        # Shared representation
        x = self.stem(x)    # (batch, num_filters, 8, 8)
        x = self.tower(x)   # (batch, num_filters, 8, 8)

        # Policy head
        p = F.relu(self.policy_bn(self.policy_conv(x)))  # (batch, 2, 8, 8)
        p = p.view(p.size(0), -1)                        # (batch, 128)
        policy_logits = self.policy_fc(p)                # (batch, NUM_ACTIONS)

        # Value head
        v = F.relu(self.value_bn(self.value_conv(x)))    # (batch, 1, 8, 8)
        v = v.view(v.size(0), -1)                        # (batch, 64)
        v = F.relu(self.value_fc1(v))                    # (batch, 256)
        value = torch.tanh(self.value_fc2(v))            # (batch, 1) in [-1, 1]

        return policy_logits, value

    def get_action_probs(
        self,
        state: torch.Tensor,
        legal_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Get masked move probabilities for a single position.

        Illegal moves are given probability 0 by setting their logits
        to a very large negative number before softmax.

        Args:
            state:      Board tensor of shape (1, 17, 8, 8).
            legal_mask: Boolean tensor of shape (NUM_ACTIONS,).

        Returns:
            Probability distribution over legal moves. Shape: (NUM_ACTIONS,).
        """
        logits, _ = self.forward(state)
        logits = logits.squeeze(0)  # (NUM_ACTIONS,)

        # Mask illegal moves: set their logit to -infinity
        logits = logits.masked_fill(~legal_mask, float("-inf"))

        return F.softmax(logits, dim=-1)
