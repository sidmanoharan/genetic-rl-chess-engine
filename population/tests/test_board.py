"""
tests/test_board.py
-------------------
Unit tests for the board encoding and move index utilities.

Run with:
    pytest tests/test_board.py -v
"""

import pytest
import chess
import numpy as np
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chess_engine.board import (
    board_to_tensor,
    get_legal_move_mask,
    move_to_index,
    index_to_move,
    NUM_ACTIONS,
    MOVE_TO_IDX,
)


class TestBoardToTensor:
    """Tests for the board state encoding."""

    def test_output_shape(self):
        """Tensor must always be (17, 8, 8)."""
        board = chess.Board()
        tensor = board_to_tensor(board)
        assert tensor.shape == (17, 8, 8), f"Expected (17, 8, 8), got {tensor.shape}"

    def test_output_dtype(self):
        """Tensor values must be float32."""
        tensor = board_to_tensor(chess.Board())
        assert tensor.dtype == np.float32

    def test_starting_position_white_pieces(self):
        """White pawns should be in plane 0, row 1 (rank 2)."""
        board = chess.Board()
        tensor = board_to_tensor(board)

        # White pawns (plane 0) should be on rank 2 (row index 1)
        assert tensor[0, 1, :].sum() == 8, "Expected 8 white pawns on rank 2"

    def test_starting_position_black_pieces(self):
        """Black pawns should be in plane 6, row 6 (rank 7)."""
        board = chess.Board()
        tensor = board_to_tensor(board)
        assert tensor[6, 6, :].sum() == 8, "Expected 8 black pawns on rank 7"

    def test_white_turn_plane(self):
        """Plane 12 should be all 1s when it's white's turn."""
        board = chess.Board()  # White moves first
        tensor = board_to_tensor(board)
        assert tensor[12, :, :].all(), "Plane 12 should be all 1s for white's turn"

    def test_black_turn_plane(self):
        """Plane 12 should be all 0s when it's black's turn."""
        board = chess.Board()
        board.push_san("e4")  # White plays, now black's turn
        tensor = board_to_tensor(board)
        assert not tensor[12, :, :].any(), "Plane 12 should be all 0s for black's turn"

    def test_empty_board_mostly_zeros(self):
        """An empty board should have zeros in piece planes."""
        board = chess.Board(fen=None)  # Empty board
        tensor = board_to_tensor(board)
        # Piece planes 0-11 should be all zeros
        assert tensor[:12].sum() == 0, "Piece planes should be zero for empty board"

    def test_tensor_binary_values(self):
        """All tensor values should be 0.0 or 1.0."""
        board = chess.Board()
        tensor = board_to_tensor(board)
        unique_vals = np.unique(tensor)
        assert set(unique_vals).issubset({0.0, 1.0}), f"Non-binary values found: {unique_vals}"

    def test_castling_rights_initial(self):
        """Initial position should have all 4 castling rights."""
        board = chess.Board()
        tensor = board_to_tensor(board)
        assert tensor[13, 0, 0] == 1.0, "White kingside castling should be available"
        assert tensor[14, 0, 0] == 1.0, "White queenside castling should be available"
        assert tensor[15, 0, 0] == 1.0, "Black kingside castling should be available"
        assert tensor[16, 0, 0] == 1.0, "Black queenside castling should be available"


class TestMoveIndex:
    """Tests for the move encoding/decoding utilities."""

    def test_num_actions_positive(self):
        """NUM_ACTIONS should be a large positive number."""
        assert NUM_ACTIONS > 4000, f"Expected > 4000 actions, got {NUM_ACTIONS}"

    def test_move_to_idx_and_back(self):
        """move_to_index followed by index_to_move should be the identity."""
        board = chess.Board()
        for move in list(board.legal_moves)[:10]:
            idx = move_to_index(move)
            recovered = index_to_move(idx)
            assert recovered.uci() == move.uci(), (
                f"Round-trip failed: {move.uci()} -> {idx} -> {recovered.uci()}"
            )

    def test_all_starting_moves_in_index(self):
        """All legal moves from the starting position should be in the move index."""
        board = chess.Board()
        for move in board.legal_moves:
            assert move.uci() in MOVE_TO_IDX, f"Move {move.uci()} not in move index"


class TestLegalMoveMask:
    """Tests for the legal move masking."""

    def test_mask_shape(self):
        """Mask must have shape (NUM_ACTIONS,)."""
        board = chess.Board()
        mask = get_legal_move_mask(board)
        assert mask.shape == (NUM_ACTIONS,), f"Expected ({NUM_ACTIONS},), got {mask.shape}"

    def test_mask_dtype(self):
        """Mask must be boolean."""
        mask = get_legal_move_mask(chess.Board())
        assert mask.dtype == bool

    def test_starting_position_has_20_legal_moves(self):
        """Starting position has exactly 20 legal moves."""
        board = chess.Board()
        mask = get_legal_move_mask(board)
        assert mask.sum() == 20, f"Expected 20 legal moves, got {mask.sum()}"

    def test_mask_legal_moves_are_true(self):
        """Every legal move should be True in the mask."""
        board = chess.Board()
        mask = get_legal_move_mask(board)
        for move in board.legal_moves:
            idx = move_to_index(move)
            assert mask[idx], f"Legal move {move.uci()} is False in mask"
