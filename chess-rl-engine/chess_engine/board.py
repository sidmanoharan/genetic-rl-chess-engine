"""
board.py
--------
Board state representation and encoding for the neural network.

The key challenge in chess RL is converting the board into a format
the neural network can understand. We use a multi-plane tensor approach
similar to AlphaZero:

    Shape: (17, 8, 8)  — 17 feature planes, each an 8×8 grid

The 17 planes are:
    Planes  0– 5: White pieces  (P, N, B, R, Q, K) — 1 where piece exists
    Planes  6–11: Black pieces  (P, N, B, R, Q, K) — 1 where piece exists
    Plane  12: Whose turn       (all 1s if white, all 0s if black)
    Plane  13: White kingside castling right
    Plane  14: White queenside castling right
    Plane  15: Black kingside castling right
    Plane  16: Black queenside castling right

This gives the neural network all the information it needs to evaluate
a position and choose a move.
"""

import chess
import numpy as np
from typing import List, Tuple


# Mapping from chess piece type to plane index (for both colours)
# White pieces occupy planes 0–5, black pieces occupy planes 6–11
PIECE_TO_PLANE = {
    (chess.PAWN,   chess.WHITE): 0,
    (chess.KNIGHT, chess.WHITE): 1,
    (chess.BISHOP, chess.WHITE): 2,
    (chess.ROOK,   chess.WHITE): 3,
    (chess.QUEEN,  chess.WHITE): 4,
    (chess.KING,   chess.WHITE): 5,
    (chess.PAWN,   chess.BLACK): 6,
    (chess.KNIGHT, chess.BLACK): 7,
    (chess.BISHOP, chess.BLACK): 8,
    (chess.ROOK,   chess.BLACK): 9,
    (chess.QUEEN,  chess.BLACK): 10,
    (chess.KING,   chess.BLACK): 11,
}


def board_to_tensor(board: chess.Board) -> np.ndarray:
    """
    Convert a chess.Board into a (17, 8, 8) NumPy tensor.

    This is the function called before every neural network forward pass.
    The agent "sees" the board through this tensor.

    Args:
        board: A python-chess Board object representing the current position.

    Returns:
        NumPy array of shape (17, 8, 8) with dtype float32.
    """
    tensor = np.zeros((17, 8, 8), dtype=np.float32)

    # ── Piece planes (0–11) ───────────────────────────────────────────────
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is not None:
            plane = PIECE_TO_PLANE[(piece.piece_type, piece.color)]
            # Convert square index (0–63) to (row, col)
            row = square // 8
            col = square % 8
            tensor[plane, row, col] = 1.0

    # ── Turn plane (12) ───────────────────────────────────────────────────
    # Fill entire plane with 1 if it's white's turn, 0 if black's turn.
    if board.turn == chess.WHITE:
        tensor[12, :, :] = 1.0

    # ── Castling rights planes (13–16) ────────────────────────────────────
    if board.has_kingside_castling_rights(chess.WHITE):
        tensor[13, :, :] = 1.0
    if board.has_queenside_castling_rights(chess.WHITE):
        tensor[14, :, :] = 1.0
    if board.has_kingside_castling_rights(chess.BLACK):
        tensor[15, :, :] = 1.0
    if board.has_queenside_castling_rights(chess.BLACK):
        tensor[16, :, :] = 1.0

    return tensor


# ---------------------------------------------------------------------------
# Move encoding: chess move <-> integer index
# ---------------------------------------------------------------------------
# We need to map every legal chess move to a unique integer index (0–4671)
# so the neural network policy head can output probabilities over all moves.
#
# We use UCI move strings ("e2e4", "e7e8q" for promotions) as the canonical
# representation, then map them to integers via a pre-built lookup table.

def build_move_index() -> Tuple[dict, dict]:
    """
    Build a bidirectional mapping between UCI move strings and integer indices.

    Chess has at most 4672 possible moves from any position (this is the
    standard number used in AlphaZero). We enumerate all possible from-to
    square combinations plus promotion pieces.

    Returns:
        move_to_idx: Dict mapping UCI string -> integer index
        idx_to_move: Dict mapping integer index -> UCI string
    """
    moves = []

    # All from-square to to-square combinations
    for from_sq in chess.SQUARES:
        for to_sq in chess.SQUARES:
            if from_sq == to_sq:
                continue
            moves.append(chess.Move(from_sq, to_sq).uci())

    # Pawn promotions (to queen, rook, bishop, knight)
    for from_sq in chess.SQUARES:
        for to_sq in chess.SQUARES:
            for promo in [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]:
                move = chess.Move(from_sq, to_sq, promotion=promo)
                uci = move.uci()
                if uci not in moves:
                    moves.append(uci)

    move_to_idx = {uci: i for i, uci in enumerate(moves)}
    idx_to_move = {i: uci for i, uci in enumerate(moves)}

    return move_to_idx, idx_to_move


# Build the lookup tables once at module load time
MOVE_TO_IDX, IDX_TO_MOVE = build_move_index()

# Total number of possible moves (used as the policy head output size)
NUM_ACTIONS = len(MOVE_TO_IDX)


def get_legal_move_mask(board: chess.Board) -> np.ndarray:
    """
    Return a boolean mask of which actions are legal in the current position.

    Shape: (NUM_ACTIONS,) — True where the move is legal.

    The agent uses this mask to zero out illegal moves before sampling.
    This is critical: without masking, the agent would sometimes try to
    make illegal moves, which is undefined behaviour.

    Args:
        board: Current board position.

    Returns:
        Boolean NumPy array of shape (NUM_ACTIONS,).
    """
    mask = np.zeros(NUM_ACTIONS, dtype=bool)

    for move in board.legal_moves:
        uci = move.uci()
        if uci in MOVE_TO_IDX:
            mask[MOVE_TO_IDX[uci]] = True

    return mask


def move_to_index(move: chess.Move) -> int:
    """Convert a chess.Move object to its integer index."""
    return MOVE_TO_IDX.get(move.uci(), 0)


def index_to_move(idx: int) -> chess.Move:
    """Convert an integer index back to a chess.Move object."""
    uci = IDX_TO_MOVE.get(idx, "a1a1")  # fallback to null move
    return chess.Move.from_uci(uci)
