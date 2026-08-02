"""Shared win-condition logic for the 5x5 Tic Tac Toe board.

The board is a flat list of 25 cells. A player wins by getting 4 marks
in a row, column, or diagonal. This logic used to be duplicated (and
had drifted out of sync) between `TicTacToe.winner` in tic_tac_toe.py
and `AIPlayer._check_winner` in player.py -- the latter incorrectly
checked for 5-in-a-row instead of 4-in-a-row, which meant the AI's
"take the winning move" / "block the opponent" heuristics never
actually triggered under the game's real win condition. Both call
sites now share this single implementation.
"""


def check_winner(board, square, letter):
    """Return True if placing `letter` at `square` completes 4 in a row.

    `board` is the flat 25-cell board *after* the move has been made.
    """
    row_ind = square // 5
    row_start = row_ind * 5
    for i in range(2):  # 4-in-a-row within a 5-wide row
        if all(board[row_start + j] == letter for j in range(i, i + 4)):
            return True

    col_ind = square % 5
    for i in range(2):  # 4-in-a-column
        if all(board[col_ind + j * 5] == letter for j in range(i, i + 4)):
            return True

    # Diagonals (top-left to bottom-right)
    for r_offset in range(-3, 1):
        for c_offset in range(-3, 1):
            start_row = row_ind + r_offset
            start_col = col_ind + c_offset
            if 0 <= start_row <= 1 and 0 <= start_col <= 1:
                if all(board[(start_row + k) * 5 + (start_col + k)] == letter for k in range(4)):
                    return True

    # Diagonals (top-right to bottom-left)
    for r_offset in range(-3, 1):
        for c_offset in range(0, 4):
            start_row = row_ind + r_offset
            start_col = col_ind + c_offset
            if 0 <= start_row <= 1 and 3 <= start_col <= 4:
                if all(board[(start_row + k) * 5 + (start_col - k)] == letter for k in range(4)):
                    return True

    return False
