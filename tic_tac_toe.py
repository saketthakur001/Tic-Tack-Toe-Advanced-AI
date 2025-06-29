import random
import torch
from player import AIPlayer
from model import TicTacToeNet

class TicTacToe:
    def __init__(self):
        self.board = [' ' for _ in range(25)]
        self.current_winner = None

    def print_board(self):
        for r in range(5):
            row = []
            for c in range(5):
                square_index = r * 5 + c
                if self.board[square_index] == 'X':
                    row.append('❌')
                elif self.board[square_index] == 'O':
                    row.append('⭕')
                else:
                    row.append(str(square_index + 1).rjust(2)) # Right-align numbers for consistent spacing
            print('| ' + ' | '.join(row) + ' |')

    def available_moves(self):
        return [i for i, spot in enumerate(self.board) if spot == ' ']

    def empty_squares(self):
        return ' ' in self.board

    def num_empty_squares(self):
        return self.board.count(' ')

    def make_move(self, square, letter):
        if self.board[square] == ' ':
            self.board[square] = letter
            if self.winner(square, letter):
                self.current_winner = letter
            return True
        return False

    def winner(self, square, letter):
        # Check rows
        row_ind = square // 5
        row_start = row_ind * 5
        for i in range(2): # Check for 4 in a row in a 5x5 grid
            if all(self.board[row_start + j] == letter for j in range(i, i + 4)):
                return True

        # Check columns
        col_ind = square % 5
        for i in range(2): # Check for 4 in a column in a 5x5 grid
            if all(self.board[col_ind + j*5] == letter for j in range(i, i + 4)):
                return True

        # Check diagonals (top-left to bottom-right)
        # Check if the square is part of any possible 4-in-a-row diagonal
        for r_offset in range(-3, 1): # Iterate through possible starting rows for a 4-in-a-row diagonal
            for c_offset in range(-3, 1): # Iterate through possible starting columns
                start_row = row_ind + r_offset
                start_col = col_ind + c_offset
                if 0 <= start_row <= 1 and 0 <= start_col <= 1: # Ensure starting point is within 2x2 top-left subgrid
                    if all(self.board[(start_row + k) * 5 + (start_col + k)] == letter for k in range(4)):
                        return True

        # Check diagonals (top-right to bottom-left)
        for r_offset in range(-3, 1):
            for c_offset in range(0, 4): # Iterate through possible starting columns
                start_row = row_ind + r_offset
                start_col = col_ind + c_offset
                if 0 <= start_row <= 1 and 3 <= start_col <= 4: # Ensure starting point is within 2x2 top-right subgrid
                    if all(self.board[(start_row + k) * 5 + (start_col - k)] == letter for k in range(4)):
                        return True
        return False

def play(game, x_player, o_player, print_game=True):
    if print_game:
        game.print_board() # Initial board display

    current_player_letter = 'X' # Human starts as X

    while game.empty_squares():
        # Human's turn
        if current_player_letter == x_player.letter:
            square = x_player.get_move(game)
            if game.make_move(square, x_player.letter):
                if print_game:
                    print(f"{x_player.letter} makes a move to square {square}")
                if game.current_winner:
                    if print_game:
                        game.print_board() # Print final board
                        print(f"{x_player.letter} wins!")
                    return x_player.letter
                current_player_letter = o_player.letter # Switch to AI
            # If make_move returns False, HumanPlayer.get_move will re-prompt, so no else needed here.

        # AI's turn (only if game is not over and it's AI's turn)
        if game.empty_squares() and not game.current_winner and current_player_letter == o_player.letter:
            square = o_player.get_move(game)
            if game.make_move(square, o_player.letter):
                if print_game:
                    print(f"{o_player.letter} makes a move to square {square}")
                if game.current_winner:
                    if print_game:
                        game.print_board() # Print final board
                        print(f"{o_player.letter} wins!")
                    return o_player.letter
                current_player_letter = x_player.letter # Switch back to Human

        # Print board after both players (or one if game ends) have moved
        if print_game and not game.current_winner and game.empty_squares():
            game.print_board()

    if print_game and not game.current_winner: # If loop ends and no winner, it's a tie
        game.print_board() # Print final board for tie
        print('It\'s a tie!')
    return None # Return None for tie or if game ends without a winner being explicitly returned


class HumanPlayer:
    def __init__(self, letter):
        self.letter = letter

    def get_move(self, game):
        valid_square = False
        val = None
        while not valid_square:
            square = input(self.letter + '\'s turn. Input move (1-25): ')
            try:
                val = int(square) - 1
                if val not in game.available_moves():
                    raise ValueError
                valid_square = True
            except ValueError:
                print('Invalid square. Try again.')
        return val

class RandomComputerPlayer:
    def __init__(self, letter):
        self.letter = letter

    def get_move(self, game):
        square = random.choice(game.available_moves())
        return square

if __name__ == '__main__':
    t = TicTacToe()
    model = TicTacToeNet()
    model.load_state_dict(torch.load('tictactoe_net.pth'))
    model.eval()
    x_player = HumanPlayer('X')
    o_player = AIPlayer('O', model)
    play(t, x_player, o_player, print_game=True)
