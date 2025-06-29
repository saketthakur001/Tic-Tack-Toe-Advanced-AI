import random
import torch
from player import AIPlayer
from model import TicTacToeNet

class TicTacToe:
    def __init__(self):
        self.board = [' ' for _ in range(25)]
        self.current_winner = None

    def print_board(self):
        for row in [self.board[i*5:(i+1)*5] for i in range(5)]:
            print('| ' + ' | '.join(row) + ' |')

    @staticmethod
    def print_board_nums():
        number_board = [[str(i) for i in range(j*5+1, (j+1)*5+1)] for j in range(5)]
        for row in number_board:
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
        # check row
        row_ind = square // 5
        row = self.board[row_ind*5 : (row_ind + 1) * 5]
        if all([s == letter for s in row]):
            return True
        # check column
        col_ind = square % 5
        column = [self.board[col_ind + i*5] for i in range(5)]
        if all([s == letter for s in column]):
            return True
        # check diagonals
        if square % 6 == 0:
            diagonal1 = [self.board[i] for i in [0, 6, 12, 18, 24]]
            if all([s == letter for s in diagonal1]):
                return True
        if square % 4 == 0 and square != 0 and square != 24:
            diagonal2 = [self.board[i] for i in [4, 8, 12, 16, 20]]
            if all([s == letter for s in diagonal2]):
                return True
        return False

def play(game, x_player, o_player, print_game=True):
    if print_game:
        game.print_board_nums()

    letter = 'X'
    while game.empty_squares():
        if letter == 'O':
            square = o_player.get_move(game)
        else:
            square = x_player.get_move(game)

        if game.make_move(square, letter):
            if print_game:
                print(letter + f' makes a move to square {square}')
                game.print_board()
                print('')

            if game.current_winner:
                if print_game:
                    print(letter + ' wins!')
                return letter
            letter = 'O' if letter == 'X' else 'X'

    if print_game:
        print('It\'s a tie!')

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
    x_player = AIPlayer('X', model)
    o_player = HumanPlayer('O')
    play(t, x_player, o_player, print_game=True)
