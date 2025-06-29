
import torch
import numpy as np

class AIPlayer:
    def __init__(self, letter, model):
        self.letter = letter
        self.model = model

    def get_move(self, game):
        board_state = torch.FloatTensor(self._get_board_state(game))
        with torch.no_grad():
            output = self.model(board_state)
        
        available_moves = game.available_moves()
        # Prioritize winning moves
        for move in available_moves:
            next_state = game.board[:]
            next_state[move] = self.letter
            if self._check_winner(next_state, self.letter):
                return move

        # Block opponent's winning moves
        opponent_letter = 'O' if self.letter == 'X' else 'X'
        for move in available_moves:
            next_state = game.board[:]
            next_state[move] = opponent_letter
            if self._check_winner(next_state, opponent_letter):
                return move

        # Choose the best move from the model's output
        while True:
            pred = output.argmax().item()
            if pred in available_moves:
                return pred
            else:
                output[pred] = -np.inf

    def _get_board_state(self, game):
        board_state = []
        for spot in game.board:
            if spot == self.letter:
                board_state.append(1)
            elif spot == ' ':
                board_state.append(0)
            else:
                board_state.append(-1)
        return board_state

    def _check_winner(self, board, letter):
        # Check rows
        for i in range(0, 25, 5):
            if all(board[j] == letter for j in range(i, i + 5)):
                return True
        # Check columns
        for i in range(5):
            if all(board[j] == letter for j in range(i, 25, 5)):
                return True
        # Check diagonals
        if all(board[i] == letter for i in [0, 6, 12, 18, 24]):
            return True
        if all(board[i] == letter for i in [4, 8, 12, 16, 20]):
            return True
        return False
