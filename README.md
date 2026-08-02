# AI Tic Tac Toe (5x5)

This project implements an AI player for a 5x5 Tic Tac Toe game using a trained neural network. The AI can play against a human player or a random computer player.

## Features

*   **5x5 Game Board:** A larger Tic Tac Toe board for more complex gameplay.
*   **AI Player:** An intelligent AI opponent powered by a pre-trained PyTorch neural network.
*   **Human Player:** Allows a human to play against the AI or another human.
*   **Random Computer Player:** A basic computer opponent that makes random valid moves.
*   **ONNX Export:** The trained PyTorch model can be exported to ONNX format for deployment or inference with other runtimes.

## Setup and Installation

To run this project, you need Python 3.x, PyTorch, and ONNX.

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/saketthakur001/Human-vs-AI-Game.git
    cd Human-vs-AI-Game
    ```

2.  **Install dependencies:**
    ```bash
    pip install torch numpy onnx
    ```

## Project Structure

*   `tic_tac_toe.py`: Contains the core game logic, including the `TicTacToe` class, `HumanPlayer`, `RandomComputerPlayer`, and the main `play` function.
*   `player.py`: Defines the `AIPlayer` class, which uses the trained neural network to determine moves.
*   `model.py`: Defines the `TicTacToeNet` neural network architecture used by the AI.
*   `win_checker.py`: Shared 4-in-a-row win-detection logic used by both the game and the AI player.
*   `train.py`: Trains `TicTacToeNet` via a simple DQN-style loop and saves the weights to `tictactoe_net.pth`.
*   `convert_to_onnx.py`: A utility script to convert the PyTorch model (`.pth`) to ONNX format (`.onnx`).
*   `gui_game.py`: A Tkinter-based graphical version of the game.
*   `README.md`: This file, providing an overview of the project.

Note: `tictactoe_net.pth` and `tictactoe_net.onnx` are **not** checked into the repo (see `.gitignore`) because trained model weights are binary artifacts that don't belong in git history. You need to generate `tictactoe_net.pth` yourself before playing (see below).

## How to Run the Game

1.  **Train the model first** (only needs to be done once; this creates `tictactoe_net.pth`):
    ```bash
    python train.py
    ```
    This runs 50,000 self-play episodes against a random opponent and can take a while on CPU.

2.  **Play against the AI** in the terminal:
    ```bash
    python tic_tac_toe.py
    ```
    Or, for a graphical version:
    ```bash
    python gui_game.py
    ```

The game will prompt you for your moves. The human plays as 'X', the AI as 'O', by default in the `if __name__ == '__main__':` block of each script. You can modify that block to change players or roles.

## How the AI Works

The AI player (`AIPlayer` in `player.py`) uses a pre-trained neural network (`TicTacToeNet` from `model.py`) to evaluate the game board and predict the best next move. The network takes a flattened representation of the 5x5 board as input and outputs scores for each possible move. The AI prioritizes winning moves and blocking opponent's winning moves before consulting the neural network's output.
