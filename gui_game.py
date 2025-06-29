import tkinter as tk
from tkinter import messagebox
import torch
from tic_tac_toe import TicTacToe, HumanPlayer, RandomComputerPlayer, play
from player import AIPlayer
from model import TicTacToeNet # Assuming model.py is in the same directory

class TicTacToeGUI:
    def __init__(self, master):
        self.master = master
        master.title("AI Tic Tac Toe (5x5)")

        self.game = TicTacToe()
        self.load_ai_model()

        self.player_x = AIPlayer('X', self.ai_model)
        self.player_o = HumanPlayer('O') # Human player for 'O'

        self.current_player = self.player_x.letter # AI starts as 'X'

        self.buttons = {}
        self.create_board_buttons()
        self.create_status_label()
        self.create_reset_button()

        self.update_status()
        self.master.after(100, self.ai_make_move_if_needed) # Start AI's turn if AI is X

    def load_ai_model(self):
        try:
            self.ai_model = TicTacToeNet()
            self.ai_model.load_state_dict(torch.load('tictactoe_net.pth'))
            self.ai_model.eval()
        except FileNotFoundError:
            messagebox.showerror("Model Error", "tictactoe_net.pth not found. Please ensure the AI model is trained and available.")
            self.master.destroy()
            return

    def create_board_buttons(self):
        for i in range(5):
            for j in range(5):
                square_index = i * 5 + j
                button = tk.Button(self.master, text=" ", font=('Arial', 24), width=4, height=2,
                                   command=lambda idx=square_index: self.make_move(idx))
                button.grid(row=i, column=j, padx=2, pady=2)
                self.buttons[square_index] = button

    def create_status_label(self):
        self.status_label = tk.Label(self.master, text="", font=('Arial', 16))
        self.status_label.grid(row=5, column=0, columnspan=5, pady=10)

    def create_reset_button(self):
        self.reset_button = tk.Button(self.master, text="Reset Game", font=('Arial', 14), command=self.reset_game)
        self.reset_button.grid(row=6, column=0, columnspan=5, pady=10)

    def update_board_display(self):
        for i, letter in enumerate(self.game.board):
            self.buttons[i].config(text=letter)

    def update_status(self):
        if self.game.current_winner:
            self.status_label.config(text=f"Player {self.game.current_winner} wins!")
        elif not self.game.empty_squares():
            self.status_label.config(text="It's a tie!")
        else:
            self.status_label.config(text=f"Player {self.current_player}'s turn")

    def make_move(self, square_index):
        if self.game.current_winner or not self.game.empty_squares():
            return # Game is over

        if self.current_player == self.player_o.letter: # Only human can make a move via click
            if self.game.make_move(square_index, self.player_o.letter):
                self.update_board_display()
                if self.game.current_winner:
                    self.update_status()
                    messagebox.showinfo("Game Over", f"Player {self.game.current_winner} wins!")
                elif not self.game.empty_squares():
                    self.update_status()
                    messagebox.showinfo("Game Over", "It's a tie!")
                else:
                    self.current_player = self.player_x.letter # Switch to AI's turn
                    self.update_status()
                    self.master.after(500, self.ai_make_move_if_needed) # Delay AI move for better UX
            else:
                messagebox.showwarning("Invalid Move", "This square is already taken!")
        else:
            messagebox.showinfo("AI's Turn", "It's the AI's turn. Please wait.")


    def ai_make_move_if_needed(self):
        if self.game.current_winner or not self.game.empty_squares():
            return # Game is over

        if self.current_player == self.player_x.letter: # If it's AI's turn
            ai_move = self.player_x.get_move(self.game)
            if self.game.make_move(ai_move, self.player_x.letter):
                self.update_board_display()
                if self.game.current_winner:
                    self.update_status()
                    messagebox.showinfo("Game Over", f"Player {self.game.current_winner} wins!")
                elif not self.game.empty_squares():
                    self.update_status()
                    messagebox.showinfo("Game Over", "It's a tie!")
                else:
                    self.current_player = self.player_o.letter # Switch to Human's turn
                    self.update_status()

    def reset_game(self):
        self.game = TicTacToe()
        self.current_player = self.player_x.letter # AI starts as 'X'
        self.update_board_display()
        self.update_status()
        self.master.after(100, self.ai_make_move_if_needed) # Start AI's turn if AI is X

if __name__ == "__main__":
    root = tk.Tk()
    gui = TicTacToeGUI(root)
    root.mainloop()
