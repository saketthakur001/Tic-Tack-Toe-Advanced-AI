import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque

from model import TicTacToeNet
from tic_tac_toe import TicTacToe

# Hyperparameters
LEARNING_RATE = 0.001
GAMMA = 0.99 # Discount factor for future rewards
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.995
BATCH_SIZE = 64
BUFFER_SIZE = 10000
NUM_EPISODES = 10000

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

def get_board_state_for_nn(game, player_letter):
    board_state = []
    for spot in game.board:
        if spot == player_letter:
            board_state.append(1)
        elif spot == ' ':
            board_state.append(0)
        else:
            board_state.append(-1)
    return torch.FloatTensor(board_state).unsqueeze(0) # Add batch dimension

def train_model():
    model = TicTacToeNet()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()
    replay_buffer = ReplayBuffer(BUFFER_SIZE)

    epsilon = EPSILON_START

    for episode in range(NUM_EPISODES):
        game = TicTacToe()
        state = get_board_state_for_nn(game, 'X') # AI plays as 'X'
        done = False
        episode_reward = 0

        while not done:
            # Epsilon-greedy action selection
            if random.random() < epsilon:
                action = random.choice(game.available_moves())
            else:
                with torch.no_grad():
                    q_values = model(state)
                    # Mask out invalid moves
                    for move_idx in range(25):
                        if move_idx not in game.available_moves():
                            q_values[0, move_idx] = -float('inf')
                    action = q_values.argmax().item()

            # Take action and observe new state and reward
            original_board = list(game.board) # Store original board for reward calculation
            game.make_move(action, 'X')
            next_state = get_board_state_for_nn(game, 'X')

            reward = 0
            if game.current_winner == 'X':
                reward = 1
                done = True
            elif not game.empty_squares(): # Tie
                reward = 0.1
                done = True
            else:
                # Opponent's turn (random move for simplicity in training)
                opponent_moves = game.available_moves()
                if opponent_moves:
                    opponent_move = random.choice(opponent_moves)
                    game.make_move(opponent_move, 'O')
                    if game.current_winner == 'O':
                        reward = -1 # Penalize for opponent winning
                        done = True
                    elif not game.empty_squares(): # Tie after opponent's move
                        reward = 0.1
                        done = True

            replay_buffer.push(state, action, reward, next_state, done)
            state = next_state

            episode_reward += reward

            # Train the model if buffer has enough samples
            if len(replay_buffer) > BATCH_SIZE:
                batch = replay_buffer.sample(BATCH_SIZE)
                states, actions, rewards, next_states, dones = zip(*batch)

                states = torch.cat(states)
                next_states = torch.cat(next_states)
                actions = torch.LongTensor(actions)
                rewards = torch.FloatTensor(rewards)
                dones = torch.FloatTensor(dones)

                current_q_values = model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
                next_q_values = model(next_states).max(1)[0].detach()
                target_q_values = rewards + (1 - dones) * GAMMA * next_q_values

                loss = criterion(current_q_values, target_q_values)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        
        epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)

        if episode % 100 == 0:
            print(f"Episode {episode}/{NUM_EPISODES}, Epsilon: {epsilon:.2f}, Reward: {episode_reward:.2f}")

    torch.save(model.state_dict(), 'tictactoe_net.pth')
    print("Training complete. Model saved to tictactoe_net.pth")

if __name__ == "__main__":
    train_model()
