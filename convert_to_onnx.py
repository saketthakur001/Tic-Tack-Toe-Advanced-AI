
import torch
import torch.nn as nn
from model import TicTacToeNet # Assuming model.py is in the same directory

def convert_to_onnx(model_path, output_path):
    # Load the trained model
    model = TicTacToeNet()
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Create a dummy input for ONNX export
    # The input to the TicTacToeNet is a flattened 5x5 board, so 25 features
    dummy_input = torch.randn(1, 25)

    # Export the model
    torch.onnx.export(model, dummy_input, output_path, verbose=True, input_names=['input'], output_names=['output'])
    print(f"Model successfully converted to ONNX and saved at {output_path}")

if __name__ == "__main__":
    model_path = "tictactoe_net.pth"
    output_path = "tictactoe_net.onnx"
    convert_to_onnx(model_path, output_path)
