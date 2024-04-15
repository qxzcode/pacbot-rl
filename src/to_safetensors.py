"""
A simple script that converts a PyTorch Q-net checkpoint to a safetensors file.

Example usage:
    python3 to_safetensors.py checkpoints/q_net-latest.pt export_directory/output.safetensors
"""

import sys

import torch
from safetensors.torch import save_file


[_, in_file, out_file] = sys.argv

q_net = torch.load(in_file, map_location="cpu")
print(q_net)
print()

save_file(q_net.state_dict(), out_file)
print(f"Saved {out_file}")
