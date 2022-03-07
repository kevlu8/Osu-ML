import torch
import torch.nn as nn
import torch.optim as optim

import torchvision
import PIL.ImageGrab as ImageGrab

import pynput
import keyboard

import defs
import osu

# Code a reinforcement neural network to learn to play osu!
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)

if not osu.is_running():
    exit("osu! is not running. Please start osu! (on a offline game) and try again.")

print("osu! is running. Make sure you are running the game offline (not on osu! servers: check the Github readme to learn how). We are not responsible if you get banned from official osu! servers.")
agree = input("Are you ready to start? (y/N): ")

if agree != "y":
    exit("Exiting.")