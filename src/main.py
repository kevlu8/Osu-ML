import tensorflow as tf
from tensorflow.python.keras import models

import pynput
import keyboard

import defs
import osu

# code a reinforcement learning agent
# the agent will be trained to play osu!
model = models.Sequential()


if not osu.is_running():
    exit("osu! is not running. Please start osu! (on a offline game) and try again.")

print("osu! is running. Make sure you are running the game offline (not on osu! servers: check the Github readme to learn how). We are not responsible if you get banned from official osu! servers.")
agree = input("Are you ready to start? (y/N): ")

if agree != "y":
    exit("Exiting.")
    
