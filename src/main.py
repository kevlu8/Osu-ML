from time import time
import pynput
import keyboard
import argparse

argparser = argparse.ArgumentParser()
argparser.add_argument('--epochs', type=int, default=100)
argparser.add_argument('--cuda', action='store_true')
argparser.add_argument('--saveevery', type=int, default=10)
argparser.add_argument('--override', action='store_true')
argparser.parse_args()
args = argparser.parse_args()

from defs import *
import osu

if not args.override:
	if not osu.is_running():
		exit("osu! is not running. Please start osu! (on a offline game) and try again.")

	print("osu! is running. Make sure you are running the game offline (not on osu! servers: check the Github readme to learn how). We are not responsible if you get banned from official osu! servers.")
	agree = input("Are you ready to start? (y/N): ")

	if agree != "y":
		exit("Exiting.")
	
import torch
import torch.nn as nn
import torch.optim as optim	
from torch.utils.data import Dataset
import os
import cv2
from PIL import Image, ImageGrab
import numpy as np
import dnn
import timeit

if args.cuda:
	device = "cuda" if torch.cuda.is_available() else exit("No CUDA GPU was found, but --cuda was specified.")

	import GPUtil
	GPUs = GPUtil.getGPUs()

	print("You are running on device:", GPUs[0].name)
	print("Current statistics:")
	GPUtil.showUtilization()
	print(GPUs[0].temperature, "C")
else:
	device = "cpu"

times = []

# set cv2 to use the GPU
cv2.cuda.setDevice(0)
i=0
while True:
	i+=1
	# source = cv2.cuda_GpuMat()
	image = ImageGrab.grab().resize((480, 270))
	# image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
	# convert rgb to gray
	image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
	# source.upload(cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR))
	# clahe = cv2.cuda.createCLAHE(clipLimit=5.0, tileGridSize=(8,8))
	# dest = clahe.apply(source, cv2.cuda_Stream.Null())
	img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	output = img.copy()
	# Find circles
	circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1.3, 100)
	# result = dest.download()

	if circles is not None:
		# Get the (x, y, r) as integers
		circles = np.round(circles[0, :]).astype("int")
		print(circles)

		pynput.mouse.Controller().position = (circles[0][0], circles[0][1])
		pynput.mouse.Controller().click(pynput.mouse.Button.left, 1)

		for (x, y, r) in circles:
			cv2.circle(output, (x, y), r, (0, 0, 255), 2)
	# cv2.imshow("output", output)
	# cv2.waitKey(0)
	Image.fromarray(output).save(f"output{i}.png")
