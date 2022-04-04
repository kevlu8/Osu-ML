import random
import sys
import pynput
import keyboard
import argparse
from defs import *
from subprocess import getoutput

argparser = argparse.ArgumentParser()
argparser.add_argument('--epochs', type=int, default=100)
argparser.add_argument('--cuda', action='store_true')
argparser.add_argument('--saveevery', type=int, default=10)
argparser.add_argument('--override', action='store_true')
argparser.add_argument('--train', type=str, default=None)
argparser.add_argument('--lr', type=float, default=0.001)
argparser.parse_args()
args = argparser.parse_args()

import osu

if not args.override:
	if not osu.is_running():
		sys.exit("osu! is not running. Please start osu! (on a offline game) and try again.")

	print("osu! is running. Make sure you are running the game offline (not on osu! servers: check the Github readme to learn how). We are not responsible if you get banned from official osu! servers.")
	agree = input("Are you ready to start? (y/N): ")

	if agree != "y":
		sys.exit("Exiting.")
	
import torch
import torch.nn as nn
import torch.optim as optim	
import torch.functional as F
import os
import numpy as np
import PIL
from mss import mss

if args.cuda:
	device = "cuda" if torch.cuda.is_available() else sys.exit("No CUDA GPU was found, but --cuda was specified.")

	import GPUtil
	GPUs = GPUtil.getGPUs()

	print("You are running on device:", GPUs[0].name)
	print("Current statistics:")
	GPUtil.showUtilization()
	print(GPUs[0].temperature, "C")
	torch.set_num_interop_threads(16)
	torch.set_num_threads(16)
else:
	device = "cpu"

class convNetwork(nn.Module):
	def __init__(self):
		super(convNetwork, self).__init__()
		# input is an image of size 256x144 grayscale
		self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
		self.relu1 = nn.ReLU()
		self.pool1 = nn.MaxPool2d(2, 2)
		self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
		self.relu2 = nn.ReLU()
		self.pool2 = nn.MaxPool2d(2, 2)
		self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
		self.relu3 = nn.ReLU()
		self.pool3 = nn.MaxPool2d(2, 2)
		self.conv4 = nn.Conv2d(64, 128, 3, padding=1)
		self.relu4 = nn.ReLU()
		self.pool4 = nn.MaxPool2d(2, 2)
		self.flatten = nn.Flatten(0)
		self.dense = nn.Linear(18432, 3)
		# self.tanh = nn.Tanh()

	def forward(self, x):
		x = self.conv1(x)
		# print(x.size())
		x = self.relu1(x)
		# print(x.size())
		x = self.pool1(x)
		# print(x.size())
		x = self.conv2(x)
		# print(x.size())
		x = self.relu2(x)
		# print(x.size())
		x = self.pool2(x)
		# print(x.size())
		x = self.conv3(x)
		# print(x.size())
		x = self.relu3(x)
		# print(x.size())
		x = self.pool3(x)
		# print(x.size())
		x = self.conv4(x)
		# print(x.size())
		x = self.relu4(x)
		# print(x.size())
		x = self.pool4(x)
		# print(x.size())
		x = self.flatten(x)
		# print(x.size())
		x = self.dense(x)
		# print(x.size())
		# x = self.tanh(x)
		# print(x.size())
		# return torch.Tensor((torch.sigmoid(torch.Tensor((x[0],))), *x[1:])).to(device)
		return x
		
circlenet = convNetwork() # network to return the location of circles
# returns: tensor([[ZDown, XDown, coordX, coordY]])

# scorenet = convNetwork() # network to parse the score
# returns: tensor([[score]])

# run an image through the network
from torchvision import transforms

def run_image(batch=1024):
	img = []
	indexes = []
	for i in range(batch):
		randint = random.randint(0, len(os.listdir("data/imgs/")) - 1)
		t = (PIL.Image.open("data/imgs/" + os.listdir("data/imgs")[randint]))
		# t = t.resize((256, 144))
		t = transforms.functional.pil_to_tensor(t).to(device)
		t = t.type(torch.FloatTensor)
		t = t.to(device)
		img.append(t)
		indexes.append(randint)
	return (indexes, img)

# def run_image_scorenet(batch=64):
# 	img = []
# 	indexes = []
# 	for i in range(batch):
# 		randint = random.randint(0, len(os.listdir("data/imgs")) - 1)
# 		t = (PIL.Image.open("data/imgs/" + os.listdir("data/imgs")[randint]))
# 		t = t.resize((256, 144))
# 		t = transforms.functional.pil_to_tensor(t).to(device)
# 		t = t.type(torch.FloatTensor)
# 		t = t.to(device)
# 		img.append(t)
# 		indexes.append(randint)
# 	return (indexes, img)

optimizer = optim.Adam(circlenet.parameters(), lr=args.lr)
# optimizerscore = optim.Adam(scorenet.parameters(), lr=args.lr)

if os.path.exists("weights.pth"):
	c = torch.load("weights.pth", map_location=device)
	circlenet.load_state_dict(c["model"])
	optimizer.load_state_dict(c["optimizer"])
# if os.path.exists("weights_score.pth"):
# 	c = torch.load("weights_score.pth", map_location=device)
# 	scorenet.load_state_dict(c["model"])
# 	optimizerscore.load_state_dict(c["optimizer"])

circlenet.to(device)
circlenet.train()
# scorenet.to(device)
# scorenet.train()

def optimizer_to(optim, device):
	for param in optim.state.values():
		# Not sure there are any global tensors in the state dict
		if isinstance(param, torch.Tensor):
			param.data = param.data.to(device)
			if param._grad is not None:
				param._grad.data = param._grad.data.to(device)
		elif isinstance(param, dict):
			for subparam in param.values():
				if isinstance(subparam, torch.Tensor):
					subparam.data = subparam.data.to(device)
					if subparam._grad is not None:
						subparam._grad.data = subparam._grad.data.to(device)

optimizer_to(optimizer, device)
# optimizer_to(optimizerscore, device)

data = []
with open("keypresses.txt", "r") as f:
	for line in f:
		# remove first and last characters
		line = line[1:-2]

		line = line.split(", ")
		newline = []
		for num in line:
			num = float(num)
			newline.append(num)
		data.append(newline)

dataScores = []
with open("scores.txt", "r") as f:
	for line in f:
		# remove first and last characters
		line = line[1:-2]

		line = line.split(", ")
		newline = []
		for num in line:
			num = float(num)
			newline.append(num)
		dataScores.append(newline)

if args.train == "conv":
	for epoch in range(args.epochs):
		for i, img in zip(*run_image()):
			try:
				circlenet.zero_grad()
				img = torch.tensor(img).to(device)
				output = circlenet(img)
				torch.sigmoid(output[0])
				loss = nn.functional.mse_loss(output, torch.tensor(
					data[i]
				, device=device, dtype=torch.float))
				loss = loss.to(device)
				loss.backward()
				optimizer.step()
			except:
				continue

		# test
		if epoch % 50 == 0:
			circlenet.eval()
			with torch.no_grad():
				output = circlenet(img)
				print(output)
			circlenet.train()
			print(GPUs[0].temperature, "C")
			
		
		if epoch % args.saveevery == 0:
			circlenet.cpu()
			torch.save({"model": circlenet.state_dict(), "optimizer": optimizer.state_dict()}, "weights.pth")
			circlenet.to(device)

		print("Epoch: ", epoch + 1, " Loss: ", loss)

# elif args.train == "score":
# 	for epoch in range(args.epochs):
# 		print("Epoch: ", epoch)

# 		for i, img in zip(*run_image_scorenet()):
# 			scorenet.zero_grad()
# 			img = torch.tensor(img).to(device)
# 			output = scorenet(img)
# 			nn.functional.sigmoid(output[0])
# 			loss = nn.functional.mse_loss(output, torch.tensor([
# 				dataScores[i]
# 			], device=device, dtype=torch.float))
# 			loss = loss.to(device)
# 			loss.backward()
# 			optimizerscore.step()

# 		# test
# 		if epoch % 50 == 0:
# 			scorenet.eval()
# 			with torch.no_grad():
# 				output = scorenet(img)
# 				print(output)
# 			scorenet.train()
		
# 		if epoch % args.saveevery == 0:
# 			scorenet.cpu()
# 			torch.save({"model": scorenet.state_dict(), "optimizer": optimizerscore.state_dict()}, "weights_score.pth")
# 			scorenet.to(device)

# 		print("Loss: ", loss)
elif args.train == "test":
	# use the network
	circlenet.eval()
	# scorenet.eval()
	img = PIL.Image.open("data/imgs/img250.jpg")
	img = img.resize((256, 144))
	img = transforms.functional.pil_to_tensor(img).to(device)
	img = img.type(torch.FloatTensor)
	img = img.to(device)
	with torch.no_grad():
		out = circlenet(img)
		print(out)
		# output = scorenet(img)
		# print(output)
	# label the image of the circle and then show it
else:
	def capture_screenshot():
		# Capture entire screen
		with mss() as sct:
			monitor = sct.monitors[1]
			sct_img = sct.grab(monitor)
			# Convert to PIL/Pillow Image
			return PIL.Image.frombytes('RGB', sct_img.size, sct_img.bgra, 'raw', 'BGRX')
	import pyautogui, time
	pyautogui.FAILSAFE = False
	time.sleep(5)
	circlenet.eval()
	holding = False
	while True:
		if keyboard.is_pressed('q'):
			break
		# use the network
		# img = PIL.ImageGrab.grab().resize((256, 144))
		img = capture_screenshot().resize((256, 144))
		# grayscale
		img = img.convert('L')
		img = transforms.functional.pil_to_tensor(img).to(device)
		img = img.type(torch.FloatTensor)
		img = img.to(device)
		
		with torch.no_grad():
			out = circlenet(img)
			print(out)
		out.cpu()
		out = out.flatten().tolist()
		click = round(out[0])

		# move mouse to x: out[1], y: out[2]
		pyautogui.moveTo(out[1] * 1920, out[2] * 1080)
		if click == 1:
			# mousebutton down
			if not holding:
				# key "z" down
				pyautogui.keyDown('z')
				holding = True
		elif click == 0 and holding:
			# key "z" down
			pyautogui.keyUp('z')
			holding = False
