import random
import sys
import argparse
from defs import *

argparser = argparse.ArgumentParser()
argparser.add_argument('--epochs', type=int, default=100)
argparser.add_argument('--cuda', action='store_true')
argparser.add_argument('--saveevery', type=int, default=10)
argparser.add_argument('--override', action='store_true')
argparser.add_argument('--train', type=str, default=None)
argparser.add_argument('--lr', type=float, default=0.01)
argparser.add_argument('--save_dir', type=str, default=".")
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

if args.cuda:
	device = "cuda" if torch.cuda.is_available() else sys.exit("No CUDA GPU was found, but --cuda was specified.")

	import GPUtil
	GPUs = GPUtil.getGPUs()

	print("You are running on device:", GPUs[0].name)
	print("Current statistics:")
	GPUtil.showUtilization()
	print(GPUs[0].temperature, "C")
else:
	device = "cpu"

class convNetwork(nn.Module):
	def __init__(self):
		super(convNetwork, self).__init__()
		# input is an image of size 256x144 grayscale
		self.main = nn.Sequential(
			nn.Conv2d(1, 16, 3, padding=1),
			nn.ReLU(),
			nn.MaxPool2d(2, 2),
			nn.Conv2d(16, 32, 3, padding=1),
			nn.ReLU(),
			nn.MaxPool2d(2, 2),
			nn.Conv2d(32, 64, 3, padding=1),
			nn.ReLU(),
			nn.MaxPool2d(2, 2),
			nn.Conv2d(64, 128, 3, padding=1),
			nn.ReLU(),
			nn.MaxPool2d(2, 2),
			nn.AdaptiveAvgPool2d(1),
			nn.Flatten(0),
			nn.Linear(128, 3)
		)

	def forward(self, x):
		return self.main(x)
		
circlenet = convNetwork() # network to return the location of circles
# returns: tensor([[ZDown, XDown, coordX, coordY]])

# run an image through the network
from torchvision import transforms
'''
def run_image(batch=1):
	img = []
	indexes = []
	for i in range(batch):
		randint = random.randint(0, len(os.listdir("data/imgs/")) - 1)
		while randint in indexes:
			randint = random.randint(0, len(os.listdir("data/imgs/")) - 1)
		# randint = 123
		t = (PIL.Image.open("data/imgs/" + os.listdir("data/imgs")[randint]))
		# t = t.resize((256, 144))
		t = transforms.functional.pil_to_tensor(t).to(device)
		t = t.type(torch.FloatTensor)
		t = t.to(device)
		img.append(t)
		indexes.append(randint)
	return (indexes, img)'''

class osuDataSet(torch.utils.data.Dataset):
	def __init__(self, batch=1):
		self.indexes = []
		self.imgs = []
		for i in range(batch):
			for img in os.listdir("data/imgs/"):
				index = img.split(".")[0]
				# remove all non-numeric characters
				index = "".join(filter(str.isdigit, index))
				self.indexes.append(int(index))
				# self.imgs.append(PIL.Image.open("data/imgs/" + img))
				self.imgs.append("data/imgs/" + img)
		
	def __len__(self):
		return len(self.indexes) if self.indexes is not None else 0
	
	def __getitem__(self, index):
		for i in range(len(self.indexes)):
			if self.indexes[i] == index:
				t = self.imgs[i]
				t = (PIL.Image.open(t))
				t = transforms.functional.pil_to_tensor(t).to(device)
				t = t.type(torch.FloatTensor)
				t = t.to(device)
				return (self.indexes[i], t)

trainDataloader = torch.utils.data.DataLoader(osuDataSet(), batch_size=64, shuffle=True)

optimizer = optim.SGD(circlenet.parameters(), lr=args.lr, momentum=0.9)
#optimizer = optim.Adam(circlenet.parameters(), lr=args.lr)

if os.path.exists("weights.pth"):
	c = torch.load("weights.pth", map_location=device)
	circlenet.load_state_dict(c["model"])
	optimizer.load_state_dict(c["optimizer"])

circlenet.to(device)
circlenet.train()

for param in optimizer.state.values(): # set optimizer to device
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

if args.train == "conv":
	losses = []
	for epoch in range(args.epochs):
		# for i, img in zip(*run_image()):
		# 	try:
		# 		circlenet.zero_grad()
		# 		optimizer.zero_grad()
		# 		img = torch.tensor(img).to(device)
		# 		output = circlenet(img)
		# 		# torch.sigmoid(output[0])
		# 		loss = nn.functional.mse_loss(output, torch.tensor(
		# 			data[i]
		# 		, device=device, dtype=torch.float))
		# 		loss = loss.to(device)
		# 		loss.backward()
		# 		optimizer.step()
		# 	except: # somehow data[i] is out of range sometimes?
		# 		continue
		for index, img in enumerate(trainDataloader):
			try:
				circlenet.zero_grad()
				optimizer.zero_grad()
				img = img.to(device)
				output = circlenet(img)
				# torch.sigmoid(output[0])
				loss = nn.functional.mse_loss(output, torch.tensor(
					data[index]
				, device=device, dtype=torch.float))
				loss = loss.to(device)
				loss.backward()
				optimizer.step()
			except: # somehow data[i] is out of range sometimes?
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
			torch.save({"model": circlenet.state_dict(), "optimizer": optimizer.state_dict()}, f"{args.save_dir}/weights.pth")
			circlenet.to(device)

		try:
			losses.append(loss.item())
			print("Epoch: ", epoch + 1, " Loss: ", loss.item()) 
		except: continue
	import matplotlib.pyplot as plt
	plt.plot(losses)
	plt.show()
elif args.train == "test":
	rand = random.randint(0, len(os.listdir("data/imgs/")) - 1)
	import cv2
	# use the network
	circlenet.eval()
	files = os.listdir("data/imgs/")
	img = (PIL.Image.open("data/imgs/" + files[rand]))
	# img = img.resize((256, 144))
	img = transforms.functional.pil_to_tensor(img).to(device)
	img = img.type(torch.FloatTensor)
	img = img.to(device)
	with torch.no_grad():
		out = circlenet(img)
		print(out)
	out = out.cpu().numpy()
	out = out.tolist()
	imgcv = cv2.imread(f"data/imgs/{files[rand]}")
	print(out)
	color = (0, 255, 0) if round(out[0]) == 1 else (0, 0, 255)
	cv2.circle(imgcv, (int(out[1] * 256), int(out[2] * 144)), 4, color, 2)
	imgcv = cv2.resize(imgcv, (480, 270))
	cv2.imshow("output", imgcv)
	cv2.waitKey(0)
else:	
	from mss import mss
	import keyboard
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
	key = 'z'
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
