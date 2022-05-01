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
argparser.add_argument('--lr', type=float, default=0.00001)
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

try:
	screenX, screenY = osu.getScreenSize()
except:
	screenX, screenY = (1920, 1080)

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
# returns: tensor([[ZDown, coordX, coordY]])

# run an image through the network
from torchvision import transforms

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
f.close()

class osuDataSet(torch.utils.data.Dataset):
	def __init__(self):
		self.indexes = []
		self.imgs = []
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
		img = PIL.Image.open(self.imgs[index])
		img = transforms.functional.pil_to_tensor(img).to(device)
		img = img.type(torch.FloatTensor)
		img = img.to(device)
		return (torch.tensor(data[index]), img)

trainDataloader = torch.utils.data.DataLoader(osuDataSet(), batch_size=64, shuffle=True)

optimizer = optim.SGD(circlenet.parameters(), lr=args.lr, momentum=0.9)

if os.path.exists("weights.pth"):
	c = torch.load("weights.pth", map_location=device)
	circlenet.load_state_dict(c["model"])
	optimizer.load_state_dict(c["optimizer"])

circlenet.to(device)

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

if args.train == "conv":
	print("Keep in mind that data is normalized. Losses are not scaled. The model is only considered to be trained if the loss is below 1e-10.")
	losses = []
	validationLosses = []
	circlenet.train()
	try:
		for epoch in range(args.epochs):
			currentValidationLosses = []
			for (index, img) in trainDataloader:
				currentlosses = []
				for i, image in enumerate(img):
					# circlenet.zero_grad()
					optimizer.zero_grad()
					output = circlenet(image)
					# print(index[i])
					loss = nn.functional.mse_loss(output, torch.tensor(
						index[i]
					, device=device, dtype=torch.float))
					loss = loss.to(device)
					loss.backward()
					optimizer.step()
					currentlosses.append(loss.item())
					# validate
					if i % 100 == 0:
						circlenet.eval()
						with torch.no_grad():
							for j, imagea in enumerate(img):
								output = circlenet(imagea)
								loss = nn.functional.mse_loss(output, torch.tensor(
									index[j]
								, device=device, dtype=torch.float))
								loss = loss.to(device)
								currentValidationLosses.append(loss.item())
								# # plot the results on cv2
								# import cv2
								# # first convert imagea to cv2
								# imagea = imagea.to("cpu")
								# imagea = imagea.numpy()
								# imagea = imagea.transpose(1, 2, 0)
								# imagea = imagea.astype(np.uint8)
								# imagea = cv2.cvtColor(imagea, cv2.COLOR_RGB2BGR)
								# # then circle the output
								# cv2.circle(imagea, (int(output[1].item() * 256), int(output[2].item() * 144)), 4, (0, 255, 0), 2)
								# # then show the image
								# cv2.imshow("image", imagea)
								# cv2.waitKey(0)
							print("Epoch:", epoch, "Loss:", loss.item(), "Validation Loss:", sum(currentValidationLosses) / len(currentValidationLosses))
			validationLosses.append(sum(currentValidationLosses) / len(currentValidationLosses))

			if epoch % 50 == 0:
				if args.cuda:
					GPUs = GPUtil.getGPUs()
					print(GPUs[0].temperature, "C")
			
			if epoch % args.saveevery == 0:
				circlenet.cpu()
				torch.save({"model": circlenet.state_dict(), "optimizer": optimizer.state_dict()}, f"{args.save_dir}/weights.pth")
				circlenet.to(device)

			losses.append(sum(currentlosses)/len(currentlosses))
			print(f"Epoch: {epoch + 1: <6} Loss: {sum(currentlosses)/len(currentlosses)}") 
	except KeyboardInterrupt:
		torch.save({"model": circlenet.state_dict(), "optimizer": optimizer.state_dict()}, f"{args.save_dir}/weights.pth")
	import matplotlib.pyplot as plt
	plt.plot(losses, label="Training Loss")
	plt.plot(validationLosses, label="Validation Loss")
	plt.show()
elif args.train == "test":
	rand = random.randint(0, len(os.listdir("data/imgs/")) - 1)
	import cv2
	# use the network
	circlenet.eval()
	img = (PIL.Image.open(f"data/imgs/img{rand}.jpg"))
	img = transforms.functional.pil_to_tensor(img).to(device)
	img = img.type(torch.FloatTensor)
	img = img.to(device)
	with torch.no_grad():
		out = circlenet(img)
	out = out.cpu().numpy()
	out = out.tolist()
	imgcv = cv2.imread(f"data/imgs/img{rand}.jpg")
	print("Output: ", out)
	print(rand)
	# remove first and last characters
	ans = data[rand - 1] 
	print("Answer: ", ans)
	loss = nn.functional.mse_loss(torch.tensor(out, dtype=torch.float, device=device), torch.tensor(ans, dtype=torch.float, device=device))
	print("Loss: ", loss.item())
	cv2.circle(imgcv, (round(ans[1] * 256), round(ans[2] * 144)), 2, (255, 255, 0), 2) # answer
	color = (0, 255, 0) if round(out[0]) == 1 else (0, 0, 255)
	cv2.circle(imgcv, (round(out[1] * 256), round(out[2] * 144)), 4, color, 2)
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
		pyautogui.moveTo(out[1] * screenX, out[2] * screenY)
		if click == 1:
			# mousebutton down
			if not holding:
				# key "z" down
				pyautogui.keyDown(key)
				holding = True
		elif click == 0 and holding:
			# key "z" down
			pyautogui.keyUp(key)
			holding = False
