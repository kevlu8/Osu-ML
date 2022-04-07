import PIL.ImageGrab, PIL.Image
from mss import mss
from keyboard import is_pressed
from pyautogui import position
i = 0
from time import sleep
sleep(5)

# delete all the files in the folder
import os
# for filename in os.listdir("data/imgs"):
# 	os.remove("data/imgs/" + filename)
# # wipe keypresses.txt
# with open("keypresses.txt", "w") as f:
# 	f.write("")
i = len(os.listdir("data/imgs"))

with open("keypresses.txt", "a") as f:
	while True:
		i += 1
		img = PIL.ImageGrab.grab().resize((256, 144))
		img = img.convert('L')
		img.save(f"data/imgs/img{i}.jpg", "JPEG")
		if is_pressed("z") or is_pressed("x"):
			f.write(f"[1, {position()[0] / 1920}, {position()[1] / 1080}]\n")
		elif is_pressed('q'):
			break
		else:
			f.write(f"[0, {position()[0] / 1920}, {position()[1] / 1080}]\n")
f.close()