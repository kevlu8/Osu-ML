import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import os
import argparse
import cv2

argparser = argparse.ArgumentParser()
argparser.add_argument('--epochs', type=int, default=100)
argparser.add_argument('--cuda', action='store_true')
argparser.add_argument('--saveevery', type=int, default=10)
argparser.parse_args()
args = argparser.parse_args()

if args.cuda:
    device = "cuda" if torch.cuda.is_available() else exit("No CUDA GPU was found, but --cuda was set")
else:
    device = "cpu"

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

batch_size = 4

classes = ('circle', 'slider', 'spinner')

class ImageDataset(Dataset):
    def __init__(self):
        with open("trainlabels.txt") as f:
            self.labels = f.readlines()
        imgs = []
        for image in os.listdir("./imgs/"):
            if image.endswith(".png"):
                imgs.append(image[:-4])
        self.imgs = imgs
        self.img_dir = "imgs/"
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, i):   
        img_path = os.path.join(self.img_dir, self.imgs[i]) + ".png"
        image = torchvision.transforms.functional.to_tensor(cv2.imread(img_path))
        if image.shape != (3, 256, 256):
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        # label = torch.Tensor(self.target_transform(set(self.labels[os.path.basename(img_path).split(".")[0]])))
        label = torch.Tensor(self.labels[os.path.basename(img_path)[:-4]])
        return image, label
    
    def target_transform(self, labels):
        label = [0.] * len(self.tags)
        for i, t in enumerate(self.tags):
            if t in labels:
                label[i] = 1.
        return label

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

net = CNN().to(device)

optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

if os.path.exists("cnn.pth"):
    m = torch.load("cnn.pth")
    net.load_state_dict(m["model"])
    optimizer.load_state_dict(m["optimizer"])

criterion = nn.CrossEntropyLoss()

net.train()

trainloader = torch.utils.data.DataLoader(
    ImageDataset(),
    batch_size=batch_size, shuffle=True)

for epoch in range(args.epochs):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 2000 == 1999:
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0
        if i % args.saveevery == 0:
            torch.save({"model": net.state_dict(), "optimizer": optimizer.state_dict()}, 'cnn.pth')

print('Finished Training')
torch.save({"model": net.state_dict(), "optimizer": optimizer.state_dict()}, 'cnn.pth')