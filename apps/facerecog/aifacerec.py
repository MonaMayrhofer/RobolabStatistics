# My Interpretation of https://hackernoon.com/facial-similarity-with-siamese-networks-in-pytorch-9642aa9db2f7
from torch import nn
import torch.nn.functional as F
import torch
import robolib.datamanager.atntfaces as data
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torchvision.datasets as dset
from torchvision import transforms
from numpy import random
from torch import optim
import random
import numpy as np
from PIL import Image
import PIL
import matplotlib.pyplot as plt
import time

data.get_data("AtnTFaces", True)


class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.cnn1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(1, 4, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(4),
            nn.Dropout2d(p=.2),

            nn.ReflectionPad2d(1),
            nn.Conv2d(4, 8, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(8),
            nn.Dropout2d(p=.2),

            nn.ReflectionPad2d(1),
            nn.Conv2d(8, 8, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(8),
            nn.Dropout2d(p=.2),
        )

        self.fc1 = nn.Sequential(
            nn.Linear(8 * 100 * 100, 500),
            nn.ReLU(inplace=True),

            nn.Linear(500, 500),
            nn.ReLU(inplace=True),

            nn.Linear(500, 5)
        )

    def forward_once(self, x):
        output = self.cnn1(x)
        output = output.view(output.size()[0], -1)
        output = self.fc1(output)
        return output

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)

        return output1, output2


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, should_label):
        euclidean_distance = F.pairwise_distance(output1, output2)
        return torch.mean((1 - should_label) * torch.pow(euclidean_distance, 2) +
                          should_label * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))


class SiameseNetworkDataset(Dataset):
    def __init__(self, image_folder_dataset, transform=None, should_invert=True):
        self.imageFolderDataset = image_folder_dataset
        self.transform = transform
        self.should_invert = should_invert

    def __getitem__(self, index):
        img0_tuple = random.choice(self.imageFolderDataset.imgs)
        # we need to make sure approx 50% of images are in the same class
        should_get_same_class = random.randint(0, 1)
        if should_get_same_class:
            while True:
                # keep looping till the same class image is found
                img1_tuple = random.choice(self.imageFolderDataset.imgs)
                if img0_tuple[1] == img1_tuple[1]:
                    break
        else:
            img1_tuple = random.choice(self.imageFolderDataset.imgs)

        img0 = Image.open(img0_tuple[0])
        img1 = Image.open(img1_tuple[0])
        img0 = img0.convert("L")
        img1 = img1.convert("L")

        if self.should_invert:
            img0 = PIL.ImageOps.invert(img0)
            img1 = PIL.ImageOps.invert(img1)

        if self.transform is not None:
            img0 = self.transform(img0)
            img1 = self.transform(img1)

        return img0, img1, torch.from_numpy(np.array([int(img1_tuple[1] != img0_tuple[1])], dtype=np.float32))

    def __len__(self):
        return len(self.imageFolderDataset.imgs)


folder_dataset_train = dset.ImageFolder("ModelData_AtnTFaces")
siamese_dataset = SiameseNetworkDataset(image_folder_dataset=folder_dataset_train,
                                        transform=transforms.Compose([transforms.Scale((100, 100)),
                                                                      transforms.ToTensor()
                                                                      ]), should_invert=False)

test_dataloader = DataLoader(siamese_dataset, num_workers=6, batch_size=1, shuffle=True)

net = SiameseNetwork().cuda()
criterion = ContrastiveLoss()
optimizer = optim.Adam(net.parameters(), lr=0.0005)

counter = []
loss_history = []
iteration_number = 0

for epoch in range(0, 100):
    for i, data in enumerate(test_dataloader, 0):
        i0, i1, label = data
        i0, i1, label = Variable(i0).cuda(), Variable(i1).cuda(), Variable(label).cuda()
        out1, out2 = net(i0, i1)
        optimizer.zero_grad()
        loss_contrastive = criterion(out1, out2, label)
        loss_contrastive.backward()
        optimizer.step()
        if i % 10 == 0:
            print("Epoch number {}\n Current loss {}\n".format(epoch, loss_contrastive.data[0]))
            iteration_number += 10
            counter.append(iteration_number)
            loss_history.append(loss_contrastive.data[0])

# show_plot(counter, loss_history)
print(counter)
print(loss_history)
