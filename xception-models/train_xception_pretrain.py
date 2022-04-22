# %%
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.utils import save_image
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
import glob
import PIL
from PIL import Image
from torch.utils import data as D
from torch.utils.data.sampler import SubsetRandomSampler
import random
import os
import json

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from tqdm import tqdm

# %%
def main():
    batch_size = 8

    # %%
    transform_train = transforms.Compose([
            transforms.Resize(299),
            transforms.ToTensor(),
            ])


    transform_test = transforms.Compose([
            transforms.Resize(299),
            transforms.ToTensor(),
            ])


    data_root = os.path.abspath(os.path.join(os.getcwd()))  # get data root path
    image_path = os.path.join(data_root, "hotdog-nothotdog")  # flower data set path
    assert os.path.exists(image_path), "{} path does not exist.".format(image_path)
    trainset = datasets.ImageFolder(root=os.path.join(image_path, "train"),
                                         transform=transform_train)
    testset = datasets.ImageFolder(root=os.path.join(image_path, "test"),
                                         transform=transform_test)


    num_train = len(trainset)

    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size,shuffle=True,num_workers=1
    )


    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=1
    )

    classes = ("hotdog","nohotdog")



    from  xception import xception
    net = xception(1000,pretrained='imagenet')

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    net.last_linear=nn.Linear(2048,2)
    net.to(device)
    epochs=25
    # %%
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    train_acc=[]
    test_acc=[]
    for epoch in range(epochs):
        net.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader)
        for i, data in enumerate(train_bar):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
                                                                     epochs,
                                                                     loss)
        net.eval()
        correct = 0
        total = 0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print('[%d epoch] Accuracy of the network on the trainset images: %d %%' %
              (epoch, 100 * correct / total)
              )
        trainset_acc=correct / total
        train_acc.append(trainset_acc)
        #testation part
        correct = 0
        total = 0
        for i, data in enumerate(test_loader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print('[%d epoch] Accuracy of the network on the test images: %d %%' %
              (epoch, 100 * correct / total)
             )
        testset_acc=correct / total
        test_acc.append(testset_acc)
    print('Finished Training')
    print(train_acc)
    print(test_acc)


if __name__ == '__main__':
    main()