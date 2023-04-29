"""
CS 5330 Final Project
Author: Thean Cheat Lim, Wenlin Fang
Date: 4/26/23

Custom CNN model for Thai food classification
"""
# import libraries
# pytorch
import torch
from torch.utils.data import Dataset
import torchvision
import torch.nn as nn
from torch.optim import Adam
# utils
import os
# image
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


# parameters
batch_size_train = 16
batch_size_test = 16
img_size = 128
num_epochs = 20
means = [0.60704023, 0.52685909, 0.4019192]
stds = [0.2553755, 0.26434593, 0.30045386]
model_path = "saved_models/custom_model.pth"
training_set_path = "THFOOD50-v1-complete/train"
test_set_path = "THFOOD50-v1-complete/test"
val_set_path = "THFOOD50-v1-complete/val"


class Network(nn.Module):
    """ Custom CNN network for Thai food classification"""

    def __init__(self):
        super(Network, self).__init__()
        # implement a network similar to VGG16
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.layer5 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer6 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.layer7 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc1 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(65536, 4096),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU()
        )
        self.fc3 = nn.Sequential(
            nn.Linear(4096, 50)
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


# Function that saves the model to the specified file path
def save_model(model):
    torch.save(model.state_dict(), model_path)


# Function to test the model with the test dataset and print the accuracy for the test images
def test_accuracy(model, test_loader):
    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(DEVICE)
    model.eval()
    accuracy = 0.0
    total = 0.0

    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            # run the model on the test set to predict labels
            outputs = model(images)
            # the label with the highest energy will be our prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            accuracy += (predicted == labels).sum().item()

    # compute the accuracy over all test images
    accuracy = (100 * accuracy / total)
    return accuracy


# Training function
def train(model, num_epochs):
    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(DEVICE)

    # transform the input images
    transform_train = torchvision.transforms.Compose([
        # random crops of images of a fixed size
        torchvision.transforms.RandomResizedCrop((img_size, img_size)),
        # random horizontal flips of input images
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        # normalize images using the means and stds calculated
        torchvision.transforms.Normalize(mean=means, std=stds)
    ])

    # transform the input images
    transform_test = torchvision.transforms.Compose([
        # random crops of images of a fixed size
        torchvision.transforms.Resize((img_size, img_size)),
        torchvision.transforms.CenterCrop((img_size, img_size)),
        torchvision.transforms.ToTensor(),
        # normalize images using the means and stds calculated
        torchvision.transforms.Normalize(mean=means, std=stds)
    ])

    # data loader for training data
    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.ImageFolder(training_set_path,
                                         transform=transform_train),
        batch_size=batch_size_train,
        shuffle=True)
    # data loader for validation set
    validation_loader = torch.utils.data.DataLoader(
        torchvision.datasets.ImageFolder(val_set_path,
                                         transform=transform_test),
        batch_size=batch_size_test,
        shuffle=False)

    # Define the loss function with Classification Cross-Entropy loss and an optimizer with Adam optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=0.001, weight_decay=0.0001)

    best_accuracy = 0.0

    for epoch in range(num_epochs):  # loop over the dataset multiple times
        model.train()
        running_loss = 0.0

        for i, (images, labels) in enumerate(train_loader, 0):
            # zero the parameter gradients
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            optimizer.zero_grad()
            # predict classes using images from the training set
            outputs = model(images)
            # compute the loss based on model output and real labels
            loss = loss_fn(outputs, labels)
            # backpropagate the loss
            loss.backward()
            # adjust parameters based on the calculated gradients
            optimizer.step()

            # print statistics for every 1,00 images
            running_loss += loss.item()  # extract the loss value
            if i % 100 == 0:
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 100))
                # zero the loss
                running_loss = 0.0
        # use validation
        # Compute and print the average accuracy for this epoch
        accuracy = test_accuracy(model, validation_loader)
        print('For epoch', epoch + 1, 'the validation accuracy over the validation set is %d %%' % accuracy)

        # we want to save the model if the accuracy is the best
        if accuracy > best_accuracy:
            save_model(model)
            best_accuracy = accuracy


# main function
def main():
    if os.path.exists(model_path):
        model = Network()
        network_state_dict = torch.load(model_path, map_location=torch.device('cpu'))
        model.load_state_dict(network_state_dict)
        train(model, num_epochs)
    else:
        model = Network()
        train(model, num_epochs)
    # test model accuracy
    transform_test = torchvision.transforms.Compose([
        # random crops of images of a fixed size
        torchvision.transforms.Resize((img_size, img_size)),
        torchvision.transforms.CenterCrop((img_size, img_size)),
        torchvision.transforms.ToTensor(),
        # normalize images using the means and stds calculated
        torchvision.transforms.Normalize(mean=means, std=stds)
    ])
    # data loader for test data
    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.ImageFolder(test_set_path,
                                         transform=transform_test),
        batch_size=batch_size_test,
        shuffle=False)
    accuracy = test_accuracy(model, test_loader)
    print('Test accuracy', accuracy)


# program entry point
if __name__ == '__main__':
    main()
