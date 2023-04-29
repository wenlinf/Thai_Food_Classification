"""
CS 5330 Final Project
Author: Thean Cheat Lim, Wenlin Fang
Date: 4/26/23

Image preprocessing util
"""
import numpy as np
import torchvision
import torch


# Function that calculate the mean and standard deviation of the Thai food images.
def calc_dataset_mean_std(training_set_path):
    size = (170, 150)
    transform_train = torchvision.transforms.Compose([
        # resize all images
        torchvision.transforms.Resize(size),
        torchvision.transforms.ToTensor()
    ])

    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.ImageFolder(training_set_path, transform=transform_train),
        batch_size=64,
        shuffle=False)

    # calculate means and stds of each batch
    batch_means = []
    batch_stds = []
    for i, data in enumerate(train_loader, 1):
        images, labels = data
        mean = torch.mean(images, dim=(0, 2, 3))
        std = torch.std(images, dim=(0, 2, 3))
        batch_means.append(mean.numpy().tolist())
        batch_stds.append(std.numpy().tolist())

    # calculate mean and std of all images
    means = np.mean(batch_means, axis=0)
    stds = np.mean(batch_stds, axis=0)
    return means, stds


# main function
def main():
    training_set_path = "THFOOD50-v1-complete/train"
    means, stds = calc_dataset_mean_std(training_set_path)
    print("Means:", means)
    print("Standard deviations:", stds)


# program entry point
if __name__ == '__main__':
    main()
