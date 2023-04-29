'''
CS 5330 Final Project
Author: Thean Cheat Lim, Wenlin Fang
Date: 4/26/23

Fine-tune ResNet and DenseNet models
'''
from PIL import ImageFile
from pytorch_model_utils import create_model, train_model, test_model
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, RandomResizedCrop, RandomHorizontalFlip, ToTensor, Normalize, Resize, CenterCrop
import sys
import torch
import torch.nn as nn

# Constants
BATCH_SIZE = 64
MEANS = [0.60704023, 0.52685909, 0.4019192]
STDS= [0.2553755, 0.26434593, 0.30045386]
IMG_SIZE = 128
LEARNING_RATE = 1e-3

def main(argv):
  model_name = argv[1]
  OUTPUT_PATH = argv[2]
  NUM_EPOCHS = int(argv[3])
  train_path, val_path, test_path = argv[4], argv[5], argv[6]

  assert model_name in ["resnet50", "densenet161"], "Invalid model_name. Pick from `resnet50`, `densenet161`"

  DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
  size = (IMG_SIZE, IMG_SIZE)

  # Define Image Transformations
  transform_train = Compose([
        RandomResizedCrop(size),
        RandomHorizontalFlip(),
        ToTensor(),
        Normalize(mean=MEANS, std=STDS)
    ])

  transform_val = Compose([
        Resize(size),
        CenterCrop(size),
        ToTensor(),
        Normalize(mean=MEANS, std=STDS)
    ])

  # DataLoaders
  train_dataloader = DataLoader(
        ImageFolder(train_path, transform=transform_train),
        batch_size=BATCH_SIZE,
        shuffle=True)

  val_dataloader = DataLoader(
          ImageFolder(val_path, transform=transform_val),
          batch_size=BATCH_SIZE,
          shuffle=False)

  test_dataloader = DataLoader(
          ImageFolder(test_path, transform=transform_val),
          batch_size=BATCH_SIZE,
          shuffle=False)
  
  model = create_model(model_name, 50, freeze=False)
  model.to(DEVICE)

  ImageFile.LOAD_TRUNCATED_IMAGES = True
  criterion = nn.CrossEntropyLoss()
  optimizer = torch.optim.AdamW([param for param in model.parameters() if param.requires_grad==True], lr=LEARNING_RATE)
  model_trained = train_model(model, OUTPUT_PATH, train_dataloader, val_dataloader, criterion, optimizer, num_epochs=NUM_EPOCHS, log_interval = 10, device = DEVICE)

  acc, loss = test_model(model_trained, test_dataloader, criterion, DEVICE)
  print("Test accuracy:", acc, ", Test Loss: ", loss)

  return

if __name__ == "__main__":
  if len(sys.argv)<7:
    print("Usage: python resnet_densenet.py resnet50 model_name_finetuned_out 2 /train /val /test")
    print("The first argument is the Model Name: `resnet50`, `densenet161`")
    print("The second argument is the Model Name.")
    print("The third argument is the Output Path.")
    print("The fourth argument is the Number of Epoch.")
    print("The fifth argument is the Training Data Path.")
    print("The sixth argument is the Validation Data Path.")
    print("The seventh argument is the Testing Data Path.")
  else:
    main(sys.argv)
