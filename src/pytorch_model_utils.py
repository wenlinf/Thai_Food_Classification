'''
CS 5330 Final Project
Author: Thean Cheat Lim, Wenlin Fang
Date: 4/26/23

Utility functions for creating/training/testing Pytorch models
'''
from torchvision import models
from tqdm.auto import tqdm
import copy
import torch
import torch.nn as nn

def create_model(model_name, num_classes, pretrained = True, freeze=True):
  '''
  Load Models (optionally with weight, frozen or unfrozen), and replace
  the classifier head with a Linear layer of `num_classes` nodes
  '''
  if model_name == "resnet50":
    model = models.resnet50(pretrained=pretrained)
    if freeze:
      for param in model.parameters():
        param.requires_grad = False
    
    model.fc = nn.Linear(2048, num_classes)
  elif model_name == "densenet161":
    model = models.densenet161(pretrained=pretrained)
    if freeze:
      for param in model.parameters():
        param.requires_grad = False

    model.classifier = nn.Linear(2208, num_classes)
  else:
    print("Invalid model_name. Pick from ", "resnet50", "densenet161")
    return

  return model

def train_model(model, model_name, train_dataloader, val_dataloader, criterion, optimizer, num_epochs=1, log_interval = 100, device = torch.device('cpu')):
  '''
  Train model. Save model in each epoch. Load and return the best model (accuracy)
  '''
  train_data_cnt = len(train_dataloader.dataset)
  total_step = len(train_dataloader)

  correct = 0
  # Keep track of the best model weights
  best_model_wts = copy.deepcopy(model.state_dict())
  best_acc = 0.0

  # train the network
  for epoch in tqdm(range(num_epochs)):
    model.train()
    for i, (images, labels) in enumerate(train_dataloader):
      images = images.to(device)
      labels = labels.to(device)

      # Forward pass
      optimizer.zero_grad()
      outputs = model(images)

      # get the number of correct predictions
      _, predicted = torch.max(outputs, 1)
      correct += (predicted == labels).sum().item()

      # backpropagation
      loss = criterion(outputs, labels)
      loss.backward()
      optimizer.step()

      if (i+1) % log_interval == 0:
        train_accuracy = 100 * (predicted == labels).sum().item() / labels.size(0)
        print('Epoch: [{}/{}],  Step: [{}/{}], Train Loss: {:.4f}, Train Accuracy: {:.4f}'
        .format(epoch+1, num_epochs, i+1, total_step, loss.item()/ labels.size(0), train_accuracy))
    #### End of one epoch ####

    # Save model at the end of each epoch
    fn = model_name+"_"+str(epoch)+".pth"
    torch.save(model.state_dict(), fn)

    if not val_dataloader: continue
    model.eval()
    with torch.no_grad():
      running_loss, running_correct = 0.0, 0
      val_data_cnt = len(val_dataloader.dataset)
      for images, labels in val_dataloader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        running_loss+=loss.item()
        _, predicted = torch.max(outputs, 1)
        running_correct+=(predicted == labels).sum().item()
    ### End of Evaluation  ###
    val_acc = running_correct/val_data_cnt*100
    print ('Epoch [{}/{}], Val Loss: {:.4f}, Val Accuracy: {:.4f}' 
                  .format(epoch+1, num_epochs, running_loss/val_data_cnt, val_acc))
    print()
    if val_acc>best_acc:
      best_acc = val_acc
      best_model_wts = copy.deepcopy(model.state_dict())

      fn = model_name+"_best.pth"
      torch.save(model.state_dict(), fn)

  #### End of Training ###
  model.load_state_dict(best_model_wts)
  print('Training accuracy: {} %'.format((correct / num_epochs*train_data_cnt) * 100))

  return model

def test_model(model, test_dataloader,criterion, device):
  '''
  Test a model and return accuracy and losses
  '''
  with torch.no_grad():
    running_loss, running_correct = 0.0, 0
    test_data_cnt = len(test_dataloader.dataset)
    for images, labels in tqdm(test_dataloader):
      images = images.to(device)
      labels = labels.to(device)
      outputs = model(images)
      loss = criterion(outputs, labels)
      running_loss+=loss.item()
      _, predicted = torch.max(outputs, 1)
      running_correct+=(predicted == labels).sum().item()
  ### End of Evaluation  ###
  test_acc = running_correct/test_data_cnt*100
  test_loss = running_loss/test_data_cnt
  return test_acc, test_loss
