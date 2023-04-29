"""
CS 5330 Final Project
Author: Thean Cheat Lim, Wenlin Fang
Date: 4/26/23

Visualization of model attention maps
"""
# import libraries
from pytorch_model_utils import create_model
from custom import Network
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision import transforms
import torchvision.transforms.functional as fn
from datasets import load_dataset
import torch
import numpy as np
import math
import cv2
from PIL import ImageFile, Image

ImageFile.LOAD_TRUNCATED_IMAGES = True


class HuggingfaceToTensorModelWrapper(torch.nn.Module):
    """ Model wrapper to return a tensor"""

    def __init__(self, model):
        super(HuggingfaceToTensorModelWrapper, self).__init__()
        self.model = model

    def forward(self, x):
        return self.model(x).logits


class ReshapeTransform:
    """Class for reshape transformation"""

    def __init__(self, width, height):
        self.width = width
        self.height = height

    def __call__(self, tensor):
        result = tensor.reshape(tensor.size(0),
                                self.height,
                                self.width,
                                tensor.size(2))
        result = result.transpose(2, 3).transpose(1, 2)
        return result


def visualize_results(models, data_indices, dataset, target_only=True, id2label=None):
    """
    models: Dict[str, nn.Module]
    data_indices: Lst[int]
    dataset: Huggingface Dataset
    """
    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    for _, model in models.items():
        model.to(DEVICE)

    for data_index in data_indices:
        image, label = dataset[data_index].values()

        img_tensor = transforms.ToTensor()(image)

        targets_for_gradcam = [ClassifierOutputTarget(label)] if target_only else None
        # Add in the batch dimension
        input_tensor = img_tensor[None, :]

        results = [image]  # images and heatmap(to be appended)
        predictions = []
        for model_name, model in models.items():
            if model_name == "resnet":
                target_layers = [model.layer4[-1]]
            elif model_name == "densenet":
                target_layers = [model.features[-1]]
            elif model_name == "swinv2":
                target_layers = [model.swinv2.layernorm]
                model = HuggingfaceToTensorModelWrapper(model)
            elif model_name == "custom":
                target_layers = [model.layer7[-4]]
                input_tensor = fn.resize(input_tensor, size=(128, 128))
                img_tensor = fn.resize(img_tensor, size=(128, 128))

            if model_name == "swinv2":
                width = math.ceil(img_tensor.shape[2] / 32.0)
                height = math.ceil(img_tensor.shape[1] / 32.0)
                reshape_transform = ReshapeTransform(width=width, height=height)
            else:
                reshape_transform = None

            cam = GradCAM(model=model, target_layers=target_layers, reshape_transform=reshape_transform,
                          use_cuda=torch.cuda.is_available())
            grayscale_cam = cam(input_tensor=input_tensor, targets=targets_for_gradcam)
            grayscale_cam = grayscale_cam[0, :]
            if model_name == "custom":
                # image resize
                image2 = image.resize((128, 128))
                visualization = show_cam_on_image(np.float32(image2) / 255, grayscale_cam, use_rgb=True)
                visualization = cv2.resize(visualization, (image.size[0], image.size[1]))
            else:
                visualization = show_cam_on_image(np.float32(image) / 255, grayscale_cam, use_rgb=True)
            results.append(visualization)
            del reshape_transform
            # Prediction
            if id2label:
                img_tensor_pred = img_tensor.unsqueeze(0).to(DEVICE)
                logits = model(img_tensor_pred)
                index = logits.cpu()[0, :].detach().numpy().argsort()[-1]
                predictions.append(id2label[index])

        img = Image.fromarray(np.hstack(results))

        # resize all images and keep aspect ratio
        basewidth = 300 * len(models) + 1
        wpercent = (basewidth / float(img.size[0]))
        hsize = int((float(img.size[1]) * float(wpercent)))
        img = img.resize((basewidth, hsize))
        # save image
        img.save('imageGradCam_' + str(data_index) + '.png')

        if id2label:
            print("Ground Truth: ", id2label[label])
            print("Model Predictions:")
            for i, (k, v) in enumerate(models.items()):
                print("\t", k, ":", predictions[i])

            print()


# main function
def main():
    # Load Dataset and Build IdtoLabel
    dataset = load_dataset("thean/THFOOD-50", split='test')
    labels = dataset.features["label"].names
    id2label = dict()
    for i, label in enumerate(labels):
        id2label[i] = label

    # Load Models
    from transformers import AutoModelForImageClassification
    swinv2 = AutoModelForImageClassification.from_pretrained("thean/swinv2-tiny-patch4-window8-256-finetuned-THFOOD-50")

    model_path = "./saved_models/resnet50-unfrozen-train10_best.pth"
    resnet = create_model("resnet50", 50, freeze=False)
    network_state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    resnet.load_state_dict(network_state_dict)

    densenet_model_path = "./saved_models/densenet161-unfrozen_best.pth"
    densenet = create_model("densenet161", 50, freeze=False)
    network_state_dict = torch.load(densenet_model_path, map_location=torch.device('cpu'))
    densenet.load_state_dict(network_state_dict)

    custom_model_path = "./saved_models/custom_model.pth"
    custom_model = Network()
    network_state_dict = torch.load(custom_model_path, map_location=torch.device('cpu'))
    custom_model.load_state_dict(network_state_dict)
    data_indices = [520, 906,  334, 445, 672, 708, 869, 990, 999, 1003, 1105]
    model_comparison = {
        "resnet": resnet,
        "densenet": densenet,
        "swinv2": swinv2,
        "custom": custom_model
    }

    visualize_results(model_comparison, data_indices, dataset, target_only=True, id2label=id2label)


# program entry point
if __name__ == '__main__':
    main()
