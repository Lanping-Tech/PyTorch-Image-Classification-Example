import torch
import torch.nn as nn
import torchvision

def get_resnet50(num_classes):
    model = torchvision.models.resnet50(num_classes=num_classes)
    return model

def get_alexnet(num_classes):
    model = torchvision.models.alexnet(num_classes=num_classes)
    return model