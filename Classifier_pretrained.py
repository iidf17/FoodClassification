import torch
import torch.nn as nn
import torchsummary
from torchvision import models


def count_lines(filename):
    classes_list = []

    with open(filename, 'r') as file:
        for line in file:
            classes_list.append(line.strip())

    with open(filename, 'r') as file:
        lines = file.readlines()

    return len(lines), classes_list

filename = "Prepared/Meta/labels.txt"

NUM_CLASSES, CLASSES_LIST = count_lines(filename)


class Classifier(nn.Module):
    def __init__(self, num_classes):
        super(Classifier, self).__init__()
        self.model = models.resnet18(pretrained=True)
        num_featrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_featrs, num_classes)

    def forward(self, x):
        return self.model(x)

if __name__ == "__main__":
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using device: {device}")

    classification = Classifier(NUM_CLASSES).to(device)

    torchsummary.summary(classification, (3, 224, 224), device=device)

    path_to_test_img = "Prepared\\test\\apple_pie\\229142.jpg"

    print(NUM_CLASSES)