import os
import torch
import torch.nn as nn
import torchsummary


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
        if torch.cuda.is_available():
            self.device = "cuda"
        else:
           self.device = "cpu"

        self.input = nn.Conv2d(3, 16, 3, 1, 1)
        self.relu = nn.ReLU()
        self.mPool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.seq = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=3)
        )
        self.flatten = nn.Flatten()
        self.lin1 = nn.Linear(32 * 37 * 37, 500)
        self.output = nn.Linear(500, num_classes)

    def forward(self, input_data):
        x = self.input(input_data)
        x = self.relu(x)
        x = self.mPool(x)
        x = self.seq(x)
        x = self.flatten(x)
        #print(x.shape)
        x = self.lin1(x)
        x = self.relu(x)
        pred = self.output(x)
        return pred

if __name__ == "__main__":
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using device: {device}")

    classification = Classifier(NUM_CLASSES).to(device)

    torchsummary.summary(classification, (3, 224, 224), device=device)

    path_to_test_img = "food-101\\images\\apple_pie\\134.jpg"

    print(NUM_CLASSES)
