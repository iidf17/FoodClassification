import torch
from torchvision import datasets
import torchvision.transforms as transforms
import torch.nn.functional as F
from PIL import Image

import time

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

from Old.Classifier import Classifier, NUM_CLASSES
from torch.utils.data import DataLoader


def plot_confusion_matrix(true_labels, predictions, class_names):
    cm = confusion_matrix(true_labels, predictions)
    plt.figure(figsize=(12, 12))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()


def predict(model, dir, device):
    t = time.localtime()
    start_time = time.strftime("%H:%M:%S", t)
    print("Started at: ", start_time)
    start_time = time.time()
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = datasets.ImageFolder(dir, transform=transform)
    loader = DataLoader(dataset, batch_size=64)

    model_state_dict = torch.load("Models/model1_augment.pth")
    model.load_state_dict(model_state_dict)
    model.eval()

    predictions = []
    true_labels = []

    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            output = model(inputs)
            smax = F.softmax(output, dim=1)
            accuracy, predicted_class = torch.max(smax, 1)
            predictions.extend(predicted_class.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    report = classification_report(true_labels, predictions, target_names=dataset.classes)
    print(report)
    plot_confusion_matrix(true_labels, predictions, dataset.classes)
    t = time.localtime()
    ended_time = time.strftime("%H:%M:%S", t)
    print(f"Ended at: {ended_time}\n"
          f"Elapsed time = {time.time() - start_time} sec")
    #print(f'Predicted class: {CLASSES_LIST[predicted_class.item()]}')
    #print(f'Accuracy: {accuracy.item() * 100} %')

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    classifier = Classifier(NUM_CLASSES).to(device)
    image = "food-101\\food-101\\images\\caesar_salad\\52578.jpg"
    image2 = "food-101\\food-101\\images\\baklava\\49917.jpg"
    test_dir = 'Prepared\\test'
    predict(classifier, test_dir, device)
