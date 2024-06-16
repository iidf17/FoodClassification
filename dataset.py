import os
import shutil
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def prepare_dataset(ds_dir, train_file, test_file):
    # разбить на train[] и test[]

    train_dir = 'Prepared\\train'
    test_dir = 'prepared\\test'
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    with open(train_file, 'r') as f:
        train_imgs = [line.strip() for line in f]

    with open(test_file, 'r') as f:
        test_imgs = [line.strip() for line in f]

    for image in train_imgs:
        src = os.path.join(ds_dir, image + '.jpg')
        dst = os.path.join(train_dir, image + '.jpg')
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy(src, dst)

    for image in test_imgs:
        src = os.path.join(ds_dir, image + '.jpg')
        dst = os.path.join(test_dir, image + '.jpg')
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy(src, dst)

def load_dataset(dir, batch_size):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = datasets.ImageFolder(dir, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return loader

if __name__ == "__main__":
    dataset_dir = 'food-101\\food-101\\images'
    train_txt_file = 'Prepared/Meta/train.txt'
    test_txt_file = 'Prepared/Meta/test.txt'
    prepare_dataset(dataset_dir, train_txt_file, test_txt_file)
