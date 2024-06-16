import time
from Classifier_pretrained import NUM_CLASSES, Classifier
from dataset import load_dataset
import torch
import torch.optim as optim
import torch.nn as nn

BATCH_SIZE = 64
LEARNING_RATE = .001
EPOCHS = 25


def train_model(model, data_loader, num_epochs, learning_rate, device):
    criterion = nn.CrossEntropyLoss()
    print(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    best_loss = 10.0

    for epoch in range(num_epochs):
        start_time = time.time()
        running_loss = .0
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

        elapsed_time = time.time() - start_time
        epoch_loss = running_loss / len(data_loader.dataset)
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}')
        print(f'Elapsed time: {elapsed_time} s.')

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(model.state_dict(), 'Models/best_model.pth')


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    prepared_dir = 'Prepared\\train'
    dataloader = load_dataset(prepared_dir, batch_size=BATCH_SIZE)
    model = Classifier(NUM_CLASSES).to(device)
    train_model(model, dataloader, num_epochs=EPOCHS, learning_rate=LEARNING_RATE, device=device)
    torch.save(model.state_dict(), 'model.pth')
    print("Trained model saved at model.pth")
    print("_________________________________________________\n\n")
