import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
import wandb

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.maxpool(self.relu(self.conv1(x)))
        x = self.maxpool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Initialize wandb
    wandb.init(
        project="sparse-bottlenecks",
        name="mnist-cnn-test",
        config={
            "architecture": "SimpleCNN",
            "dataset": "MNIST",
            "epochs": 100,
            "batch_size": 64,
            "learning_rate": 0.001,
            "optimizer": "Adam",
        }
    )

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Download dataset
    os.makedirs('./data', exist_ok=True)
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False, num_workers=4)

    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Let wandb watch the model for gradient and parameter logging
    wandb.watch(model, log="all", log_freq=100)

    epochs = 100
    print("Starting training...")
    for epoch in range(epochs):
        # --- Training ---
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            if i % 100 == 99:
                print(f"[Epoch {epoch + 1}, Batch {i + 1}] loss: {running_loss / 100:.3f}")
                running_loss = 0.0

        train_acc = 100.0 * correct / total
        train_loss = running_loss / len(train_loader)

        # --- Evaluation ---
        model.eval()
        test_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                test_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        test_acc = 100.0 * correct / total
        test_loss /= len(test_loader)

        # Log metrics to wandb
        wandb.log({
            "epoch": epoch + 1,
            "train/loss": train_loss,
            "train/accuracy": train_acc,
            "test/loss": test_loss,
            "test/accuracy": test_acc,
        })

        print(f"Epoch {epoch + 1}/{epochs} | Train Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}% | Test Loss: {test_loss:.4f}")

    print("Finished Training")
    
    # Save the model
    os.makedirs('results', exist_ok=True)
    model_path = 'results/mnist_cnn.pth'
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    # Log the saved model as a wandb artifact
    artifact = wandb.Artifact('mnist-cnn-model', type='model')
    artifact.add_file(model_path)
    wandb.log_artifact(artifact)

    wandb.finish()

if __name__ == '__main__':
    train()
