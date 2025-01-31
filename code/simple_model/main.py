import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
from resnet_models import resnet18, resnet34, resnet50, resnet101, resnet152

class TrajectoryDataset(Dataset):
    def __init__(self, data, labels, window_size):
        """
        data: numpy array of shape (num_trajectories, time_steps, 9)
        labels: numpy array of shape (num_trajectories,)
        window_size: number of time steps to consider as one sample
        """
        self.data = data
        self.labels = labels
        self.window_size = window_size
        
        # Create sliding windows
        self.windows = []
        self.window_labels = []
        
        for i in range(len(data)):
            for j in range(data[i].shape[0] - window_size + 1):
                self.windows.append(data[i][j:j+window_size])
                self.window_labels.append(labels[i])
                
        self.windows = np.array(self.windows)
        self.window_labels = np.array(self.window_labels)

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        window = torch.FloatTensor(self.windows[idx]).transpose(0, 1)  # Shape: (9, window_size)
        label = torch.FloatTensor([self.window_labels[idx]])
        return window, label

def train_model(model, train_loader, num_epochs=50, learning_rate=0.001):
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        for batch_data, batch_labels in train_loader:
            batch_data = batch_data.to(device)
            batch_labels = batch_labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_data)
            loss = criterion(outputs, batch_labels)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        if (epoch + 1) % 5 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')

def evaluate_model(model, test_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch_data, batch_labels in test_loader:
            batch_data = batch_data.to(device)
            batch_labels = batch_labels.to(device)
            
            outputs = model(batch_data)
            predicted = (outputs >= 0.5).float()
            
            total += batch_labels.size(0)
            correct += (predicted == batch_labels).sum().item()
    
    accuracy = 100 * correct / total
    return accuracy

if __name__ == "__main__":
    # Example usage with different ResNet architectures
    
    # Generate dummy data for demonstration
    num_trajectories = 100
    max_time_steps = 50
    window_size = 32  # Increased window size for ResNet
    
    # Create dummy trajectories
    trajectories = []
    labels = []
    
    for _ in range(num_trajectories):
        time_steps = np.random.randint(window_size, max_time_steps + 1)
        trajectory = np.random.randn(time_steps, 9)
        trajectories.append(trajectory)
        labels.append(np.random.randint(2))
    
    # Split data into train and test sets
    train_size = int(0.8 * num_trajectories)
    train_trajectories = trajectories[:train_size]
    train_labels = labels[:train_size]
    test_trajectories = trajectories[train_size:]
    test_labels = labels[train_size:]
    
    # Create datasets and dataloaders
    train_dataset = TrajectoryDataset(train_trajectories, train_labels, window_size)
    test_dataset = TrajectoryDataset(test_trajectories, test_labels, window_size)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Example of using different ResNet architectures
    models = {
        'ResNet18': resnet18(input_channels=9),
        # Uncomment for deeper architectures
        # 'ResNet34': resnet34(input_channels=9),
        # 'ResNet50': resnet50(input_channels=9),
        # 'ResNet101': resnet101(input_channels=9),
        # 'ResNet152': resnet152(input_channels=9)
    }
    
    # Train and evaluate each model
    for name, model in models.items():
        print(f"\nTraining {name}")
        train_model(model, train_loader)
        
        accuracy = evaluate_model(model, test_loader)
        print(f"{name} Test Accuracy: {accuracy:.2f}%")
