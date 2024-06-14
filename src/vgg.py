import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

# Define the Enhanced VGG Model with Batch Normalization and Dropout for regularization
class EnhancedVGG(nn.Module):
    def __init__(self):
        super(EnhancedVGG, self).__init__()
        # First set of convolutional layers followed by Batch Normalization and Max Pooling
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(2, 2)

        # Second set of convolutional layers with Batch Normalization and Max Pooling
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)

        # Third set of convolutional layers with Batch Normalization and Max Pooling
        self.conv5 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn5 = nn.BatchNorm2d(128)
        self.conv6 = nn.Conv2d(128, 128, 3, padding=1)
        self.bn6 = nn.BatchNorm2d(128)

        # Fully connected layers with Dropout for regularization
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 256)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(256, 10)

    def forward(self, x):
        # Forward pass through the first convolutional block
        x = self.pool(F.relu(self.bn2(self.conv2(F.relu(self.bn1(self.conv1(x)))))))
        # Forward pass through the second convolutional block
        x = self.pool(F.relu(self.bn4(self.conv4(F.relu(self.bn3(self.conv3(x)))))))
        # Forward pass through the third convolutional block and attention map generation
        attention_map = F.relu(self.bn6(self.conv6(F.relu(self.bn5(self.conv5(x))))))
        x = self.pool(attention_map)
        # Flatten the tensor for the fully connected layers
        x = x.view(-1, 128 * 4 * 4)
        # Forward pass through the fully connected layers with Dropout
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        # Final output layer
        output = self.fc3(x)
        return output, attention_map

# Apply attention masking to the inputs using the model's attention maps
def apply_attention_masking(inputs, model, attenuation_factor=0.5, dynamic_threshold=True):
    _, attention_maps = model(inputs)
    # Determine the threshold for masking
    if dynamic_threshold:
        threshold = attention_maps.quantile(0.75)
    else:
        threshold = attention_maps.mean()
    # Normalize the attention maps
    attention_maps_normalized = (attention_maps - attention_maps.min()) / (attention_maps.max() - attention_maps.min())
    size = (inputs.size(2), inputs.size(3))
    # Upsample the attention maps to match input size
    attention_maps_upsampled = F.interpolate(attention_maps_normalized, size=size, mode='bilinear', align_corners=False)
    # Create masks based on the threshold
    masks = (attention_maps_upsampled > threshold).float() * attenuation_factor
    masks = masks.mean(dim=1, keepdim=True)
    # Apply masks to the inputs
    masked_inputs = inputs * (1 - masks)
    return masked_inputs

# Calculate the accuracy of predictions
def calculate_accuracy(predictions, labels):
    _, predicted = torch.max(predictions, 1)
    correct = (predicted == labels).sum().item()
    total = labels.size(0)
    accuracy = (correct / total) * 100
    return accuracy

# Train the model for a given number of epochs
def train_model(model, trainloader, criterion, optimizer, num_epochs, device, apply_mask):
    for epoch in range(num_epochs):
        running_loss = 0.0
        total_accuracy = 0.0
        total_samples = 0

        # Loop over the training data
        for i, data in enumerate(tqdm(trainloader, desc=f"Epoch {epoch + 1}/{num_epochs}"), 0):
            inputs, labels = data[0].to(device), data[1].to(device)
            # Optionally apply attention masking
            if apply_mask:
                inputs = apply_attention_masking(inputs, model)
            optimizer.zero_grad()
            outputs, _ = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            total_accuracy += calculate_accuracy(outputs, labels)
            total_samples += labels.size(0)
            # Print statistics every 2000 mini-batches
            if (i + 1) % 2000 == 0:
                print(f'[{epoch + 1}, {total_samples}] loss: {running_loss / 2000:.3f} accuracy: {total_accuracy / 2000:.2f}%')
                running_loss = 0.0
                total_accuracy = 0.0

# Main function to setup data loaders, model, and training
if __name__ == '__main__':
    # Define the data transformations including normalization
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Load the CIFAR-10 dataset
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    # Define training parameters
    learning_rate = 0.001
    momentum = 0.9
    batch_size = 4
    num_epochs = 12

    # Setup device for training (CPU or GPU)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = EnhancedVGG().to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

    # Setup data loaders
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    # Initial training without attention masking
    print("Initial training without masking...")
    train_model(model, trainloader, criterion, optimizer, num_epochs=num_epochs, device=device, apply_mask=False)

    # Save the initial trained model
    torch.save(model.state_dict(), './vgg_initial.pth')
    print("\nInitial model trained and saved.")

    # Load the initial trained model for fine-tuning
    model.load_state_dict(torch.load('./vgg_initial.pth'))
    print("\nInitial model loaded for fine-tuning.")

    # Setup a new optimizer for fine-tuning with a smaller learning rate
    optimizer_ft = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)
    print("\nStarting fine-tuning with masking...")
    train_model(model, trainloader, criterion, optimizer_ft, num_epochs=1, device=device, apply_mask=True)

    # Save the fine-tuned model
    torch.save(model.state_dict(), './vgg_finetuned.pth')
    print("\nFine-tuning completed and model saved.")
