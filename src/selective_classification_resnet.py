import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import torchvision
import torchvision.models as models
import torch.optim as optim
from resnet50 import test_dataset, test_loader, num_classes, AttentionMaskingResNet50
from sklearn.metrics import precision_recall_fscore_support


# Function to perform selective classification using two models
def selective_classification(model_initial, model_attention, dataloader, confidence_threshold):
    model_initial.eval()  # Set the first model (without attention) to evaluation mode
    model_attention.eval()  # Set the second model (with attention) to evaluation mode
    total_samples = 0
    total_correct = 0
    rejected = 0
    agreements = 0  # Counter for cases where models agree on predictions

    with torch.no_grad():  # Disable gradient calculation for evaluation
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            # Get predictions from both models
            outputs_initial = model_initial(images)
            outputs_attention = model_attention(images)

            # Apply sigmoid to get probabilities for both models
            probs_initial = torch.sigmoid(outputs_initial)
            probs_attention = torch.sigmoid(outputs_attention)

            # Calculate maximum confidence of the attention model predictions
            confidences_attention, _ = probs_attention.max(1)

            # Check if confidences exceed the threshold
            confident_preds = confidences_attention > confidence_threshold

            # Determine binary predictions for both models
            preds_initial = (probs_initial > 0.5).float()
            preds_attention = (probs_attention > 0.5).float()

            # Check agreement between models for each label
            agreement = (preds_initial == preds_attention) & confident_preds.unsqueeze(1)
            mask_agreement = agreement.all(dim=1)  # Check agreement on all labels

            # Select labels and predictions where both models agree
            selected_labels = labels[mask_agreement]
            selected_preds = preds_attention[mask_agreement]

            if len(selected_labels) == 0:  # If no agreed predictions, continue to the next batch
                rejected += len(images)
                continue

            agreements += mask_agreement.sum().item()  # Update the count of agreement cases
            correct_preds = (selected_preds == selected_labels).float().sum(dim=1)

            total_samples += len(selected_labels)
            total_correct += correct_preds.sum().item()

    accuracy = (total_correct / total_samples) * 100 if total_samples > 0 else 0
    rejection_rate = (rejected / len(dataloader.dataset)) * 100
    agreement_rate = (agreements / len(dataloader.dataset)) * 100  # Calculate agreement rate

    print(f'Accuracy (excluding rejections): {accuracy:.2f}%')
    print(f'Rejection Rate: {rejection_rate:.2f}%')
    print(f'Agreement Rate: {agreement_rate:.2f}%')  # Print agreement rate


if __name__ == '__main__':
    # Define the device for computation (CPU or GPU)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    # Load the trained ResNet50 model without masking components
    resnet50 = models.resnet50(pretrained=False)  # Initialize ResNet50 model
    resnet50.fc = nn.Linear(resnet50.fc.in_features, num_classes)  # Adjust the last layer for num_classes
    PATH_resnet50 = './resnet50.pth'  # Path to the saved model state
    resnet50.load_state_dict(torch.load(PATH_resnet50, map_location=device))  # Load the model state
    resnet50.to(device)  # Move the model to the appropriate device

    # Load the trained AttentionMaskingResNet50 model
    attention_resnet50 = AttentionMaskingResNet50(resnet50, num_classes)  # Use the resnet50 model as a base
    PATH_attention_resnet50 = './attention_resnet50.pth'  # Path to the saved model state
    attention_resnet50.load_state_dict(torch.load(PATH_attention_resnet50, map_location=device))  # Load the model state
    attention_resnet50.to(device)  # Move the model to the appropriate device

    # Test the model with selective classification
    print("\nTesting with selective classification:")
    confidence_threshold = 0.6  # Confidence threshold for selective classification
    selective_classification(resnet50, attention_resnet50, test_loader, confidence_threshold)
