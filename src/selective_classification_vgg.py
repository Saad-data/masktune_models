import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import torchvision
import torch.optim as optim
from vgg import EnhancedVGG  # Assume EnhancedVGG is saved in a file named enhanced_vgg.py


# Function to perform selective classification using two models
def selective_classification(vgg_initial, vgg_finetuned, dataloader, base_confidence_threshold):
    vgg_initial.eval()  # Set the first model to evaluation mode
    vgg_finetuned.eval()  # Set the second model to evaluation mode

    total, correct, rejected, agreements = 0, 0, 0, 0  # Initialize counters

    # Define weights for more difficult classes (bird, cat, deer, dog) to adjust the confidence threshold
    class_difficulty_weights = {3: 0.7, 4: 0.7, 5: 0.7, 6: 0.7}

    with torch.no_grad():  # Disable gradient calculation for evaluation
        for data in dataloader:
            images, labels = data[0].to(device), data[1].to(device)

            # Get the initial model's predictions and their confidence scores
            outputs_initial_tuple = vgg_initial(images)
            outputs_initial = outputs_initial_tuple[0] if isinstance(outputs_initial_tuple,
                                                                     tuple) else outputs_initial_tuple
            softmax_scores_initial = F.softmax(outputs_initial, dim=1)
            _, predictions_initial = torch.max(softmax_scores_initial, 1)

            # Get the fine-tuned model's predictions and their confidence scores
            outputs_finetuned_tuple = vgg_finetuned(images)
            outputs_finetuned = outputs_finetuned_tuple[0] if isinstance(outputs_finetuned_tuple,
                                                                         tuple) else outputs_finetuned_tuple
            softmax_scores_finetuned = F.softmax(outputs_finetuned, dim=1)
            confidences_finetuned, predictions_finetuned = torch.max(softmax_scores_finetuned, 1)

            for idx, (confidence_finetuned, prediction_finetuned) in enumerate(
                    zip(confidences_finetuned, predictions_finetuned)):
                # Adjust confidence threshold for difficult classes
                adjusted_confidence_threshold = base_confidence_threshold * class_difficulty_weights.get(
                    prediction_finetuned.item(), 1)

                if prediction_finetuned == predictions_initial[idx]:  # Check if both models agree on the prediction
                    agreements += 1
                    if confidence_finetuned.item() >= adjusted_confidence_threshold:  # Check if confidence is above threshold
                        total += 1
                        correct += (prediction_finetuned == labels[idx]).item()  # Increment correct predictions
                    else:
                        rejected += 1  # Increment rejected predictions

    # Calculate and print the accuracy, rejection rate, and agreement rate
    accuracy = correct / total if total > 0 else 0
    rejection_rate = rejected / (total + rejected)
    agreement_rate = agreements / len(dataloader.dataset)

    print(f'Accuracy (excluding rejections): {accuracy * 100:.2f}%')
    print(f'Rejection Rate: {rejection_rate * 100:.2f}%')
    print(f'Agreement Rate: {agreement_rate * 100:.2f}%')


if __name__ == '__main__':
    # Define the transformations applied to the test images
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Load the CIFAR-10 test dataset
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=2)

    # Define the class labels for CIFAR-10
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # Initialize the Enhanced VGG models
    model_initial = EnhancedVGG()
    model_finetuned = EnhancedVGG()

    # Define the device for computation (CPU or GPU)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model_initial.to(device)
    model_finetuned.to(device)

    # Load the trained model states
    PATH_INITIAL = './vgg_initial.pth'
    PATH_FINETUNED = './vgg_finetuned.pth'
    model_initial.load_state_dict(torch.load(PATH_INITIAL, map_location=device))
    model_finetuned.load_state_dict(torch.load(PATH_FINETUNED, map_location=device))

    # Test the models with selective classification
    print("\nTesting with selective classification:")
    confidence_threshold = 0.8  # Confidence threshold for selective classification
    selective_classification(model_initial, model_finetuned, testloader, confidence_threshold)
