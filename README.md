
# MaskTune Models

This repository contains code for training and evaluating various neural network models on the CIFAR-10 and CelebA datasets. Follow the steps below to use the code.

## How to Use the Code

### 1. Clone the Repository
Start by cloning the repository to your local machine:
```sh
git clone https://github.com/Saad-data/masktune_models.git
```

### 2. Install Required Python Packages
Navigate to the project directory and install the necessary Python packages listed in `requirements.txt`:
```sh
pip install -r requirements.txt
```

### 3. Download and Prepare the CIFAR-10 Dataset
No manual download is needed for the CIFAR-10 dataset. It will be automatically downloaded and saved in the project's `data` folder.

### 4. Train the VGG Model on CIFAR-10
To load and train the VGG model on the CIFAR-10 dataset, run the `vgg.py` script:
```sh
python src/vgg.py
```
You can customize the training process by modifying parameters like `batch_size`, `num_epochs`, `learning_rate`, and `momentum` within the script.

### 5. Evaluate the VGG Model
To evaluate the accuracy of the VGG model before and after fine-tuning, run the `evaluate_vgg.py` script:
```sh
python src/evaluate_vgg.py
```
This will provide the network's accuracy on the entire dataset and on each individual class.

### 6. Run Selective Classification for the VGG Model
To perform selective classification using the VGG model, execute the `selective_classification_vgg.py` script:
```sh
python src/selective_classification_vgg.py
```

### 7. Download and Prepare the CelebA Dataset
1. Download the `img_align_celeba` folder from [this link](https://example.com).
2. Extract the contents, and you should see a folder named `archive`.
3. Move this folder into the project's `data/celeba` directory to allow the network to load the dataset correctly.

### 8. Train the AttentionMaskingResNet50 Model on CelebA
To load and train the AttentionMaskingResNet50 model on the CelebA dataset, run the `resnet50.py` script:
```sh
python src/resnet50.py
```
You can customize the training process by modifying parameters like `batch_size`, `num_epochs`, `learning_rate`, `momentum`, and `num_classes`.

To train the model on a subset of the dataset for computational resource reasons, adjust the `subset_size` variable.

### 9. Run Selective Classification for the AttentionMaskingResNet50 Model
To perform selective classification using the AttentionMaskingResNet50 model, execute the `selective_classification_resnet.py` script:
```sh
python src/selective_classification_resnet.py
```

## Repository Structure
- `data/`: Contains dataset files.
- `src/`: Contains source code files.
- `requirements.txt`: Lists the Python packages required for the project.

## Notes
- Ensure you have the necessary computational resources to train the models, especially for large datasets like CelebA.
- Modify the script parameters according to your preferences and computational constraints.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
