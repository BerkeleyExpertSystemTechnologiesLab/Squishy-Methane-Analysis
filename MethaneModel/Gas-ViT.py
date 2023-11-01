import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch
import torch.nn as nn
import pandas as pd
import torchvision.transforms as transforms
from PIL import Image
from transformers import ViTForImageClassification, ViTImageProcessor, ViTConfig
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix


class MultiClassVideoFrameDataset(Dataset):
    def __init__(self, root_dir, transform=None, processor=None):
        self.root_dir = root_dir
        self.transform = transform
        self.processor = processor
        self.classes = os.listdir(root_dir)  # Get class names from subdirectories

        self.frames = []
        self.labels = []

        for class_idx, class_name in enumerate(self.classes):
            class_dir = os.path.join(self.root_dir, class_name)
            frame_list = [os.path.join(class_dir, file) for file in os.listdir(class_dir) if file.endswith(('.jpg', '.png', '.jpeg'))]
            self.frames.extend(frame_list)
            self.labels.extend([class_idx] * len(frame_list))

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        frame_path = self.frames[idx]
        image = Image.open(frame_path)

        if self.transform:
            image = self.transform(image)
        
        image = image.convert('RGB') # ViTImageProcessor expects 3 channel data
        
        if self.processor:
            image = self.processor.preprocess(image, return_tensors="pt")

        label = self.labels[idx]

        return image, label
    

def train(model, train_dataloader, val_dataloader, device, weight=None, num_epochs=10):
    criterion = nn.CrossEntropyLoss(weight=weight) # extendable for multiclass classification as well
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # can try out lr scheduler later if needed
    # can also try out warmup ratio

    for epoch in range(num_epochs):
        model.train()
        for batch_images, batch_labels in tqdm(train_dataloader):
            batch_image_pixels, batch_labels = batch_images.pixel_values.squeeze(1).to(device), batch_labels.to(device)
            optimizer.zero_grad()
            outputs = model(batch_image_pixels).logits
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item():.4f}")

        model.eval()
        accuracy = 0
        total_samples = 0

        with torch.no_grad():
            for batch_images, batch_labels in tqdm(val_dataloader, leave=False):
                batch_image_pixels, batch_labels = batch_images.pixel_values.squeeze(1).to(device), batch_labels.to(device)
                outputs = model(batch_image_pixels).logits
                _, predicted = torch.max(outputs, 1)
                accuracy += (predicted == batch_labels).sum().item()
                total_samples += batch_labels.size(0)

        validation_accuracy = accuracy / total_samples
        print(f"Validation Accuracy: {validation_accuracy:.4f}")


def predict(model, test_dataloader, device):
    model.eval()
    accuracy = 0
    total_samples = 0
    predictions = []  # List to store the predictions
    truth_labels = []  # List to store the truth labels

    with torch.no_grad():
        for batch_images, batch_labels in tqdm(test_dataloader, leave=False):
            batch_image_pixels, batch_labels = batch_images.pixel_values.squeeze(1).to(device), batch_labels.to(device)
            outputs = model(batch_image_pixels).logits
            _, predicted = torch.max(outputs, 1)
            accuracy += (predicted == batch_labels).sum().item()
            total_samples += batch_labels.size(0)

            predictions.extend(predicted.cpu().numpy())
            truth_labels.extend(batch_labels.cpu().numpy())

    validation_accuracy = accuracy / total_samples
    print(f"Test Accuracy: {validation_accuracy:.4f}")
    return np.array(predictions), np.array(truth_labels)

def main():
    dir_path = os.path.dirname(os.path.realpath("__file__"))

    # get all raw video data directories
    data_dir = os.path.join(dir_path, 'data')

    train_data_dir = os.path.join(data_dir, 'train')
    test_data_dir = os.path.join(data_dir, 'test')

    frame_data_dir = '/home/bestlab/Desktop/Squishy-Methane-Analysis/0 - GasNet/frame_data_movingAvg'
    frame_train_data_dir = os.path.join(frame_data_dir, 'train')
    frame_test_data_dir = os.path.join(frame_data_dir, 'test')

    image_processor = ViTImageProcessor(
        "google/vit-base-patch16-224",
        do_normalize=True,
        max_size=384,
        pad_to_max_size=True
    )

    # define some transforms
    train_transform = transforms.Compose([
        # transforms.RandomHorizontalFlip(),  # Randomly flip the image horizontally
        # transforms.RandomResizedCrop(200),  # Randomly resize and crop the image
        # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Randomly adjust brightness, contrast, saturation, and hue
        # transforms.RandomApply([
        #     transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0))
        #     ], p=0.2),  # Apply random Gaussian blur
        # transforms.RandomPerspective(distortion_scale=0.3, p=0.4),
        # transforms.ToTensor()
    ])

    full_train_dataset = MultiClassVideoFrameDataset(root_dir=frame_train_data_dir, transform=train_transform, processor=image_processor)
    test_dataset = MultiClassVideoFrameDataset(root_dir=frame_test_data_dir, transform=None, processor=image_processor)

    # Define the percentage of data to use for validation
    validation_split = 0.2  # Adjust this as needed

    # Calculate the number of samples for the validation set
    num_samples = len(full_train_dataset)
    num_val_samples = int(validation_split * num_samples)
    num_train_samples = num_samples - num_val_samples

    # Create a list of indices for the full dataset
    indices = list(range(num_samples))

    # Use random sampling to split the indices into train and validation indices
    val_indices = torch.randperm(num_samples)[:num_val_samples]
    train_indices = list(set(indices) - set(val_indices))

    # Create Subset objects for train and validation
    train_dataset = Subset(full_train_dataset, train_indices)
    val_dataset = Subset(full_train_dataset, val_indices)

    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    print('Length of Entire Dataset:', len(full_train_dataset) + len(test_dataset))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    configs = ViTConfig(
        hidden_dropout_prob=0.5,
        attention_probs_dropout_prob=0.2
    )

    # Initialize the ViT feature extractor and model
    model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224', config=configs, ignore_mismatched_sizes=True)
    model.to(device)

    # Change classifier layer to have num classes consistent with dataset
    model.classifier.out_features = len(full_train_dataset.classes)

    class_weight = torch.tensor([1, 6]).float().to(device)

    train(model, train_dataloader, val_dataloader, device, class_weight, 10)

    model_name = 'no_clsweight_c0'
    # Specify the file path for saving the model
    model_path = f'vit_model_{model_name}.pth'

    # Save the model's state_dict to the specified file
    torch.save(model.state_dict(), model_path)

    predictions, truth_labels = predict(model, test_dataloader, device)

    df = pd.DataFrame({
        'ID': range(1, len(predictions) + 1),
        'Prediction': predictions,
        'Label': truth_labels})
    
    df.to_csv(f'predictions_{model_name}.csv', index=False)

    accuracy = accuracy_score(truth_labels, predictions)
    f1 = f1_score(truth_labels, predictions)
    print('Overall Accuracy:', accuracy)
    print('F1 Score:', f1)
    conf_matrix = confusion_matrix(truth_labels, predictions)
    per_class_accuracy = conf_matrix.diagonal() / conf_matrix.sum(axis=1)
    for label, acc in zip(['Leak', 'Nonleak'], per_class_accuracy):
        print(f"Class '{label}' Accuracy: {acc:.4f}")


if __name__ == "__main__":
    main()
