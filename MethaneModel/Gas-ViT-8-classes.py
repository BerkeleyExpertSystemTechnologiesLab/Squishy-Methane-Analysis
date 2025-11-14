#!/usr/bin/env python
# coding: utf-8

# ### This is a Pytorch version of the work, for easier time working with ViT

# # Step 1: Load and Preprocess the Dataset
# 
# ### Load the GasVid dataset
# ### Preprocess the data
# ### Split the dataset into training and test sets

# In[2]:


get_ipython().run_cell_magic('capture', '', '%pip install --upgrade pip\n%pip install -r ../requirements.txt\n')


# In[3]:


# if apple and want MPS acceleration do this
# %%capture
# %pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cpu


# In[4]:


# Imports

import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch
import torch.nn as nn
from torchvision import transforms
from transformers import ViTForImageClassification, ViTImageProcessor, ViTConfig
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm.notebook import tqdm
import pandas as pd

from sklearn.metrics import RocCurveDisplay, roc_curve, ConfusionMatrixDisplay, confusion_matrix

from sklearn.mixture import GaussianMixture


# ### Setting up Directories

# In[5]:


# get generic path to directory
dir_path = os.path.dirname(os.path.realpath("__file__"))

# get all raw video data directories
data_dir = os.path.join(dir_path, 'data')

train_data_dir = os.path.join(data_dir, 'train')
test_data_dir = os.path.join(data_dir, 'test')

frame_data_dir = os.path.join(dir_path, 'background_sub_movingavg8_frames')
frame_train_data_dir = os.path.join(frame_data_dir, 'train')
frame_test_data_dir = os.path.join(frame_data_dir, 'test')

for i in range(8):
    train_count = 0
    for file in os.listdir(os.path.join(frame_train_data_dir, "C%d"%i)):
        train_count += 1
    print ("C%d Training Data Count: "%i, train_count, flush=True)
    test_count = 0
    for file in os.listdir(os.path.join(frame_test_data_dir, "C%d"%i)):
        test_count += 1
    print ("C%d Testing Data Count: "%i, test_count, flush=True)


# # Step 2: Create Dataset for Ingesting Image Frames

# In[6]:


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
        image = cv2.imread(frame_path)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB

        if self.transform:
            image = self.transform(image)
        
        if self.processor:
            image = self.processor.preprocess(image, return_tensors="pt")

        label = self.labels[idx]

        return image, label


# In[7]:


image_processor = ViTImageProcessor(
    "google/vit-base-patch16-224",
    do_normalize=True,
    max_size=384,
    pad_to_max_size=True
)


# In[8]:


# define some transforms
transform = transforms.Compose([
])


# In[9]:


full_train_dataset = MultiClassVideoFrameDataset(root_dir=frame_train_data_dir, transform=transform, processor=image_processor)
test_dataset = MultiClassVideoFrameDataset(root_dir=frame_test_data_dir, transform=transform, processor=image_processor)

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


# In[10]:


len(full_train_dataset) + len(test_dataset)


# # Step 3: Build the GasViT Architecture
# 
# ### Define the GasNet architecture (GasNet-2 as mentioned in the paper)
# ### Implement the model using TensorFlow/Keras

# In[11]:


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device
# model.to(device)


# In[12]:


configs = ViTConfig(
    hidden_dropout_prob=0.5,
    attention_probs_dropout_prob=0.2,
    num_labels=8
)


# In[13]:


# Initialize the ViT feature extractor and model
model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224', config=configs, ignore_mismatched_sizes=True)
model.to(device)


# In[14]:


# Change classifier layer to have num classes consistent with dataset
model.classifier.out_features = len(full_train_dataset.classes)
model.classifier


# In[15]:


train_dataloader.dataset.dataset.classes


# In[16]:


def train(model, weight=None, num_epochs=10):
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

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item():.4f}", flush=True)

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
        print(f"Validation Accuracy: {validation_accuracy:.4f}", flush=True)


# ### Adjust classweights to account for class imbalance

# In[17]:


# Adjust Class weights here
# class_weight = torch.tensor([1]*8).float().to(device)
class_weight = torch.tensor([1, 1, 1, 1, 1, 1, 1, 1]).float().to(device)


# In[18]:


train(model, class_weight, 6)


# In[19]:


# Specify the file path for saving the model
model_path = 'vit_model_8.pth'

# Save the model's state_dict to the specified file
torch.save(model.state_dict(), model_path)


# In[20]:


model_path = 'vit_model_8.pth'
model.load_state_dict(torch.load(model_path))


# # Step 4: Evaluate the model on the Test Dataset
# 
# ### Generate evaluation metrics and plots such as confusion matrix and ROC curves, F1 score, etc.

# We are primarily concerned with high false positive rate due to the extreme class imbalance

# In[21]:


def predict(model):
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
    print(f"Test Accuracy: {validation_accuracy:.4f}", flush=True)
    return predictions, truth_labels


# In[22]:


predictions, truth_labels = predict(model)
predictions, truth_labels = np.array(predictions), np.array(truth_labels)
df_predictions = pd.DataFrame(data={"predictions": predictions, "truth_labels": truth_labels})
df_predictions.to_csv("vit_8_preds.csv", index=False)


# In[23]:


from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

#source: https://vitalflux.com/python-draw-confusion-matrix-matplotlib/

conf_matrix = confusion_matrix(y_true=truth_labels.astype(int), y_pred=predictions)
fig, ax = plt.subplots(figsize=(7.5, 7.5))
ax.matshow(conf_matrix, cmap=plt.cm.Blues, alpha=0.3)
for i in range(conf_matrix.shape[0]):
    for j in range(conf_matrix.shape[1]):
        ax.text(x=j, y=i,s=conf_matrix[i, j], va='center', ha='center', size='xx-large')
 
plt.xlabel('Predictions', fontsize=18)
plt.ylabel('Actuals', fontsize=18)
plt.title(f'Confusion Matrix (0: Leak, 1:Nonleak)', fontsize=18)
plt.savefig("vit_8_confusion.png")
plt.show()


# In[24]:


from sklearn.metrics import accuracy_score,f1_score
accuracy = accuracy_score(truth_labels, predictions)


# In[25]:


print("accuracy: ", accuracy, flush=True)


# In[26]:


f1 = f1_score(truth_labels, predictions)
print("f1 score: ", f1, flush=True)


# In[27]:


conf_matrix = confusion_matrix(truth_labels, predictions)
per_class_accuracy = conf_matrix.diagonal() / conf_matrix.sum(axis=1)


# In[28]:


for label, acc in zip(['C6', 'C7', 'C2', 'C1', 'C5', 'C4', 'C3', 'C0'], per_class_accuracy):
    print(f"Class '{label}' Accuracy: {acc:.4f}", flush=True)


# 

# 
