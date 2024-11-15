"""
Trains a PyTorch image classification model using device-agnostic code.
"""

import os
import torch
import data_setup, engine, model_builder, utils,loss_and_accuracy_curve_plotter,testing
from torchvision import transforms
from timeit import default_timer as timer 
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
from torchinfo import summary
import torch.nn as nn
import torch.nn.init as init


import torchvision


# Setup hyperparameters
NUM_EPOCHS = 50
BATCH_SIZE = 64
LEARNING_RATE = 0.00020703742384961855
# LEARNING_RATE = 1e-04

CONFIG_NAME = 50


# Setup directories
# Data
root_dir = "/home/h6x/git_projects/ornl-svi-data-processing/experiment_2/processed_data_1/npy_combined"
annotation_file_path ="/home/h6x/git_projects/ornl-svi-data-processing/experiment_2/processed_data_1/annotations_2018_npy_2_classes_only_h0h1_90_percentile_random.csv"
# annotation_file_path = "/home/h6x/git_projects/ornl-svi-data-processing/experiment_3/processed_data_1/annotations_2018_npy_5_classes_only_h0h1.csv"



# Model save and plots
model_root_dir = "/home/h6x/git_projects/ornl-overdose-modeling-per-images/Experiment_transfer_learning"
plots_dir = os.path.join(model_root_dir, "plots")


# Setup target device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Create transforms
data_transform = transforms.Compose([
  transforms.ToTensor()
])

# Create DataLoaders with help from data_setup.py
train_dataloader, validation_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(
    annotation_file_path=annotation_file_path,
    root_dir=root_dir,
    transform=data_transform,
    batch_size=BATCH_SIZE
)

# Load the pretrained EfficientNet_B0 model
# weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT
# model = torchvision.models.efficientnet_b0(weights=weights).to(device)
# model = torchvision.models.efficientnet_b0(pretrained=False).to(device)
model = torchvision.models.resnet50(pretrained=False).to(device)

# Step 1: Modify the first convolutional layer to accept 15 input channels
# Get the existing first layer (it's a Conv2d layer)
# first_conv_layer = model.features[0][0] # For EfficientNet

first_conv_layer = model.conv1 # For ResNet

print(first_conv_layer)

# Create a new Conv2d layer with 15 input channels instead of 3
# We will copy the weights for the first 3 channels and initialize the rest
new_conv = nn.Conv2d(
    in_channels=15,  # Change to 15 channels
    out_channels=first_conv_layer.out_channels,
    kernel_size=first_conv_layer.kernel_size,
    stride=first_conv_layer.stride,
    padding=first_conv_layer.padding,
    bias=first_conv_layer.bias
)

# Replace the first layer in EfficientNet
# model.features[0][0] = new_conv

# Replace the first layer in ResNet
model.conv1 = new_conv

# Step 2: Modify the classifier layer to match the number of output classes
# For instance, if you need 2 output classes, modify accordingly
# num_features = model.classifier[1].in_features # this is for EfficientNet
# model.classifier[1] = nn.Linear(num_features, 2)  # Change 2 to the number of your output classes # this is for EfficientNe

num_features = model.fc.in_features # this is for ResNet
model.fc = nn.Linear(num_features, 2)  # Change 2 to the number of your output classes # this is for ResNet


# # Freeze all layers except the first layer and the final classifier layer
# for name, param in model.named_parameters():
#     if "features.0.0" in name or "classifier.1" in name:  # Unfreeze the first and last layers
#         param.requires_grad = True
#     else:
#         param.requires_grad = False  # Freeze all other layers

# Print the modified model
# print(model)

# # Verify the new model summary
# summary(model=model, 
#         input_size=(64, 15, 224, 224), 
#         col_names=["input_size", "output_size", "num_params", "trainable"],
#         col_width=20,
#         row_settings=["var_names"]
# )

model = model.to(device) ###???????????? This needed becase we are changing the model architecture


#######################

# # Set loss and optimizer
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.RMSprop(model.parameters(),
                             lr=LEARNING_RATE)

# # add a optimizer SGD
# # optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)

start_time = timer()

# Start training with help from engine.py
results = engine.train(model=model,
             train_dataloader=train_dataloader,
             validation_dataloader=validation_dataloader,
             loss_fn=loss_fn,
             optimizer=optimizer,
             epochs=NUM_EPOCHS,
             device=device,
             use_mixed_precision=True,
             save_name="se_restnet.pth",
             save_path=os.path.join(model_root_dir, "models/"))

# End the timer and print out how long it took
end_time = timer()
print(f"Total training time: {end_time-start_time:.3f} seconds")
print(f"Total training time: {(end_time-start_time)/60:.3f} minutes")

# Plotting the results
loss_and_accuracy_curve_plotter.plot_loss_curves(results,plots_dir)

# Test the model after training
test_loss, test_acc, y_labels, y_preds  = testing.test_step(model=model,
                                  dataloader=test_dataloader,
                                  loss_fn=loss_fn,
                                  device=device,
                                  use_mixed_precision=True)

# Print out test results
print(
    f"Test results | "
    f"test_loss: {test_loss:.4f} | "
    f"test_acc: {test_acc:.4f}"
)

# Compute confusion matrix
conf_matrix = confusion_matrix(y_labels, y_preds)

# Plot confusion matrix
def plot_confusion_matrix(conf_matrix, class_names):
    plt.figure(figsize=(10, 7))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False, 
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    # plt.show()
    plt.savefig(os.path.join(plots_dir, "confusion_matrix.png"))

# Plot the confusion matrix
plot_confusion_matrix(conf_matrix, class_names)

# Function to calculate TPR and FPR for each class
def calculate_tpr_fpr(conf_matrix):
    num_classes = conf_matrix.shape[0]
    TPR = np.zeros(num_classes)
    FPR = np.zeros(num_classes)
    FNR = np.zeros(num_classes)
    
    for i in range(num_classes):
        TP = conf_matrix[i, i]
        FN = np.sum(conf_matrix[i, :]) - TP
        FP = np.sum(conf_matrix[:, i]) - TP
        TN = np.sum(conf_matrix) - (TP + FN + FP)
        
        TPR[i] = TP / (TP + FN) if (TP + FN) != 0 else 0
        FPR[i] = FP / (FP + TN) if (FP + TN) != 0 else 0
        FNR[i] = FN / (TP + FN) if (TP + FN) != 0 else 0
    
    return TPR, FPR, FNR

# Calculate TPR and FPR
TPR, FPR, FNR = calculate_tpr_fpr(conf_matrix)

# Print TPR and FPR for each class
for idx, class_name in enumerate(class_names):
    print(f"Class {class_name} - TPR: {TPR[idx]:.2f}, FPR: {FPR[idx]:.2f}, FNR: {FNR[idx]:.2f}")

