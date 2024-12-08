#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script for fine-tuning a pre-trained ResNet1D model on your dataset.

Usage:
    python fine_tune_model.py --data_dir ./data/ --checkpoint_path ./History/TFPrediction_checkpoint_WC1.pth --batch_size 32 --epochs 20

"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# **Import Custom Modules**
# Ensure that the 'Models' module is in your Python path
from Models import ResNet1D  # Assuming ResNet1D is defined in Models/ResNet1D.py


class ModelBase(nn.Module):
    """
    Encoder for classification based on ResNet1D.
    """
    def __init__(self, dim=128):
        super(ModelBase, self).__init__()
        self.net = ResNet1D.resnet18(norm_layer=None)  # Encoder using ResNet18 architecture
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(512, dim)  # Adjust '512' if necessary based on encoder's output size

    def forward(self, x):
        x = self.net(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x


def load_pretrained_encoder(model, checkpoint_path):
    """
    Load pre-trained encoder weights into the model.

    Args:
        model (nn.Module): The model to load weights into.
        checkpoint_path (str): Path to the checkpoint file.

    Returns:
        nn.Module: The model with loaded weights.
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state_dict = checkpoint

    # Extract 'encoderT' weights and adjust keys
    encoder_state_dict = {}
    for k in list(state_dict.keys()):
        if k.startswith('encoderT.'):
            new_key = k.replace('encoderT.', '')
            encoder_state_dict[new_key] = state_dict[k]

    # Remove 'fc' layer weights to avoid size mismatch
    encoder_state_dict = {k: v for k, v in encoder_state_dict.items() if not k.startswith('fc.')}

    # Load the encoder weights into the model
    missing_keys, unexpected_keys = model.load_state_dict(encoder_state_dict, strict=False)
    print(f"Missing keys: {missing_keys}")
    print(f"Unexpected keys: {unexpected_keys}")

    # Reinitialize the fully connected layer
    model.fc.weight.data.normal_(mean=0.0, std=0.01)
    model.fc.bias.data.zero_()
    print("\nEncoder loaded....")
    return model


# def prepare_data(data_dir, train_ratio=0.8):
#     """
#     Load and prepare the dataset.

#     Args:
#         data_dir (str): Directory containing 'X_test.npy' and 'y_test.npy'.
#         train_ratio (float): Ratio of data to use for training.

#     Returns:
#         tuple: Training and test DataLoaders.
#     """
#     # Load the data
#     print("\nLoading the data....")
    
#     X_data = np.load(os.path.join(data_dir, 'X_t.npy'), allow_pickle=True)  # Shape: (num_samples, channels, signal_length)
#     y_data = np.load(os.path.join(data_dir, 'y_t.npy'), allow_pickle=True)  # Shape: (num_samples,)
#     # print("Checking for NaNs or Infs in X_data...")
#     # if np.isnan(X_data).any() or np.isinf(X_data).any():
#     #     print("NaNs or Infs detected in X_data.")
#     # else:
#     #     print("X_data is clean.")



#     # Reshape data to (num_samples * num_channels, 1, signal_length)
#     num_samples, num_channels, signal_length = X_data.shape
#     X_data_reshaped = X_data.reshape(-1, 1, signal_length)
#     y_data_repeated = np.repeat(y_data, num_channels)

#     # Shuffle the data
#     num_total_samples = X_data_reshaped.shape[0]
#     indices = np.arange(num_total_samples)
#     np.random.shuffle(indices)
#     X_shuffled = X_data_reshaped[indices]
#     y_shuffled = y_data_repeated[indices]

#     # Split the data
#     num_train = int(train_ratio * num_total_samples)
#     X_train = X_shuffled[:num_train]
#     y_train = y_shuffled[:num_train]
#     X_test = X_shuffled[num_train:]
#     y_test = y_shuffled[num_train:]

#     # Convert data to PyTorch tensors
#     X_train_tensor = torch.from_numpy(X_train).float()
#     y_train_tensor = torch.from_numpy(y_train).long()
#     X_test_tensor = torch.from_numpy(X_test).float()
#     y_test_tensor = torch.from_numpy(y_test).long()

#     # Create Datasets and DataLoaders
#     train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
#     test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

#     return train_dataset, test_dataset

'''Standardization with data preprocessing'''
def prepare_data(data_dir, train_ratio=0.8):
    """
    Load and prepare the dataset with reshaping and standardization.

    Args:
        data_dir (str): Directory containing 'X_t.npy' and 'y_t.npy'.
        train_ratio (float): Ratio of data to use for training.

    Returns:
        tuple: Training and test TensorDatasets.
    """
    # Load the data
    print("\nLoading the data....")
    
    X_data = np.load(os.path.join(data_dir, 'X_t.npy'), allow_pickle=True)  # Shape: (num_samples, channels, signal_length)
    y_data = np.load(os.path.join(data_dir, 'y_t.npy'), allow_pickle=True)  # Shape: (num_samples,)

    # Verify data shapes
    assert X_data.ndim == 3, f"X_data should be 3D, but got {X_data.ndim}D"
    assert y_data.ndim == 1, f"y_data should be 1D, but got {y_data.ndim}D"
    num_samples, num_channels, signal_length = X_data.shape
    print(f"Data shape: {X_data.shape}")
    print(f"Labels shape: {y_data.shape}")

    # Reshape data to (num_samples * num_channels, 1, signal_length)
    X_data_reshaped = X_data.reshape(-1, 1, signal_length)  # New shape: (num_samples * channels, 1, signal_length)
    y_data_repeated = np.repeat(y_data, num_channels)      # Repeat labels accordingly
    print(f"Reshaped X_data shape: {X_data_reshaped.shape}")
    print(f"Reshaped y_data shape: {y_data_repeated.shape}")

    # Shuffle the data
    num_total_samples = X_data_reshaped.shape[0]
    indices = np.arange(num_total_samples)
    np.random.shuffle(indices)
    X_shuffled = X_data_reshaped[indices]
    y_shuffled = y_data_repeated[indices]
    print("Data shuffled.")

    # Split the data
    num_train = int(train_ratio * num_total_samples)
    X_train = X_shuffled[:num_train]
    y_train = y_shuffled[:num_train]
    X_test = X_shuffled[num_train:]
    y_test = y_shuffled[num_train:]
    print(f"Training samples: {X_train.shape[0]}")
    print(f"Testing samples: {X_test.shape[0]}")

    # Compute mean and std from training data (global mean and std)
    mu = X_train.mean()
    sigma = X_train.std()
    print(f"Computed mean (mu): {mu:.4f}")
    print(f"Computed std (sigma): {sigma:.4f}")

    # Avoid division by zero
    if sigma == 0:
        sigma = 1.0
        print("Standard deviation is zero. Setting sigma to 1.0 to avoid division by zero.")

    # Apply standardization
    X_train_std = (X_train - mu) / sigma
    X_test_std = (X_test - mu) / sigma
    print("Standardization applied to training and testing data.")

    # Optional: Verify standardization
    print(f"Training data mean after standardization: {X_train_std.mean():.4f}")
    print(f"Training data std after standardization: {X_train_std.std():.4f}")

    # Convert data to PyTorch tensors
    X_train_tensor = torch.from_numpy(X_train_std).float()
    y_train_tensor = torch.from_numpy(y_train).long()
    X_test_tensor = torch.from_numpy(X_test_std).float()
    y_test_tensor = torch.from_numpy(y_test).long()

    # Create TensorDatasets
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    print("TensorDatasets created.")

    return train_dataset, test_dataset

def train_epoch(model, dataloader, optimizer, criterion, device):
    """
    Train the model for one epoch.

    Args:
        model (nn.Module): The model to train.
        dataloader (DataLoader): DataLoader for training data.
        optimizer (Optimizer): Optimizer for updating weights.
        criterion (Loss): Loss function.
        device (torch.device): Device to run computations on.

    Returns:
        tuple: Training loss and accuracy.
    """
    model.train()
    running_loss = 0.0
    running_corrects = 0

    for inputs, labels in tqdm(dataloader, desc='Training', leave=False):
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # Statistics
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = running_corrects.double() / len(dataloader.dataset)

    return epoch_loss, epoch_acc.item()


def evaluate(model, dataloader, criterion, device):
    """
    Evaluate the model.

    Args:
        model (nn.Module): The model to evaluate.
        dataloader (DataLoader): DataLoader for evaluation data.
        criterion (Loss): Loss function.
        device (torch.device): Device to run computations on.

    Returns:
        tuple: Evaluation loss and accuracy.
    """
    model.eval()
    running_loss = 0.0
    running_corrects = 0

    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc='Evaluating', leave=False):
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            # Statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = running_corrects.double() / len(dataloader.dataset)

    return epoch_loss, epoch_acc.item()


def main():
    """
    Main function to run the training and evaluation.
    """
    # **Parse Command-Line Arguments**
    parser = argparse.ArgumentParser(description='Fine-tune a pre-trained ResNet1D model.')
    parser.add_argument('--data_dir', type=str, default='./data/', help='Directory containing the data.')
    parser.add_argument('--checkpoint_path', type=str, default='./History/TFPrediction_checkpoint_WC1.pth',
                        help='Path to the pre-trained checkpoint.')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training.')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs to train.')
    parser.add_argument('--train_ratio', type=float, default=0.8, help='Ratio of data to use for training.')
    parser.add_argument('--classifier_lr', type=float, default=0.1, help='Learning rate for the classifier layer.')
    parser.add_argument('--backbone_lr', type=float, default=0.01, help='Learning rate for the backbone encoder.')
    parser.add_argument('--num_workers', type=int, default=0, help='Number of workers for data loading.')
    parser.add_argument('--gpu', type=str, default='0', help='GPU device ID to use.')

    args = parser.parse_args()

    # **Set the GPU Device**
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    # **Device Configuration**
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # **Prepare Data**
    print("\nPreparing data...")
    train_dataset, test_dataset = prepare_data(args.data_dir, train_ratio=args.train_ratio)
    print("\nData prepared.")
    # Create DataLoaders
    print("\nCreating DataLoaders...")
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    print("DataLoaders created.")
    # **Initialize the Model**
    num_classes = len(torch.unique(train_dataset.tensors[1]))
    model = ModelBase(dim=num_classes).to(device)

    # **Load the Pre-trained Encoder**
    model = load_pretrained_encoder(model, args.checkpoint_path)
    print(model)

    # **Set Up Optimizer, Loss Function, and Scheduler**
    # Separate parameters for different learning rates
    classifier_parameters = [param for name, param in model.named_parameters() if name.startswith('fc')]
    backbone_parameters = [param for name, param in model.named_parameters() if not name.startswith('fc')]

    # Define optimizer
    optimizer = optim.SGD([
        {'params': classifier_parameters, 'lr': args.classifier_lr},
        {'params': backbone_parameters, 'lr': args.backbone_lr}
    ], momentum=0.9, weight_decay=5e-4)

    # Define loss function
    criterion = nn.CrossEntropyLoss().to(device)

    # Define learning rate scheduler
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # **Training Loop**
    best_acc = 0.0

    for epoch in range(args.epochs):
        print(f'Epoch {epoch+1}/{args.epochs}')

        # Train for one epoch
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)

        # Evaluate on test data
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)

        # Step the scheduler
        lr_scheduler.step()

        # Check if this is the best model so far
        if test_acc > best_acc:
            best_acc = test_acc
            # Save the best model weights
            torch.save(model.state_dict(), 'best_model.pth')

        # Print statistics
        print(f'Train Loss: {train_loss:.4f} Acc: {train_acc:.4f}')
        print(f'Test Loss: {test_loss:.4f} Acc: {test_acc:.4f}')
        print('-' * 30)

    print(f'Best Test Acc: {best_acc:.4f}')


if __name__ == "__main__":
    main()


'''use this command to run this code '''
# python fine_tune_model.py --data_dir /home/dapgrad/tenzinl2/TFPred/raw_data --checkpoint_path ./History/TFPrediction_checkpoint_WC1.pth --batch_size 32 --epochs 20
# python fine_tune_model.py --data_dir /home/dapgrad/tenzinl2/TFPred/raw_data --checkpoint_path ./History/TFPred_checkpoint.pth --batch_size 32 --epochs 20
