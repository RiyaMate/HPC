import os
import shutil
from glob import glob
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as torchmp
from torch.nn.parallel import DistributedDataParallel as DDP 
from torch.utils.data import DataLoader
from torch.utils.data import random_split, Dataset, DataLoader
from torchvision import transforms as T
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.utils import make_grid
from torchvision.datasets import ImageFolder
from time import time
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2
import warnings

num_gpus = 2
# os.environ['MKL_THREADING_LAYER'] = 'GNU'
# os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
torch.manual_seed(2024)
warnings.filterwarnings("ignore")

# Define the data directory and augmentation directory
data_directory = "/kaggle/input/plant-diseases-training-dataset/data"
aug_directory = "/kaggle/input/augmented-ds/aug"

# Defining the image size, mean, and standard deviation for normalizing
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
im_size = 224

# set batch size
batch_size = 32

# Defining the transformation function to apply on images
tfs = T.Compose([
    T.Resize((im_size, im_size)), 
    T.ToTensor(), 
    T.Normalize(mean = mean, std = std)
])

class BotanyDataset(Dataset):
    def __init__(self, roots, has_aug_directory = False, transformations = None):
        self.transformations = transformations
        self.class_names = {}
        self.class_counts = {}
        count = 0

        # Determine which directories to include based on `has_aug_directory`
        if has_aug_directory:
            # Include all roots since roots contains data_directory and aug_directory
            included_roots = roots
        else:
            # Only include the root directory
            included_roots = [roots]

        self.image_paths = []
        temp_path = []

        for root in included_roots:
            temp_path = sorted(glob(f"{root}/*/*"))
            for _ , image_path in enumerate(temp_path):
                class_name = self.get_classname(image_path)
                if class_name not in self.class_names:
                    self.class_names[class_name] = count
                    self.class_counts[class_name] = 1
                    count += 1
                else:
                    self.class_counts[class_name] += 1
            self.image_paths.extend(temp_path)
            
    def get_classname(self, path): 
        return os.path.dirname(path).split("/")[-1]
    
    def __len__(self): 
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")
        ground_truth = self.class_names[self.get_classname(image_path)]
        
        if self.transformations is not None: 
            image = self.transformations(image)
        
        return image, ground_truth

def get_dls(rank, world_size, root, aug_root, has_aug_directory = False, transformations = None, batch_size = None, split = [0.9, 0.05, 0.05], workers = 4):
    # Create dataset with original and augmented directories
    ds = BotanyDataset(
        roots = [root, aug_root], 
        has_aug_directory = has_aug_directory, 
        transformations = transformations
    )
    
    # Calculate the lengths of respective sets
    dataset_length = len(ds)
    train_dataset_length = int(dataset_length * split[0])
    validation_dataset_length = int(dataset_length * split[1])
    test_dataset_length = dataset_length - (train_dataset_length + validation_dataset_length)

    # print("length of dataset:", dataset_length)
    # print("length of training_data:", train_dataset_length)
    # print("length of validation_data:", validation_dataset_length)
    # print("length of testing_data:", test_dataset_length)

    # Split the dataset
    train_dataset, validation_dataset, test_dataset = random_split(
        dataset=ds,
        lengths=[train_dataset_length, validation_dataset_length, test_dataset_length],
        generator=torch.Generator().manual_seed(42)
    )

    # Create data samplers for each GPU
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset,
        num_replicas = world_size,
        rank = rank,
        shuffle = True
    )
    val_sampler = torch.utils.data.distributed.DistributedSampler(
        validation_dataset,
        num_replicas = world_size,
        rank = rank,
        shuffle = True
    )
    
    # Create DataLoaders for respective sets
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=workers
    )
    validation_dataloader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        sampler=val_sampler,
        num_workers=workers
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=workers
    )
    
    return train_dataloader, validation_dataloader, test_dataloader, ds.class_names

# Get data loaders for each GPU rank
train_dataloader, validation_dataloader, test_dataloader, classes = get_dls(
    rank = 0,
    world_size = num_gpus,
    root = data_directory,
    aug_root = aug_directory,
    has_aug_directory = True,
    transformations = tfs,
    batch_size = 32
)

# print("Number of samples in training data: ", len(train_dataloader))
# print("Number of samples in validation data: ", len(validation_dataloader))
# print("Number of samples in testing data: ", len(test_dataloader))

class ResidualBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        
    def forward(self, x):
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.conv2(out)
        return self.relu2(out) + x # ReLU can be applied before or after adding the input

# Calculate the accuracy
def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

# Architecture for training
# convolution block with BatchNormalization
def ConvolutionalBlock(in_channels, out_channels, pool=False):
    layers = [
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    ]

    if pool:
        layers.append(nn.MaxPool2d(4))

    return nn.Sequential(*layers)

# ResNet9 model definition
class ResNet9(nn.Module):
    def __init__(self, in_channels, num_infection_classes):
        super().__init__()
        
        self.conv1 = ConvolutionalBlock(in_channels, 64)
        self.conv2 = ConvolutionalBlock(64, 128, pool=True)  # out_dim: 128 x 128 x 128 
        self.res1 = nn.Sequential(ConvolutionalBlock(128, 128), ConvolutionalBlock(128, 128))
        
        self.conv3 = ConvolutionalBlock(128, 256, pool=True)  # out_dim: 256 x 64 x 64
        self.conv4 = ConvolutionalBlock(256, 512, pool=True)  # out_dim: 512 x 32 x 32
        self.res2 = nn.Sequential(ConvolutionalBlock(512, 512), ConvolutionalBlock(512, 512))
        
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, num_infection_classes)
        )
    
    def forward(self, xb):  # xb is the loaded batch
        out = self.conv1(xb)
        out = self.conv2(out)
        out = self.res1(out)
        out = F.interpolate(out, scale_factor=2, mode='nearest')  # Upsample
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out)
        out = F.interpolate(out, scale_factor=2, mode='nearest')  # Upsample
        out = self.classifier(out)
        return out
    
def training_step(model, batch, device):
    images, labels = batch
    images = images.to(device)
    labels = labels.to(device)
    out = model(images)                  # Generate predictions
    loss = F.cross_entropy(out, labels)  # Calculate loss
    return loss
    
def validation_step(model, batch, device):
    images, labels = batch
    images = images.to(device)
    labels = labels.to(device)
    out = model(images)                   # Generate prediction
    loss = F.cross_entropy(out, labels)   # Calculate loss
    acc = accuracy(out, labels)           # Calculate accuracy
    return {"val_loss": loss.detach(), "val_accuracy": acc}

def validation_epoch_end(outputs, model):
    batch_losses = [x["val_loss"] for x in outputs]
    batch_accuracy = [x["val_accuracy"] for x in outputs]
    epoch_loss = torch.stack(batch_losses).mean()       # Combine loss
    epoch_accuracy = torch.stack(batch_accuracy).mean()
    return {"val_loss": epoch_loss, "val_accuracy": epoch_accuracy}  # Combine accuracies

def epoch_end(epoch, result, rank):
    print("Epoch [{}], \
          last_lr: {:.5f}, \
          train_loss: {:.4f}, \
          val_loss: {:.4f}, \
          val_acc: {:.4f}" .format(
              epoch + 1,
              result['lrs'][-1],
              result['train_loss'],
              result['val_loss'],
              result['val_accuracy']
            )
    )
    
    # Write this data to file for future processing
    output_string = "{}, {:.5f}, {:.4f}, {:.4f}, {:.4f}".format(
        epoch + 1,
        result['lrs'][-1],
        result['train_loss'],
        result['val_loss'],
        result['val_accuracy']
    )
    file_name = 'rank0_output.txt' if rank == 0 else 'rank1_output.txt'
    with open(file_name, 'a') as file:
            file.write(output_string + "\n")

# for training
@torch.no_grad()
def evaluate(model, val_loader, device):
    model.to(device)
    model.eval()
    outputs = [validation_step(model.module, batch, device) for batch in val_loader]
    return validation_epoch_end(outputs, model)

def get_learning_rate(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def train_model(
        rank,
        epochs, 
        max_lr, 
        model, 
        train_loader, 
        val_loader, 
        weight_decay = 0, 
        grad_clip = None, 
        opt_func = torch.optim.SGD,
        world_size = 1
    ):
    # Initialize the process group for DDP.
    print(f"Setting up DDP on rank: {rank}")
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(
        'nccl',
        rank = rank,
        world_size = world_size
    )

    print(f"Moving model to rank: {rank}")
    device = torch.device(f"cuda:{rank}")
    model.to(device)
    model = DDP(model, device_ids=[rank])
    print(f"Moving completed on rank: {rank}")
    
    history = []
    optimizer = opt_func(model.parameters(), max_lr, weight_decay = weight_decay)
    _scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr, epochs = epochs, steps_per_epoch = len(train_loader))

    print(f"Starting training on rank: {rank}")
    for epoch in range(epochs):
        # Track epoch start time
        epoch_start_time = time()
        
        # Train the model
        model.train()
        train_losses = []
        lrs = []
        progress_bar = tqdm(train_loader, desc=f'Rank: {rank} Epoch {epoch + 1}/{epochs}', leave=False)
        
        for batch in progress_bar:
            loss = training_step(model.module, batch, device)
            train_losses.append(loss)
            loss.backward()
            
            # Gradient clipping
            if grad_clip: 
                nn.utils.clip_grad_value_(model.parameters(), grad_clip)
                
            optimizer.step()
            optimizer.zero_grad()
            
            # Record and update learning rates
            lrs.append(get_learning_rate(optimizer))
            _scheduler.step()

            # Update progress bar with loss
            progress_bar.set_postfix({'loss': loss.item()})

        # Calculate epoch duration
        epoch_end_time = time()
        epoch_duration = (epoch_end_time - epoch_start_time) 

        # Validate the model
        result = evaluate(model, val_loader, device)
        result['train_loss'] = torch.stack(train_losses).mean().item()
        result['lrs'] = lrs
        epoch_end(epoch, result, rank)
        history.append(result)

        # Display epoch duration in progress bar
        progress_bar.set_postfix({
            'epoch_duration': epoch_duration
        })

    # model_save_path = os.path.join(os.getcwd(), "saved_models", "trained_model.pth")
    # torch.save(model.state_dict(), model_save_path)

    dist.destroy_process_group()

def main(world_size):

    # Setup Hyperparameters and initialize the model
    world_size = num_gpus
    epochs = 5
    max_lr = 0.01
    grad_clip = 0.1
    weight_decay = 1e-4
    opt_func = torch.optim.Adam
    model = ResNet9(3, len(classes))
    
    # Initialize start time
    start_time = time()

    # Maintain the spawned process list
    processes = []
    for rank in range(world_size):
        p = torchmp.Process(target=train_model, args=(
            rank,
            epochs,
            max_lr,
            model,
            train_dataloader,
            validation_dataloader,
            grad_clip,
            weight_decay,
            opt_func,
            world_size
        ))
        p.start()
        processes.append(p)

    # Wait for processes to finish
    for p in processes:
        p.join()

    # Save end time
    end_time = time()
        
    # Calculate elapsed time to minutes
    training_time_seconds = end_time - start_time
    training_time_minutes = training_time_seconds / 60
        
    print(f"Overall Training Time: {training_time_seconds:.2f} seconds ({training_time_minutes:.2f} minutes)")

if __name__ == "__main__":
    
    # Set Pytorch multiprocessing to spawn processes
    torchmp.set_start_method('spawn')
    
    # Check number of GPUs available and list them
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print("Number of available GPUs:", num_gpus)
        for i in range(num_gpus):
            print("GPU", i, ":", torch.cuda.get_device_name(i))
            
    # Initiate training process with DDP
    main(world_size = num_gpus)
    