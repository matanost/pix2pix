import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os


# Define the INR MLP Model
class INRModel(nn.Module):
    def __init__(self, layer_dims):
        super(INRModel, self).__init__()
        layers = []
        for i in range(len(layer_dims) - 1):
            layers.append(nn.Linear(layer_dims[i], layer_dims[i + 1]))
            if i < len(layer_dims) - 2:
                layers.append(nn.ReLU())
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)

    def generate_image(self, resolution):
        h, w = resolution
        # Create a grid of coordinates
        coords = torch.stack(torch.meshgrid(torch.linspace(0, 1, w), torch.linspace(0, 1, h)), -1).reshape(-1, 2)

        # Pass through the model
        with torch.no_grad():
            colors = self(coords).reshape(h, w, 3).cpu().numpy()

        # Clip the colors to be in the valid range [0, 1]
        colors = np.clip(colors, 0, 1)

        # Plot and save the image
        plt.imshow(colors)
        plt.axis('off')
        plt.savefig('output_image.png', bbox_inches='tight', pad_inches=0)
        plt.show()


# Define the Image Dataset
class ImageDataset(Dataset):
    def __init__(self, image_tensor):
        self.image_tensor = image_tensor
        self.h, self.w = image_tensor.shape[1], image_tensor.shape[2]

    def __len__(self):
        return self.h * self.w

    def __getitem__(self, idx):
        x = idx % self.w
        y = idx // self.w
        coords = torch.tensor([x / self.w, y / self.h], dtype=torch.float32)
        color = self.image_tensor[:, x, y]  # Corrected indexing
        return coords, color


# Define the PyTorch Lightning module
class INRTrainer(pl.LightningModule):
    def __init__(self, layer_dims, image_tensor):
        super(INRTrainer, self).__init__()
        self.model = INRModel(layer_dims)
        self.criterion = nn.L1Loss()
        self.dataset = ImageDataset(image_tensor)
        self.train_losses = []

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        coords, colors = batch
        preds = self(coords)
        loss = self.criterion(preds, colors)
        self.train_losses.append(loss.item())
        self.log('train_loss', loss, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def train_dataloader(self):
        return DataLoader(self.dataset, batch_size=32, shuffle=True)

    def train_inr(self, max_epochs=100):
        trainer = pl.Trainer(max_epochs=max_epochs)
        trainer.fit(self)
        self.plot_loss()
        return self.model

    def plot_loss(self):
        plt.figure()
        plt.plot(self.train_losses, label='L1 Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss Over Epochs')
        plt.legend()
        plt.show()


# Function to show the original image
def my_imshow(image_path):
    image = Image.open(image_path)
    h, w = image.size
    # Extract the part of the path starting from pix2pix
    relative_path = os.path.join('pix2pix', os.path.relpath(image_path, start=image_path.split('pix2pix')[0]))
    plt.imshow(image)
    plt.title(f'{relative_path} - {w}x{h}')
    plt.axis('off')
    plt.show()


# Function to read the image, split into source and target, and return the source
def read_and_split_image(image_path):
    image = Image.open(image_path).convert('RGB')
    w, h = image.size
    source_image = image.crop((0, 0, w // 2, h))
    target_image = image.crop((w // 2, 0, w, h))
    source_tensor = transforms.ToTensor()(source_image)
    target_tensor = transforms.ToTensor()(target_image)
    return source_tensor, target_tensor


# Example usage
if __name__ == "__main__":
    image_path = '/home/matano/pix2pix/datasets/facades/train/1.jpg'
    my_imshow(image_path)  # Show the original image with title

    # Read and split image into source and target
    source_tensor, target_tensor = read_and_split_image(image_path)

    layer_dims = [2, 128, 128, 128, 3]  # Example dimensions
    model = INRTrainer(layer_dims, source_tensor).train_inr(max_epochs=100)

    # Generate an image from the trained INR
    resolution = (256, 256)  # Example resolution
    model.generate_image(resolution)