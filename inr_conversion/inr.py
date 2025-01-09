from datetime import datetime
import torch
from matplotlib.scale import scale_factory
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
    def __init__(self, layer_dims, encoding='RGB'):
        if encoding == 'RGB':
            layer_dims.append(2 * 3)
        super(INRModel, self).__init__()
        layers = []
        for i in range(len(layer_dims) - 1):
            layers.append(nn.Linear(layer_dims[i], layer_dims[i + 1]))
            if i < len(layer_dims) - 2:
                layers.append(nn.ReLU())
        self.mlp = nn.Sequential(*layers)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, x):
        reshaped = torch.reshape(self.mlp(x), (-1, 3, 2))
        return self.softmax(reshaped)[:, :, 0]

    def generate_image_tensor(self, resolution):
        h, w = resolution
        # Create a grid of coordinates
        coords = torch.stack(torch.meshgrid(torch.linspace(0, 1, w), torch.linspace(0, 1, h)), -1).reshape(-1, 2)

        # Pass through the model
        with torch.no_grad():
            colors = self(coords).reshape(h, w, 3).cpu().numpy()

        # Clip the colors to be in the valid range [0, 1]
        colors = np.clip(colors, 0, 1)

        return colors


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
        color = self.image_tensor[:, x, y]
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
        return self.model


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


def tensor_imshow(tensor_image):
    plt.imshow(tensor_image.permute(1, 2, 0))
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


# Function to plot results
def plot_results(trainer, model, title, resolution, layer_dims, chosen_sample):
    # Plot the training loss
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 2, 1)
    plt.plot(trainer.train_losses, label='L1 Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Epochs')
    plt.legend()

    # Show the original image
    plt.subplot(2, 2, 2)
    plt.imshow(trainer.dataset.image_tensor.permute(1, 2, 0).cpu().numpy())
    plt.title('Original Image')
    plt.axis('off')

    # Generate the INR image
    inr_image = model.generate_image_tensor(resolution)

    # Show the INR image
    plt.subplot(2, 2, 3)
    plt.imshow(inr_image)
    plt.title('INR Image')
    plt.axis('off')

    # Compute and show the heatmap of L1 pixel distance
    original_image = trainer.dataset.image_tensor.permute(1, 2, 0).cpu().numpy()
    l1_distance = np.abs(original_image - inr_image).sum(axis=-1)
    plt.subplot(2, 2, 4)
    plt.imshow(l1_distance, cmap='hot', interpolation='nearest')
    plt.title('Heatmap of L1 Pixel Distance')
    plt.colorbar()
    plt.axis('off')

    plt.tight_layout()
    now = datetime.now()
    date = now.date().isoformat()
    dir_name = os.path.join('results', str(chosen_sample), date)
    os.makedirs(dir_name, exist_ok=True)
    file_name = f'{"-".join(map(str, layer_dims))}_{now.strftime('%H-%m-%s')}-{title}.png'
    plt.savefig(os.path.join(dir_name, file_name))
    plt.show()

    print(f'original shape={original_image.shape}')
    print(f'reconstructed shape={inr_image.shape}')


def target_downsample(target_tensor, scale_factor=0.125):
    return F.interpolate(target_tensor.unsqueeze(0), scale_factor=(scale_factor, scale_factor), mode='nearest-exact').squeeze(0)


def source_downsample(source_tensor, scale_factor=0.125):
    return F.interpolate(source_tensor.unsqueeze(0), scale_factor=(scale_factor, scale_factor), mode='nearest-exact').squeeze(0)


# Example usage
if __name__ == "__main__":
    chosen_sample = '2'
    image_path = f'/home/matano/pix2pix/datasets/facades/train/{chosen_sample}.jpg'
    my_imshow(image_path)  # Show the original image with title

    # Read and split image into source and target
    source_tensor, target_tensor = read_and_split_image(image_path)

    scale_factor = 0.5
    source_tensor = source_downsample(source_tensor, scale_factor=scale_factor)
    target_tensor = target_downsample(target_tensor, scale_factor=scale_factor)
    tensor_imshow(source_tensor)
    tensor_imshow(target_tensor)

    # layer_dims = [2, 128, 128, 128, 128, 128]  # Example dimensions
    # layer_dims = [2, 256, 256, 256, 256, 256, 256, 256]  # Example dimensions
    layer_dims = [2, 256, 256, 256, 256, 256, 256, 256]  # Example dimensions
    # layer_dims = [2, 256, 256, 256]  # Example dimensions
    for tensor, title in [(source_tensor, 'src'), (target_tensor, 'tgt')]:
        trainer = INRTrainer(layer_dims, tensor)
        model = trainer.train_inr(max_epochs=100)
        # Plot the results
        resolution = (tensor.shape[1], tensor.shape[2])
        plot_results(trainer, model, title, resolution, layer_dims, chosen_sample)
        plt.close('all')
