import shutil
from datetime import datetime
import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os


HOMEDIR = os.getcwd()


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
                # layers.append(nn.Dropout())
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
    plt.imshow(image)
    plt.title(f'{image_path.split('/')[-1]} - {w}x{h}')
    plt.axis('off')
    plt.show()


def tensor_imshow(tensor_image):
    plt.imshow(tensor_image.permute(1, 2, 0))
    plt.show()

# Function to read the image, split into source and target, and return the source
def read_and_split_image(image_path):
    image = Image.open(image_path).convert('RGB')
    w, h = image.size
    # source_image = image.crop((0, 0, w // 2, h))
    # target_image = image.crop((w // 2, 0, w, h))
    target_image = image.crop((0, 0, w // 2, h))
    source_image = image.crop((w // 2, 0, w, h))
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


def input_file_path(dataset, chosen_sample):
    return os.path.join(HOMEDIR, f'datasets/facades/dataset/{chosen_sample}.jpg')

def result_dir(chosen_sample):
    dir_name = os.path.join(HOMEDIR, f'generated_inrs/{chosen_sample}')
    os.makedirs(dir_name, exist_ok=True)
    return dir_name

def train_inr(dataset, chosen_sample, layer_dims, max_epochs=100, verbose=True):
    image_path = input_file_path(dataset, chosen_sample)
    output_dir = result_dir(chosen_sample)
    shutil.copy(image_path, os.path.join(output_dir, f'{chosen_sample}_original.jpg'))
    source_tensor, target_tensor = read_and_split_image(image_path)
    scale_factor = 0.5
    scale_factor = 1
    source_tensor = source_downsample(source_tensor, scale_factor=scale_factor)
    target_tensor = target_downsample(target_tensor, scale_factor=scale_factor)

    reconstructed_inr_images = {}
    for tensor, title in [(source_tensor, 'src'), (target_tensor, 'tgt')]:
        resolution = (tensor.shape[1], tensor.shape[2])
        if verbose:
            tensor_imshow(tensor)
        trainer = INRTrainer(layer_dims, tensor)
        model = trainer.train_inr(max_epochs=max_epochs)
        reconstructed_inr_images[title] = model.generate_image_tensor(resolution)
        torch.save(model.state_dict(), os.path.join(output_dir, f'{title}_model.pt'))
        # Plot the results
        plot_results(trainer, model, title, resolution, layer_dims, chosen_sample)
        plt.close('all')

    reconstructed_src = reconstructed_inr_images['src']
    reconstructed_tgt = reconstructed_inr_images['tgt']
    stacked_tensor = np.concatenate((reconstructed_src, reconstructed_tgt), axis=2)
    # Save the stacked image as a PNG file
    save_image(torch.tensor(stacked_tensor), os.path.join(output_dir, f'{chosen_sample}_reconstructed.png'))
    return output_dir


def list_all_files_flat(directory):
    return [os.path.join(directory, f) for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]


# Example usage
if __name__ == "__main__":
    datasets = ['train', 'val', 'test']
    samples = {dataset: list_all_files_flat(os.path.join(HOMEDIR, 'datasets/facades')) for dataset in datasets}
    # chosen_sample = '2'

    # layer_dims = [2, 128, 128, 128, 128, 128]  # Example dimensions
    layer_dims = [2] + ([256] * 6)  # Example dimensions
    # layer_dims = [2, 256, 256, 256, 256, 256, 256, 256]  # Example dimensions
    # layer_dims = [2, 256, 256, 256, 256, 256, 256, 256]  # Example dimensions
    # layer_dims = [2, 256, 256, 256]  # Example dimensions

    for dataset, samples in samples:
        for i, sample in enumerate(samples):
            if i > 0:
                break
            sample_num = sample.split('/')[-1].split('.')[0]
            train_inr(dataset, sample_num, layer_dims, max_epochs=100, verbose=True)

    # image_path = input_file_path(chosen_sample)
    # my_imshow(image_path)  # Show the original image with title
