import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.optim import Adam
import h5py
from vqvae import VQVAE_LunarLander  # Assuming this is the module containing your model

# Custom Dataset for loading saved observations, actions, and rewards from HDF5
class LunarLanderHDF5Dataset(Dataset):
    def __init__(self, hdf5_file, data_dir):
        """
        Args:
            hdf5_file (str): Path to the HDF5 file where the observation, action, and reward data are stored.
        """
        self.hdf5_file = hdf5_file
        self.hf = h5py.File(self.hdf5_file, 'r')

        # Total number of observations
        self.total_obs = len(self.hf['image_observations'])
        print(self.total_obs)

        # Transformations for image preprocessing
        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.Resize(size=(64, 64))
        ])
        
        with open(os.path.join(data_dir, "metadata.txt"), "r") as f:
            self.metadata = int(f.readline()[8:])

    def __len__(self):
        return self.metadata

    def __getitem__(self, idx):
        # Load observation, action, and reward from HDF5
        
        file_idx = idx // (self.metadata // self.total_obs)
        in_file_idx = idx // self.total_obs
        
        image_obs = self.hf[f'image_observations'][f'image_observations_{file_idx}'][in_file_idx]
        raw_obs = self.hf[f'raw_observations'][f'raw_observations_{file_idx}'][in_file_idx]
        actions = self.hf[f'actions'][f'actions_{file_idx}'][in_file_idx]
        rewards = self.hf[f'rewards'][f'rewards_{file_idx}'][in_file_idx]

        # Preprocess the image (normalize and reshape)
        image_obs = self.transforms(image_obs)  # Normalize to [0, 1]
        raw_obs = torch.tensor(raw_obs, dtype=torch.float32)

        action = torch.tensor(actions, dtype=torch.float32)  # Convert actions to tensor
        reward = torch.tensor(rewards, dtype=torch.float32)  # Convert rewards to tensor

        return image_obs, raw_obs, action, reward

    def close(self):
        """ Close the HDF5 file when done """
        self.hf.close()


# LightningModule for VQVAE Training
class VQVAE_LunarLander_Lightning(pl.LightningModule):
    def __init__(self, num_embeddings=512, latent_dim=64, action_dim=1, reward_dim=1, commitment_cost=0.25, lr=1e-4):
        super(VQVAE_LunarLander_Lightning, self).__init__()
        self.vqvae_model = VQVAE_LunarLander(num_embeddings, latent_dim, action_dim, reward_dim, commitment_cost)
        self.lr = lr

    def forward(self, image_obs, raw_obs, actions, rewards):
        return self.vqvae_model(image_obs, raw_obs, actions, rewards)

    def training_step(self, batch, batch_idx):
        image_obs, raw_obs, actions, rewards = batch

        # Forward pass through the VQ-VAE
        recon_image, recon_raw, recon_action, recon_reward, vq_loss = self(image_obs, raw_obs, actions, rewards)

        # Compute reconstruction loss
        recon_image_loss = F.mse_loss(recon_image, image_obs)
        recon_raw_loss = F.mse_loss(recon_raw, raw_obs)
        recon_action_loss = F.mse_loss(recon_action, actions)
        recon_reward_loss = F.mse_loss(recon_reward, rewards)

        total_loss = recon_image_loss + recon_raw_loss + recon_action_loss + recon_reward_loss + vq_loss

        self.log('train_loss', total_loss, on_step=True, on_epoch=True, prog_bar=True)

        return total_loss

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.lr)
        return optimizer


# Function to create the HDF5 file from numpy data
def create_hdf5(data_dir, output_file):
    image_files = sorted([f for f in os.listdir(data_dir) if 'image_observations' in f and f.endswith('.npy')])
    raw_files = sorted([f for f in os.listdir(data_dir) if 'raw_observations' in f and f.endswith('.npy')])
    action_files = sorted([f for f in os.listdir(data_dir) if 'actions' in f and f.endswith('.npy')])
    reward_files = sorted([f for f in os.listdir(data_dir) if 'rewards' in f and f.endswith('.npy')])

    with h5py.File(output_file, 'w') as hf:
        # Create groups for observations, actions, and rewards
        image_obs_group = hf.create_group('image_observations')
        raw_obs_group = hf.create_group('raw_observations')
        act_group = hf.create_group('actions')
        rew_group = hf.create_group('rewards')

        # Load and store each observation, action, and reward into HDF5
        for i, (img_file, raw_file, act_file, rew_file) in enumerate(zip(image_files, raw_files, action_files, reward_files)):
            
            image_obs = np.load(os.path.join(data_dir, img_file))
            raw_obs = np.load(os.path.join(data_dir, raw_file))
            actions = np.load(os.path.join(data_dir, act_file))
            rewards = np.load(os.path.join(data_dir, rew_file))

            # Store each into the HDF5 file
            image_obs_group.create_dataset(f'image_observations_{i}', data=image_obs)
            raw_obs_group.create_dataset(f'raw_observations_{i}', data=raw_obs)
            act_group.create_dataset(f'actions_{i}', data=actions)
            rew_group.create_dataset(f'rewards_{i}', data=rewards)

    print(f"HDF5 file created at {output_file}")
