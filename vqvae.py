import torch
import torch.nn as nn
import torch.nn.functional as F

class ImageEncoder(nn.Module):
    def __init__(self, in_channels=3, latent_dim=64):
        super(ImageEncoder, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=4, stride=2, padding=1)  # 32x32x32
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)  # 16x16x64
        self.conv3 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)  # 8x8x128
        self.conv4 = nn.Conv2d(128, latent_dim, kernel_size=4, stride=2, padding=1)  # 4x4xlatent_dim

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.conv4(x)  # No activation
        return x  # Shape (B, latent_dim, 4, 4)

class RawObservationEncoder(nn.Module):
    def __init__(self, input_dim=8, latent_dim=64):
        super(RawObservationEncoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, latent_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x  # Shape (B, latent_dim)

class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        super(VectorQuantizer, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost

        # Initialize the embedding vectors
        self.embeddings = nn.Embedding(num_embeddings, embedding_dim)
        self.embeddings.weight.data.uniform_(-1 / num_embeddings, 1 / num_embeddings)

    def forward(self, x):
        # Flatten the input to be (batch_size * height * width, embedding_dim)
        flat_x = x.view(-1, self.embedding_dim)

        # Compute the distances to each embedding
        distances = torch.sum(flat_x ** 2, dim=1, keepdim=True) + torch.sum(self.embeddings.weight ** 2, dim=1) - 2 * torch.matmul(flat_x, self.embeddings.weight.t())

        # Get the indices of the closest embeddings
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)

        # Get the embeddings corresponding to the closest encoding indices
        quantized = torch.index_select(self.embeddings.weight, dim=0, index=encoding_indices.view(-1))

        # Reshape the quantized embeddings back to the input shape
        quantized = quantized.view_as(x)

        # Compute the commitment loss
        e_latent_loss = F.mse_loss(quantized.detach(), x)
        q_latent_loss = F.mse_loss(quantized, x.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss

        # Straight-through estimator
        quantized = x + (quantized - x).detach()

        return quantized, loss

class ImageDecoder(nn.Module):
    def __init__(self, latent_dim=64, out_channels=3):
        super(ImageDecoder, self).__init__()
        self.conv1 = nn.ConvTranspose2d(latent_dim, 128, kernel_size=4, stride=2, padding=1)  # 8x8x128
        self.conv2 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)  # 16x16x64
        self.conv3 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)  # 32x32x32
        self.conv4 = nn.ConvTranspose2d(32, out_channels, kernel_size=4, stride=2, padding=1)  # 64x64x3

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = torch.sigmoid(self.conv4(x))  # Sigmoid activation for pixel values
        return x

class RawObservationDecoder(nn.Module):
    def __init__(self, latent_dim=64, output_dim=8):
        super(RawObservationDecoder, self).__init__()
        self.fc1 = nn.Linear(latent_dim, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x  # Shape (B, output_dim)
    
    
class VQVAE_LunarLander(nn.Module):
    def __init__(self, num_embeddings=512, latent_dim=64, action_dim=1, reward_dim=1, commitment_cost=0.25):
        super(VQVAE_LunarLander, self).__init__()
        self.image_encoder = ImageEncoder(latent_dim=latent_dim)
        self.raw_encoder = RawObservationEncoder(latent_dim=latent_dim)

        # Action and reward embedding layers
        self.action_embedding = nn.Linear(action_dim, latent_dim)  # Embed action into latent_dim
        self.reward_embedding = nn.Linear(reward_dim, latent_dim)  # Embed reward into latent_dim

        # Vector Quantizer for quantizing the combined embedding
        self.quantizer = VectorQuantizer(num_embeddings=num_embeddings, embedding_dim=1024 + latent_dim * 3, commitment_cost=commitment_cost)

        # Decoders for reconstructing from the quantized latent space
        self.image_decoder = ImageDecoder(latent_dim=latent_dim)
        self.raw_decoder = RawObservationDecoder(latent_dim=latent_dim)

        # Optional decoder for actions and rewards if you want to reconstruct them
        self.action_decoder = nn.Linear(latent_dim, action_dim)
        self.reward_decoder = nn.Linear(latent_dim, reward_dim)

    def forward(self, image, raw_obs, action, reward):
        # Encode image and raw observations
        image_latent = self.image_encoder(image)  # Shape: (B, latent_dim, 4, 4)
        raw_latent = self.raw_encoder(raw_obs)  # Shape: (B, latent_dim)

        # Embed actions and rewards
        action_latent = F.relu(self.action_embedding(action))  # Shape: (B, latent_dim)
        reward_latent = F.relu(self.reward_embedding(reward))  # Shape: (B, latent_dim)

        # Flatten the image latent
        image_latent_flat = image_latent.view(image_latent.size(0), -1)  # Shape: (B, 1024)

        # Concatenate all embeddings into a single combined embedding
        combined_embedding = torch.cat([image_latent_flat, raw_latent, action_latent, reward_latent], dim=1)  # Shape: (B, 1216)

        # Quantize the combined embedding
        quantized_embedding, vq_loss = self.quantizer(combined_embedding)  # Shape: (B, 1216)

        # Decode the quantized embedding back into individual components
        # The image decoder expects (B, latent_dim, H, W) so we reshape
        quantized_image = quantized_embedding[:, :1024].view(image_latent.size())  # Shape: (B, 64, 4, 4)
        quantized_raw = quantized_embedding[:, 1024:1024 + 64]  # Shape: (B, 64)
        quantized_action = quantized_embedding[:, 1024 + 64:1024 + 128]  # Shape: (B, 64)
        quantized_reward = quantized_embedding[:, 1024 + 128:]  # Shape: (B, 64)

        # Reconstruct image and raw observations
        recon_image = self.image_decoder(quantized_image)  # Shape: (B, 3, 64, 64)
        recon_raw = self.raw_decoder(quantized_raw)  # Shape: (B, 8)

        # Optionally, reconstruct actions and rewards (if desired)
        recon_action = self.action_decoder(quantized_action)  # Shape: (B, 1)
        recon_reward = self.reward_decoder(quantized_reward)  # Shape: (B, 1)

        return recon_image, recon_raw, recon_action, recon_reward, vq_loss
    
    def get_quantized_embedding_with_id(self, image, raw_obs, action, reward):
        """
        Given observation (image, raw_obs, action, reward), return the quantized embedding and its ID.
        """
        # Encode image and raw observations
        image_latent = self.image_encoder(image)  # Shape: (B, latent_dim, 4, 4)
        raw_latent = self.raw_encoder(raw_obs)  # Shape: (B, latent_dim)

        # Embed actions and rewards
        action_latent = F.relu(self.action_embedding(action))  # Shape: (B, latent_dim)
        reward_latent = F.relu(self.reward_embedding(reward))  # Shape: (B, latent_dim)

        # Flatten the image latent
        image_latent_flat = image_latent.view(image_latent.size(0), -1)  # Shape: (B, 1024)

        # Concatenate all embeddings into a single combined embedding
        combined_embedding = torch.cat([image_latent_flat, raw_latent, action_latent, reward_latent], dim=1)  # Shape: (B, 1216)

        # Compute quantized embedding and find the closest embedding (ID)
        flat_combined = combined_embedding.view(-1, self.quantizer.embedding_dim)  # Flatten for quantization
        distances = torch.sum(flat_combined ** 2, dim=1, keepdim=True) + torch.sum(self.quantizer.embeddings.weight ** 2, dim=1) - 2 * torch.matmul(flat_combined, self.quantizer.embeddings.weight.t())

        # Get the indices of the closest embeddings
        encoding_indices = torch.argmin(distances, dim=1).view(image.size(0), -1)  # Shape: (B, 1)

        # Quantize the embedding
        quantized_embedding, _ = self.quantizer(combined_embedding)  # Shape: (B, 1216)

        return quantized_embedding, encoding_indices
