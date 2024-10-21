# Import the custom environment wrapper and policy
import gymnasium as gym
from stable_baselines3 import PPO
from custom_arch import CustomCNNExtractor
from custom_lander import LunarLanderImageAndStateWrapper
from callback import SaveTrainingDataCallback
from trainer import *


env = LunarLanderImageAndStateWrapper(gym.make('LunarLander-v2', render_mode='rgb_array'))

# Define the custom policy using PPO and the custom CNN+MLP extractor
policy_kwargs = dict(
    features_extractor_class=CustomCNNExtractor,
    features_extractor_kwargs=dict(cnn_output_dim=64),  # Adjust CNN output dimension as needed
)

# Instantiate the PPO agent with the custom policy
model = PPO("MultiInputPolicy", env, policy_kwargs=policy_kwargs, verbose=1)
callback = SaveTrainingDataCallback()

# Train the model and collect data
model.learn(total_timesteps=100000, callback=callback)

data_dir = './training_data'
output_file = 'lunarlander_data.h5'
create_hdf5(data_dir, output_file)

hdf5_file = 'lunarlander_data.h5'  # Path to the HDF5 file
torch.set_float32_matmul_precision('medium')

# Initialize the dataset
dataset = LunarLanderHDF5Dataset(hdf5_file, data_dir)

# Initialize the Lightning DataLoader
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)  # Using multiple workers for faster loading

# Initialize the VQVAE Lightning model
vqvae_lightning_model = VQVAE_LunarLander_Lightning(num_embeddings=512, latent_dim=64, action_dim=1, reward_dim=1, commitment_cost=0.25, lr=1e-4)

# Initialize the PyTorch Lightning Trainer
trainer = pl.Trainer(max_epochs=50, devices=1, accelerator="gpu")

# Train the model
trainer.fit(vqvae_lightning_model, dataloader)

# Close the HDF5 file after training
dataset.close()

trajectories = []
for i_episode in range(10):
    traj = []
    done = False
    trunc = False
    state, info = env.reset()
    while not done and not trunc:
        action = model.predict(state, deterministic=True)
        next_state, reward, done, trunc, info = env.step(action[0])
        
        traj.append((state, action[0], reward))
        
    trajectories.append(traj)
    

for traj in trajectories:
    
    tokens = []
    for (obs, action, reward) in traj:
        image_obs = obs['image']
        image_obs_tensor = torch.from_numpy(image_obs).permute(2, 0, 1)  # Change to (C, H, W)
        resized_image = callback.transform(image_obs_tensor)  # Apply resizing
        
        _, token = vqvae_lightning_model.vqvae_model.get_quantized_embedding_with_id(
            resized_image.reshape(-1, *resized_image.shape), 
            torch.Tensor(obs['raw_observation']).reshape(1, -1), 
            torch.Tensor(np.array([action])).reshape(1, -1), 
            torch.Tensor(np.array([reward])).reshape(1, -1)
        )
        tokens.append(token[0].item())
        
    print(tokens)