import numpy as np
import os
from stable_baselines3.common.callbacks import BaseCallback
from torchvision import transforms
import torch

class SaveTrainingDataCallback(BaseCallback):
    """
    Custom callback for saving the training data (observations, actions, rewards)
    at the end of each rollout to a file.
    """
    def __init__(self, save_path='./training_data', verbose=0):
        super(SaveTrainingDataCallback, self).__init__(verbose)
        self.save_path = save_path
        self.image_observations = []
        self.raw_observations = []
        self.actions = []
        self.rewards = []
        self.current_len = 0

        # Create the directory if it does not exist
        os.makedirs(self.save_path, exist_ok=True)

        # Rollout counter for unique file names
        self.rollout_counter = 0

        # Define transformation for image resizing
        self.transform = transforms.Compose([
            transforms.ToPILImage(),  # Convert from NumPy array to PIL Image
            transforms.Resize((64, 64)),  # Resize to 64x64
            transforms.ToTensor()  # Convert back to Tensor
        ])
        

    def _on_step(self) -> bool:
        # PPO uses a rollout buffer to store data
        
        # Get current rollout buffer contents
        obs = self.locals['env'].get_attr('observation')
        actions = self.locals['env'].get_attr('action')
        rewards = self.locals['env'].get_attr('reward')
        
        image_obs = obs[0]['image']
        image_obs_tensor = torch.from_numpy(image_obs).permute(2, 0, 1)  # Change to (C, H, W)
        resized_image = self.transform(image_obs_tensor)  # Apply resizing
        
        # Save data
        self.image_observations.append(resized_image.permute(1, 2, 0).numpy())  # Append resized image observations
        self.raw_observations.append(obs[0]['raw_observation'])
        self.actions.append(actions)
        self.rewards.append(rewards)
        
        return True
    
    def _on_rollout_end(self) -> None:
        """
        Called at the end of each rollout. Save the collected data to disk and reset buffers.
        """
        # Save observations, actions, and rewards to files
        np.save(os.path.join(self.save_path, f'image_observations_{self.rollout_counter}.npy'), np.array(self.image_observations))
        np.save(os.path.join(self.save_path, f'raw_observations_{self.rollout_counter}.npy'), np.array(self.raw_observations))
        np.save(os.path.join(self.save_path, f'actions_{self.rollout_counter}.npy'), np.array(self.actions))
        np.save(os.path.join(self.save_path, f'rewards_{self.rollout_counter}.npy'), np.array(self.rewards))
        self.current_len += len(self.actions)
        
        # Reset data buffers after saving to free memory
        self.image_observations = []
        self.raw_observations = []
        self.actions = []
        self.rewards = []

        # Increment rollout counter
        self.rollout_counter += 1
    
    def _on_training_end(self) -> None:
        """
        This event is triggered before exiting the `learn()` method.
        """
        if len(self.image_observations) == 0:
            with open(os.path.join(self.save_path, 'metadata.txt'), 'w') as f:
                f.write(f"Length: {self.current_len}")
            
            print("WRITTEN TO FILE")
            return
        np.save(os.path.join(self.save_path, f'image_observations_{self.rollout_counter}.npy'), np.array(self.image_observations))
        np.save(os.path.join(self.save_path, f'raw_observations_{self.rollout_counter}.npy'), np.array(self.raw_observations))
        np.save(os.path.join(self.save_path, f'actions_{self.rollout_counter}.npy'), np.array(self.actions))
        np.save(os.path.join(self.save_path, f'rewards_{self.rollout_counter}.npy'), np.array(self.rewards))
        self.current_len += len(self.actions)
        
        with open(os.path.join(self.save_path, 'metadata.txt'), 'w') as f:
            f.write(f"Length: {self.current_len}")
        
        print("WRITTEN TO FILE")
        
        
        
        # Reset data buffers after saving to free memory
        self.image_observations = []
        self.raw_observations = []
        self.actions = []
        self.rewards = []
