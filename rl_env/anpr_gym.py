import gymnasium as gym
from gymnasium import spaces
import numpy as np
import os
import pandas as pd
import time
from main import run_pipeline

class ANPROptimizationEnv(gym.Env):
    def __init__(self, video_path="sample.mp4"):
        super(ANPROptimizationEnv, self).__init__()
        self.video_path = video_path
        self.video_name = os.path.splitext(os.path.basename(video_path))[0]
        
        # Action: [Vehicle Confidence (0.1-0.9), Image Size (320-960)]
        self.action_space = spaces.Box(low=np.array([0.1, 320]), high=np.array([0.9, 960]), dtype=np.float32)
        
        # Observation: [Accuracy Proxy %, Latency ms]
        self.observation_space = spaces.Box(low=0, high=1000, shape=(2,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Reset state to zero before first action
        observation = np.array([0.0, 0.0], dtype=np.float32)
        return observation, {}

    def step(self, action):
        conf, imgsz = action
        start_time = time.time()
        
        # Execute the main pipeline
        run_pipeline(self.video_path, det_conf=float(conf), det_imgsz=int(imgsz))
        
        latency = (time.time() - start_time) * 1000 # Convert to ms
        accuracy = self._get_accuracy_metric()
        
        # Reward: Accuracy - Latency Penalty
        # Heavy penalty if accuracy is 0 to prevent reward hacking via empty results
        reward = accuracy - (latency * 0.05)
        if accuracy == 0:
            reward -= 50

        observation = np.array([accuracy, latency], dtype=np.float32)
        terminated = accuracy > 95.0
        
        return observation, reward, terminated, False, {}

    def _get_accuracy_metric(self):
        """Reads the interpolated CSV to determine unique plate yield per 100 frames."""
        csv_path = f"{self.video_name}_interpolated.csv"
        if not os.path.exists(csv_path):
            return 0.0
        
        df = pd.read_csv(csv_path)
        if df.empty:
            return 0.0
            
        unique_plates = df['license_number'].nunique()
        total_frames = df['frame_nmr'].max() or 1
        
        # Proxy: Success is finding distinct valid plates consistently
        yield_score = (unique_plates / (total_frames / 100)) * 10
        return min(100.0, yield_score)