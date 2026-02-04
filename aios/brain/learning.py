import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
import os
from gymnasium import spaces

class OS1OptimizationEnv(gym.Env):
    """
    Custom Environment where OS1 learns to optimize its response parameters
    (Verbosity, Creativity/Temperature, Tone) based on user feedback.
    """
    def __init__(self):
        super(OS1OptimizationEnv, self).__init__()
        
        # Actions:
        # 0: Decrease Temperature (More precise)
        # 1: Increase Temperature (More creative)
        # 2: Decrease Verbosity
        # 3: Increase Verbosity
        self.action_space = spaces.Discrete(4)
        
        # Observation: [Current Satisfaction (0-1), Avg Response Time, Last Sentiment (-1 to 1)]
        self.observation_space = spaces.Box(low=np.array([0, 0, -1]), high=np.array([1, 100, 1]), dtype=np.float32)
        
        self.state = np.array([0.5, 1.0, 0.0], dtype=np.float32) # Neutral start

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = np.array([0.5, 1.0, 0.0], dtype=np.float32)
        return self.state, {}

    def step(self, action):
        # In a real scenario, this would apply changes to config.yaml
        # Here we simulate the effect for training
        reward = 0
        done = False
        
        # Mock logic: If sentiment is negative, and we change parameters, we might get a reward
        current_sentiment = self.state[2]
        
        if current_sentiment < 0 and action in [0, 2]: # If user unhappy, trying precision/brevity
            reward = 1.0
        elif current_sentiment > 0:
            reward = 0.5 # Sustain
            
        # Update state randomly to simulate new interaction
        self.state = np.random.uniform(low=[0,0,-1], high=[1,5,1]).astype(np.float32)
        
        return self.state, reward, done, False, {}

class RLAgent:
    def __init__(self):
        self.env = OS1OptimizationEnv()
        self.model_path = "root/db/rl_model"
        
        if os.path.exists(self.model_path + ".zip"):
            self.model = PPO.load(self.model_path, env=self.env)
        else:
            self.model = PPO("MlpPolicy", self.env, verbose=1)
            
    def update_policy(self, user_feedback_score):
        """
        Run a training step based on real interaction.
        user_feedback_score: -1 (Bad) to 1 (Good)
        """
        # Inject current state into env (hacky for stateless API, but works for prototype)
        self.env.state[2] = user_feedback_score 
        self.model.learn(total_timesteps=100)
        self.model.save(self.model_path)
        
    def get_optimization_action(self):
        action, _ = self.model.predict(self.env.state)
        actions_map = {0: "Decr Temp", 1: "Incr Temp", 2: "Decr Verbosity", 3: "Incr Verbosity"}
        return actions_map[action]