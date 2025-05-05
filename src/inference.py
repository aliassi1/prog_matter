import torch
import numpy as np
import pygame
import time
import glob
from datetime import datetime
import os
import torch.nn.functional as F

from mappo_agent import MAPPOAgent
from level_manager import LevelManager
from prog_matter_env import ProgrammableMatterEnv

class ModelInference:
    def __init__(
        self,
        checkpoint_path=None,
        visualize=True,
        num_episodes=10,
        delay=0.1,  # Delay between steps in seconds
        num_test_episodes=5,  # Number of test episodes to run per level
        temperature=0.1,  # Temperature for action selection
        level=1  # Add level parameter
    ):
        self.visualize = visualize
        self.num_episodes = num_episodes
        self.delay = delay
        self.num_test_episodes = num_test_episodes
        self.temperature = temperature
        self.paused = False
        self.level = level
        
        # Load the checkpoint
        if checkpoint_path is None:
            # For levels 1 and 2, use specific checkpoints
            if level == 1:
                checkpoint_path = self.find_checkpoint_for_level(1)
            elif level == 2:
                checkpoint_path = self.find_checkpoint_for_level(2)
            else:
                checkpoint_path = self.find_latest_checkpoint()
            
            if checkpoint_path is None:
                raise ValueError(f"No checkpoint found for level {level}. Please train the model first or specify a checkpoint path.")
        
        # Load checkpoint and determine level
        self.checkpoint = torch.load(checkpoint_path)
        print(f"Loaded checkpoint from {checkpoint_path}")
        
        # Initialize environment with parameters from checkpoint
        self.level_manager = LevelManager(
            base_grid_size=(30, 30),
            base_num_agents=1,
            max_levels=10,
            success_threshold=0.9,
            min_episodes_per_level=100
        )
        
        # Set level from parameter
        self.current_level = level
        self.level_manager.current_level = self.current_level
        self.env = self.level_manager.create_level_env()
        
        # Initialize agent
        self.agent = MAPPOAgent(
            num_agents=self.env.num_agents,
            action_dim=len(self.env.directions),
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        
        # Load agent weights
        self.agent.load_state_dict(self.checkpoint["agent_state_dict"])
        # Set networks to evaluation mode
        self.agent.actor.eval()
        self.agent.critic.eval()
        
        # Initialize Pygame if visualization is enabled
        if self.visualize:
            self.init_visualization()
    
    def init_visualization(self):
        """Initialize Pygame visualization"""
        pygame.init()
        self.cell_size = 20
        self.margin = 50
        self.screen_width = self.env.n * self.cell_size + 2 * self.margin
        self.screen_height = self.env.m * self.cell_size + 2 * self.margin
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption(f"Programmable Matter Inference - Level {self.current_level} ({self.env.num_agents} Agents)")
        self.font = pygame.font.SysFont('Arial', 16)
    
    def find_latest_checkpoint(self):
        """Find the most recent checkpoint across all directories"""
        # Get the absolute path to the checkpoints directory
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        checkpoint_dirs = glob.glob(os.path.join(base_dir, "checkpoints/*"))
        if not checkpoint_dirs:
            print("No checkpoint directories found")
            return None
        
        latest_dir = max(checkpoint_dirs, key=os.path.getmtime)
        print(f"Found latest checkpoint directory: {latest_dir}")
        
        # First try to find the latest.pt file
        latest_file = os.path.join(latest_dir, "latest.pt")
        if os.path.exists(latest_file):
            print(f"Using latest checkpoint: {latest_file}")
            return latest_file
        
        # If no latest.pt, try to find level files
        level_files = glob.glob(os.path.join(latest_dir, "level_*.pt"))
        if not level_files:
            print("No checkpoint files found in the latest directory")
            return None
        
        latest_file = max(level_files, key=os.path.getmtime)
        print(f"Using latest level checkpoint: {latest_file}")
        return latest_file

    def find_checkpoint_for_level(self, level):
        """Find the checkpoint for a specific level"""
        # Get the absolute path to the checkpoints directory
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        checkpoint_dirs = glob.glob(os.path.join(base_dir, "checkpoints/*"))
        if not checkpoint_dirs:
            print("No checkpoint directories found")
            return None
        
        latest_dir = max(checkpoint_dirs, key=os.path.getmtime)
        level_path = os.path.join(latest_dir, f"level_{level}.pt")
        if os.path.exists(level_path):
            print(f"Found checkpoint for level {level}: {level_path}")
            return level_path
        
        level_files = glob.glob(os.path.join(latest_dir, "level_*.pt"))
        if not level_files:
            print("No checkpoint files found in the latest directory")
            return None
        
        latest_file = max(level_files, key=os.path.getmtime)
        print(f"Using latest level checkpoint: {latest_file}")
        return latest_file
    
    def preprocess_observations(self, obs_dict):
        """Batch process observations for all agents"""
        processed = {}
        for key in obs_dict[list(obs_dict.keys())[0]].keys():
            values = [obs[key] for obs in obs_dict.values()]
            processed[key] = torch.stack([torch.FloatTensor(v) for v in values]).to(self.agent.device)
        return processed
    
    def select_actions(self, obs, temperature=0.1):
        """Select actions with temperature-based exploration"""
        with torch.no_grad():
            action_probs = self.agent.actor(obs)
            if temperature > 0:
                action_probs = F.softmax(action_probs / temperature, dim=1)
            actions = torch.multinomial(action_probs, 1).squeeze(-1)
        return actions
    
    def handle_visualization_events(self):
        """Handle visualization events"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    self.paused = not self.paused
                elif event.key == pygame.K_ESCAPE:
                    return False
        return True
    
    def render(self):
        """Render the current environment state"""
        if not self.visualize:
            return
        
        # Clear screen
        self.screen.fill((255, 255, 255))
        
        # Draw grid
        for x in range(self.env.n):
            for y in range(self.env.m):
                rect = pygame.Rect(
                    self.margin + x * self.cell_size,
                    self.margin + y * self.cell_size,
                    self.cell_size,
                    self.cell_size
                )
                pygame.draw.rect(self.screen, (200, 200, 200), rect, 1)
        
        # Draw obstacles
        for x in range(self.env.n):
            for y in range(self.env.m):
                if self.env.grid[x, y] == 0:
                    rect = pygame.Rect(
                        self.margin + x * self.cell_size,
                        self.margin + y * self.cell_size,
                        self.cell_size,
                        self.cell_size
                    )
                    pygame.draw.rect(self.screen, (100, 100, 100), rect)
        
        # Draw agents and their targets
        for agent_id in self.env.agents:
            # Draw agent
            x, y = self.env.agent_positions[agent_id]
            rect = pygame.Rect(
                self.margin + x * self.cell_size,
                self.margin + y * self.cell_size,
                self.cell_size,
                self.cell_size
            )
            pygame.draw.rect(self.screen, (0, 0, 255), rect)
            
            # Draw target
            x, y = self.env.agent_targets[agent_id]
            rect = pygame.Rect(
                self.margin + x * self.cell_size,
                self.margin + y * self.cell_size,
                self.cell_size,
                self.cell_size
            )
            pygame.draw.rect(self.screen, (0, 255, 0), rect)
        
        # Draw info text
        info_texts = [
            f"Level: {self.current_level}",
            f"Number of Agents: {self.env.num_agents}",
            f"Steps: {self.env.steps_taken}/{self.env.max_steps}",
            f"Status: {'Paused' if self.paused else 'Running'}"
        ]
        
        for i, text in enumerate(info_texts):
            text_surface = self.font.render(text, True, (0, 0, 0))
            self.screen.blit(text_surface, (10, 10 + i * 20))
        
        pygame.display.flip()
    
    def run_episode(self):
        """Run a single episode of inference with detailed information"""
        obs = self.env.reset()
        done = False
        total_reward = 0
        steps = 0
        
        # Get initial positions for all agents
        start_positions = {agent_id: self.env.agent_positions[agent_id] for agent_id in self.env.agents}
        target_positions = {agent_id: self.env.agent_targets[agent_id] for agent_id in self.env.agents}
        
        print("\nStarting new episode:")
        print(f"Agent start positions: {start_positions}")
        print(f"Target positions: {target_positions}")
        print("Press SPACE to start/pause, ESC to exit...")
        
        if self.visualize:
            self.render()
            while True:
                if not self.handle_visualization_events():
                    return None
                if not self.paused:
                    break
                self.render()
                time.sleep(0.1)
        
        # Action mapping for better understanding
        action_names = {
            0: "UP",
            1: "DOWN",
            2: "LEFT",
            3: "RIGHT",
            4: "UP-LEFT",
            5: "UP-RIGHT",
            6: "DOWN-LEFT",
            7: "DOWN-RIGHT",
            8: "STAY"
        }
        
        while not done:
            # Process observations in batch
            processed_obs = self.preprocess_observations(obs)
            
            # Get actions for all agents
            actions = self.select_actions(processed_obs, temperature=self.temperature)
            actions = {agent_id: action.item() for agent_id, action in zip(self.env.agents, actions)}
            
            # Step environment
            next_obs, rewards, dones, _ = self.env.step(actions)
            
            # Print step information for all agents
            print(f"\nStep {steps + 1}:")
            for agent_id in self.env.agents:
                current_pos = self.env.agent_positions[agent_id]
                target_pos = target_positions[agent_id]
                print(f"Agent {agent_id}:")
                print(f"  Current position: {current_pos}")
                print(f"  Action taken: {actions[agent_id]} ({action_names[actions[agent_id]]})")
                print(f"  Reward received: {rewards[agent_id]:.2f}")
                print(f"  Distance to target: {abs(current_pos[0] - target_pos[0]) + abs(current_pos[1] - target_pos[1])}")
            
            # Update state
            obs = next_obs
            total_reward += sum(rewards.values())
            done = dones.get("__all__", False)  # Use global done flag instead of checking all individual dones
            steps += 1
            
            # Handle visualization
            if self.visualize:
                self.render()
                while self.paused:
                    if not self.handle_visualization_events():
                        return None
                    self.render()
                    time.sleep(0.1)
                time.sleep(self.delay)
            
            # Check for episode completion
            if done or steps >= self.env.max_steps:
                print(f"\nEpisode completed in {steps} steps")
                print(f"Total reward: {total_reward:.2f}")
                break
        
        return total_reward
    
    def run_inference(self):
        """Run inference for multiple episodes"""
        try:
            total_rewards = []
            for episode in range(self.num_episodes):
                print(f"\nRunning episode {episode + 1}/{self.num_episodes}")
                reward = self.run_episode()
                if reward is None:  # User requested exit
                    break
                total_rewards.append(reward)
            
            if total_rewards:
                avg_reward = sum(total_rewards) / len(total_rewards)
                print(f"\nAverage reward over {len(total_rewards)} episodes: {avg_reward:.2f}")
        
        finally:
            self.cleanup()
    
    def run_inference_all_levels(self):
        """Run inference for all levels sequentially"""
        try:
            for level in range(1, self.level_manager.max_levels + 1):
                print(f"\nRunning inference for Level {level}")
                
                # Find and load the appropriate checkpoint for this level
                checkpoint_path = self.find_checkpoint_for_level(level)
                if checkpoint_path is None:
                    print(f"No checkpoint found for level {level}. Skipping...")
                    continue
                
                # Load checkpoint and update level
                self.checkpoint = torch.load(checkpoint_path)
                self.current_level = self.checkpoint.get('level', level)
                self.level_manager.current_level = self.current_level
                self.env = self.level_manager.create_level_env()
                
                # Update agent for new number of agents
                self.agent = MAPPOAgent(
                    num_agents=self.env.num_agents,
                    action_dim=len(self.env.directions),
                    device="cuda" if torch.cuda.is_available() else "cpu"
                )
                
                # Load agent weights
                self.agent.load_state_dict(self.checkpoint["agent_state_dict"])
                self.agent.actor.eval()
                self.agent.critic.eval()
                
                # Update visualization if enabled
                if self.visualize:
                    self.init_visualization()
                
                # Run episodes for current level
                total_rewards = []
                for episode in range(self.num_episodes):
                    print(f"\nRunning episode {episode + 1}/{self.num_episodes} for Level {level}")
                    reward = self.run_episode()
                    if reward is None:  # User requested exit
                        return
                    total_rewards.append(reward)
                
                if total_rewards:
                    avg_reward = sum(total_rewards) / len(total_rewards)
                    print(f"\nLevel {level} Summary:")
                    print(f"Average reward: {avg_reward:.2f}")
                    print("-" * 50)
                
                # Wait for user to continue to next level
                if self.visualize:
                    print("\nPress SPACE to continue to next level, ESC to exit...")
                    while True:
                        if not self.handle_visualization_events():
                            return
                        if not self.paused:
                            break
                        self.render()
                        time.sleep(0.1)
        
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if self.visualize:
            pygame.quit()

if __name__ == "__main__":
    # Example usage
    inference = ModelInference(
        checkpoint_path=None,  # Will find latest checkpoint
        visualize=True,
        num_episodes=5,  # Number of episodes per level
        delay=0.1,
        temperature=0.1,
        level=1  # For level 1
    )
    inference.run_inference_all_levels()  # Run inference for all levels 