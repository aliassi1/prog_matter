import torch
import numpy as np
import pygame
import time
import glob
from datetime import datetime
import os

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
        num_test_episodes=5  # Number of test episodes to run per level
    ):
        self.visualize = visualize
        self.num_episodes = num_episodes
        self.delay = delay
        self.num_test_episodes = num_test_episodes
        
        # Load the level_1 checkpoint if none specified
        if checkpoint_path is None:
            checkpoint_path = self.find_latest_checkpoint()
            if checkpoint_path is None:
                raise ValueError("No checkpoint found. Please train the model first or specify a checkpoint path.")
        
        # Load checkpoint
        self.checkpoint = torch.load(checkpoint_path)
        print(f"Loaded checkpoint from {checkpoint_path}")
        
        # Initialize environment with simple level 1 parameters
        self.level_manager = LevelManager(
            base_grid_size=(30, 30),  # 30x30 grid
            base_num_agents=1,        # Single agent
            max_levels=10,
            success_threshold=0.9,
            min_episodes_per_level=100
        )
        
        # Force level 1
        self.level_manager.current_level = 1
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
            pygame.init()
            self.cell_size = 20  # Adjusted for 30x30 grid
            self.margin = 50
            self.screen_width = self.env.n * self.cell_size + 2 * self.margin
            self.screen_height = self.env.m * self.cell_size + 2 * self.margin
            self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
            pygame.display.set_caption(f"Programmable Matter Inference - Level 1 (1 Agent)")
            self.font = pygame.font.SysFont('Arial', 16)
    
    def find_latest_checkpoint(self):
        """Find the most recent checkpoint across all directories"""
        # Get all checkpoint directories
        checkpoint_dirs = glob.glob("checkpoints/*")
        if not checkpoint_dirs:
            return None
        
        # Find the latest checkpoint directory
        latest_dir = max(checkpoint_dirs, key=os.path.getmtime)
        
        # Find all level checkpoints in the latest directory
        level_files = glob.glob(f"{latest_dir}/level_*.pt")
        if not level_files:
            return None
        
        # Return the most recent checkpoint
        return max(level_files, key=os.path.getmtime)

    def find_checkpoint_for_level(self, level):
        """Find the checkpoint for a specific level"""
        # Get all checkpoint directories
        checkpoint_dirs = glob.glob("checkpoints/*")
        if not checkpoint_dirs:
            return None
        
        # Find the latest checkpoint directory
        latest_dir = max(checkpoint_dirs, key=os.path.getmtime)
        
        # Look for the specific level checkpoint
        level_path = f"{latest_dir}/level_{level}.pt"
        if os.path.exists(level_path):
            return level_path
        
        # If specific level checkpoint not found, return the latest checkpoint
        level_files = glob.glob(f"{latest_dir}/level_*.pt")
        if not level_files:
            return None
        
        return max(level_files, key=os.path.getmtime)
    
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
            f"Level: {self.level_manager.current_level}",
            f"Number of Agents: {self.env.num_agents}",
            f"Steps: {self.env.steps_taken}/{self.env.max_steps}"
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
        print("Press any key to start...")
        if self.visualize:
            self.render()
            pygame.event.wait()  # Wait for key press
        
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
            # Get actions for all agents
            actions = {}
            for agent_id in self.env.agents:
                # Convert observation dictionary to tensors
                obs_tensors = {}
                for key, value in obs[agent_id].items():
                    obs_tensors[key] = torch.FloatTensor(value).unsqueeze(0).to(self.agent.device)
                
                # Get action probabilities
                with torch.no_grad():
                    action_probs = self.agent.actor(obs_tensors)
                    action = torch.argmax(action_probs).item()
                
                actions[agent_id] = action
            
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
            done = all(dones.values())
            steps += 1
            
            # Render and wait
            if self.visualize:
                self.render()
                time.sleep(self.delay)
        
        # Print episode summary
        print("\nEpisode Summary:")
        print(f"Total steps: {steps}")
        print(f"Total reward: {total_reward:.2f}")
        
        # Check success for all agents
        success = True
        for agent_id in self.env.agents:
            if self.env.agent_positions[agent_id] != target_positions[agent_id]:
                success = False
                break
        print(f"Success: {'Yes' if success else 'No'}")
        
        return total_reward, steps
    
    def find_all_level_checkpoints(self):
        """Find all available level_X.pt checkpoints, sorted by level number."""
        checkpoint_dirs = glob.glob("checkpoints/*")
        level_ckpts = []
        for dir_path in checkpoint_dirs:
            level_files = glob.glob(f"{dir_path}/level_*.pt")
            for file in level_files:
                # Extract level number
                try:
                    level_num = int(os.path.basename(file).split('_')[-1].split('.')[0])
                    level_ckpts.append((level_num, file))
                except Exception:
                    continue
        # Sort by level number
        level_ckpts.sort()
        return level_ckpts

    def run_inference_all_levels(self):
        """Run inference for all levels"""
        for level in range(1, self.level_manager.max_levels + 1):
            print(f"\nRunning inference for Level {level}")
            
            # Find and load the appropriate checkpoint for this level
            checkpoint_path = self.find_checkpoint_for_level(level)
            if checkpoint_path is None:
                print(f"No checkpoint found for level {level}. Skipping...")
                continue
            
            checkpoint = torch.load(checkpoint_path)
            print(f"Loaded checkpoint from {checkpoint_path}")
            
            # Set up level manager and environment
            self.level_manager.current_level = level
            self.env = self.level_manager.create_level_env()
            
            # Update agent for new number of agents
            self.agent = MAPPOAgent(
                num_agents=self.env.num_agents,
                action_dim=len(self.env.directions),
                device="cuda" if torch.cuda.is_available() else "cpu"
            )
            
            # Load agent weights from the level-specific checkpoint
            self.agent.load_state_dict(checkpoint["agent_state_dict"])
            self.agent.actor.eval()
            self.agent.critic.eval()
            
            # Update visualization if enabled
            if self.visualize:
                self.cell_size = 20
                self.margin = 50
                self.screen_width = self.env.n * self.cell_size + 2 * self.margin
                self.screen_height = self.env.m * self.cell_size + 2 * self.margin
                self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
                pygame.display.set_caption(f"Programmable Matter Inference - Level {level} ({self.env.num_agents} Agents)")
            
            # Run episodes for current level
            total_rewards = []
            total_steps = []
            successes = 0
            
            for episode in range(self.num_test_episodes):
                print(f"\nEpisode {episode + 1}/{self.num_test_episodes} for Level {level}")
                reward, steps = self.run_episode()
                total_rewards.append(reward)
                total_steps.append(steps)
                
                # Check if episode was successful
                agent_id = list(self.env.agents)[0]
                if self.env.agent_positions[agent_id] == self.env.agent_targets[agent_id]:
                    successes += 1
            
            # Print level summary
            print(f"\nLevel {level} Summary:")
            print(f"Average reward: {np.mean(total_rewards):.2f}")
            print(f"Average steps: {np.mean(total_steps):.2f}")
            print(f"Success rate: {successes/self.num_test_episodes:.2%}")
            print("-" * 50)

if __name__ == "__main__":
    inference = ModelInference(
        checkpoint_path=None,  # Will load the latest checkpoint
        visualize=True,        # Enable visualization
        num_episodes=1,       # Only one episode per level
        delay=0.1,
        num_test_episodes=5   # Run 5 test episodes for level 1
    )
    inference.run_inference_all_levels() 