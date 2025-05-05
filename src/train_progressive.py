import torch
import numpy as np
from tqdm import tqdm
from datetime import datetime
import os
import pygame
import time
import torch.multiprocessing as mp
from torch.nn.parallel import DataParallel
import glob
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from mappo_agent import MAPPOAgent
from level_manager import LevelManager
from prog_matter_env import ProgrammableMatterEnv

class ProgressiveTrainer:
    def __init__(
        self,
        base_grid_size=(30, 30),
        base_num_agents=8,
        max_levels=10,
        success_threshold=0.9,
        min_episodes_per_level=100,
        device="cuda" if torch.cuda.is_available() else "cpu",
        visualize=True,
        num_parallel_envs=4,  # Number of parallel environments
        load_checkpoint=None  # Path to checkpoint to load
    ):
        self.device = device
        self.visualize = visualize
        self.num_parallel_envs = num_parallel_envs
        
        # Initialize Pygame if visualization is enabled
        if self.visualize:
            pygame.init()
            self.cell_size = 20
            self.margin = 50
            self.screen_width = base_grid_size[0] * self.cell_size + 2 * self.margin
            self.screen_height = base_grid_size[1] * self.cell_size + 2 * self.margin
            self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
            pygame.display.set_caption("Programmable Matter Training")
            self.font = pygame.font.SysFont('Arial', 16)
            
            # Initialize matplotlib figures for graphs
            plt.ion()  # Turn on interactive mode
            self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(10, 8))
            self.reward_lines = []
            self.target_time_lines = []
            self.episode_rewards = [[] for _ in range(num_parallel_envs)]
            self.target_times = [[] for _ in range(num_parallel_envs)]
            self.episodes = []
        
        # Initialize level manager
        self.level_manager = LevelManager(
            base_grid_size=base_grid_size,
            base_num_agents=base_num_agents,
            max_levels=max_levels,
            success_threshold=success_threshold,
            min_episodes_per_level=min_episodes_per_level
        )
        
        # Create parallel environments
        self.envs = [self.level_manager.create_level_env() for _ in range(num_parallel_envs)]
        
        # Create agent with DataParallel if using GPU
        self.agent = MAPPOAgent(
            num_agents=self.envs[0].num_agents,
            action_dim=len(self.envs[0].directions),
            device=device
        )
        
        if torch.cuda.is_available() and num_parallel_envs > 1:
            self.agent.actor = DataParallel(self.agent.actor)
            self.agent.critic = DataParallel(self.agent.critic)
        
        # Initialize score tracking
        self.total_reward = 0
        self.episode_reward = 0
        self.best_reward = float('-inf')
        
        # Load checkpoint if specified
        if load_checkpoint:
            self.load_checkpoint(load_checkpoint)
            print(f"Loaded checkpoint from {load_checkpoint}")
            print(f"Continuing from level {self.level_manager.current_level}")
        else:
            # Create new checkpoint directory
            self.checkpoint_dir = f"checkpoints/{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            os.makedirs(self.checkpoint_dir, exist_ok=True)

    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path)
        
        # Load level manager state
        self.level_manager.current_level = checkpoint["level"]
        self.level_manager.episode_count = checkpoint["level_stats"]["episodes_played"]
        self.level_manager.success_count = int(checkpoint["level_stats"]["success_rate"] * self.level_manager.episode_count)
        
        # Load agent state
        self.agent.load_state_dict(checkpoint["agent_state_dict"])
        
        # Update environments for current level
        self.envs = [self.level_manager.create_level_env() for _ in range(self.num_parallel_envs)]
        
        # Set checkpoint directory to the parent of the loaded checkpoint
        self.checkpoint_dir = os.path.dirname(checkpoint_path)
        
        print(f"Loaded checkpoint from level {checkpoint['level']}")
        print(f"Success rate: {checkpoint['level_stats']['success_rate']:.2f}")
        print(f"Episodes played: {checkpoint['level_stats']['episodes_played']}")
        
        # If we're continuing from a previous level, ensure we're at the right level
        if self.level_manager.current_level > 1:
            print(f"Continuing training from level {self.level_manager.current_level}")
            print(f"Level focus: {self.level_manager.level_focus[self.level_manager.current_level]['name']}")

    def render(self, env_idx=0):
        """Render the current environment state"""
        if not self.visualize:
            return

        # Clear screen
        self.screen.fill((255, 255, 255))

        # Draw grid
        for x in range(self.envs[env_idx].n):
            for y in range(self.envs[env_idx].m):
                rect = pygame.Rect(
                    self.margin + x * self.cell_size,
                    self.margin + y * self.cell_size,
                    self.cell_size,
                    self.cell_size
                )
                pygame.draw.rect(self.screen, (200, 200, 200), rect, 1)

        # Draw obstacles (where grid is 0)
        for x in range(self.envs[env_idx].n):
            for y in range(self.envs[env_idx].m):
                if self.envs[env_idx].grid[x, y] == 0:
                    rect = pygame.Rect(
                        self.margin + x * self.cell_size,
                        self.margin + y * self.cell_size,
                        self.cell_size,
                        self.cell_size
                    )
                    pygame.draw.rect(self.screen, (100, 100, 100), rect)

        # Draw agents
        for agent_id, pos in self.envs[env_idx].agent_positions.items():
            x, y = pos
            rect = pygame.Rect(
                self.margin + x * self.cell_size,
                self.margin + y * self.cell_size,
                self.cell_size,
                self.cell_size
            )
            pygame.draw.rect(self.screen, (0, 0, 255), rect)

        # Draw targets
        for agent_id, target in self.envs[env_idx].agent_targets.items():
            x, y = target
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
            f"Episode: {self.level_manager.episode_count}",
            f"Success Rate: {self.level_manager.success_count / max(1, self.level_manager.episode_count):.2f}",
            f"Agents: {self.envs[env_idx].num_agents}",
            f"Grid Size: {self.envs[env_idx].n}x{self.envs[env_idx].m}",
            f"Steps: {self.envs[env_idx].steps_taken}/{self.envs[env_idx].max_steps}",
            f"Episode Reward: {self.episode_reward:.2f}",
            f"Total Reward: {self.total_reward:.2f}",
            f"Best Reward: {self.best_reward:.2f}",
            f"Showing Environment: {env_idx + 1}/{self.num_parallel_envs}"
        ]

        for i, text in enumerate(info_texts):
            text_surface = self.font.render(text, True, (0, 0, 0))
            self.screen.blit(text_surface, (10, 10 + i * 20))

        pygame.display.flip()

    def update_graphs(self, episode, rewards, target_times):
        """Update the reward and target time graphs"""
        if not self.visualize:
            return
            
        self.episodes.append(episode)
        
        # Update reward graph
        for i, reward in enumerate(rewards):
            self.episode_rewards[i].append(reward)
            if len(self.reward_lines) <= i:
                line, = self.ax1.plot(self.episodes, self.episode_rewards[i], 
                                    label=f'Agent {i+1}')
                self.reward_lines.append(line)
            else:
                self.reward_lines[i].set_data(self.episodes, self.episode_rewards[i])
        
        # Update target time graph
        for i, target_time in enumerate(target_times):
            self.target_times[i].append(target_time)
            if len(self.target_time_lines) <= i:
                line, = self.ax2.plot(self.episodes, self.target_times[i], 
                                    label=f'Agent {i+1}')
                self.target_time_lines.append(line)
            else:
                self.target_time_lines[i].set_data(self.episodes, self.target_times[i])
        
        # Update axes limits and labels
        self.ax1.relim()
        self.ax1.autoscale_view()
        self.ax1.set_title('Episode Rewards')
        self.ax1.set_xlabel('Episode')
        self.ax1.set_ylabel('Reward')
        self.ax1.legend()
        
        self.ax2.relim()
        self.ax2.autoscale_view()
        self.ax2.set_title('Target Reaching Time')
        self.ax2.set_xlabel('Episode')
        self.ax2.set_ylabel('Steps to Target')
        self.ax2.legend()
        
        plt.tight_layout()
        plt.draw()
        plt.pause(0.01)

    def train_parallel_episode(self, env_idx):
        """Train a single episode in parallel"""
        env = self.envs[env_idx]
        obs = env.reset()
        done = False
        episode_reward = 0
        steps = 0
        target_reached_times = {agent_id: -1 for agent_id in env.agents}
        
        while not done:
            # Get actions for all agents in parallel
            actions = {}
            for agent_id in env.agents:
                if not isinstance(obs[agent_id], dict):
                    obs[agent_id] = self.agent.preprocess_observation(obs[agent_id])
                elif 'local_grid' not in obs[agent_id]:
                    obs[agent_id] = self.agent.preprocess_observation(obs[agent_id])
                
                action = self.agent.select_action(obs[agent_id], evaluate=False)
                actions[agent_id] = action
            
            # Step environment
            next_obs, rewards, dones, infos = env.step(actions)
            
            # Store transitions
            for agent_id in env.agents:
                self.agent.store_transition(
                    obs[agent_id],
                    actions[agent_id],
                    rewards[agent_id],
                    dones[agent_id]
                )
                
                # Track when agents reach their targets
                if infos[agent_id]["target_reached"] and target_reached_times[agent_id] == -1:
                    target_reached_times[agent_id] = steps
            
            obs = next_obs
            step_reward = sum(rewards.values())
            episode_reward += step_reward
            done = all(dones.values())
            steps += 1
            
            # Print step information
            print(f"\rEnv {env_idx + 1}: Step {steps}, Reward: {step_reward:.2f}, Total: {episode_reward:.2f}", end="")
        
        print()  # New line after episode
        
        # Check if episode was successful (all agents reached their targets)
        success = all(target_reached_times[agent_id] != -1 for agent_id in env.agents)
        return episode_reward, success, target_reached_times

    def train(self, num_episodes=10000):
        """Train the agent with parallel environments"""
        best_env_idx = 0  # Track the best performing environment
        
        for episode in tqdm(range(num_episodes)):
            print(f"\nEpisode {episode + 1}/{num_episodes}")
            print(f"Current Level: {self.level_manager.current_level}")
            print(f"Success Rate: {self.level_manager.success_count / max(1, self.level_manager.episode_count):.2f}")
            print(f"Episodes in Level: {self.level_manager.episode_count}/{self.level_manager.min_episodes_per_level}")
            print("-" * 50)
            
            # Train in parallel environments
            episode_rewards = []
            episode_successes = []
            target_times = []
            
            for env_idx in range(self.num_parallel_envs):
                print(f"\nEnvironment {env_idx + 1}/{self.num_parallel_envs}")
                reward, success, target_reached_times = self.train_parallel_episode(env_idx)
                episode_rewards.append(reward)
                episode_successes.append(success)
                target_times.append(max(target_reached_times.values()))  # Use max time as metric
                print(f"Environment {env_idx + 1} Final Reward: {reward:.2f}, Success: {success}")
            
            # Update graphs every epoch
            if self.visualize:
                self.update_graphs(episode, episode_rewards, target_times)
            
            # Find the best performing environment
            best_env_idx = np.argmax(episode_rewards)
            
            # Update level progress based on all environments' successes
            for success in episode_successes:
                self.level_manager.update_level_progress(success)
            
            # Average rewards across parallel environments
            avg_reward = sum(episode_rewards) / self.num_parallel_envs
            self.total_reward += avg_reward
            
            # Update networks
            self.agent.update()
            
            # Print episode summary
            print("\nEpisode Summary:")
            print(f"Average Reward: {avg_reward:.2f}")
            print(f"Best Environment: {best_env_idx + 1} (Reward: {episode_rewards[best_env_idx]:.2f})")
            print(f"Total Reward: {self.total_reward:.2f}")
            print(f"Current Level: {self.level_manager.current_level}")
            print(f"Success Rate: {self.level_manager.success_count / max(1, self.level_manager.episode_count):.2f}")
            print(f"Episodes in Level: {self.level_manager.episode_count}/{self.level_manager.min_episodes_per_level}")
            
            # Save checkpoint if best reward or every 10 episodes
            if avg_reward > self.best_reward or episode % 10 == 0:
                if avg_reward > self.best_reward:
                    self.best_reward = avg_reward
                    print("New best reward! Saving checkpoint...")
                self.save_checkpoint()
            
            # Check for level progression
            if (self.level_manager.episode_count >= self.level_manager.min_episodes_per_level and 
                self.level_manager.success_count / self.level_manager.episode_count >= self.level_manager.success_threshold):
                # Save checkpoint before advancing level
                self.save_checkpoint()
                self.level_manager.advance_level()
                self.envs = [self.level_manager.create_level_env() for _ in range(self.num_parallel_envs)]
                print(f"\nLevel Up! Now at level {self.level_manager.current_level}")
                print(f"New Level Success Rate: 0.00 (Starting fresh)")
                # Save checkpoint after advancing level
                self.save_checkpoint()

    def evaluate_success_rate(self, num_eval_episodes=10):
        """Evaluate success rate across parallel environments"""
        success_count = 0
        for _ in range(num_eval_episodes):
            for env in self.envs:
                obs = env.reset()
                done = False
                while not done:
                    actions = {}
                    for agent_id in env.agents:
                        # Ensure observation has all required fields
                        if not isinstance(obs[agent_id], dict):
                            obs[agent_id] = self.agent.preprocess_observation(obs[agent_id])
                        elif 'local_grid' not in obs[agent_id]:
                            obs[agent_id] = self.agent.preprocess_observation(obs[agent_id])
                        
                        action = self.agent.select_action(obs[agent_id], evaluate=True)
                        actions[agent_id] = action
                    next_obs, _, dones, _ = env.step(actions)
                    obs = next_obs
                    done = all(dones.values())
                    if done:
                        success_count += 1
        return success_count / (num_eval_episodes * self.num_parallel_envs)

    def save_checkpoint(self):
        """Save current model checkpoint"""
        # Ensure checkpoint directory exists
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        checkpoint = {
            "level": self.level_manager.current_level,
            "agent_state_dict": self.agent.state_dict(),
            "level_stats": self.level_manager.get_level_stats()
        }
        
        # Save checkpoint with level number
        checkpoint_path = f"{self.checkpoint_dir}/level_{self.level_manager.current_level}.pt"
        torch.save(checkpoint, checkpoint_path)
        print(f"Saved checkpoint to {checkpoint_path}")
        
        # Also save a latest checkpoint
        latest_path = f"{self.checkpoint_dir}/latest.pt"
        torch.save(checkpoint, latest_path)
        print(f"Saved latest checkpoint to {latest_path}")

def find_latest_checkpoint():
    """Find the most recent checkpoint directory"""
    checkpoint_dirs = glob.glob("checkpoints/*")
    if not checkpoint_dirs:
        return None
    
    latest_dir = max(checkpoint_dirs, key=os.path.getmtime)
    level_files = glob.glob(f"{latest_dir}/level_*.pt")
    if not level_files:
        return None
    
    # Find the highest level checkpoint
    latest_checkpoint = max(level_files, key=lambda x: int(x.split('_')[-1].split('.')[0]))
    return latest_checkpoint

if __name__ == "__main__":
    # Try to find and load the latest checkpoint
    latest_checkpoint = find_latest_checkpoint()
    
    trainer = ProgressiveTrainer(
        base_grid_size=(30, 30),
        base_num_agents=8,
        max_levels=10,
        success_threshold=0.9,
        min_episodes_per_level=100,
        visualize=True,  # Enable visualization
        num_parallel_envs=4,  # Number of parallel environments
        load_checkpoint=latest_checkpoint  # Load latest checkpoint if available
    )
    trainer.train(num_episodes=10000) 