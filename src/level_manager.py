import numpy as np
from typing import Dict, List, Tuple, Optional
from prog_matter_env import ProgrammableMatterEnv
from collections import deque
import pygame

class LevelManager:
    def __init__(
        self,
        base_grid_size: Tuple[int, int] = (20, 20),
        base_num_agents: int = 1,  # Start with 1 agent for single agent training
        max_levels: int = 9,  # Increased to include single agent training
        success_threshold: float = 0.8,
        min_episodes_per_level: int = 100,
    ):
        self.base_grid_size = base_grid_size
        self.base_num_agents = base_num_agents
        self.max_levels = max_levels
        self.success_threshold = success_threshold
        self.min_episodes_per_level = min_episodes_per_level
        
        self.current_level = 1
        self.episode_count = 0
        self.success_count = 0
        self.level_history = []
        
        # Level progression parameters
        self.grid_size_increase = (5, 5)
        self.agent_increase = 1  # Start with 1 agent increase after single agent training
        self.obstacle_density_increase = 0.05
        
        # Level-specific parameters
        self.level_focus = {
            1: {
                "name": "Single Agent Training",
                "description": "Teach individual agents to reach targets independently",
                "obstacle_density": 0.0,
                "target_pattern": "single",
                "connectivity_weight": 0.0,
                "progress_weight": 2.0,
                "step_penalty": 0.1,
                "num_agents": 1
            },
            2: {
                "name": "Random Face",
                "description": "Agents form a random connected shape",
                "obstacle_density": 0.0,
                "target_pattern": "random_connected",
                "connectivity_weight": 1.5,
                "progress_weight": 1.2,
                "step_penalty": 0.1,
                "num_agents": 6
            },
            3: {
                "name": "Square Formation",
                "description": "Form a square shape with increased agents",
                "obstacle_density": 0.0,
                "target_pattern": "square",
                "connectivity_weight": 1.2,
                "progress_weight": 1.0,
                "step_penalty": 0.1,
                "num_agents": 10
            },
            4: {
                "name": "Complex Navigation",
                "description": "Navigate through obstacles to form complex shapes",
                "obstacle_density": 0.0,
                "target_pattern": "square",
                "connectivity_weight": 1.0,
                "progress_weight": 1.0,
                "step_penalty": 0.1,
                "num_agents": 12
            },
            5: {
                "name": "Circle Formation",
                "description": "Form a circular pattern with many agents",
                "obstacle_density": 0.0,
                "target_pattern": "circle",
                "connectivity_weight": 1.0,
                "progress_weight": 1.0,
                "step_penalty": 0.1,
                "num_agents": 14
            },
            # 8: {
            #     "name": "Diamond Formation",
            #     "description": "Form a diamond shape with obstacles",
            #     "obstacle_density": 0.0,
            #     "target_pattern": "diamond",
            #     "connectivity_weight": 1.0,
            #     "progress_weight": 1.0,
            #     "step_penalty": 0.1,
            #     "num_agents": 16
            # },
            # 9: {
            #     "name": "Complex Pattern",
            #     "description": "Form complex patterns with many agents and obstacles",
            #     "obstacle_density": 0.0,
            #     "target_pattern": "complex",
            #     "connectivity_weight": 1.0,
            #     "progress_weight": 1.0,
            #     "step_penalty": 0.1,
            #     "num_agents": 18
            # }
        }
        
        self.success_window = deque(maxlen=self.min_episodes_per_level)
        
    def get_current_level_params(self) -> Dict:
        """Get parameters for current level"""
        level_focus = self.level_focus[self.current_level]
        
        # Use constant grid size for all levels
        grid_size = self.base_grid_size
        
        # Use level-specific number of agents
        num_agents = level_focus['num_agents']
        
        # Calculate completion reward based on level difficulty
        base_completion_reward = 50.0  # Increased base reward
        level_multiplier = 1.0 + (self.current_level - 1) * 0.5  # Increases by 50% per level
        completion_reward = base_completion_reward * level_multiplier
        
        return {
            'grid_size': grid_size,
            'num_agents': num_agents,
            'obstacle_density': level_focus['obstacle_density'],
            'target_pattern': level_focus['target_pattern'],
            'connectivity_weight': level_focus['connectivity_weight'],
            'progress_weight': level_focus['progress_weight'],
            'step_penalty': level_focus['step_penalty'],
            'completion_reward': completion_reward,
            'level_name': level_focus['name'],
            'level_description': level_focus['description']
        }
    
    def _generate_target_positions(self, grid_size: Tuple[int, int], num_agents: int, pattern: str) -> List[Tuple[int, int]]:
        """Generate target positions based on the specified pattern"""
        n, m = grid_size
        targets = []
        margin = 2
        
        if pattern == "single":
            x = np.random.randint(margin, n - margin)
            y = np.random.randint(margin, m - margin)
            targets.append((x, y))
            return targets
            
        elif pattern == "line":
            y = m // 2
            spacing = (n - 2 * margin) / (num_agents - 1)
            for i in range(num_agents):
                x = margin + int(i * spacing)
                targets.append((x, y))
            return targets
            
        # All other patterns: connected random walk
        x = np.random.randint(margin, n - margin)
        y = np.random.randint(margin, m - margin)
        targets.append((x, y))
        visited = set(targets)
        while len(targets) < num_agents:
            tx, ty = targets[np.random.randint(0, len(targets))]
            neighbors = [
                (tx + 1, ty), (tx - 1, ty), (tx, ty + 1), (tx, ty - 1)
            ]
            np.random.shuffle(neighbors)
            for nx, ny in neighbors:
                if (margin <= nx < n - margin and margin <= ny < m - margin and
                    (nx, ny) not in visited):
                    targets.append((nx, ny))
                    visited.add((nx, ny))
                    break
        
        return targets
    
    def create_level_env(self) -> ProgrammableMatterEnv:
        """Create environment for current level"""
        params = self.get_current_level_params()
        
        # Generate obstacles based on density
        grid_area = params['grid_size'][0] * params['grid_size'][1]
        num_obstacles = int(grid_area * params['obstacle_density'])
        
        # Generate random obstacle positions
        obstacle_positions = []
        
        # Generate target positions based on pattern
        target_positions = self._generate_target_positions(
            params['grid_size'],
            params['num_agents'],
            params['target_pattern']
        )
        
        return ProgrammableMatterEnv(
            grid_size=params['grid_size'],
            num_agents=params['num_agents'],
            max_steps=200,  # Increased max steps for more complex levels
            obstacle_positions=obstacle_positions,
            target_positions=target_positions,
            connectivity_weight=params['connectivity_weight'],
            progress_weight=params['progress_weight'],
            step_penalty=params['step_penalty'],
            completion_reward=params['completion_reward']
        )
    
    def update_level_progress(self, success: bool):
        """Update level progress based on episode outcome"""
        self.episode_count += 1
        if success:
            self.success_count += 1
        self.success_window.append(success)
        # Use moving window for success rate
        window_success_rate = sum(self.success_window) / len(self.success_window)
        if (len(self.success_window) == self.min_episodes_per_level and 
            window_success_rate >= self.success_threshold):
            self.advance_level()
    
    def advance_level(self):
        """Advance to the next level if possible"""
        if self.current_level < self.max_levels:
            # Calculate success rate with safety check
            success_rate = self.success_count / max(1, self.episode_count)
            # Record level completion stats
            self.level_history.append({
                'level': self.current_level,
                'episodes': self.episode_count,
                'successes': self.success_count,
                'success_rate': success_rate
            })
            # Reset counters for new level
            self.episode_count = 0
            self.success_count = 0
            self.success_window.clear()
            # Advance to next level
            self.current_level += 1
            print(f"Advancing to level {self.current_level}")
        else:
            print("Maximum level reached!")
    
    def get_level_stats(self) -> Dict:
        """Get statistics for current level"""
        params = self.get_current_level_params()
        return {
            'current_level': self.current_level,
            'level_name': params['level_name'],
            'level_description': params['level_description'],
            'episodes_played': self.episode_count,
            'success_rate': self.success_count / max(1, self.episode_count),
            'level_history': self.level_history,
            'current_target_pattern': params['target_pattern'],
            'grid_size': params['grid_size'],
            'num_agents': params['num_agents'],
            'obstacle_density': params['obstacle_density']
        }
    
    def print_level_details(self, level: int):
        """Prints the configuration of a specific level."""
        if level not in self.level_focus:
            print(f"Level {level} is not defined.")
            return
        prev_level = self.current_level
        self.current_level = level
        params = self.get_current_level_params()
        print(f"\nLevel {level}: {params['level_name']}")
        print(f"  Description: {params['level_description']}")
        print(f"  Grid Size: {params['grid_size']}")
        print(f"  Number of Agents: {params['num_agents']}")
        print(f"  Target Pattern: {params['target_pattern']}")
        print(f"  Obstacle Density: {params['obstacle_density']}")
        print(f"  Connectivity Weight: {params['connectivity_weight']}")
        print(f"  Progress Weight: {params['progress_weight']}")
        print(f"  Step Penalty: {params['step_penalty']}")
        print(f"  Completion Reward: {params['completion_reward']}")
        self.current_level = prev_level

    def plot_level(self, level: int, cell_size: int = 30):
        """Visualize the grid, obstacles, and targets for a given level using pygame."""
        if level not in self.level_focus:
            print(f"Level {level} is not defined.")
            return

        prev_level = self.current_level
        self.current_level = level
        params = self.get_current_level_params()
        grid_size = params['grid_size']
        num_agents = params['num_agents']
        pattern = params['target_pattern']
        obstacle_density = params['obstacle_density']

        # Generate obstacles and targets as in create_level_env
        grid_area = grid_size[0] * grid_size[1]
        num_obstacles = int(grid_area * obstacle_density)
        obstacle_positions = []
        target_positions = self._generate_target_positions(grid_size, num_agents, pattern)

        # Pygame setup
        pygame.init()
        width, height = grid_size[0] * cell_size, grid_size[1] * cell_size
        screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption(f"Level {level}: {params['level_name']}")
        clock = pygame.time.Clock()

        # Colors
        BG_COLOR = (240, 240, 240)
        GRID_COLOR = (200, 200, 200)
        OBSTACLE_COLOR = (80, 80, 80)
        TARGET_COLOR = (200, 50, 50)
        
        running = True
        next_level = False
        while running and not next_level:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE or event.key == pygame.K_RIGHT:
                        next_level = True

            screen.fill(BG_COLOR)

            # Draw grid
            for x in range(grid_size[0]):
                for y in range(grid_size[1]):
                    rect = pygame.Rect(x * cell_size, y * cell_size, cell_size, cell_size)
                    pygame.draw.rect(screen, GRID_COLOR, rect, 1)

            # Draw obstacles
            for (x, y) in obstacle_positions:
                rect = pygame.Rect(x * cell_size, y * cell_size, cell_size, cell_size)
                pygame.draw.rect(screen, OBSTACLE_COLOR, rect)

            # Draw targets
            for (x, y) in target_positions:
                rect = pygame.Rect(x * cell_size, y * cell_size, cell_size, cell_size)
                pygame.draw.ellipse(screen, TARGET_COLOR, rect)

            pygame.display.flip()
            clock.tick(30)

        pygame.quit()
        self.current_level = prev_level

if __name__ == "__main__":
    lm = LevelManager()
    for level in range(1, lm.max_levels + 1):
        lm.print_level_details(level)
        lm.plot_level(level) 