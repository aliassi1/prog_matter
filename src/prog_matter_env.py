import numpy as np
import random
import math
import pygame
from gym import spaces
from pettingzoo import ParallelEnv
from typing import Dict, List, Tuple, Any
from scipy.optimize import linear_sum_assignment

# Constants for directions
DIRECTIONS = [
    (-1, 0),  # up
    (1, 0),  # down
    (0, -1),  # left
    (0, 1),  # right
    (-1, -1),  # up-left
    (-1, 1),  # up-right
    (1, -1),  # down-left
    (1, 1),  # down-right
    (0, 0),  # stay
]

class ProgrammableMatterEnv(ParallelEnv):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "name": "programmable_matter_v0",
    }

    def __init__(
        self,
        grid_size: Tuple[int, int] = (30, 30),
        num_agents: int = 8,
        max_steps: int = 200,
        render_mode: str = None,
        connectivity_weight: float = 1.0,
        progress_weight: float = 0.1,
        step_penalty: float = 0.01,
        completion_reward: float = 10.0,
        obstacle_positions=None,
    ):
        """Initialize the programmable matter environment."""
        super().__init__()
        self.n, self.m = grid_size
        self._num_agents = num_agents  # Change this line to use a protected attribute
        self.max_steps = max_steps
        self.render_mode = render_mode

        # Reward weights
        self.connectivity_weight = connectivity_weight
        self.progress_weight = progress_weight
        self.step_penalty = step_penalty
        self.completion_reward = completion_reward

        # Directions: Up, Down, Left, Right, Diagonals, Stay
        self.directions = DIRECTIONS

        # Initialize agents
        self.agents = {}
        self.agent_positions = {}
        self.agent_targets = {}
        self.previous_distances = {}
        self.steps_taken = 0

        # Define agent IDs
        self.possible_agents = [
            f"agent_{i}" for i in range(self._num_agents)
        ]  # Use _num_agents here
        self.agents = {agent_id: None for agent_id in self.possible_agents}

        # Define observation and action spaces
        # Local observation: 7x7 grid around agent (3 channels: agents, obstacles, targets)
        # Plus agent's position (2), target position (2), and distance to target (1)
        self.observation_spaces = {
            agent_id: spaces.Dict(
                {
                    "local_grid": spaces.Box(
                        low=0, high=1, shape=(7, 7, 3), dtype=np.float32
                    ),
                    "agent_pos": spaces.Box(
                        low=0, high=max(self.n, self.m), shape=(2,), dtype=np.float32
                    ),
                    "target_pos": spaces.Box(
                        low=0, high=max(self.n, self.m), shape=(2,), dtype=np.float32
                    ),
                    "target_distance": spaces.Box(
                        low=0,
                        high=math.sqrt(self.n**2 + self.m**2),
                        shape=(1,),
                        dtype=np.float32,
                    ),
                }
            )
            for agent_id in self.possible_agents
        }

        # Action space: 9 possible moves (8 directions + stay)
        self.action_spaces = {
            agent_id: spaces.Discrete(len(self.directions))
            for agent_id in self.possible_agents
        }

        # Create the grid (1 for free space, 0 for obstacles)
        self.grid = np.ones((self.n, self.m), dtype=np.int8)

        # Add obstacles if provided
        if obstacle_positions:
            for pos in obstacle_positions:
                x, y = pos
                if 0 <= x < self.n and 0 <= y < self.m:
                    self.grid[x, y] = 0

        # Setup for rendering
        self.renderer = None

    def reset(self, seed=None, return_info=False, options=None):
        """Reset the environment to initial state."""
        if seed is not None:
            np.random.seed(seed)

        # Reset step counter
        self.steps_taken = 0

        # Reset agents
        self.agents = self.possible_agents.copy()

        # Random initialization of agent positions (ensuring connectivity)
        self.agent_positions = {}
        self.agent_targets = {}

        # Create a connected initial configuration
        self._place_connected_agents()

        # Assign targets to agents
        self._assign_targets()

        # Calculate initial distances to targets
        self.previous_distances = {
            agent_id: self._calculate_distance(
                self.agent_positions[agent_id], self.agent_targets[agent_id]
            )
            for agent_id in self.agents
        }

        # Compute observations
        observations = {}
        for agent_id in self.agents:
            observations[agent_id] = self._get_observation(agent_id)

        if return_info:
            infos = {agent_id: {} for agent_id in self.agents}
            return observations, infos

        return observations

    def step(self, actions):
        """Execute one step for all agents based on their actions."""
        self.steps_taken += 1

        # Store proposed new positions
        proposed_positions = {}
        for agent_id, action in actions.items():
            dx, dy = self.directions[action]
            x, y = self.agent_positions[agent_id]
            new_x, new_y = x + dx, y + dy

            # Check if position is valid (in bounds and not an obstacle)
            if (
                0 <= new_x < self.n
                and 0 <= new_y < self.m
                and self.grid[new_x, new_y] == 1
            ):
                proposed_positions[agent_id] = (new_x, new_y)
            else:
                proposed_positions[agent_id] = (x, y)  # Stay in place if invalid

        # Resolve collisions and ensure connectivity
        self._resolve_moves(proposed_positions)

        # Calculate rewards
        rewards = {}
        for agent_id in self.agents:
            rewards[agent_id] = self._calculate_reward(agent_id)

        # Update previous distances
        for agent_id in self.agents:
            self.previous_distances[agent_id] = self._calculate_distance(
                self.agent_positions[agent_id], self.agent_targets[agent_id]
            )

        # Check if done
        dones = {}
        all_at_target = True
        for agent_id in self.agents:
            at_target = self.agent_positions[agent_id] == self.agent_targets[agent_id]
            if not at_target:
                all_at_target = False
            dones[agent_id] = at_target or self.steps_taken >= self.max_steps

        # If max steps reached, all agents are done
        if self.steps_taken >= self.max_steps:
            dones = {agent_id: True for agent_id in self.agents}

        # Global termination if all agents reached targets
        dones["__all__"] = all_at_target or self.steps_taken >= self.max_steps

        # Compute observations
        observations = {}
        for agent_id in self.agents:
            observations[agent_id] = self._get_observation(agent_id)

        # Additional info
        infos = {agent_id: {} for agent_id in self.agents}

        # Optionally render
        if self.render_mode == "human":
            self.render()

        return observations, rewards, dones, infos

    def _place_connected_agents(self):
        """Place agents on the grid in a connected configuration."""
        # Start with one agent
        while True:
            start_x = np.random.randint(1, self.n - 1)
            start_y = np.random.randint(1, self.m - 1)
            if self.grid[start_x, start_y] == 1:  # Only place on free cells
                break

        self.agent_positions[self.possible_agents[0]] = (start_x, start_y)

        # Place remaining agents to ensure connectivity
        placed_agents = [self.possible_agents[0]]

        for i in range(1, self.num_agents):
            agent_id = self.possible_agents[i]

            # Try to find a valid adjacent position to an existing agent
            valid_positions = []
            for placed_agent in placed_agents:
                x, y = self.agent_positions[placed_agent]

                for dx, dy in self.directions[:-1]:  # Exclude 'stay'
                    new_x, new_y = x + dx, y + dy

                    # Check if position is valid, not occupied, and not an obstacle
                    if (
                        0 <= new_x < self.n
                        and 0 <= new_y < self.m
                        and self.grid[new_x, new_y] == 1
                        and all(
                            (new_x, new_y) != self.agent_positions.get(a)
                            for a in placed_agents
                        )
                    ):
                        valid_positions.append((new_x, new_y))

            if not valid_positions:
                # Fallback: place close to existing agents but not necessarily adjacent
                placed_positions = list(self.agent_positions.values())
                while not valid_positions:
                    x, y = placed_positions[np.random.randint(len(placed_positions))]
                    search_radius = 2

                    for dx in range(-search_radius, search_radius + 1):
                        for dy in range(-search_radius, search_radius + 1):
                            new_x, new_y = x + dx, y + dy

                            if (
                                0 <= new_x < self.n
                                and 0 <= new_y < self.m
                                and self.grid[new_x, new_y] == 1
                                and all(
                                    (new_x, new_y) != self.agent_positions.get(a)
                                    for a in placed_agents
                                )
                            ):
                                valid_positions.append((new_x, new_y))

                    search_radius += 1
                    if search_radius > 5:  # Avoid infinite loop
                        # If we can't find valid positions, restart the whole process
                        self.agent_positions = {}
                        return self._place_connected_agents()

            # Choose a random valid position
            position = valid_positions[np.random.randint(len(valid_positions))]
            self.agent_positions[agent_id] = position
            placed_agents.append(agent_id)

    def _assign_targets(self, clustering_factor=0.5):
        """
        Assign targets to agents with optional clustering.

        Args:
            clustering_factor: 0-1 value where 0 means random targets and
                             1 means highly clustered targets
        """
        # Create target positions (with clustering)
        target_positions = []

        if clustering_factor > 0:
            # Start with a central point
            center_x = np.random.randint(self.n // 4, 3 * self.n // 4)
            center_y = np.random.randint(self.m // 4, 3 * self.m // 4)

            # Maximum distance from center (scaled by clustering factor)
            max_radius = min(self.n, self.m) * (1 - clustering_factor) * 0.5

            while len(target_positions) < self.num_agents:
                # Generate random angle and distance
                angle = 2 * math.pi * random.random()
                distance = max_radius * random.random()

                # Convert to coordinates
                x = int(center_x + distance * math.cos(angle))
                y = int(center_y + distance * math.sin(angle))

                # Ensure coordinates are within bounds and not obstacles
                if (
                    0 <= x < self.n
                    and 0 <= y < self.m
                    and self.grid[x, y] == 1
                    and (x, y) not in target_positions
                ):
                    target_positions.append((x, y))
        else:
            # Completely random targets
            while len(target_positions) < self.num_agents:
                x = np.random.randint(0, self.n)
                y = np.random.randint(0, self.m)

                if (
                    self.grid[x, y] == 1
                    and (x, y) not in target_positions
                    and not any((x, y) == pos for pos in self.agent_positions.values())
                ):
                    target_positions.append((x, y))

        # Assign targets using Hungarian algorithm for optimal assignment
        cost_matrix = np.zeros((self.num_agents, self.num_agents))
        for i, agent_id in enumerate(self.agents):
            agent_pos = self.agent_positions[agent_id]
            for j, target_pos in enumerate(target_positions):
                cost_matrix[i, j] = self._calculate_distance(agent_pos, target_pos)

        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        for i, agent_id in enumerate(self.agents):
            self.agent_targets[agent_id] = target_positions[col_ind[i]]

    def _get_observation(self, agent_id):
        """Create observation for an agent."""
        agent_x, agent_y = self.agent_positions[agent_id]
        target_x, target_y = self.agent_targets[agent_id]

        # Local grid observation (7x7 area around agent)
        local_grid = np.zeros((7, 7, 3), dtype=np.float32)

        # Fill local grid (channel 0: agents, channel 1: obstacles, channel 2: targets)
        for i in range(-3, 4):
            for j in range(-3, 4):
                grid_x, grid_y = agent_x + i, agent_y + j

                if 0 <= grid_x < self.n and 0 <= grid_y < self.m:
                    # Check for agents
                    for other_id, pos in self.agent_positions.items():
                        if pos == (grid_x, grid_y) and other_id != agent_id:
                            local_grid[i + 3, j + 3, 0] = 1

                    # Check for obstacles (if grid cell is 0)
                    if self.grid[grid_x, grid_y] == 0:
                        local_grid[i + 3, j + 3, 1] = 1

                    # Check for targets
                    for other_id, target in self.agent_targets.items():
                        if target == (grid_x, grid_y):
                            local_grid[i + 3, j + 3, 2] = 1
                else:
                    # Out of bounds is considered an obstacle
                    local_grid[i + 3, j + 3, 1] = 1

        # Agent's own target
        local_grid[3, 3, 2] = (
            1 if (agent_x, agent_y) == self.agent_targets[agent_id] else 0
        )

        # Calculate distance to target
        distance = self._calculate_distance((agent_x, agent_y), (target_x, target_y))

        return {
            "local_grid": local_grid,
            "agent_pos": np.array([agent_x, agent_y], dtype=np.float32),
            "target_pos": np.array([target_x, target_y], dtype=np.float32),
            "target_distance": np.array([distance], dtype=np.float32),
        }

    def _calculate_distance(self, pos1, pos2):
        """Calculate Euclidean distance between two positions."""
        return math.sqrt((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2)

    def _calculate_reward(self, agent_id):
        """Calculate reward for an agent."""
        reward = 0.0

        # Get current position and target
        current_pos = self.agent_positions[agent_id]
        target_pos = self.agent_targets[agent_id]

        # Calculate current distance to target
        current_distance = self._calculate_distance(current_pos, target_pos)

        # Progress reward (positive if agent got closer to target)
        previous_distance = self.previous_distances[agent_id]
        progress = previous_distance - current_distance
        reward += progress * self.progress_weight

        # Completion reward if agent reached its target
        if current_pos == target_pos:
            reward += self.completion_reward

        # Connectivity penalty (check if agents are still connected)
        all_positions = list(self.agent_positions.values())
        if not self._is_connected(all_positions):
            reward -= self.connectivity_weight

        # Step penalty to encourage efficiency
        reward -= self.step_penalty

        return reward

    def _is_connected(self, positions):
        """Check if all positions are connected (8-connectivity)."""
        if not positions:
            return True

        # Create a set of positions for O(1) lookup
        pos_set = set(positions)

        # Start BFS from the first position
        from collections import deque

        visited = set()
        queue = deque([positions[0]])
        visited.add(positions[0])

        # 8-directional connectivity check
        while queue:
            x, y = queue.popleft()
            for dx, dy in self.directions[:-1]:  # Exclude 'stay'
                nx, ny = x + dx, y + dy
                if (nx, ny) in pos_set and (nx, ny) not in visited:
                    visited.add((nx, ny))
                    queue.append((nx, ny))

        # If all positions are visited, the graph is connected
        return len(visited) == len(positions)

    def _resolve_moves(self, proposed_positions):
        """Resolve conflicting moves and ensure connectivity constraints."""
        # Sort agents by priority (could be distance to target or other metrics)
        agent_priorities = {}
        for agent_id in self.agents:
            distance = self._calculate_distance(
                self.agent_positions[agent_id], self.agent_targets[agent_id]
            )
            agent_priorities[agent_id] = distance

        # Sort agents by priority (higher priority first)
        sorted_agents = sorted(self.agents, key=lambda a: agent_priorities[a])

        # Process agents in order of priority
        final_positions = self.agent_positions.copy()

        for agent_id in sorted_agents:
            if agent_id not in proposed_positions:
                continue

            new_pos = proposed_positions[agent_id]
            old_pos = self.agent_positions[agent_id]

            # Skip if position is unchanged
            if new_pos == old_pos:
                continue

            # Check for collisions with other agents' final positions
            if new_pos in final_positions.values():
                continue

            # Tentatively accept the move
            tentative_positions = final_positions.copy()
            tentative_positions[agent_id] = new_pos

            # Check if this breaks connectivity
            if self._is_connected(list(tentative_positions.values())):
                final_positions[agent_id] = new_pos
            # else: revert to old position (already in final_positions)

        # Update agent positions
        self.agent_positions = final_positions

    def render(self):
        """Render the environment."""
        if self.render_mode is None:
            return

        if self.renderer is None:
            pygame.init()

            # Define cell size and screen dimensions
            cell_size = 20
            screen_width = self.m * cell_size
            screen_height = self.n * cell_size

            self.renderer = {
                "screen": pygame.display.set_mode((screen_width, screen_height)),
                "cell_size": cell_size,
                "clock": pygame.time.Clock(),
                "font": pygame.font.SysFont(None, 15),
            }

            pygame.display.set_caption("Programmable Matter Simulation")

        screen = self.renderer["screen"]
        cell_size = self.renderer["cell_size"]

        # Clear screen
        screen.fill((255, 255, 255))

        # Draw grid
        for i in range(self.n):
            for j in range(self.m):
                if self.grid[i, j] == 0:  # Obstacle
                    pygame.draw.rect(
                        screen,
                        (100, 100, 100),
                        pygame.Rect(j * cell_size, i * cell_size, cell_size, cell_size),
                    )

        # Draw targets
        for agent_id, target in self.agent_targets.items():
            tx, ty = target
            target_color = (0, 150, 0)  # Green for targets
            pygame.draw.rect(
                screen,
                target_color,
                pygame.Rect(ty * cell_size, tx * cell_size, cell_size, cell_size),
                2,
            )

        # Draw agents
        for agent_id, pos in self.agent_positions.items():
            x, y = pos
            agent_color = (0, 0, 255)  # Blue for agents

            # Draw circle for agent
            pygame.draw.circle(
                screen,
                agent_color,
                (y * cell_size + cell_size // 2, x * cell_size + cell_size // 2),
                cell_size // 3,
            )

            # Draw agent ID
            text = self.renderer["font"].render(
                str(agent_id.split("_")[1]), True, (255, 255, 255)
            )
            text_rect = text.get_rect(
                center=(y * cell_size + cell_size // 2, x * cell_size + cell_size // 2)
            )
            screen.blit(text, text_rect)

        # Draw connections between agents
        all_positions = list(self.agent_positions.values())
        for i, (x1, y1) in enumerate(all_positions):
            for j, (x2, y2) in enumerate(all_positions[i + 1 :], i + 1):
                # Check if agents are adjacent (8-connected)
                if max(abs(x1 - x2), abs(y1 - y2)) <= 1:
                    pygame.draw.line(
                        screen,
                        (200, 200, 255),
                        (
                            y1 * cell_size + cell_size // 2,
                            x1 * cell_size + cell_size // 2,
                        ),
                        (
                            y2 * cell_size + cell_size // 2,
                            x2 * cell_size + cell_size // 2,
                        ),
                        2,
                    )

        pygame.display.flip()
        self.renderer["clock"].tick(10)  # 10 FPS

        if self.render_mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(screen)), axes=(1, 0, 2)
            )

    # Add these property methods to the ProgrammableMatterEnv class:
    @property
    def num_agents(self):
        return self._num_agents

    @num_agents.setter
    def num_agents(self, value):
        self._num_agents = value
        # Update possible_agents if needed
        self.possible_agents = [f"agent_{i}" for i in range(self._num_agents)]
