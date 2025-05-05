import numpy as np
import random
import math
import pygame
from gym import spaces
from pettingzoo import ParallelEnv
from typing import Dict, List, Tuple, Any
from scipy.optimize import linear_sum_assignment
import heapq

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
        step_penalty: float = 0.1,
        completion_reward: float = 100.0,
        obstacle_positions=None,
        target_positions=None,
        fast_mode: bool = False,
    ):
        """Initialize the programmable matter environment."""
        super().__init__()
        self.n, self.m = grid_size
        self._num_agents = num_agents
        self.max_steps = max_steps
        self.render_mode = render_mode
        self.target_positions = target_positions

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

        # Add visited positions
        self.visited_positions = {agent_id: set() for agent_id in self.agents}

        self.deadlock_counter = 0
        self.last_positions = None
        self.fast_mode = fast_mode

        self.reached_target_once = {agent_id: False for agent_id in self.agents}

    def reset(self, seed=None, return_info=False, options=None):
        """Reset the environment to initial state."""
        if seed is not None:
            np.random.seed(seed)

        # Reset step counter
        self.steps_taken = 0

        # Reset agents
        self.agents = list(self.possible_agents)  # Ensure this is a list, not a dict

        # Random initialization of agent positions (ensuring connectivity)
        self.agent_positions = {}
        self.agent_targets = {}

        # Create a connected initial configuration
        self._place_connected_agents()

        # For level 2, always generate a line for targets
        if hasattr(self, 'level') and self.level == 2:
            self.target_positions = self._generate_target_positions(self.num_agents, pattern='line')
        self._assign_targets()

        if not self.fast_mode:
            print("Initial agent positions and targets:")
            for agent_id in self.agents:
                print(f"  {agent_id}: position={self.agent_positions[agent_id]}, target={self.agent_targets[agent_id]}")

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
        # Store previous positions for deadlock detection and overlap revert
        prev_positions = self.agent_positions.copy()
        # Store proposed new positions
        proposed_positions = {}
        for agent_id, action in actions.items():
            dx, dy = self.directions[action]
            x, y = self.agent_positions[agent_id]
            new_x, new_y = x + dx, y + dy
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
        # Deadlock detection: if no agent moved, increment counter
        if self.last_positions is not None and all(self.agent_positions[aid] == self.last_positions[aid] for aid in self.agents):
            self.deadlock_counter += 1
        else:
            self.deadlock_counter = 0
        self.last_positions = self.agent_positions.copy()
        # If deadlock for 10 steps, randomly move one agent (to a truly free cell)
        if self.deadlock_counter >= 10:
            agent_id = random.choice(list(self.agents))
            possible_moves = [i for i, (dx, dy) in enumerate(self.directions)
                             if 0 <= self.agent_positions[agent_id][0] + dx < self.n
                             and 0 <= self.agent_positions[agent_id][1] + dy < self.m
                             and self.grid[self.agent_positions[agent_id][0] + dx, self.agent_positions[agent_id][1] + dy] == 1
                             and (self.agent_positions[agent_id][0] + dx, self.agent_positions[agent_id][1] + dy) not in self.agent_positions.values()]
            if possible_moves:
                move = random.choice(possible_moves)
                dx, dy = self.directions[move]
                x, y = self.agent_positions[agent_id]
                new_pos = (x + dx, y + dy)
                self.agent_positions[agent_id] = new_pos
                if not self.fast_mode:
                    print(f"[DEADLOCK BREAK] Randomly moved {agent_id} to {new_pos} to break deadlock.")
            self.deadlock_counter = 0
        # FINAL ATOMIC ANTI-OVERLAP CHECK
        pos_counts = {}
        for aid, pos in self.agent_positions.items():
            pos_counts[pos] = pos_counts.get(pos, 0) + 1
        overlap_agents = [aid for aid, pos in self.agent_positions.items() if pos_counts[pos] > 1]
        if overlap_agents:
            if not self.fast_mode:
                print(f"[FINAL ANTI-OVERLAP] Overlap detected after all moves! Reverting moves for agents: {overlap_agents}")
            for aid in overlap_agents:
                self.agent_positions[aid] = prev_positions[aid]
        if len(set(self.agent_positions.values())) != len(self.agent_positions):
            print("Agent positions (overlap detected after step):", self.agent_positions)
        assert len(set(self.agent_positions.values())) == len(self.agent_positions), "Agent overlap detected after all moves!"
        # Calculate rewards
        rewards = {}
        for agent_id in self.agents:
            pos = self.agent_positions[agent_id]
            rewards[agent_id] = self._calculate_reward(agent_id)
            if pos in self.visited_positions[agent_id]:
                rewards[agent_id] -= 0.5
            self.visited_positions[agent_id].add(pos)
        # Add large negative penalty if max_steps reached and agent did not reach goal
        if self.steps_taken >= self.max_steps:
            for agent_id in self.agents:
                if self.agent_positions[agent_id] != self.agent_targets[agent_id]:
                    # Proportional penalty based on distance to target
                    dist = self._calculate_distance(self.agent_positions[agent_id], self.agent_targets[agent_id])
                    # Penalty is -50 if far, less if close
                    penalty = -10 - 40 * (dist / (self.n + self.m))
                    rewards[agent_id] += penalty
        # Update previous distances
        for agent_id in self.agents:
            self.previous_distances[agent_id] = self._calculate_distance(
                self.agent_positions[agent_id], self.agent_targets[agent_id]
            )
        # Check if done
        dones = {}
        target_reached = {}
        all_at_target = True
        for agent_id in self.agents:
            at_target = self.agent_positions[agent_id] == self.agent_targets[agent_id]
            target_reached[agent_id] = at_target
            if not at_target:
                all_at_target = False
            # In level 1, agents are done when they reach their target
            # In other levels, agents stay active until all reach targets
            if getattr(self, 'level', 1) == 1:
                dones[agent_id] = at_target or self.steps_taken >= self.max_steps
            else:
                # In level 2+, agents are never done individually
                dones[agent_id] = False
        # If max steps reached, all agents are done
        if self.steps_taken >= self.max_steps:
            dones = {agent_id: True for agent_id in self.agents}
        # Global termination if all agents reached targets (only in level 1)
        if getattr(self, 'level', 1) == 1:
            dones["__all__"] = all_at_target or self.steps_taken >= self.max_steps
        else:
            dones["__all__"] = self.steps_taken >= self.max_steps
        # Compute observations
        observations = {}
        for agent_id in self.agents:
            observations[agent_id] = self._get_observation(agent_id)
        # Additional info
        infos = {agent_id: {"target_reached": target_reached[agent_id]} for agent_id in self.agents}
        # Optionally render
        if self.render_mode == "human" and not self.fast_mode:
            self.render()
        return observations, rewards, dones, infos

    def _place_connected_agents(self):
        """Place agents on the grid in a connected configuration."""
        # For level 2, place agents in a straight line
        if hasattr(self, 'level') and self.level == 2:
            self._place_agents_in_line()
            # Assert no overlap after placement
            assert len(set(self.agent_positions.values())) == len(self.agent_positions), "Agent overlap detected after line placement!"
            return
        # Start with one agent
        while True:
            start_x = np.random.randint(1, self.n - 1)
            start_y = np.random.randint(1, self.m - 1)
            if self.grid[start_x, start_y] == 1:  # Only place on free cells
                break
        self.agent_positions[self.possible_agents[0]] = (start_x, start_y)
        placed_agents = [self.possible_agents[0]]
        for i in range(1, self.num_agents):
            agent_id = self.possible_agents[i]
            valid_positions = []
            for placed_agent in placed_agents:
                x, y = self.agent_positions[placed_agent]
                for dx, dy in self.directions[:-1]:  # Exclude 'stay'
                    new_x, new_y = x + dx, y + dy
                    if (
                        0 <= new_x < self.n
                        and 0 <= new_y < self.m
                        and self.grid[new_x, new_y] == 1
                        and all((new_x, new_y) != self.agent_positions.get(a) for a in placed_agents)
                    ):
                        valid_positions.append((new_x, new_y))
            if not valid_positions:
                placed_positions = list(self.agent_positions.values())
                search_radius = 2
                while not valid_positions:
                    x, y = placed_positions[np.random.randint(len(placed_positions))]
                    for dx in range(-search_radius, search_radius + 1):
                        for dy in range(-search_radius, search_radius + 1):
                            new_x, new_y = x + dx, y + dy
                            if (
                                0 <= new_x < self.n
                                and 0 <= new_y < self.m
                                and self.grid[new_x, new_y] == 1
                                and all((new_x, new_y) != self.agent_positions.get(a) for a in placed_agents)
                            ):
                                valid_positions.append((new_x, new_y))
                    search_radius += 1
                    if search_radius > 5:
                        self.agent_positions = {}
                        return self._place_connected_agents()
            position = valid_positions[np.random.randint(len(valid_positions))]
            self.agent_positions[agent_id] = position
            placed_agents.append(agent_id)
        # Assert no overlap after placement
        assert len(set(self.agent_positions.values())) == len(self.agent_positions), "Agent overlap detected after connected placement!"

    def _place_agents_in_line(self):
        self.agent_positions = {}  # Clear previous positions!
        margin = 2
        y_agents = self.m // 2
        if hasattr(self, 'level') and self.level == 2:
            num_agents = 6
            spacing = 1
            start_x = (self.n - num_agents) // 2
            self.agents = self.possible_agents[:num_agents]  # <--- ADD THIS LINE
            for i, agent_id in enumerate(self.agents):
                x = start_x + i * spacing
                if not (0 <= x < self.n and 0 <= y_agents < self.m):
                    raise ValueError(f"Agent {agent_id} position ({x},{y_agents}) out of bounds!")
                if self.grid[x, y_agents] != 1:
                    raise ValueError(f"Agent {agent_id} position ({x},{y_agents}) is an obstacle!")
                if (x, y_agents) in self.agent_positions.values():
                    raise ValueError(f"Agent {agent_id} position ({x},{y_agents}) would overlap!")
                self.agent_positions[agent_id] = (x, y_agents)
            if len(set(self.agent_positions.values())) != len(self.agent_positions):
                print("Agent positions (overlap detected):", self.agent_positions)
            assert len(set(self.agent_positions.values())) == len(self.agent_positions), "Agent overlap detected after line placement!"
            return
        # For other levels, use the original placement logic
        spacing = 1
        used_positions = set()
        for i, agent_id in enumerate(self.possible_agents):
            x = margin + int(i * spacing)
            while (x, y_agents) in used_positions or self.grid[x, y_agents] != 1:
                x += 1
                if x >= self.n - margin:
                    x = margin
                    y_agents += 1
                    if y_agents >= self.m - margin:
                        y_agents = self.m // 2
            if (x, y_agents) in self.agent_positions.values():
                raise ValueError(f"Agent {agent_id} position ({x},{y_agents}) would overlap!")
            self.agent_positions[agent_id] = (x, y_agents)
            used_positions.add((x, y_agents))
        if len(set(self.agent_positions.values())) != len(self.agent_positions):
            print("Agent positions (overlap detected):", self.agent_positions)
        assert len(set(self.agent_positions.values())) == len(self.agent_positions), "Agent overlap detected after line placement!"

    def _assign_targets(self, clustering_factor=0.5):
        self.agent_targets = {
            agent_id: self.target_positions[i]
            for i, agent_id in enumerate(self.agents)
        }
        if self.target_positions is not None:
            # For level 2, ensure targets are a line and do not overlap with agents
            if hasattr(self, 'level') and self.level == 2:
                overlap = set(self.agent_positions.values()) & set(self.target_positions)
                if overlap:
                    raise ValueError(f"Agent(s) initialized on target(s): {overlap}. Please check placement logic.")
                assert len(set(self.target_positions)) == self.num_agents, "Target overlap detected!"
                # Assign targets in order (no Hungarian for a line)
                self.agent_targets = {
                    agent_id: self.target_positions[i]
                    for i, agent_id in enumerate(self.agents)
                }
                return
            # ... rest of your code for other levels ...
        else:
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

                    # Ensure coordinates are within bounds and not obstacles or agent positions
                    if (
                        0 <= x < self.n
                        and 0 <= y < self.m
                        and self.grid[x, y] == 1
                        and (x, y) not in target_positions
                        and (x, y) not in self.agent_positions.values()
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
                        and (x, y) not in self.agent_positions.values()
                    ):
                        target_positions.append((x, y))

            # Assign targets using Hungarian algorithm for optimal assignment
            cost_matrix = np.zeros((self.num_agents, self.num_agents))
            for i, agent_id in enumerate(self.agents):
                agent_pos = self.agent_positions[agent_id]
                for j, target_pos in enumerate(target_positions):
                    cost_matrix[i, j] = self._calculate_distance(agent_pos, target_pos)

            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            self.agent_targets = {
                agent_id: target_positions[col_ind[i]]
                for i, agent_id in enumerate(self.agents)
            }
            self.target_positions = target_positions
            # Assert all targets are unique
            assert len(set(self.target_positions)) == len(self.target_positions), "Target overlap detected!"

    def _generate_additional_targets(self, num_additional, clustering_factor):
        """Generate additional target positions when needed."""
        additional_targets = []
        
        if clustering_factor > 0:
            # Use existing target positions to determine clustering
            if self.target_positions:
                # Calculate center from existing targets
                center_x = sum(x for x, _ in self.target_positions) / len(self.target_positions)
                center_y = sum(y for _, y in self.target_positions) / len(self.target_positions)
            else:
                center_x = self.n // 2
                center_y = self.m // 2

            max_radius = min(self.n, self.m) * (1 - clustering_factor) * 0.5

            while len(additional_targets) < num_additional:
                angle = 2 * math.pi * random.random()
                distance = max_radius * random.random()
                x = int(center_x + distance * math.cos(angle))
                y = int(center_y + distance * math.sin(angle))

                if (
                    0 <= x < self.n
                    and 0 <= y < self.m
                    and self.grid[x, y] == 1
                    and (x, y) not in self.target_positions
                    and (x, y) not in additional_targets
                ):
                    additional_targets.append((x, y))
        else:
            # Random additional targets
            while len(additional_targets) < num_additional:
                x = np.random.randint(0, self.n)
                y = np.random.randint(0, self.m)

                if (
                    self.grid[x, y] == 1
                    and (x, y) not in self.target_positions
                    and (x, y) not in additional_targets
                    and not any((x, y) == pos for pos in self.agent_positions.values())
                ):
                    additional_targets.append((x, y))

        return additional_targets

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
        reward = 0.0
        current_pos = self.agent_positions[agent_id]
        target_pos = self.agent_targets[agent_id]
        current_distance = self._calculate_distance(current_pos, target_pos)
        previous_distance = self.previous_distances[agent_id]
        progress = previous_distance - current_distance

        # Step penalty (reduced)
        reward -= self.step_penalty * 0.5

        # Progress reward (increased)
        if current_pos != target_pos:
            if progress > 0:
                reward += progress * self.progress_weight * 3.0
            elif progress < 0:
                reward += progress * self.progress_weight * 1.0
            else:
                reward -= self.step_penalty * 0.2

            # Distance-based reward (increased)
            max_possible_distance = math.sqrt(self.n**2 + self.m**2)
            distance_reward = (max_possible_distance - current_distance) / max_possible_distance
            reward += distance_reward * self.progress_weight * 1.0

            # Exploration bonus
            if current_pos not in self.visited_positions[agent_id]:
                reward += 0.2

            # Small positive living reward
            reward += 0.05
        else:
            if not self.reached_target_once[agent_id]:
                reward += 100.0  # Large bonus for reaching the goal (first time only)
                self.reached_target_once[agent_id] = True

        # Connectivity penalty (reduced)
        all_positions = list(self.agent_positions.values())
        if not self._is_connected(all_positions):
            reward -= self.connectivity_weight * 0.5

        # For level 2: line straightness and spacing (keep, but scale down penalty)
        if hasattr(self, 'level') and self.level == 2:
            if self._is_connected(all_positions):
                if len(all_positions) >= 2:
                    sorted_positions = sorted(all_positions, key=lambda p: p[0])
                    avg_y = sum(p[1] for p in sorted_positions) / len(sorted_positions)
                    y_deviation = sum(abs(p[1] - avg_y) for p in sorted_positions)
                    max_deviation = len(sorted_positions) * (self.m // 2)
                    line_straightness = 1.0 - (y_deviation / max_deviation)
                    reward += line_straightness * self.connectivity_weight * 1.0

                    min_spacing = 1
                    spacing_penalty = 0
                    for i in range(len(sorted_positions) - 1):
                        spacing = sorted_positions[i + 1][0] - sorted_positions[i][0]
                        if spacing < min_spacing:
                            spacing_penalty += (min_spacing - spacing) * 0.2
                    reward -= spacing_penalty
            else:
                reward -= self.connectivity_weight * 1.0  # Reduced penalty

        self.previous_distances[agent_id] = current_distance
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
        """Resolve conflicting moves and ensure connectivity constraints, including only pairwise swaps and strict anti-overlap."""
        agent_priorities = {}
        for agent_id in self.agents:
            distance = self._calculate_distance(
                self.agent_positions[agent_id], self.agent_targets[agent_id]
            )
            agent_priorities[agent_id] = distance
        sorted_agents = sorted(self.agents, key=lambda a: agent_priorities[a])
        final_positions = self.agent_positions.copy()
        moved_agents = set()
        swap_pairs = set()
        # Track which agents are involved in swaps
        swap_involved = set()
        for agent_id in sorted_agents:
            if agent_id not in proposed_positions:
                continue
            new_pos = proposed_positions[agent_id]
            old_pos = self.agent_positions[agent_id]
            if new_pos == old_pos:
                continue
            # Check if the new position is already occupied by any agent
            if new_pos in final_positions.values():
                # Check for possible pairwise swap
                swap_agent = None
                for other_id, other_old_pos in self.agent_positions.items():
                    if other_id != agent_id and other_old_pos == new_pos:
                        if proposed_positions.get(other_id, other_old_pos) == old_pos:
                            pair = tuple(sorted([agent_id, other_id]))
                            if pair not in swap_pairs and agent_id not in swap_involved and other_id not in swap_involved:
                                swap_agent = other_id
                                swap_pairs.add(pair)
                                swap_involved.add(agent_id)
                                swap_involved.add(other_id)
                                break
                if swap_agent is not None:
                    # Perform the swap
                    final_positions[agent_id] = new_pos
                    final_positions[swap_agent] = old_pos
                    moved_agents.add(agent_id)
                    moved_agents.add(swap_agent)
                continue
            # Check if the move would break connectivity
            connected = False
            for other_id, other_pos in final_positions.items():
                if other_id != agent_id and self._are_adjacent(new_pos, other_pos):
                    connected = True
                    break
            if not connected:
                tentative_positions = final_positions.copy()
                tentative_positions[agent_id] = new_pos
                if not self._is_connected(list(tentative_positions.values())):
                    continue
            # Accept the move
            final_positions[agent_id] = new_pos
            moved_agents.add(agent_id)
        # Strict anti-overlap: check for overlaps after all moves
        pos_counts = {}
        for aid, pos in final_positions.items():
            pos_counts[pos] = pos_counts.get(pos, 0) + 1
        overlap_agents = [aid for aid, pos in final_positions.items() if pos_counts[pos] > 1]
        if overlap_agents:
            if not self.fast_mode:
                print(f"[WARNING] Overlap detected after move resolution! Reverting moves for agents: {overlap_agents}")
            # Revert moves for involved agents
            for aid in overlap_agents:
                final_positions[aid] = self.agent_positions[aid]
        self.agent_positions = final_positions
        assert len(set(self.agent_positions.values())) == len(self.agent_positions), "Agent overlap detected after move resolution!"

    def render(self):
        """Render the environment."""
        if self.render_mode is None or getattr(self, 'fast_mode', False):
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

    def _generate_target_positions(self, num_targets, pattern, clustering_factor=0.5):
        """Generate connected target positions based on pattern."""
        target_positions = []
        center_x = self.n // 2
        center_y = self.m // 2
        if pattern == "single":
            target_positions.append((center_x, center_y))
        elif pattern == "line":
            if hasattr(self, 'level') and self.level == 2:
                num_agents = 6
                spacing = 1
                y_targets = self.m // 2 - 2  # Place targets 2 rows above agents, for example
                start_x = (self.n - num_agents) // 2
                targets = []
                for i in range(num_agents):
                    x = start_x + i
                    if not (0 <= x < self.n and 0 <= y_targets < self.m):
                        continue
                    if self.grid[x, y_targets] != 1:
                        continue
                    targets.append((x, y_targets))
                if len(set(targets)) != num_agents:
                    print("Target positions (overlap or missing):", targets)
                assert len(set(targets)) == num_agents, "Target overlap or missing targets!"
                return targets
            # ... rest of your code for other levels ...
        elif pattern == "square":
            # Square formation
            side_length = int(np.sqrt(num_targets))
            if side_length * side_length < num_targets:
                side_length += 1
                
            start_x = center_x - side_length // 2
            start_y = center_y - side_length // 2
            
            for i in range(side_length):
                for j in range(side_length):
                    if len(target_positions) >= num_targets:
                        break
                    x = start_x + i
                    y = start_y + j
                    if (0 <= x < self.n and 0 <= y < self.m and 
                        self.grid[x, y] == 1 and (x, y) not in target_positions):
                        target_positions.append((x, y))
                        
        elif pattern == "circle":
            # Circular formation
            radius = min(self.n, self.m) // 4
            for i in range(num_targets):
                angle = 2 * np.pi * i / num_targets
                x = int(center_x + radius * np.cos(angle))
                y = int(center_y + radius * np.sin(angle))
                if (0 <= x < self.n and 0 <= y < self.m and 
                    self.grid[x, y] == 1 and (x, y) not in target_positions):
                    target_positions.append((x, y))
                    
        elif pattern == "diamond":
            # Diamond formation
            size = min(self.n, self.m) // 3
            for i in range(num_targets):
                angle = 2 * np.pi * i / num_targets
                x = int(center_x + size * np.cos(angle) * np.abs(np.cos(angle)))
                y = int(center_y + size * np.sin(angle) * np.abs(np.sin(angle)))
                if (0 <= x < self.n and 0 <= y < self.m and 
                    self.grid[x, y] == 1 and (x, y) not in target_positions):
                    target_positions.append((x, y))
                    
        elif pattern == "complex":
            # Complex pattern combining multiple shapes
            # Start with a circle and add branches
            radius = min(self.n, self.m) // 4
            for i in range(num_targets):
                angle = 2 * np.pi * i / num_targets
                # Vary the radius to create a more complex shape
                r = radius * (0.7 + 0.3 * np.sin(3 * angle))
                x = int(center_x + r * np.cos(angle))
                y = int(center_y + r * np.sin(angle))
                if (0 <= x < self.n and 0 <= y < self.m and 
                    self.grid[x, y] == 1 and (x, y) not in target_positions):
                    target_positions.append((x, y))
        
        # Ensure targets are connected
        if not self._is_connected(target_positions):
            # If not connected, try to connect them
            connected_targets = self._connect_targets(target_positions)
            if connected_targets:
                target_positions = connected_targets
            else:
                # If can't connect, generate new positions
                return self._generate_target_positions(num_targets, pattern, clustering_factor)
        
        return target_positions
    
    def _connect_targets(self, targets):
        """Connect disconnected target positions."""
        if not targets:
            return targets
            
        connected = [targets[0]]
        remaining = targets[1:]
        
        while remaining:
            found_connection = False
            for i, target in enumerate(remaining):
                for connected_target in connected:
                    if self._are_adjacent(target, connected_target):
                        connected.append(target)
                        remaining.pop(i)
                        found_connection = True
                        break
                if found_connection:
                    break
                    
            if not found_connection:
                # Try to find a path between disconnected components
                for i, target in enumerate(remaining):
                    for connected_target in connected:
                        path = self._find_path(target, connected_target)
                        if path:
                            connected.extend(path)
                            remaining.pop(i)
                            found_connection = True
                            break
                    if found_connection:
                        break
                        
            if not found_connection:
                return None
                
        return connected
    
    def _are_adjacent(self, pos1, pos2):
        """Check if two positions are adjacent (8-connectivity)."""
        return max(abs(pos1[0] - pos2[0]), abs(pos1[1] - pos2[1])) <= 1
    
    def _find_path(self, start, end):
        """Find a path between two positions using A*."""
        # Implementation of A* pathfinding
        open_set = [(0, start)]
        came_from = {}
        g_score = {start: 0}
        f_score = {start: self._calculate_distance(start, end)}
        
        while open_set:
            current = heapq.heappop(open_set)[1]
            
            if self._are_adjacent(current, end):
                path = [end]
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                return path[::-1]
                
            for dx, dy in self.directions[:-1]:  # Exclude 'stay'
                neighbor = (current[0] + dx, current[1] + dy)
                
                if (0 <= neighbor[0] < self.n and 0 <= neighbor[1] < self.m and 
                    self.grid[neighbor[0], neighbor[1]] == 1):
                    
                    tentative_g_score = g_score[current] + 1
                    
                    if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                        came_from[neighbor] = current
                        g_score[neighbor] = tentative_g_score
                        f_score[neighbor] = tentative_g_score + self._calculate_distance(neighbor, end)
                        heapq.heappush(open_set, (f_score[neighbor], neighbor))
                        
        return None

    def select_action(self, obs, evaluate=False):
        epsilon = 0.1  # 10% random moves
        if not evaluate and random.random() < epsilon:
            # Take a random action
            return random.randint(0, len(self.directions) - 1)
        else:
            # Usual policy action
            return self.policy_action(obs)

    def policy_action(self, obs):
        # Implementation of the policy action
        pass

    def _random_move(self):
        if self.steps_taken % 50 == 0:  # Every 50 steps
            for agent_id in self.agents:
                possible_moves = [i for i, (dx, dy) in enumerate(self.directions)
                                 if 0 <= self.agent_positions[agent_id][0] + dx < self.n
                                 and 0 <= self.agent_positions[agent_id][1] + dy < self.m
                                 and self.grid[self.agent_positions[agent_id][0] + dx, self.agent_positions[agent_id][1] + dy] == 1
                                 and (self.agent_positions[agent_id][0] + dx, self.agent_positions[agent_id][1] + dy) not in self.agent_positions.values()]
                if possible_moves:
                    move = random.choice(possible_moves)
                    dx, dy = self.directions[move]
                    x, y = self.agent_positions[agent_id]
                    self.agent_positions[agent_id] = (x + dx, y + dy)

