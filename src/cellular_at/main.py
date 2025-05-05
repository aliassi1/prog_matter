import pygame
import numpy as np
import sys
import os
from typing import List, Tuple, Dict
from collections import Counter

# Add the Base_Agent directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'Base_Agent'))
from last_trail_sarah import BlockAgent, GridEnvironment, SupervisorAgent
from render import PygameRenderer

# Define directions for 8-connected neighborhood
DIRECTIONS = [
    (-1, 0),  # up
    (1, 0),   # down
    (0, -1),  # left
    (0, 1),   # right
    (-1, -1), # up-left
    (-1, 1),  # up-right
    (1, -1),  # down-left
    (1, 1)    # down-right
]

class CellularAgent(BlockAgent):
    def __init__(self, agent_id: int, x: int, y: int, rules: Dict):
        super().__init__(agent_id, x, y)
        self.rules = rules
        self.state = 1  # Always alive
        self.target = (x + 5, y + 5)  # Set a target position for each agent
        
    def decide_action(self, env) -> Tuple[int, int]:
        """Decide next move using wave filling and BFS pathfinding if blocked."""
        x, y = self.get_position()
        tx, ty = self.target

        if (x, y) == (tx, ty):
            return x, y  # Already at target

        # --- Wave Filling Logic ---
        # Only move into target if all neighbors closer to target are at their targets
        closer_neighbors = []
        for dx, dy in DIRECTIONS:
            nx, ny = x + dx, y + dy
            if 0 <= nx < env.n and 0 <= ny < env.m:
                for other in env.agents.values():
                    if other.agent_id != self.agent_id and other.get_position() == (nx, ny):
                        # Is this neighbor closer to its target than I am to mine?
                        if abs(nx - tx) + abs(ny - ty) < abs(x - tx) + abs(y - ty):
                            closer_neighbors.append(other)
        # If all closer neighbors are at their targets, allow move
        if all((n.x, n.y) == n.target for n in closer_neighbors):
            # Try to move directly toward target if not blocked
            dx = np.sign(tx - x)
            dy = np.sign(ty - y)
            move = (x + dx, y + dy)
            if 0 <= move[0] < env.n and 0 <= move[1] < env.m:
                occupied = False
                for other in env.agents.values():
                    if other.agent_id != self.agent_id and other.get_position() == move:
                        occupied = True
                        break
                if not occupied:
                    return move
        # --- BFS Pathfinding Fallback ---
        from collections import deque
        visited = set()
        queue = deque()
        queue.append(((x, y), []))
        occupied_positions = {other.get_position() for other in env.agents.values() if other.agent_id != self.agent_id}
        while queue:
            (cx, cy), path = queue.popleft()
            if (cx, cy) == (tx, ty):
                if path:
                    return path[0]  # Take the first step toward the target
                else:
                    return x, y  # Already at target
            for dx, dy in DIRECTIONS:
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < env.n and 0 <= ny < env.m and (nx, ny) not in visited and (nx, ny) not in occupied_positions:
                    visited.add((nx, ny))
                    queue.append(((nx, ny), path + [(nx, ny)]))
        # If no path found, stay in place
        return x, y

    def update_state(self, neighbors: List['CellularAgent']):
        """Update state based on cellular automata rules - no death"""
        # Agents always stay alive
        self.state = 1

class CellularEnvironment(GridEnvironment):
    def __init__(self, n: int, m: int, rules: Dict):
        super().__init__(n, m)
        self.rules = rules
        self.n = n
        self.m = m
        self.agents = {}  # Store agents in a dictionary
        
    def add_agent(self, agent: CellularAgent):
        """Add an agent to the environment"""
        self.agents[agent.agent_id] = agent
        
    def get_neighbors(self, agent: CellularAgent) -> List[CellularAgent]:
        """Get 8-connected neighbors of an agent"""
        x, y = agent.get_position()
        neighbors = []
        
        for dx, dy in DIRECTIONS:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.n and 0 <= ny < self.m:
                for other in self.agents.values():
                    if other.get_position() == (nx, ny):
                        neighbors.append(other)
        return neighbors
        
    def step(self):
        """Perform one step of the cellular automata simulation"""
        # First pass: decide moves for all agents
        proposed_moves = {}
        for agent in self.agents.values():
            new_pos = agent.decide_action(self)
            proposed_moves[agent.agent_id] = new_pos
        
        # Second pass: update states based on current positions
        for agent in self.agents.values():
            neighbors = self.get_neighbors(agent)
            agent.update_state(neighbors)
        
        # Third pass: apply moves
        for agent in self.agents.values():
            if agent.agent_id in proposed_moves:
                x, y = proposed_moves[agent.agent_id]
                agent.x = x
                agent.y = y

def create_pattern(env: CellularEnvironment, pattern_type: str, start_x: int, start_y: int):
    """Create different cellular automata patterns with unique agent and target positions"""
    if pattern_type == "glider":
        positions = [
            (start_x, start_y),
            (start_x + 1, start_y + 1),
            (start_x + 1, start_y + 2),
            (start_x, start_y + 2),
            (start_x - 1, start_y + 2)
        ]
    elif pattern_type == "blinker":
        positions = [
            (start_x, start_y),
            (start_x, start_y + 1),
            (start_x, start_y + 2)
        ]
    elif pattern_type == "beacon":
        positions = [
            (start_x, start_y),
            (start_x + 1, start_y),
            (start_x, start_y + 1),
            (start_x + 1, start_y + 1),
            (start_x + 2, start_y + 2),
            (start_x + 3, start_y + 2),
            (start_x + 2, start_y + 3),
            (start_x + 3, start_y + 3)
        ]
    else:
        print(f"Unknown pattern type: {pattern_type}. Using glider pattern.")
        positions = [
            (start_x, start_y),
            (start_x + 1, start_y + 1),
            (start_x + 1, start_y + 2),
            (start_x, start_y + 2),
            (start_x - 1, start_y + 2)
        ]

    # Remove duplicate positions
    unique_positions = list(dict.fromkeys(positions))
    # Create target positions (shifted to form a square)
    target_positions = [(x + 10, y + 10) for x, y in unique_positions]
    # Remove duplicate targets
    unique_target_positions = list(dict.fromkeys(target_positions))
    # Ensure the number of agents matches the number of unique targets
    min_len = min(len(unique_positions), len(unique_target_positions))
    for i in range(min_len):
        x, y = unique_positions[i]
        target = unique_target_positions[i]
        agent = CellularAgent(i, x, y, env.rules)
        agent.target = target  # Set the target position
        env.add_agent(agent)

def main(pattern_type="glider"):
    # Initialize pygame
    pygame.init()
    
    # Set up the environment with smaller grid and cell size
    grid_size = (20, 20)  # Smaller grid
    cell_size = 30  # Smaller cells
    
    rules = {
        'birth': [3],  # Number of neighbors needed for birth
        'survival': [1,2, 3]  # Number of neighbors needed for survival
    }
    
    env = CellularEnvironment(grid_size[0], grid_size[1], rules)
    
    # Create renderer with custom cell size
    renderer = PygameRenderer(env, cell_size=cell_size)
    
    # Create initial pattern
    create_pattern(env, pattern_type, 5, 5)
    
    # Main simulation loop
    clock = pygame.time.Clock()
    running = True
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                
        # Update simulation
        env.step()
        
        # Render
        renderer.render()  # No need to pass env, it's already stored in renderer
        
        # Control simulation speed
        clock.tick(10)
        
    # After the simulation loop, before pygame.quit()
    final_positions = set((agent.x, agent.y) for agent in env.agents.values())
    target_positions = [agent.target for agent in env.agents.values()]  # List instead of set to count duplicates
    unique_targets = set(target_positions)
    missing_targets = unique_targets - final_positions
    overlapping_agents = len(final_positions) < len(env.agents)

    print(f"Final agent positions: {final_positions}")
    print(f"Target positions: {target_positions}")
    print(f"Unique target positions: {unique_targets}")
    print(f"Missing targets: {missing_targets}")
    print(f"Any overlapping agents? {overlapping_agents}")
    print(f"Number of agents: {len(env.agents)} | Number of unique positions: {len(final_positions)}")
    print(f"Number of target positions (including duplicates): {len(target_positions)}")
    print(f"Number of unique target positions: {len(unique_targets)}")

    # Count agent positions
    final_positions_list = [ (agent.x, agent.y) for agent in env.agents.values() ]
    position_counts = Counter(final_positions_list)
    overlapping = {pos: count for pos, count in position_counts.items() if count > 1}
    if overlapping:
        print("Overlapping agent positions (position: count):", overlapping)
    else:
        print("No overlapping agent positions.")

    # Check for unreached targets
    targets = [agent.target for agent in env.agents.values()]
    unreached_targets = set(targets) - set(final_positions_list)
    if unreached_targets:
        print("Unreached targets:", unreached_targets)
    else:
        print("All targets reached.")

    for agent in env.agents.values():
        if (agent.x, agent.y) != agent.target:
            print(f"Agent {agent.agent_id} did not reach its target: at {agent.x, agent.y}, target {agent.target}")

    pygame.quit()

if __name__ == "__main__":
    main() 