from collections import deque
import random
from typing import List, Tuple, Dict
import heapq
from collections import defaultdict
import math
import numpy as np

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

class BlockAgent:
    """
    Each block is an independent agent.
    Decides how to move based on its target or local logic.
    """
    def __init__(self, agent_id: int, x: int, y: int):
        self.agent_id = agent_id
        self.x = x
        self.y = y
        self.onroad = False
        self.path = None  # Store the path for reuse
        self.past_positions = []
        self.priority = 0  # Initialize priority to 0

    def get_position(self) -> Tuple[int,int]:
        return (self.x, self.y)
    
    
    def set_target(self, target: Tuple[int, int]):
        self.target = target
        
    
    def astar_path(self, grid, start_x, start_y, target_x, target_y):
        """
        A* pathfinding algorithm implementation
        
        Args:
            grid: 2D list where 0 represents obstacles and 1 represents free cells
            start_x, start_y: Starting coordinates
            target_x, target_y: Target coordinates
            
        Returns:
            List of (x,y) coordinates representing the path, or None if no path exists
        """
        # Helper functions for heuristic and neighbors
        def heuristic(x, y):
            return abs(x - target_x) + abs(y - target_y)  # Manhattan distance
            
        def get_neighbors(x, y):
            neighbors = []
            for dx, dy in DIRECTIONS:  # Make sure DIRECTIONS includes diagonals if needed
                new_x, new_y = x + dx, y + dy
                if (0 <= new_x < len(grid) and 
                    0 <= new_y < len(grid[0]) and 
                    grid[new_x][new_y] == 1):
                    # For diagonal moves, ensure the path isn't blocked by corner obstacles
                    if dx != 0 and dy != 0:  # Diagonal move
                        if grid[x][new_y] == 0 or grid[new_x][y] == 0:
                            continue  # Skip if diagonal path is blocked by corner obstacle
                    neighbors.append((new_x, new_y))
            return neighbors

        # Priority queue for open set
        open_set = [(0, start_x, start_y)]
        came_from = {}
        
        # Cost from start to each node
        g_score = defaultdict(lambda: float('inf'))
        g_score[(start_x, start_y)] = 0
        
        # Estimated total cost from start to goal through node
        f_score = defaultdict(lambda: float('inf'))
        f_score[(start_x, start_y)] = heuristic(start_x, start_y)
        
        while open_set:
            _, current_x, current_y = heapq.heappop(open_set)
            
            if current_x == target_x and current_y == target_y:
                # Reconstruct path
                path = []
                current = (current_x, current_y)
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append((start_x, start_y))
                return path[::-1]
                
            for next_x, next_y in get_neighbors(current_x, current_y):
                tentative_g_score = g_score[(current_x, current_y)] + 1
                
                if tentative_g_score < g_score[(next_x, next_y)]:
                    came_from[(next_x, next_y)] = (current_x, current_y)
                    g_score[(next_x, next_y)] = tentative_g_score
                    f_score[(next_x, next_y)] = tentative_g_score + heuristic(next_x, next_y)
                    heapq.heappush(open_set, (f_score[(next_x, next_y)], next_x, next_y))
        
        return None  # No path found



    def steer_local_3x3(self, env, tx, ty):
        # Create a 3x3 local occupancy grid around the agent
        local_grid = [['Sarah' for _ in range(3)] for _ in range(3)]
        #visibility of the agent 


        #fill the cooridnates around the local grid
        for i in range(3):
            for j in range(3):
                gx = self.x + (i - 1)  # i - 1 => -1, 0, +1
                gy = self.y + (j - 1)
                if 0 <= gx < env.n and 0 <= gy < env.m:
                    if env.is_free(gx, gy):
                        local_grid[i][j] = 1
                    else:
                        local_grid[i][j] = 0
                else:
                    local_grid[i][j] = 0

        # Store the current position as the last position before moving
        if not hasattr(self, 'last_position'):
            self.last_position = None
        
        # Calculate directional vector toward the target
        Dx = tx - self.x
        Dy = ty - self.y

        # If the agent is already at the target, no move is needed.
        if Dx == 0 and Dy == 0:
            return (0, 0)

        # Variables to track the best move and best distance
        best_move = (0, 0)
        best_distance = float('inf')  # Start with infinity
        
        # Detect oscillation (going back and forth between positions)
        oscillation_detected = False
        if len(self.past_positions) >= 4:
            # Check last 4 positions for pattern like A -> B -> A -> B
            if (self.past_positions[0] == self.past_positions[2] and 
                self.past_positions[1] == self.past_positions[3]):
                oscillation_detected = True
        
        # Loop over local grid (skipping the center cell)
        for i in range(3):
            for j in range(3):
                if i == 1 and j == 1:
                    continue  # Skip center (no move)

                if local_grid[i][j] == 1:
                    # local offset from center
                    dr = i - 1
                    dc = j - 1
                    
                    # Calculate new position if we take this move
                    new_x = self.x + dr
                    new_y = self.y + dc
                    
                    # Skip this move if it leads back to the previous position
                    if self.last_position and (new_x, new_y) == self.last_position:
                        continue
                    
                    # If oscillation detected, avoid positions we've visited recently
                    if oscillation_detected and any((new_x, new_y) == pos for pos in self.past_positions[-4:]):
                        continue
                    
                    # Calculate distance to target from this new position
                    new_distance = math.sqrt((tx - new_x) ** 2 + (ty - new_y) ** 2)
                    
                    # Choose the neighbor with the minimum distance to target
                    if new_distance < best_distance:
                        best_distance = new_distance
                        best_move = (dr, dc)
        
        # If we couldn't find a valid move, then take a random valid move to break the deadlock
        if best_move == (0, 0) and self.x != tx and self.y != ty:
            valid_moves = []
            for i in range(3):
                for j in range(3):
                    if i == 1 and j == 1:
                        continue  # Skip center
                    if local_grid[i][j] == 1:
                        valid_moves.append((i-1, j-1))
            
            if valid_moves:
                best_move = random.choice(valid_moves)
                print('random move')
        
        # Calculate the final new position with the best move
        new_x = self.x + best_move[0]
        new_y = self.y + best_move[1]
        
        # Update the last position and add to position history
        self.last_position = (self.x, self.y)
        self.past_positions.append((new_x, new_y))
        
        # Limit history size to prevent memory growth
        if len(self.past_positions) > 10:
            self.past_positions.pop(0)
   
        return best_move

    # def check_if_need_pushing(self, env):
    #     # Check if the agent is being pushed by another agent
        

    def decide_action(self, env) -> Tuple[int, int]:
        """Decide how to move in a decentralized manner."""
        # Check if we're currently resolving a deadlock
        if hasattr(self, 'resolving_deadlock') and self.resolving_deadlock:
            print(f"Agent {self.agent_id} is waiting for deadlock resolution.")
            return (0, 0)  # Don't move during deadlock resolution
        
        if self.target is None:
            # If no target, do nothing
            return (0, 0)
        
        path = self.steer_local_3x3(env, self.target[0], self.target[1])
        
        (dx, dy) = path
        old_x, old_y = self.x, self.y
        new_x, new_y = old_x + dx, old_y + dy

        # Build a hypothetical arrangement of positions
        hypothetical_positions = []
        for aid, ag in env.agents.items():
            if aid == self.agent_id:
                hypothetical_positions.append((new_x, new_y))
            else:
                hypothetical_positions.append(ag.get_position())

        # Check if that arrangement is still connected
        if env.is_connected(hypothetical_positions):
            # If it's connected, propose this move
            return (dx, dy)
        else:
            print(f"Agent {self.agent_id}: move {path} rejected due to connectivity")
            return (0, 0)

    # def detect_deadlock(self, env, history_length=30):  # Reduced from 60 to 30
    #     """Detect if agent is stuck in a deadlock by checking position history"""
    #     # First check: if agent is already at target, it's not deadlocked
    #     if self.target is not None and (self.x, self.y) == self.target:
    #         return False
            
    #     if not hasattr(self, 'position_history'):
    #         self.position_history = []
    #         self.stuck_timer = 0  # Add a timer for being stuck
        
    #     current_pos = (self.x, self.y)
    #     self.position_history.append(current_pos)
        
    #     # Keep history limited to recent positions
    #     if len(self.position_history) > history_length:
    #         self.position_history.pop(0)
        
    #     # Check if positions are cycling or agent hasn't moved
    #     if len(self.position_history) >= history_length:
    #         unique_positions = set(self.position_history)
    #         if len(unique_positions) <= 2:  # Agent oscillating between positions or stuck
    #             return True
        
    #     # Alternative detection: if we haven't moved in 10 steps, consider it a deadlock
    #     if len(self.position_history) >= 10:
    #         if all(pos == current_pos for pos in self.position_history[-10:]):
    #             self.stuck_timer += 1
    #             if self.stuck_timer > 3:  # After 3 consecutive checks (30+ steps total)
    #                 return True
    #     else:
    #         self.stuck_timer = 0
        
    #     return False

    # def resolve_deadlock(self, env):
    #     """
    #     Attempt to resolve a deadlock by finding an alternative path
    #     that considers the positions of all other blocks
    #     """
    #     # Get positions of all other blocks
    #     print(f'Agent {self.agent_id} resolving deadlock')
    #     other_blocks = self.get_all_block_positions(env)
        
    #     # Build a grid representation of the environment
    #     grid = [[1 for _ in range(env.m)] for _ in range(env.n)]
        
    #     # Mark positions of other blocks as obstacles (0)
    #     for _, pos in other_blocks.items():
    #         grid[pos[0]][pos[1]] = 0
        
    #     # Set our current position as traversable
    #     grid[self.x][self.y] = 1
        
    #     # Try direct path to target first
    #     astar_path = self.astar_path(grid, self.x, self.y, self.target[0], self.target[1])
        
    #     # If direct path doesn't work, try finding temporary detour points
    #     if not astar_path:
    #         print(f"Agent {self.agent_id}: No direct path to target, trying detour")
    #         # Find an empty space at increasing distances from the agent
    #         for radius in range(3, max(env.n, env.m), 2):
    #             detour_points = []
    #             for i in range(-radius, radius+1):
    #                 for j in range(-radius, radius+1):
    #                     if abs(i) + abs(j) <= radius:  # Manhattan distance <= radius
    #                         nx, ny = self.x + i, self.y + j
    #                         if (0 <= nx < env.n and 0 <= ny < env.m and 
    #                             grid[nx][ny] == 1):  # Empty space
    #                             detour_points.append((nx, ny))
                
    #             # If we found potential detour points, try them
    #             if detour_points:
    #                 # Sort by Manhattan distance to target
    #                 detour_points.sort(key=lambda p: abs(p[0]-self.target[0]) + abs(p[1]-self.target[1]))
                    
    #                 # Try each detour point until we find a path
    #                 for detour in detour_points[:3]:  # Try up to 3 best detours
    #                     temp_path = self.astar_path(grid, self.x, self.y, detour[0], detour[1])
    #                     if temp_path and len(temp_path) > 1:
    #                         print(f"Agent {self.agent_id} found detour to {detour}")
    #                         # Remember original target so we can return to it later
    #                         if not hasattr(self, 'original_target'):
    #                             self.original_target = self.target
    #                         self.temp_target = detour
    #                         return temp_path
                    
    #                 # If we found detour points but couldn't find paths to any of them,
    #                 # we'll just keep trying with a larger radius
                
    #             # If no detour points in this radius, try a larger radius
        
    #     print(f"Target {self.target} and current = {self.x,self.y}")
    #     return astar_path
       


    def get_all_block_positions(self, env, include_self=False) -> Dict[int, Tuple[int, int]]:
        """
        Get positions of all blocks in the environment
        
        Parameters:
            env: The grid environment containing all agents
            include_self: Whether to include this agent's position in the result
            
        Returns:
            Dictionary mapping agent_id to position (x, y)
        """
        positions = {}
        for agent_id, agent in env.agents.items():
            if not include_self and agent_id == self.agent_id:
                continue
            positions[agent_id] = agent.get_position()
        return positions

class GridEnvironment:
    """
    The environment maintains the global state:
      - Grid dimensions
      - Positions of each agent
      - Connectivity checks, etc.
    """
    def __init__(self, n: int, m: int):
        self.n = n
        self.m = m
        self.agents: Dict[int, BlockAgent] = {}
        
    def is_free(self, x: int, y: int) -> bool:
        """
        Check if a cell is free (not blocked)
        """
        for agent_id, agent in self.agents.items():
            if (agent.x, agent.y) == (x, y):
                return False
        return True
    
    def add_agent(self, agent: BlockAgent):
        # Make sure no agent has same ID
        if agent.agent_id in self.agents:
            raise ValueError(f"Agent with ID {agent.agent_id} already exists.")

        if not (0 <= agent.x < self.n and 0 <= agent.y < self.m):
            raise ValueError("Agent initial position out of bounds.")
        

        for a in self.agents.values():
            if (a.x, a.y) == (agent.x, agent.y):
                raise ValueError("Collision on agent add. Two agents in same cell.")
        self.agents[agent.agent_id] = agent

    
    
    def get_positions(self) -> List[Tuple[int,int]]:
        """
        Return a list of all agent positions. 
        """
        return [(agent.x, agent.y) for agent in self.agents.values()]
    
    def propose_moves(self) -> Dict[int, Tuple[int,int]]:
        """
        In a parallel sense, each agent decides its move. 
        Return a dict: agent_id -> (dx, dy).
        """
        moves = {}
        for agent_id, agent in self.agents.items():
            dx, dy = agent.decide_action(self)
            moves[agent_id] = (dx, dy)

        return moves
    
    def calculate_target_centroid(self):
        """Calculate the centroid of all agent targets"""
        targets = [agent.target for agent in self.agents.values() if agent.target is not None]
        if not targets:
            return None
        
        sum_x = sum(t[0] for t in targets)
        sum_y = sum(t[1] for t in targets)
        return (sum_x / len(targets), sum_y / len(targets))
    
    def update_agent_priorities(self):
        """Update priorities for all agents based on two factors:
        1. Distance from target to centroid (closer = higher priority)
        2. Density of targets around the agent's target (more targets = higher priority)
        """
        centroid = self.calculate_target_centroid()
        if not centroid:
            return
        
        # Get all targets
        targets = [agent.target for agent in self.agents.values() if agent.target is not None]
        
        # Calculate distance from each agent's target to the centroid
        for agent_id, agent in self.agents.items():
            if agent.target is None:
                agent.priority = 0  # No target = lowest priority
                continue
                
            # Factor 1: Distance to centroid (closer = higher value)
            dist_to_centroid = math.sqrt((agent.target[0] - centroid[0])**2 + 
                             (agent.target[1] - centroid[1])**2)
            centroid_priority = 1.0 / (dist_to_centroid + 0.1)  # Adding 0.1 to avoid division by zero
            
            # Factor 2: Target density (count nearby targets)
            nearby_count = 0
            target_radius = 3  # Consider targets within this Manhattan distance
            
            for other_target in targets:
                if other_target == agent.target:
                    continue  # Skip self
                
                # Calculate Manhattan distance between targets
                target_dist = abs(agent.target[0] - other_target[0]) + abs(agent.target[1] - other_target[1])
                if target_dist <= target_radius:
                    nearby_count += 1
            
            # Normalize density factor (1.0 + bonus for density)
            density_factor = 1.0 + (nearby_count * 0.2)  # 20% bonus per nearby target
            
            # Combine factors: primary is distance to centroid, secondary is target density
            agent.priority = centroid_priority * density_factor
                
    def apply_moves(self, moves: Dict[int, Tuple[int,int]]):
        # 1) Have all agents propose moves simultaneously
        proposed_positions = {}
        
        for agent_id, agent in self.agents.items():
            dx, dy = moves[agent_id]
            nx, ny = agent.x + dx, agent.y + dy
            # Only propose valid moves (within bounds)
            if 0 <= nx < self.n and 0 <= ny < self.m:
                proposed_positions[agent_id] = (nx, ny)
            else:
                proposed_positions[agent_id] = (agent.x, agent.y)
        
        # 2) Update agent priorities based on target distance to centroid
        self.update_agent_priorities()
        
        # 3) Sort agents by priority (highest priority first)
        agent_order = list(self.agents.keys())
        agent_order.sort(key=lambda agent_id: self.agents[agent_id].priority, reverse=True)
        
        final_positions = {}
        for agent_id, agent in self.agents.items():
            final_positions[agent_id] = (agent.x, agent.y)
            
        # 4) Try moves one by one, prioritizing agents with higher priority
        for agent_id in agent_order:
            agent = self.agents[agent_id]
            dx, dy = moves[agent_id]
            nx, ny = agent.x + dx, agent.y + dy

            # Check boundary
            if not (0 <= nx < self.n and 0 <= ny < self.m):
                # revert – do nothing
                continue
           
            # Check collision with accepted moves
            if (nx, ny) in final_positions.values():
                # revert – do nothing
                continue

            # Tentatively accept
            old_pos = final_positions[agent_id]
            final_positions[agent_id] = (nx, ny)

            # 4) Check connectivity with this new position
            all_positions = list(final_positions.values())
            if not self.is_connected(all_positions):
                # revert only this agent
                final_positions[agent_id] = old_pos

        # 5) Commit final positions
        for agent_id, (x, y) in final_positions.items():
            self.agents[agent_id].x = x
            self.agents[agent_id].y = y

    def is_connected(self, positions: List[Tuple[int,int]]) -> bool:
        """
        Check 8-direction connectivity for the given list of positions.
        """
        if not positions:
            return True
        pos_set = set(positions)
        visited = set()
        
        from collections import deque
        queue = deque([positions[0]])
        visited.add(positions[0])
        
        dirs = DIRECTIONS
        while queue:
            cx, cy = queue.popleft()
            for dx, dy in dirs:
                nx, ny = cx + dx, cy + dy
                if (nx, ny) in pos_set and (nx, ny) not in visited:
                    visited.add((nx, ny))
                    queue.append((nx, ny))
        return len(visited) == len(positions)


        
class SupervisorAgent:
    def __init__(self, env: GridEnvironment):
        self.env = env

    def check_paths(self):
        # Create a temporary grid representation of the environment
        grid = [[1 for _ in range(self.env.m)] for _ in range(self.env.n)]
        
        # Mark positions of all agents as obstacles (0)
        for agent_id, agent in self.env.agents.items():
            grid[agent.x][agent.y] = 0
            
        for agent_id, agent in self.env.agents.items():
            if agent.target is not None:
                # Temporarily mark the agent's position as free
                grid[agent.x][agent.y] = 1
                
                # Check if a path exists
                path = agent.astar_path(grid, agent.x, agent.y, agent.target[0], agent.target[1])
                
                # Reset agent's position as obstacle
                grid[agent.x][agent.y] = 0
                
                # Initialize stuck_count if it doesn't exist
                if not hasattr(agent, 'stuck_count'):
                    agent.stuck_count = 0
                    
                # Initialize target_history if it doesn't exist
                if not hasattr(agent, 'target_history'):
                    agent.target_history = set()
                    
                if path is None:
                    # Agent is stuck with no path to target
                    agent.stuck_count += 1
                    agent.target_history.add(agent.target)
                    
                    # Only calculate direction vector if not already stored
                        # Get direction vector from agent to target
                    dx = agent.target[0] - agent.x
                    dy = agent.target[1] - agent.y
                    dir_x = 1 if dx > 0 else (-1 if dx < 0 else 0)
                    dir_y = 1 if dy > 0 else (-1 if dy < 0 else 0)
                        
                        # Store the direction vector on the agent object
                    print(f"Agent {agent_id} is stuck (count: {agent.stuck_count}).")
                    
                    
                    # If agent gets stuck too many times, try more aggressive approach
                    if agent.stuck_count > 3:
                        # Looking for any agent, not just in the direction
                        print(f"Agent {agent_id} has been stuck {agent.stuck_count} times, trying more aggressive approach")
                        closest_agent = None
                        min_dist = float('inf')
                        for other_id, other in self.env.agents.items():
                            if other_id == agent_id:
                                continue
                            # Skip agents that were already swapped with
                            if hasattr(other, 'target_history') and agent.target in other.target_history:
                                continue
                            # Calculate distance
                            dist = math.sqrt((other.x - agent.x)**2 + (other.y - agent.y)**2)
                            if dist < min_dist:
                                min_dist = dist
                                closest_agent = other
                    else:
                        # Regular approach - find agent in correct direction
                        closest_agent = None
                        min_dist = float('inf')
                        for other_id, other in self.env.agents.items():
                            if other_id == agent_id:
                                continue
                            # Skip agents that were already swapped with
                            if hasattr(other, 'target_history') and agent.target in other.target_history:
                                continue
                            
                            # Check if other agent is roughly in the direction we want
                            other_dx = other.x - agent.x
                            other_dy = other.y - agent.y
                            if (other_dx * dir_x >= 0) and (other_dy * dir_y >= 0):
                                dist = math.sqrt(other_dx**2 + other_dy**2)  # Euclidean distance
                                if dist < min_dist:
                                    min_dist = dist
                                    closest_agent = other
                    
                    if closest_agent:
                        # Swap targets between agent and closest agent
                        temp_target = agent.target
                        agent.target = closest_agent.target
                        closest_agent.target = temp_target
                        
                        # Update target histories
                        agent.target_history.add(agent.target)
                        if not hasattr(closest_agent, 'target_history'):
                            closest_agent.target_history = set()
                        closest_agent.target_history.add(closest_agent.target)
                        
                       
                        
                        # Reset stuck count for both agents
                        agent.stuck_count = 0
                        if hasattr(closest_agent, 'stuck_count'):
                            closest_agent.stuck_count = 0
                        
                        print(f"Swapped targets: Agent {agent_id} now targets {agent.target}, Agent {closest_agent.agent_id} now targets {closest_agent.target}")
                    else:
                        print(f"WARNING: No path exists for agent {agent_id} at {agent.get_position()} to reach target {agent.target} and no swap possible")
                else:
                    # Path exists, don't reset stuck_direction completely, but reset stuck_count
                    agent.stuck_count = 0

def run_simulation(env: GridEnvironment, supervisor: SupervisorAgent, max_steps=1000):
    """
    Run the decentralized simulation for up to max_steps.
    """
    step = 0
    stagnation_count = 0
    last_positions = None
    while step < max_steps:
        step += 1
        print(f"\n=== Step {step} ===")
        
        # 1) Agents propose moves
        moves = env.propose_moves()
        
        # 2) Environment tries to apply moves
        env.apply_moves(moves)
        if step % 6 == 0:
            supervisor.check_paths()
        
        
        # Track positions for stagnation detection
        current_positions = {agent_id: agent.get_position() for agent_id, agent in env.agents.items()}
        
        # Check for stagnation (no agent moved)
        if last_positions == current_positions:
            stagnation_count += 1
            if stagnation_count > 5:  # Stuck for 5 consecutive turns
                print("WARNING: Agents appear to be stuck. Consider adjusting parameters.")
                # Optional: implement a "nudge" mechanism here to break deadlocks
        else:
            stagnation_count = 0
            
        last_positions = current_positions
        
        # Print current positions
        for agent_id, agent in env.agents.items():
            target_dist = "N/A" if agent.target is None else \
                          round(math.sqrt((agent.x-agent.target[0])**2 + (agent.y-agent.target[1])**2), 2)
            print(f"Agent {agent_id} at {agent.get_position()}, distance to target: {target_dist}")
        
        # Check if done
        all_at_target = all(
            agent.target is None or (agent.x, agent.y) == agent.target 
            for agent in env.agents.values()
        )
        
        if all_at_target:
            print("All agents reached their targets. Stopping simulation.")
            break
    if all_at_target:
            print("All agents reached their targets. Stopping simulation.")


if __name__ == "__main__":
    # Example usage
    n, m = 20, 20
    env = GridEnvironment(n, m)

    # Create some agents
    # We'll place them in top-left corner area, 
    # and assign some distinct target in the bottom-right corner.

    agents_data = [
        (0, 1, 1, (9,10)),  # (agent_id, x, y, target)
        (1, 1, 2, (9,11)),
        (2, 1, 3, (8,10)),
        (3, 1, 4, (8,11)),
        (4, 1, 5, (7,10)),
        (5, 1, 6, (7,11)),
        (6, 1, 7, (10,10)),
        (7, 1, 8, (10,11)),
    ]
    for (aid, x, y, tgt) in agents_data:
        block = BlockAgent(aid, x, y)
        block.set_target(tgt)
        env.add_agent(block)
    
    # Run the decentralized simulation
    run_simulation(env, max_steps=30)
