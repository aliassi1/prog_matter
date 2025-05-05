import numpy as np
from typing import List, Tuple, Dict
import math

class GradientAgent:
    """
    A decentralized agent that uses gradient fields for navigation.
    Each agent follows a potential field that combines:
    1. Attraction to its target
    2. Repulsion from other agents
    3. Repulsion from obstacles
    """
    def __init__(self, agent_id: int, x: int, y: int):
        self.agent_id = agent_id
        self.x = x
        self.y = y
        self.target = None
        self.radius = 2  # Influence radius for repulsion
        self.attraction_strength = 1.0
        self.repulsion_strength = 0.5
        self.obstacle_strength = 0.8
        self.last_position = (self.x, self.y)
        
    def set_target(self, target: Tuple[int, int]):
        self.target = target
        
    def get_position(self) -> Tuple[int, int]:
        return (self.x, self.y)
        
    def calculate_attraction_force(self) -> Tuple[float, float]:
        """Calculate force attracting agent to its target"""
        if self.target is None:
            return (0, 0)
            
        dx = self.target[0] - self.x
        dy = self.target[1] - self.y
        distance = math.sqrt(dx*dx + dy*dy)
        
        if distance < 0.1:  # Prevent division by zero
            return (0, 0)
            
        # Normalize and scale by attraction strength
        fx = (dx/distance) * self.attraction_strength
        fy = (dy/distance) * self.attraction_strength
        return (fx, fy)
        
    def calculate_repulsion_force(self, other_agents: Dict[int, Tuple[int, int]]) -> Tuple[float, float]:
        """Calculate repulsion force from other agents"""
        fx, fy = 0, 0
        
        for agent_id, (other_x, other_y) in other_agents.items():
            if agent_id == self.agent_id:
                continue
                
            dx = self.x - other_x
            dy = self.y - other_y
            distance = math.sqrt(dx*dx + dy*dy)
            
            if distance < self.radius:
                # Strong repulsion when too close
                if distance < 0.1:  # Prevent division by zero
                    distance = 0.1
                fx += (dx/distance) * self.repulsion_strength * (1/distance)
                fy += (dy/distance) * self.repulsion_strength * (1/distance)
                
        return (fx, fy)
        
    def calculate_obstacle_force(self, grid: np.ndarray) -> Tuple[float, float]:
        """Calculate repulsion force from obstacles"""
        fx, fy = 0, 0
        n, m = grid.shape
        
        # Check 8 surrounding cells
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                    
                new_x, new_y = self.x + dx, self.y + dy
                if 0 <= new_x < n and 0 <= new_y < m:
                    if grid[new_x, new_y] == 0:  # Obstacle
                        # Repel from obstacle
                        fx -= dx * self.obstacle_strength
                        fy -= dy * self.obstacle_strength
                        
        return (fx, fy)
        
    def decide_action(self, env) -> Tuple[int, int]:
        """Decide next move based on combined forces"""
        # Get grid and other agents
        grid = env.grid
        other_agents = {id: agent.get_position() 
                       for id, agent in env.agents.items() 
                       if id != self.agent_id}
        
        # Calculate all forces
        att_fx, att_fy = self.calculate_attraction_force()
        rep_fx, rep_fy = self.calculate_repulsion_force(other_agents)
        obs_fx, obs_fy = self.calculate_obstacle_force(grid)
        
        # Combine forces
        total_fx = att_fx + rep_fx + obs_fx
        total_fy = att_fy + rep_fy + obs_fy
        
        # Normalize force vector
        magnitude = math.sqrt(total_fx*total_fx + total_fy*total_fy)
        if magnitude < 0.5:
            return (self.x, self.y)
        if magnitude > 0:
            total_fx /= magnitude
            total_fy /= magnitude
            
        # Convert force to discrete movement
        new_x = self.x + round(total_fx)
        new_y = self.y + round(total_fy)
        
        # Ensure new position is valid
        if env.is_free(new_x, new_y):
            if self.target and math.dist((self.x, self.y), self.target) < 1.0:
                return (self.x, self.y)
            return (new_x, new_y)
        return (self.x, self.y)

class GradientEnvironment:
    """
    Environment for gradient-based agents
    """
    def __init__(self, n: int, m: int):
        self.n = n
        self.m = m
        self.grid = np.ones((n, m))  # 1 for free space, 0 for obstacles
        self.agents = {}
        
    def is_free(self, x: int, y: int) -> bool:
        """Check if cell (x,y) is free"""
        return (0 <= x < self.n and 
                0 <= y < self.m and 
                self.grid[x, y] == 1)
                
    def add_agent(self, agent: GradientAgent):
        """Add an agent to the environment"""
        self.agents[agent.agent_id] = agent
        
    def propose_moves(self) -> Dict[int, Tuple[int, int]]:
        """Get proposed moves from all agents"""
        moves = {}
        for agent_id, agent in self.agents.items():
            moves[agent_id] = agent.decide_action(self)
        return moves
        
    def apply_moves(self, moves: Dict[int, Tuple[int, int]]):
        """Apply proposed moves if valid"""
        # First check all moves are valid
        valid_moves = {}
        for agent_id, (new_x, new_y) in moves.items():
            if self.is_free(new_x, new_y):
                valid_moves[agent_id] = (new_x, new_y)
                
        # Apply valid moves
        for agent_id, (new_x, new_y) in valid_moves.items():
            agent = self.agents[agent_id]
            agent.x = new_x
            agent.y = new_y 