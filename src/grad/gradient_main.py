import pygame
import numpy as np
import sys
import os

# Add the Base_Agent directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'Base_Agent'))

from gradient_agent import GradientAgent, GradientEnvironment
from render import PygameRenderer

def generate_source_positions(num_blocks, grid_size=30):
    """Generate starting positions for blocks at the bottom of the grid"""
    source_positions = []
    
    # Generate starting positions at the bottom of the grid
    for i in range(4):
        row = grid_size - 5 - i  # Start from bottom rows
        for j in range(grid_size - 4):
            col = 2 + j
            source_positions.append((row, col))
            if len(source_positions) >= num_blocks:
                return source_positions
                
    # If we need more positions, use another row higher up
    row = grid_size - 10
    for j in range(4):
        col = 8 + j
        source_positions.append((row, col))
        if len(source_positions) >= num_blocks:
            return source_positions
            
    return source_positions

def create_heart_shape():
    """Create a heart shape pattern of target positions"""
    heart_positions = []
    # Heart shape pattern
    pattern = [
        (0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
        (0, 0, 1, 1, 0, 0, 1, 1, 0, 0),
        (0, 1, 1, 1, 1, 1, 1, 1, 1, 0),
        (0, 1, 1, 1, 1, 1, 1, 1, 1, 0),
        (0, 1, 1, 1, 1, 1, 1, 1, 1, 0),
        (0, 0, 1, 1, 1, 1, 1, 1, 0, 0),
        (0, 0, 0, 1, 1, 1, 1, 0, 0, 0),
        (0, 0, 0, 0, 1, 1, 0, 0, 0, 0),
        (0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    ]
    
    # Convert pattern to coordinates
    for i in range(len(pattern)):
        for j in range(len(pattern[0])):
            if pattern[i][j] == 1:
                heart_positions.append((i, j))
                
    return heart_positions

def main():
    # Initialize pygame
    pygame.init()
    
    # Create environment with a 30x30 grid
    n, m = 30, 30
    env = GradientEnvironment(n, m)
    
    # Create heart shape target positions
    target_positions = create_heart_shape()
    
    # Calculate center offset to place heart in center of grid
    center_row = (n - 10) // 2
    center_col = (m - 10) // 2
    
    # Adjust target positions to be centered
    targets = [(y + center_row, x + center_col) for y, x in target_positions]
    
    # Generate source positions
    sources = generate_source_positions(len(targets), n)
    
    # Create agents with assigned targets
    for i, (source, target) in enumerate(zip(sources, targets)):
        y, x = source
        agent_id = i + 1
        
        # Create gradient agent
        agent = GradientAgent(agent_id, x, y)
        agent.set_target((target[1], target[0]))  # Note: target is (y,x) but we need (x,y)
        
        env.add_agent(agent)
    
    # Create renderer
    cell_size = 20
    renderer = PygameRenderer(env, cell_size=cell_size)
    
    # For controlling simulation speed
    clock = pygame.time.Clock()
    
    running = True
    steps = 0
    max_steps = 500
    
    # Main simulation loop
    while running:
        steps += 1
        if steps > max_steps:
            print("Max steps reached. Exiting.")
            running = False
            
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                
        # Agents propose moves
        moves = env.propose_moves()
        
        # Environment applies moves
        env.apply_moves(moves)
        
        # Check if all agents reached their targets
        all_at_target = True
        for agent in env.agents.values():
            if agent.target is not None and (agent.x, agent.y) != agent.target:
                all_at_target = False
                break
                
        # Render the current state
        renderer.render()
        
        if all_at_target:
            print("All agents reached their targets!")
            pygame.time.wait(2000)  # Wait 2 seconds
            running = False
            
        # Limit simulation speed
        clock.tick(20)
        
    # Print final positions
    print("Final agent positions:")
    for agent_id, agent in env.agents.items():
        print(f"Agent {agent_id}: Position ({agent.x}, {agent.y}), Target {agent.target}")
        
    pygame.quit()

if __name__ == "__main__":
    main() 