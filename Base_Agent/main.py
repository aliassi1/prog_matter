"""
Main file that:
  1) Uses trail_interface to load an image and get target positions
  2) Applies selected algorithm to assign agents to targets
  3) Creates the environment with agents
  4) Runs the simulation with PygameRenderer
"""

import pygame
import numpy as np
from scipy.optimize import linear_sum_assignment
from last_trail_sarah import GridEnvironment, BlockAgent, SupervisorAgent
from render import PygameRenderer
from trail_interface import TrailInterface, process_image
from path_planning import (
    apply_minimax_algorithm,
    apply_alpha_beta_algorithm,
    apply_expectimax_algorithm,
    apply_gradient_based_algorithm,
    apply_cellular_automata_algorithm,
    apply_astar_greedy_local,
    apply_heuristic_movement
)

def generate_source_positions(num_blocks, grid_size=30):
    """Generate starting positions for blocks at the bottom of the grid"""
    source_positions = []
    
    # Generate starting positions at the bottom of the grid
    for i in range(4):
        row = grid_size - 5 - i  # Start from bottom rows (row 55 for a 60x60 grid)
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

def apply_hungarian_algorithm(sources, targets):
    """
    Apply the Hungarian algorithm to find optimal assignment of agents to targets.
    
    Args:
        sources: List of (x, y) coordinates of agents
        targets: List of (x, y) coordinates of targets
        
    Returns:
        List of (agent_idx, target_coordinate) pairs
    """
    # Create cost matrix (Manhattan distance between each agent and each target)
    cost_matrix = np.zeros((len(sources), len(targets)))
    
    for i, (s_x, s_y) in enumerate(sources):
        for j, (t_x, t_y) in enumerate(targets):
            # Manhattan distance
            cost_matrix[i, j] = abs(s_y - t_y) + abs(s_x - t_x)
    
    # Apply Hungarian algorithm
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    # Create assignments
    assignments = []
    for i, j in zip(row_ind, col_ind):
        assignments.append((i, targets[j]))
    
    return assignments

def main():
    # Create the trail interface to load an image
    interface = TrailInterface()
    
    # Show welcome screen and get user choice along with configuration
    choice, config = interface.run()
    
    # Extract configuration options
    topology = config["topology"]
    decision_making = config["decision_making"]
    movement_type = config["movement_type"]
    algorithm = config["algorithm"]
    
    print(f"\nSelected Configuration:")
    print(f"Topology: {topology}")
    print(f"Decision Making: {decision_making}")
    print(f"Movement Type: {movement_type}")
    print(f"Algorithm: {algorithm}")
    
    # Process user choice
    if choice == "heart":
        # Use heart shape, no image needed
        print("Using heart shape pattern")
        pixel_coordinates = interface.create_heart_shape()
        player_image = None
    else:  # choice == "image"
        # Load and process image
        print("Loading image...")
        player_image = interface.load_image()
        if player_image is None:
            print("Error: No image loaded")
            return
        pixel_coordinates = process_image(player_image)
    
    if not pixel_coordinates:
        print("Error: No valid pixels detected")
        return
        
    print(f"Number of pixels detected: {len(pixel_coordinates)}")
    
    # Create environment with a 30 x 30 grid
    n, m = 30, 30
    env = GridEnvironment(n, m)
    
    # Set environment configuration based on user choices
    env.set_topology(topology)
    env.set_decision_making(decision_making)
    env.set_movement_type(movement_type)
    
    # Calculate the center offset to place image in center of grid
    center_row = (n - 20) // 2
    center_col = (m - 20) // 2
    
    # Create target positions (centered)
    targets = [(y + center_row, x + center_col) for y, x in pixel_coordinates]
    
    # Generate source positions (where agents start)
    sources = generate_source_positions(len(targets), n)
    
    # Make sure we don't have more sources than targets
    if len(sources) > len(targets):
        sources = sources[:len(targets)]
    
    # Apply path planning algorithm based on user choice and decision making type
    print("Applying path planning algorithm...")
    if decision_making == "Centralized":
        if algorithm == "Minimax":
            assignments = apply_minimax_algorithm(sources, targets)
        elif algorithm == "Alpha-Beta":
            assignments = apply_alpha_beta_algorithm(sources, targets)
        else:  # Expectimax
            assignments = apply_expectimax_algorithm(sources, targets)
    else:  # Distributed
        if algorithm == "Gradient-based":
            assignments = apply_gradient_based_algorithm(sources, targets)
        elif algorithm == "Cellular Automata":
            assignments = apply_cellular_automata_algorithm(sources, targets)
        elif algorithm == "A*/Greedy":
            assignments = apply_astar_greedy_local(sources, targets)
        else:  # Heuristic
            assignments = apply_heuristic_movement(sources, targets)
    
    if not assignments:
        print("Error: No valid assignments generated")
        return
        
    print(f"Generated {len(assignments)} assignments")
    
    # Create agents with assigned targets
    for i, (agent_idx, target) in enumerate(assignments):
        y, x = sources[agent_idx]  # Sources are (y, x)
        agent_id = i + 1  # Start agent IDs from 1
        
        # BlockAgent constructor takes (x, y) not (y, x)
        block = BlockAgent(agent_id, x, y)
        
        # Target should be in format expected by BlockAgent (x, y)
        block.set_target((target[1], target[0]))
        
        env.add_agent(block)

    # Create a supervisor agent for the environment
    supervisor = SupervisorAgent(env)

    # Create a renderer for this environment
    cell_size = 20
    if choice == "heart":
        # No background image for heart shape
        renderer = PygameRenderer(env, cell_size=cell_size)
    else:
        # Use background image for custom image
        renderer = PygameRenderer(env, cell_size=cell_size, background_image=player_image)

    # For controlling the simulation speed
    clock = pygame.time.Clock()

    running = True
    steps = 0
    max_steps = 500

    print("Starting simulation...")
    
    # Main simulation loop
    while running:
        # Check if we've hit a maximum steps or user wants to close
        steps += 1
        if steps > max_steps:
            print("Max steps reached. Exiting.")
            running = False

        # Handle any user events (close button, etc.)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Agents propose moves based on movement type
        if movement_type == "Sequential":
            moves = env.propose_moves_sequential()
        elif movement_type == "Parallel":
            moves = env.propose_moves_parallel()
        else:  # Asynchronous
            moves = env.propose_moves_async()

        # Environment applies moves based on decision making type
        if decision_making == "Centralized":
            env.apply_moves_centralized(moves)
        else:  # Distributed
            env.apply_moves_distributed(moves)
        
        # Use the supervisor to check paths periodically
        if steps % 9 == 0:
            supervisor.check_paths()

        # Check if all agents reached their targets
        all_at_target = True
        for agent in env.agents.values():
            if agent.target is not None and (agent.x, agent.y) != agent.target:
                all_at_target = False
                break
                
        # Render current state
        renderer.render()
        
        # If all agents reached targets, wait a bit then exit
        if all_at_target:
            pygame.time.wait(2000)  # Wait 2 seconds
            running = False

        # Control simulation speed
        clock.tick(30)  # 30 FPS

    pygame.quit()

if __name__ == "__main__":
    main()
