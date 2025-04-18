"""
Main file that:
  1) Uses trail_interface to load an image and get target positions
  2) Applies Hungarian algorithm to assign agents to targets
  3) Creates the environment with agents
  4) Runs the simulation with PygameRenderer
"""

import pygame
import numpy as np
from scipy.optimize import linear_sum_assignment
from last_trail_sarah import GridEnvironment, BlockAgent, SupervisorAgent
from render import PygameRenderer
from trail_interface import TrailInterface, process_image

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
    
    # Show welcome screen and get user choice
    choice = interface.start_menu()  # Capture the return value
    
    # Process user choice
    if choice == "heart":
        # Use heart shape, no image needed
        print("Using heart shape pattern")
        pixel_coordinates = interface.create_heart_shape()
    else:  # choice == "image"
        # Load an image
        player_image = None
        while player_image is None:
            player_image = interface.load_image()
            if player_image is None:
                print("Please select an image to continue.")
        
        # Process the image to get pixel coordinates
        pixel_coordinates = process_image(player_image)
    
    print(f"Number of pixels detected: {len(pixel_coordinates)}")
    
    # Create environment with a 60 x 60 grid
    n, m = 30, 30
    env = GridEnvironment(n, m)
    
    # Calculate the center offset to place image in center of grid
    # Now the image is 20x20 instead of 10x10
    center_row = (n - 20) // 2
    center_col = (m - 20) // 2
    
    # Create target positions (centered)
    # Note: pixel_coordinates are (y, x) but we need to convert to (x, y) for targets
    targets = [(y + center_row, x + center_col) for y, x in pixel_coordinates]
    
    # Generate source positions (where agents start)
    sources = generate_source_positions(len(targets), n)
    
    # Make sure we don't have more sources than targets
    if len(sources) > len(targets):
        sources = sources[:len(targets)]
    
    # Apply Hungarian algorithm to get optimal assignments
    assignments = apply_hungarian_algorithm(sources, targets)
    
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

        # Agents propose moves
        moves = env.propose_moves()

        # Environment applies moves
        env.apply_moves(moves)
        
        # Use the supervisor to check paths periodically
        if steps % 9 == 0:
            supervisor.check_paths()

        # Check if all agents reached their targets
        all_at_target = True
        for agent in env.agents.values():
            if agent.target is not None and (agent.x, agent.y) != agent.target:
                all_at_target = False
                break
                
        # Standard rendering during the formation process
        if not all_at_target:
            # Render the current state with agents
            renderer.render()
        else:
            # All agents reached targets - smooth transition to the image
            if choice == "image":
                # Only do image transition if we have an image
                # First, find the min and max coordinates of the targets
                min_x = min_y = float('inf')
                max_x = max_y = float('-inf')
                
                for agent in env.agents.values():
                    if agent.target:
                        target_x, target_y = agent.target
                        min_x = min(min_x, target_x)
                        max_x = max(max_x, target_x)
                        min_y = min(min_y, target_y)
                        max_y = max(max_y, target_y)
                
                # Calculate the grid size
                grid_width = max_x - min_x + 1
                grid_height = max_y - min_y + 1
                
                # Calculate pixel positions and sizes
                pixel_x = min_x * cell_size
                pixel_y = min_y * cell_size
                pixel_width = grid_width * cell_size
                pixel_height = grid_height * cell_size
                
                # Resize the image to fit the exact target area
                target_img = pygame.transform.scale(player_image, (pixel_width, pixel_height))
                
                # Create a surface with per-pixel alpha for smooth fading
                alpha_surface = pygame.Surface((pixel_width, pixel_height), pygame.SRCALPHA)
                
                # Perform the smooth transition
                for alpha in range(0, 256, 5):  # Increase by 5 for faster transition
                    # Render the current state with agents
                    renderer.render()
                    
                    # Set the alpha for this frame
                    alpha_surface.fill((255, 255, 255, 0))  # Clear with transparent
                    target_img_copy = target_img.copy()
                    target_img_copy.set_alpha(alpha)
                    alpha_surface.blit(target_img_copy, (0, 0))
                    
                    # Overlay the semi-transparent image
                    renderer.screen.blit(alpha_surface, (pixel_x, pixel_y))
                    
                    # Update the display
                    pygame.display.flip()
                    
                    # Short delay for the transition
                    pygame.time.delay(20)  # 20ms delay per frame
                
                # Display the final image for a moment
                pygame.time.wait(2000)  # 2 seconds
                print("All agents reached their targets. Exiting.")
                running = False
            else:
                # For heart shape, just display the final formation for a moment
                pygame.time.wait(2000)  # 2 seconds
                print("All agents reached their targets. Exiting.")
                running = False

        # Limit the simulation speed
        clock.tick(20)

    # Print final positions
    print("Final agent positions:")
    for agent_id, agent in env.agents.items():
        print(f"Agent {agent_id}: Position ({agent.x}, {agent.y}), Target {agent.target}")

    pygame.quit()


if __name__ == "__main__":
    main()
