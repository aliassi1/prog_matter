"""
Main file that:
  1) Uses trail_interface to load an image and get target positions
  2) Applies Hungarian algorithm to assign agents to targets
  3) Creates the environment with agents
  4) Runs the simulation with PygameRenderer
"""

import pygame
import numpy as np
import sys
import os
from scipy.optimize import linear_sum_assignment
from last_trail_sarah import GridEnvironment, BlockAgent, SupervisorAgent
from render import PygameRenderer
from trail_interface import TrailInterface, process_image

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src', 'grad'))
from gradient_agent import GradientAgent, GradientEnvironment

# Add the src directory to the Python path for RL
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from inference import ModelInference

def generate_source_positions(num_blocks, grid_size=30):
    """Generate exactly num_blocks unique starting positions for blocks at the bottom of the grid"""
    source_positions = []
    used = set()
    row = grid_size - 2
    col = 1
    while len(source_positions) < num_blocks:
        pos = (row, col)
        if pos not in used:
            source_positions.append(pos)
            used.add(pos)
        col += 1
        if col >= grid_size - 1:
            col = 1
            row -= 1
            if row < 0:
                row = grid_size - 2  # wrap around if needed
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
    mode, shape, level = interface.start_menu()  # Capture mode, shape, and level
    
    print(f"DEBUG: Selected mode: {mode}")
    
    # If RL mode, run the inference interface directly with the selected level
    if mode == "rl":
        print(f"Running RL inference interface for level {level}...")
        inference = ModelInference(
            checkpoint_path=None,  # Will find latest checkpoint
            visualize=True,
            num_episodes=5,  # Number of episodes to run
            delay=0.1,  # Delay between steps
            temperature=0.1,  # Temperature for action selection
            level=level  # Pass the selected level
        )
        inference.run_inference()
        return
    
    # Handle manual placement mode
    if shape == "manual":
        print("Using manual placement from interface.")
        agent_positions = interface.placed_agents
        target_positions = interface.placed_destinations

        n, m = 20, 20  # or whatever grid size you want

        # Choose the appropriate environment and agent classes based on mode
        if mode == "base":
            env = GridEnvironment(n, m)
            AgentClass = BlockAgent
        elif mode == "cellular":
            sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src', 'cellular_at'))
            from cellular_at.main import CellularEnvironment, CellularAgent
            env = CellularEnvironment(n, m, {
                'birth': [],  # No birth rules - we don't want to create new cells
                'survival': [0, 1, 2, 3, 4, 5, 6, 7, 8]  # All cells survive regardless of neighbors
            })
            AgentClass = CellularAgent
        else:
            env = GridEnvironment(n, m)
            AgentClass = BlockAgent

        # Create agents and assign targets
        for i, (agent_pos, target_pos) in enumerate(zip(agent_positions, target_positions)):
            agent_id = i + 1
            if mode == "cellular":
                agent = AgentClass(agent_id, agent_pos[0], agent_pos[1], env.rules)
            else:
                agent = AgentClass(agent_id, agent_pos[0], agent_pos[1])
            agent.set_target((target_pos[0], target_pos[1]))
            env.add_agent(agent)

        # Render and run the simulation
        renderer = PygameRenderer(env, cell_size=20)
        
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

            # Update simulation based on mode
            if mode == "cellular":
                # For cellular mode, use the cellular automata step
                env.step()
            else:
                # For other modes, use the standard move proposal
                moves = env.propose_moves()
                env.apply_moves(moves)

            # Check if all agents reached their targets
            all_at_target = True
            for agent in env.agents.values():
                if agent.target is not None and (agent.x, agent.y) != agent.target:
                    all_at_target = False
                    break

            if all_at_target:
                print("All agents reached their targets. Exiting.")
                running = False

            # Render the current state
            renderer.render()
            
            # Limit the simulation speed
            clock.tick(20)

        # Print final positions
        print("Final agent positions:")
        for agent_id, agent in env.agents.items():
            print(f"Agent {agent_id}: Position ({agent.x}, {agent.y}), Target {agent.target}")

        pygame.quit()
        return  # Exit after running manual mode
    
    # Initialize player_image as None
    player_image = None
    
    # Process user choice
    if shape == "heart":
        # Use heart shape, no image needed
        print("Using heart shape pattern")
        pixel_coordinates = interface.create_heart_shape()
    elif shape == "square":
        # Use square shape
        print("Using square shape pattern")
        pixel_coordinates = interface.create_square_shape()
    elif shape == "triangle":
        # Use triangle shape
        print("Using triangle shape pattern")
        pixel_coordinates = interface.create_triangle_shape()
    elif shape == "circle":
        # Use circle shape
        print("Using circle shape pattern")
        pixel_coordinates = interface.create_circle_shape()
    else:  # shape == "image"
        # Load an image
        while player_image is None:
            player_image = interface.load_image()
            if player_image is None:
                print("Please select an image to continue.")
        
        # Process the image to get pixel coordinates
        pixel_coordinates = process_image(player_image)
    
    print(f"Number of pixels detected: {len(pixel_coordinates)}")
    
    # Create environment with a 60 x 60 grid
    n, m = 30, 30
    
    # Choose the appropriate environment and agent classes based on mode
    if mode == "base":
        env = GridEnvironment(n, m)
        AgentClass = BlockAgent
    elif mode == "gradient":
        env = GradientEnvironment(n, m)
        AgentClass = GradientAgent
    elif mode == "cellular":
        # Import cellular automata code
        sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src', 'cellular_at'))
        from cellular_at.main import CellularEnvironment, CellularAgent
        env = CellularEnvironment(n, m, {
            'birth': [],  # No birth rules - we don't want to create new cells
            'survival': [0, 1, 2, 3, 4, 5, 6, 7, 8]  # All cells survive regardless of neighbors
        })
        AgentClass = CellularAgent
    else:
        print(f"Unknown mode: {mode}. Using base mode.")
        env = GridEnvironment(n, m)
        AgentClass = BlockAgent
    
    # Calculate the center offset to place image in center of grid
    center_row = (n - 20) // 2
    center_col = (m - 20) // 2
    
    # Create target positions (centered)
    targets = [(y + center_row, x + center_col) for y, x in pixel_coordinates]
    
    # Generate source positions (where agents start)
    sources = generate_source_positions(len(targets), n)
    print(f"DEBUG: Number of sources: {len(sources)} | Number of targets: {len(targets)}")
    print(f"DEBUG: Unique sources: {len(set(sources))}")
    print(f"DEBUG: Example sources: {sources[:10]}")
    print(f"DEBUG: Example targets: {targets[:10]}")

    # Apply Hungarian algorithm to get optimal assignments
    assignments = apply_hungarian_algorithm(sources, targets)
    print(f"DEBUG: Number of assignments: {len(assignments)}")

    # Create agents with assigned targets (guarantee one agent per target)
    for i, (agent_idx, target) in enumerate(assignments):
        y, x = sources[agent_idx]  # Sources are (y, x)
        agent_id = i + 1  # Start agent IDs from 1
        if mode == "cellular":
            agent = AgentClass(agent_id, x, y, env.rules)
        else:
            agent = AgentClass(agent_id, x, y)
        agent.set_target((target[1], target[0]))
        env.add_agent(agent)
    print(f"DEBUG: Number of agents created: {len(env.agents)}")

    # Create a supervisor agent for the environment (only for base mode)
    if mode == "base":
        supervisor = SupervisorAgent(env)

    # Create a renderer for this environment
    cell_size = 20
    # Only pass background_image if we have one (for custom image mode)
    renderer = PygameRenderer(env, cell_size=cell_size, background_image=player_image if shape == "image" else None)

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
        if mode == "cellular":
            # For cellular mode, use the cellular automata step
            env.step()
        else:
            # For other modes, use the standard move proposal
            moves = env.propose_moves()
            env.apply_moves(moves)
        
        # Use the supervisor to check paths periodically (only for base mode)
        if mode == "base" and steps % 9 == 0:
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
            if shape == "image":
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
                # For predefined shapes, just display the final formation for a moment
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
