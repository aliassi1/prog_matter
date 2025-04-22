import pygame
import numpy as np
from tkinter import Tk, filedialog
from PIL import Image
from last_trail_sarah import GridEnvironment, BlockAgent
from scipy.optimize import linear_sum_assignment

class PygameRenderer:
    """
    Renders the GridEnvironment on a Pygame window.
    
    """
    def __init__(self, env: GridEnvironment=None, cell_size=50, background_image=None):
        # Initialize Pygame
        pygame.init()
        pygame.display.set_caption("Block Agent Simulation")
        
        self.cell_size = cell_size
        self.env = env
        
        # Will be set later if not provided
        if env:

            self.grid_width = env.n
            self.grid_height = env.m
            self.screen_width = self.grid_width * self.cell_size
            self.screen_height = self.grid_height * self.cell_size
            self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))


        else:
            self.grid_width = 80
            self.grid_height = 80
            self.screen_width = self.grid_width * self.cell_size
            self.screen_height = self.grid_height * self.cell_size
            self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))


    def load_image(self):
        """Load an image using file dialog"""
        root = Tk()
        root.withdraw()  # Hide the root window
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")])
        if file_path:
            return pygame.image.load(file_path)
        return None

    def get_non_zero_coordinates(self, image):
        """Extract non-zero pixel coordinates from an image with better detection"""
        # Convert to PIL Image for processing
        img_str = pygame.image.tostring(image, 'RGB')
        img_size = image.get_size()
        pil_img = Image.frombytes('RGB', img_size, img_str)
        
        resize_dim = 10  
        
        # Save the processed image for display
        self.processed_pil_img = pil_img
        
        # Convert to grayscale
        if pil_img.mode != 'L':
            pil_img = pil_img.convert('L')
        
        # Apply stronger contrast enhancement
        import PIL.ImageOps
        pil_img = PIL.ImageOps.autocontrast(pil_img, cutoff=5)  # Reduced cutoff for more detail
        
        # Add debug visualization - save processed image to file
        pil_img.save("processed_image.png")
        
        # Convert to numpy array
        image_array = np.array(pil_img)
        
        # Try multiple threshold values
        # Use a lower threshold to capture more detail
        threshold = max(np.mean(image_array) * 0.5, 20)  # Lower threshold multiplier
        
        # Create binary image
        binary_image = image_array > threshold
        
        # Save binary version for debugging
        Image.fromarray(binary_image.astype(np.uint8) * 255).save("binary_image.png")
        
        # Find non-zero pixels (foreground)
        y_coords, x_coords = np.nonzero(binary_image)
        
        # If we got very few pixels, try a lower threshold
        if len(y_coords) < 10:
            threshold = max(np.mean(image_array) * 0.3, 10)
            binary_image = image_array > threshold
            y_coords, x_coords = np.nonzero(binary_image)
            # Save the new binary version
            Image.fromarray(binary_image.astype(np.uint8) * 255).save("binary_image_low_threshold.png")
        
        # If we got too many pixels (>90% of image), invert the selection
        if len(y_coords) > (resize_dim * resize_dim * 0.9):
            binary_image = ~binary_image
            y_coords, x_coords = np.nonzero(binary_image)
            # Save the inverted binary version
            Image.fromarray(binary_image.astype(np.uint8) * 255).save("binary_image_inverted.png")
        
        # Save binary image for visualization (debug)
        self.binary_image = binary_image
        
        # Debug output
        print(f"Detected {len(y_coords)} pixels with threshold {threshold}")
        
        # Return list of (y, x) tuples to be consistent with grid coordinates
        return list(zip(y_coords, x_coords))

    def generate_source_positions(self, num_blocks, grid_size=20):
        """Generate starting positions for blocks in the bottom-right corner"""
        source_positions = []
        
        # Simulate the same starting positions as in main.py
        for i in range(4):
            row = 15 + i  # Start from row 15
            for j in range(grid_size - 4):
                col = 2 + j
                source_positions.append((row, col))
                if len(source_positions) >= num_blocks:
                    return source_positions
                    
        # If we need more positions
        row = 14
        for j in range(4):
            col = 8 + j
            source_positions.append((row, col))
            if len(source_positions) >= num_blocks:
                return source_positions
                
        return source_positions

    def apply_hungarian_algorithm(self, sources, targets):
        """Apply Hungarian algorithm to assign agents to targets optimally"""
        # Create cost matrix with consistent coordinate format
        cost_matrix = np.zeros((len(sources), len(targets)))
        
        for i, (s_row, s_col) in enumerate(sources):
            for j, (t_row, t_col) in enumerate(targets):
                # Both are now in (row,col) = (y,x) format
                cost_matrix[i, j] = abs(s_row - t_row) + abs(s_col - t_col)
        
        # Apply Hungarian algorithm
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        # Create assignments
        assignments = []
        for i, j in zip(row_ind, col_ind):
            assignments.append((i, targets[j]))
        
        return assignments

    def setup_from_image(self):
        """Set up environment based on an image"""
        # Load image
        image = self.load_image()
        if not image:
            print("No image selected. Exiting.")
            return None
            
        # Process image to get target coordinates
        pixel_coords = self.get_non_zero_coordinates(image)
        
        if not pixel_coords:
            print("No pixels found in image. Exiting.")
            return None
        
        # Calculate center offset to place image in center of grid
        resize_dim = 30  # Same as in get_non_zero_coordinates
        center_row = (self.grid_height - resize_dim) // 2
        center_col = (self.grid_width - resize_dim) // 2
        
        print(f"Debug - Center offsets: row={center_row}, col={center_col}")
        
        # Create centered target positions - pixel_coords are now (y,x)
        target_positions = []
        for y, x in pixel_coords:
            target_row = y + center_row
            target_col = x + center_col
            # Ensure coordinates are within grid bounds
            if 0 <= target_row < self.grid_height and 0 <= target_col < self.grid_width:
                target_positions.append((target_row, target_col))
        
        # Print sample of target positions for debugging
        if target_positions:
            print(f"Debug - First 5 target positions (row,col): {target_positions[:5]}")
        
        # Fix debug grid visualization - consistent with (row,col) format
        debug_grid = np.zeros((self.grid_height, self.grid_width), dtype=np.uint8)
        for row, col in target_positions:
            debug_grid[row, col] = 255
        
        Image.fromarray(debug_grid).save("target_positions.png")
        
        print(f"Created {len(target_positions)} target positions from image")
        
        # Generate source positions
        source_positions = self.generate_source_positions(len(target_positions))
        if source_positions:
            print(f"Debug - First 5 source positions (row,col): {source_positions[:5]}")
        
        # Create environment
        env = GridEnvironment(self.grid_width, self.grid_height)
        
        # Apply Hungarian algorithm to assign agents to targets
        assignments = self.apply_hungarian_algorithm(source_positions, target_positions)
        
        # Debug assignments
        if assignments:
            print(f"Debug - First 5 assignments (idx, (row,col)): {assignments[:5]}")
        
        # Create agents with assigned targets
        for i, (agent_idx, target) in enumerate(assignments):
            source_row, source_col = source_positions[agent_idx]
            agent_id = i + 1
            
            # Important: BlockAgent takes (x,y) so we swap (row,col) to (col,row)
            block = BlockAgent(agent_id, source_col, source_row)
            
            target_row, target_col = target
            # Same for target - swap to (x,y) format
            block.set_target((target_col, target_row))
            
            print(f"Debug - Agent {agent_id}: Source (row,col)=({source_row},{source_col}), Source (x,y)=({source_col},{source_row})")
            print(f"Debug - Agent {agent_id}: Target (row,col)=({target_row},{target_col}), Target (x,y)=({target_col},{target_row})")
            
            env.add_agent(block)
        
        # Update the environment reference
        self.env = env
        
        return env

    def run_simulation(self, max_steps=500):
        """Run the simulation with the current environment"""
        if not self.env:
            print("No environment set up. Please call setup_from_image first.")
            return
        
        # Print initial agent positions and targets for debugging
        print("\nInitial agent positions and targets:")
        for agent_id, agent in self.env.agents.items():
            target_str = f"({agent.target[0]}, {agent.target[1]})" if agent.target else "None"
            print(f"Agent {agent_id}: Position (x,y)=({agent.x}, {agent.y}), Target (x,y)={target_str}")
        
        # For controlling the simulation speed
        clock = pygame.time.Clock()
        
        running = True
        steps = 0
        
        while running:
            # Check if we've hit maximum steps
            steps += 1
            if steps > max_steps:
                print("Max steps reached. Exiting.")
                running = False
                
            # Handle user events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    
            # Print positions every 50 steps for debugging
            if steps % 50 == 0:
                print(f"\nStep {steps} agent positions:")
                for agent_id, agent in self.env.agents.items():
                    target_str = f"({agent.target[0]}, {agent.target[1]})" if agent.target else "None"
                    print(f"Agent {agent_id}: Position ({agent.x}, {agent.y}), Target {target_str}")
                    
            # Agents propose moves
            moves = self.env.propose_moves()
            
            # Environment applies moves
            self.env.apply_moves(moves)
            
            # Render current state
            self.render()
            
            # Check if all agents reached their targets
            all_at_target = True
            for agent_id, agent in self.env.agents.items():
                if agent.target is not None:
                    tx, ty = agent.target
                    if agent.x != tx or agent.y != ty:
                        all_at_target = False
                        if steps % 50 == 0:  # Print debug info for misaligned agents periodically
                            print(f"Agent {agent_id} not at target: Position ({agent.x}, {agent.y}), Target ({tx}, {ty})")
                        break
            
            if all_at_target:
                print("All agents reached their targets. Exiting.")
                running = False
                
            # Limit simulation speed
            clock.tick(30)
            
        # Print final positions
        print("Final agent positions:")
        for agent_id, agent in self.env.agents.items():
            print(f"Agent {agent_id}: Position ({agent.x}, {agent.y}), Target {agent.target}")
            
        pygame.quit()

    def render(self):
        # Fill background with white
        self.screen.fill((255, 255, 255))
        
        # Draw grid lines
        for x in range(self.grid_width):
            for y in range(self.grid_height):
                rect = pygame.Rect(
                    x * self.cell_size,
                    y * self.cell_size,
                    self.cell_size,
                    self.cell_size
                )
                # Draw a light gray outline for each cell
                pygame.draw.rect(self.screen, (200, 200, 200), rect, 1)

        # Draw targets as light red squares
        if self.env:
            for agent in self.env.agents.values():
                if agent.target:
                    tx, ty = agent.target
                    target_rect = pygame.Rect(
                        tx * self.cell_size + 4,
                        ty * self.cell_size + 4,
                        self.cell_size - 8,
                        self.cell_size - 8
                    )
                    pygame.draw.rect(self.screen, (255, 200, 200), target_rect)

        # Draw each agent as a smaller square (blue block) within its cell
        if self.env:
            padding = 4  # Padding to keep the square inside the cell
            for agent in self.env.agents.values():
                ax, ay = agent.x, agent.y
                block_rect = pygame.Rect(
                    ax * self.cell_size + padding,
                    ay * self.cell_size + padding,
                    self.cell_size - 2 * padding,
                    self.cell_size - 2 * padding
                )
                pygame.draw.rect(self.screen, (0, 0, 255), block_rect)

        # Flip (update) the display
        pygame.display.flip()

if __name__ == "__main__":
    # When run directly, use the image-based setup
    renderer = PygameRenderer(cell_size=20)  # Using smaller cell size for 20x20 grid
    env = renderer.setup_from_image()
    if env:
        renderer.run_simulation()
