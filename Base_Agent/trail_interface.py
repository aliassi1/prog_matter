import pygame
import sys
from tkinter import Tk, filedialog
from PIL import Image
import numpy as np
import math
import os
import random

def get_non_zero_coordinates(image: Image.Image):
    """
    Get the coordinates of all pixels in an image where the pixel values are not 0.
    """
    # Convert the image to grayscale to simplify processing
    if image.mode != 'L':
        image = image.convert('L')
    
    # Convert to numpy array
    image_array = np.array(image)
    
    # Find non-zero pixels
    y_coords, x_coords = np.nonzero(image_array)
    
    # Return list of (y, x) tuples
    return list(zip(y_coords, x_coords))


def process_image(img):
    """
    Process an image and return coordinates of non-zero pixels
    """
    # Convert pygame surface to PIL Image
    img_str = pygame.image.tostring(img, 'RGB')
    img_size = img.get_size()
    pil_img = Image.frombytes('RGB', img_size, img_str)
    
    # Resize while preserving aspect ratio
    max_size = 10
    width, height = pil_img.size
    ratio = min(max_size/width, max_size/height)
    new_size = (int(width * ratio), int(height * ratio))
    pil_img = pil_img.resize(new_size, Image.NEAREST)
    
    # Create a blank 50x50 image
    background = Image.new('RGB', (max_size, max_size), (0, 0, 0))
    # Paste the resized image in the center
    offset = ((max_size - new_size[0]) // 2, (max_size - new_size[1]) // 2)
    background.paste(pil_img, offset)
    pil_img = background
    
    # Get coordinates of non-transparent pixels
    coordinates = get_non_zero_coordinates(pil_img)
    
    # Ensure we have valid coordinates
    if not coordinates:
        # If no coordinates found, return at least one coordinate
        return [(0, 0)]
    
    return coordinates

class TrailInterface:
    def __init__(self):
        pygame.init()

        # Game Constants
        self.WIDTH, self.HEIGHT = 1200, 800  # Increased from 800x600 to 1200x800
        
        # Futuristic Elegant Garden Theme
        self.WHITE = (240, 245, 255)  # Bright white with slight blue tint
        self.BLACK = (25, 25, 35)  # Deep space black with slight purple tint
        self.PRIMARY = (90, 175, 110)  # Slightly darker emerald green for better contrast
        self.SECONDARY = (240, 130, 170)  # Vibrant pink
        self.TERTIARY = (160, 210, 150)  # Soft mint green
        self.QUATERNARY = (255, 190, 145)  # Warm peach
        self.BUTTON_COLOR = (55, 130, 75)  # Darker green for better contrast with white text
        self.BUTTON_HOVER = (70, 155, 90)  # Slightly lighter but still dark enough for contrast
        self.BACKGROUND_COLOR = (15, 50, 40)  # Deep emerald green background
        self.PANEL_COLOR = (25, 65, 50)  # Slightly lighter emerald green
        self.TEXT_SHADOW = (10, 30, 25, 100)  # Dark emerald shadow
        
        # Configuration options - simplified to 2 choices each
        self.topologies = ["Von Neumann", "Moore"]
        self.decision_making = ["Centralized", "Distributed"]
        self.movement_types = ["Sequential", "Parallel"]
        
        # Algorithm options based on decision making - simplified to 2 choices each
        self.centralized_algorithms = ["Minimax", "Expectimax"]
        self.distributed_algorithms = ["Gradient-based", "Cellular Automata"]
        
        # Default selections
        self.selected_topology = self.topologies[0]
        self.selected_decision = self.decision_making[0]
        self.selected_movement = self.movement_types[0]
        self.selected_algorithm = self.centralized_algorithms[0]  # Default to centralized algorithms
        
        # Create game window
        self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
        pygame.display.set_caption("Programmable Matter")
        
        # Fonts - using default fonts but with different sizes for a cleaner look
        pygame.font.init()
        self.title_font = pygame.font.Font(None, 72)
        self.button_font = pygame.font.Font(None, 30)  # Changed from 36 to 32
        self.subtitle_font = pygame.font.Font(None, 32)
        self.config_font = pygame.font.Font(None, 28)
        
        # Background elements
        self.bg_particles = self.create_particles(100)
        
        # Animation variables
        self.animation_tick = 0
        self.animation_speed = 0.05
        
        # Store previous mouse position for interactive effects
        self.prev_mouse_pos = pygame.mouse.get_pos()

    def create_particles(self, count):
        """Create floating particles for background effect"""
        particles = []
        for _ in range(count):
            x = random.randint(0, self.WIDTH)
            y = random.randint(0, self.HEIGHT)
            size = random.uniform(1, 5)
            speed = random.uniform(0.2, 1)
            color_type = random.choice(["primary", "secondary", "tertiary", "quaternary"])
            particles.append({
                "x": x, 
                "y": y, 
                "size": size, 
                "speed": speed,
                "color_type": color_type,
                "angle": random.uniform(0, math.pi * 2)
            })
        return particles
    
    def draw_particles(self):
        """Draw and update floating particles"""
        for particle in self.bg_particles:
            # Update position with a gentle floating motion
            particle["x"] += math.sin(particle["angle"]) * particle["speed"]
            particle["y"] += math.cos(particle["angle"]) * particle["speed"]
            particle["angle"] += 0.01
            
            # Wrap around screen
            if particle["x"] < 0:
                particle["x"] = self.WIDTH
            elif particle["x"] > self.WIDTH:
                particle["x"] = 0
            if particle["y"] < 0:
                particle["y"] = self.HEIGHT
            elif particle["y"] > self.HEIGHT:
                particle["y"] = 0
            
            # Determine color based on type
            if particle["color_type"] == "primary":
                color = self.PRIMARY
            elif particle["color_type"] == "secondary":
                color = self.SECONDARY
            elif particle["color_type"] == "tertiary":
                color = self.TERTIARY
            else:
                color = self.QUATERNARY
                
            # Add transparency
            alpha = random.randint(30, 100)
            particle_color = (*color, alpha)
            
            # Draw particle with glow effect
            glow_surface = pygame.Surface((particle["size"] * 3, particle["size"] * 3), pygame.SRCALPHA)
            pygame.draw.circle(
                glow_surface, 
                (*color, 40), 
                (glow_surface.get_width() // 2, glow_surface.get_height() // 2), 
                particle["size"] * 1.5
            )
            self.screen.blit(
                glow_surface, 
                (particle["x"] - glow_surface.get_width() // 2, 
                 particle["y"] - glow_surface.get_height() // 2)
            )
            
            pygame.draw.circle(
                self.screen, 
                particle_color, 
                (int(particle["x"]), int(particle["y"])), 
                particle["size"]
            )

    def draw_background(self):
        """Draw a modern gradient background with floating particles"""
        # Create gradient background from top to bottom
        for y in range(0, self.HEIGHT, 1):
            # Create a smooth gradient from dark to slightly lighter
            factor = y / self.HEIGHT
            r = min(255, int(self.BACKGROUND_COLOR[0] + (10 * factor)))
            g = min(255, int(self.BACKGROUND_COLOR[1] + (10 * factor)))
            b = min(255, int(self.BACKGROUND_COLOR[2] + (15 * factor)))
            pygame.draw.line(self.screen, (r, g, b), (0, y), (self.WIDTH, y))
        
        # Add a subtle grid pattern
        grid_color = (min(255, self.BACKGROUND_COLOR[0] + 10), 
                      min(255, self.BACKGROUND_COLOR[1] + 10), 
                      min(255, self.BACKGROUND_COLOR[2] + 20), 15)
        grid_spacing = 30
        
        for x in range(0, self.WIDTH, grid_spacing):
            pygame.draw.line(self.screen, grid_color, (x, 0), (x, self.HEIGHT))
        for y in range(0, self.HEIGHT, grid_spacing):
            pygame.draw.line(self.screen, grid_color, (0, y), (self.WIDTH, y))
        
        # Draw floating particles
        self.draw_particles()

    def draw_text_centered(self, text, font, color, y_offset, glow=False, shadow=True):
        """Function to display text centered horizontally with optional effects"""
        # Create text surface
        text_surface = font.render(text, True, color)
        text_rect = text_surface.get_rect(center=(self.WIDTH // 2, y_offset))
        
        # Add shadow for depth
        if shadow:
            shadow_surface = font.render(text, True, self.TEXT_SHADOW)
            shadow_rect = shadow_surface.get_rect(center=(self.WIDTH // 2 + 2, y_offset + 2))
            self.screen.blit(shadow_surface, shadow_rect)
        
        # Add glow effect
        if glow:
            glow_color = (*color[:3], 40)  # Transparent version of the text color
            for i in range(3):
                glow_surface = font.render(text, True, glow_color)
                glow_rect = glow_surface.get_rect(center=(self.WIDTH // 2, y_offset))
                # Expand the glow
                glow_surface = pygame.transform.scale(
                    glow_surface, 
                    (glow_rect.width + i*4, glow_rect.height + i*4)
                )
                glow_rect = glow_surface.get_rect(center=(self.WIDTH // 2, y_offset))
                self.screen.blit(glow_surface, glow_rect)
        
        # Draw main text
        self.screen.blit(text_surface, text_rect)
        return text_rect

    def draw_button(self, text, y, width, height, color=None, icon=None):
        """Draw a modern, futuristic button with hover effects"""
        mouse_x, mouse_y = pygame.mouse.get_pos()
        x = self.WIDTH // 2 - width // 2  # Center horizontally
        button_rect = pygame.Rect(x, y, width, height)
        
        # Determine colors
        button_color = color if color else self.BUTTON_COLOR
        hover_color = self.BUTTON_HOVER
        
        # Interactive effects when hovering
        is_hovering = button_rect.collidepoint(mouse_x, mouse_y)
        
        # Base shape
        if is_hovering:
            # Draw expanded glow when hovering
            for i in range(4):
                glow_rect = button_rect.inflate(i*6, i*4)
                s = pygame.Surface((glow_rect.width, glow_rect.height), pygame.SRCALPHA)
                glow_alpha = 120 - i*30
                pygame.draw.rect(s, (*hover_color, glow_alpha), s.get_rect(), border_radius=15)
                self.screen.blit(s, glow_rect)
            
            # Draw main button with hover color
            main_color = hover_color
        else:
            # Draw base button
            main_color = button_color
        
        # Draw main button with rounded corners
        pygame.draw.rect(self.screen, main_color, button_rect, border_radius=15)
        
        # Add glass effect (gradient overlay)
        gradient_surface = pygame.Surface((width, height//2), pygame.SRCALPHA)
        for i in range(height//2):
            alpha = 40 - int(i * 80 / height)
            if alpha > 0:
                pygame.draw.line(gradient_surface, (255, 255, 255, alpha), (0, i), (width, i))
        
        # Apply gradient to top half of button
        gradient_rect = gradient_surface.get_rect(topleft=(x, y))
        self.screen.blit(gradient_surface, gradient_rect)
        
        # Draw subtle border
        pygame.draw.rect(self.screen, (255, 255, 255, 100), button_rect, width=2, border_radius=15)
        
        # Add decorative elements for futuristic look
        if is_hovering:
            # Decorative corner marks
            corner_size = 10
            line_width = 2
            
            # Top left corner
            pygame.draw.line(self.screen, self.SECONDARY, 
                            (x, y + corner_size), 
                            (x, y), 
                            line_width)
            pygame.draw.line(self.screen, self.SECONDARY, 
                            (x, y), 
                            (x + corner_size, y), 
                            line_width)
            
            # Top right corner
            pygame.draw.line(self.screen, self.SECONDARY, 
                            (x + width - corner_size, y), 
                            (x + width, y), 
                            line_width)
            pygame.draw.line(self.screen, self.SECONDARY, 
                            (x + width, y), 
                            (x + width, y + corner_size), 
                            line_width)
            
            # Bottom left corner
            pygame.draw.line(self.screen, self.SECONDARY, 
                            (x, y + height - corner_size), 
                            (x, y + height), 
                            line_width)
            pygame.draw.line(self.screen, self.SECONDARY, 
                            (x, y + height), 
                            (x + corner_size, y + height), 
                            line_width)
            
            # Bottom right corner
            pygame.draw.line(self.screen, self.SECONDARY, 
                            (x + width - corner_size, y + height), 
                            (x + width, y + height), 
                            line_width)
            pygame.draw.line(self.screen, self.SECONDARY, 
                            (x + width, y + height), 
                            (x + width, y + height - corner_size), 
                            line_width)
        
        # Render text with shadow for depth
        text_color = self.WHITE
        
        # Shadow
        shadow_surface = self.button_font.render(text, True, self.TEXT_SHADOW)
        shadow_rect = shadow_surface.get_rect(center=button_rect.center)
        shadow_rect.y += 2  # Offset shadow slightly
        self.screen.blit(shadow_surface, shadow_rect)
        
        # Main text
        text_surface = self.button_font.render(text, True, text_color)
        text_rect = text_surface.get_rect(center=button_rect.center)
        
        # Add a slight bounce effect when hovering
        if is_hovering:
            text_rect.y -= 2
        
        self.screen.blit(text_surface, text_rect)
        
        # Add icon if provided
        if icon:
            icon_size = height - 20
            icon_rect = pygame.Rect(x + 15, y + (height - icon_size) // 2, icon_size, icon_size)
            self.screen.blit(icon, icon_rect)
            
        return button_rect

    def load_image(self):
        """Function to load an image"""
        root = Tk()
        root.withdraw()  # Hide Tkinter root window
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")])
        if file_path:
            return pygame.image.load(file_path)
        return None

    def start_menu(self):
        """Start menu function with improved game-like UI and mode selection"""
        self.animation_tick = 0
        mode = None
        shape = None
        level = None
        modes = [
            ("base", "Base"),
            ("gradient", "Gradient"),
            ("rl", "RL"),
            ("cellular", "Cellular Automata")
        ]
        while mode is None:
            self.animation_tick += self.animation_speed
            self.screen.fill(self.BACKGROUND_COLOR)
            self.draw_background()  # Use new background system
            pulse = (math.sin(self.animation_tick) + 1) / 2
            banner_height = 120
            pygame.draw.rect(self.screen, (20, 60, 45), (0, 0, self.WIDTH, banner_height))
            highlight_height = 5 + int(pulse * 3)
            pygame.draw.rect(self.screen, self.SECONDARY, (0, banner_height, self.WIDTH, highlight_height))
            glow_size = 2 + int(pulse * 3)
            title_shadow = self.title_font.render("Start Game", True, self.TEXT_SHADOW)
            title_shadow_rect = title_shadow.get_rect(center=(self.WIDTH//2 + glow_size, banner_height//2 + glow_size))
            self.screen.blit(title_shadow, title_shadow_rect)
            title_surface = self.title_font.render("Start Game", True, self.WHITE)
            title_rect = title_surface.get_rect(center=(self.WIDTH//2, banner_height//2))
            self.screen.blit(title_surface, title_rect)
            subtitle_color = (
                int(180 + 75 * pulse),
                int(180 + 75 * pulse),
                255
            )
            self.draw_text_centered("Choose Your Mode", self.subtitle_font, subtitle_color, banner_height + 60)
            # Draw mode buttons
            btn_width, btn_height = 200, 60
            btn_spacing = 20
            first_btn_y = banner_height + 120
            mode_buttons = []
            for i, (mode_key, mode_label) in enumerate(modes):
                btn_y = first_btn_y + i * (btn_height + btn_spacing)
                btn = self.draw_button(mode_label, btn_y, btn_width, btn_height)
                mode_buttons.append((btn, mode_key))
            pygame.display.flip()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                if event.type == pygame.MOUSEBUTTONDOWN:
                    for btn, mode_key in mode_buttons:
                        if btn.collidepoint(event.pos):
                            mode = mode_key
                            break

        # If RL mode is selected, show level selection
        if mode == "rl":
            level = self.show_level_selection()
            if level is None:
                pygame.quit()
                sys.exit()
            return mode, None, level  # Return mode, None for shape, and selected level

        # Now shape selection as before, but after mode selection
        while shape is None:
            self.animation_tick += self.animation_speed
            self.screen.fill(self.BACKGROUND_COLOR)
            self.draw_background()  # Use new background system
            pulse = (math.sin(self.animation_tick) + 1) / 2
            banner_height = 120
            pygame.draw.rect(self.screen, (20, 60, 45), (0, 0, self.WIDTH, banner_height))
            highlight_height = 5 + int(pulse * 3)
            pygame.draw.rect(self.screen, self.SECONDARY, (0, banner_height, self.WIDTH, highlight_height))
            glow_size = 2 + int(pulse * 3)
            title_shadow = self.title_font.render("Start Game", True, self.TEXT_SHADOW)
            title_shadow_rect = title_shadow.get_rect(center=(self.WIDTH//2 + glow_size, banner_height//2 + glow_size))
            self.screen.blit(title_shadow, title_shadow_rect)
            title_surface = self.title_font.render("Start Game", True, self.WHITE)
            title_rect = title_surface.get_rect(center=(self.WIDTH//2, banner_height//2))
            self.screen.blit(title_surface, title_rect)
            subtitle_color = (
                int(180 + 75 * pulse),
                int(180 + 75 * pulse),
                255
            )
            self.draw_text_centered(f"Mode: {mode.upper()} - Choose Shape", self.subtitle_font, subtitle_color, banner_height + 60)
            btn_width, btn_height = 300, 60
            btn_spacing = 20
            first_btn_y = banner_height + 120
            square_shape_button = self.draw_button("Use Square Shape", first_btn_y, btn_width, btn_height)
            triangle_shape_button = self.draw_button("Use Triangle Shape", first_btn_y + btn_height + btn_spacing, btn_width, btn_height)
            placement_button = self.draw_button("Manual Placement", first_btn_y + 2 * (btn_height + btn_spacing), btn_width, btn_height, self.TERTIARY)
            quit_button = self.draw_button("Quit", first_btn_y + 3 * (btn_height + btn_spacing), btn_width, btn_height)
            
            # Draw stars below the buttons
            for i in range(5):
                star_x = self.WIDTH * (0.1 + (i * 0.2))
                star_y = first_btn_y + 4 * (btn_height + btn_spacing) + 50  # Position stars below all buttons
                star_size = 15 + int(pulse * 5)
                self.draw_star(star_x, star_y, star_size, (220, 220, 100, 150))
            
            pygame.display.flip()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                if event.type == pygame.MOUSEBUTTONDOWN:
                    if square_shape_button.collidepoint(event.pos):
                        print("Square button pressed")
                        shape = "square"
                    elif triangle_shape_button.collidepoint(event.pos):
                        print("Triangle button pressed")
                        shape = "triangle"
                    elif placement_button.collidepoint(event.pos):
                        print("Placement button pressed")
                        self.placement_mode = True
                        self.placing_agents = True
                        self.placed_agents = []
                        self.placed_destinations = []
                        self.handle_placement_mode()
                        shape = "manual"  # Set shape to manual to indicate custom placement
                    elif quit_button.collidepoint(event.pos):
                        print("Quit button pressed")
                        pygame.quit()
                        sys.exit()
        return mode, shape, None

    def show_level_selection(self):
        """Show level selection screen for RL mode"""
        level = None
        max_levels = 10  # Maximum number of levels to show
        
        while level is None:
            self.animation_tick += self.animation_speed
            self.screen.fill(self.BACKGROUND_COLOR)
            self.draw_background()  # Use new background system
            pulse = (math.sin(self.animation_tick) + 1) / 2
            
            # Draw header
            banner_height = 120
            pygame.draw.rect(self.screen, (20, 60, 45), (0, 0, self.WIDTH, banner_height))
            highlight_height = 5 + int(pulse * 3)
            pygame.draw.rect(self.screen, self.SECONDARY, (0, banner_height, self.WIDTH, highlight_height))
            
            # Draw title
            title_surface = self.title_font.render("Select Level", True, self.WHITE)
            title_rect = title_surface.get_rect(center=(self.WIDTH//2, banner_height//2))
            self.screen.blit(title_surface, title_rect)
            
            # Draw subtitle
            subtitle_color = (int(180 + 75 * pulse), int(180 + 75 * pulse), 255)
            self.draw_text_centered("Choose a level to run the RL model", self.subtitle_font, subtitle_color, banner_height + 60)
            
            # Draw level buttons in a grid
            btn_width, btn_height = 100, 60
            btn_spacing = 20
            cols = 5  # Number of columns in the grid
            rows = (max_levels + cols - 1) // cols  # Calculate number of rows needed
            
            # Calculate starting position for the grid
            grid_width = cols * (btn_width + btn_spacing) - btn_spacing
            grid_height = rows * (btn_height + btn_spacing) - btn_spacing
            start_x = (self.WIDTH - grid_width) // 2
            start_y = banner_height + 120
            
            # Draw level buttons
            level_buttons = []
            for i in range(max_levels):
                row = i // cols
                col = i % cols
                x = start_x + col * (btn_width + btn_spacing)
                y = start_y + row * (btn_height + btn_spacing)
                btn = self.draw_button(f"Level {i+1}", y, btn_width, btn_height, self.PRIMARY)
                level_buttons.append((btn, i+1))
            
            # Draw back button
            back_btn = self.draw_button("Back", self.HEIGHT - 100, 200, 60, self.SECONDARY)
            
            pygame.display.flip()
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                if event.type == pygame.MOUSEBUTTONDOWN:
                    # Check level buttons
                    for btn, lvl in level_buttons:
                        if btn.collidepoint(event.pos):
                            level = lvl
                            break
                    # Check back button
                    if back_btn.collidepoint(event.pos):
                        return None
        
        return level

    def draw_star(self, x, y, size, color):
        """Draw a decorative star"""
        points = []
        for i in range(10):
            angle = math.pi * 2 * i / 10
            radius = size if i % 2 == 0 else size / 2
            points.append((x + radius * math.cos(angle), y + radius * math.sin(angle)))
        pygame.draw.polygon(self.screen, color, points)

    def handle_placement_mode(self):
        """Handle manual placement of agents and destinations on a grid"""
        done_button = None
        switch_button = None
        running = True
        
        # Grid settings
        grid_size = 20  # Size of each cell
        grid_width = 20  # Number of cells horizontally
        grid_height = 20  # Number of cells vertically
        grid_start_x = (self.WIDTH - grid_width * grid_size) // 2
        grid_start_y = 150
        
        while running:
            self.animation_tick += self.animation_speed
            self.screen.fill(self.BACKGROUND_COLOR)
            self.draw_background()  # Use new background system
            
            # Draw header
            banner_height = 120
            pygame.draw.rect(self.screen, (20, 60, 45), (0, 0, self.WIDTH, banner_height))
            title_surface = self.title_font.render("Manual Placement", True, self.WHITE)
            title_rect = title_surface.get_rect(center=(self.WIDTH//2, banner_height//2))
            self.screen.blit(title_surface, title_rect)
            
            # Draw instructions
            mode_text = "Placing Agents" if self.placing_agents else "Placing Destinations"
            self.draw_text_centered(mode_text, self.subtitle_font, self.WHITE, banner_height + 40)
            self.draw_text_centered("Click to place, Right-click to remove", self.subtitle_font, self.WHITE, banner_height + 80)
            
            # Draw grid
            for x in range(grid_width + 1):
                pygame.draw.line(self.screen, (100, 100, 100), 
                               (grid_start_x + x * grid_size, grid_start_y),
                               (grid_start_x + x * grid_size, grid_start_y + grid_height * grid_size))
            for y in range(grid_height + 1):
                pygame.draw.line(self.screen, (100, 100, 100),
                               (grid_start_x, grid_start_y + y * grid_size),
                               (grid_start_x + grid_width * grid_size, grid_start_y + y * grid_size))
            
            # Draw placed agents and destinations
            for agent in self.placed_agents:
                cell_x = grid_start_x + agent[0] * grid_size
                cell_y = grid_start_y + agent[1] * grid_size
                pygame.draw.rect(self.screen, self.PRIMARY, (cell_x, cell_y, grid_size, grid_size))
            
            for dest in self.placed_destinations:
                cell_x = grid_start_x + dest[0] * grid_size
                cell_y = grid_start_y + dest[1] * grid_size
                pygame.draw.rect(self.screen, self.SECONDARY, (cell_x, cell_y, grid_size, grid_size))
            
            # Draw buttons below the grid
            button_y = grid_start_y + grid_height * grid_size + 20
            switch_button = self.draw_button("Switch Mode", button_y, 200, 60, self.TERTIARY)
            done_button = self.draw_button("Done", button_y + 80, 200, 60, self.PRIMARY)
            
            pygame.display.flip()
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                if event.type == pygame.MOUSEBUTTONDOWN:
                    mouse_x, mouse_y = event.pos
                    # Check if click is within grid
                    if (grid_start_x <= mouse_x < grid_start_x + grid_width * grid_size and
                        grid_start_y <= mouse_y < grid_start_y + grid_height * grid_size):
                        # Convert mouse position to grid coordinates
                        grid_x = (mouse_x - grid_start_x) // grid_size
                        grid_y = (mouse_y - grid_start_y) // grid_size
                        
                        if event.button == 1:  # Left click
                            # Add new point
                            if self.placing_agents:
                                self.placed_agents.append((grid_x, grid_y))
                            else:
                                self.placed_destinations.append((grid_x, grid_y))
                        elif event.button == 3:  # Right click
                            # Remove point at this grid position
                            if self.placing_agents:
                                self.placed_agents = [p for p in self.placed_agents if p != (grid_x, grid_y)]
                            else:
                                self.placed_destinations = [p for p in self.placed_destinations if p != (grid_x, grid_y)]
                    elif event.button == 1:  # Left click on buttons
                        if switch_button.collidepoint(event.pos):
                            self.placing_agents = not self.placing_agents
                        elif done_button.collidepoint(event.pos):
                            running = False

    def create_triangle_shape(self):
        """Create a triangle shape pattern of target positions"""
        triangle_positions = []
        # Triangle shape pattern (equilateral triangle)
        pattern = [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 1, 1, 1, 0, 0, 0],
            [0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
            [0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        ]
        
        # Convert pattern to coordinates
        for i in range(len(pattern)):
            for j in range(len(pattern[0])):
                if pattern[i][j] == 1:
                    triangle_positions.append((i, j))
                    
        return triangle_positions

    def create_square_shape(self):
        """Create a square shape pattern of target positions"""
        square_positions = []
        # Square shape pattern (8x8 square)
        pattern = [
            [1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1]
        ]
        
        # Convert pattern to coordinates
        for i in range(len(pattern)):
            for j in range(len(pattern[0])):
                if pattern[i][j] == 1:
                    square_positions.append((i, j))
                    
        return square_positions

    def run(self):
        """Main game loop with improved game-like UI"""
        # Show start menu
        mode, shape, level = self.start_menu()
        print(f"DEBUG: mode={mode}, shape={shape}, level={level}")
        
        if shape == "manual":
            print("Using manual placement from interface.")
            agent_positions = self.placed_agents
            target_positions = self.placed_destinations

            n, m = 20, 20  # or whatever grid size you want

            # Choose the appropriate environment and agent classes based on mode
            if mode == "base":
                env = GridEnvironment(n, m)
                AgentClass = BlockAgent
                print(f"Using agent: {AgentClass.__name__}")
            elif mode == "gradient":
                sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
                from grad.gradient_module import GradientEnvironment, GradientAgent
                env = GradientEnvironment(n, m)
                AgentClass = GradientAgent
                print(f"Using agent: {AgentClass.__name__}")
            elif mode == "cellular":
                sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
                from cellular_at.main import CellularEnvironment, CellularAgent
                env = CellularEnvironment(n, m, {
                    'birth': [],
                    'survival': [0, 1, 2, 3, 4, 5, 6, 7, 8]
                })
                AgentClass = CellularAgent
                print(f"Using agent: {AgentClass.__name__}")
            else:
                # fallback or error
                ...

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
            renderer.run_simulation()
            return  # Exit after running manual mode
        else:
            # Handle other shape selections as before
            pixel_coordinates = []
            if shape == "heart":
                pixel_coordinates = self.create_heart_shape()
                message = f"Heart shape selected! (Mode: {mode}, Level: {level})"
            elif shape == "square":
                pixel_coordinates = self.create_square_shape()
                message = f"Square shape selected! (Mode: {mode}, Level: {level})"
            elif shape == "triangle":
                pixel_coordinates = self.create_triangle_shape()
                message = f"Triangle shape selected! (Mode: {mode}, Level: {level})"
            elif shape == "circle":
                pixel_coordinates = self.create_circle_shape()
                message = f"Circle shape selected! (Mode: {mode}, Level: {level})"
            else:
                print("No valid shape selected. Exiting.")
                pygame.quit()
                sys.exit()
            
            # Display the completion message with animation
            running = True
            self.animation_tick = 0
            
            while running:
                self.animation_tick += self.animation_speed
                pulse = (math.sin(self.animation_tick) + 1) / 2
                
                # Draw background
                self.screen.fill(self.BACKGROUND_COLOR)
                self.draw_background()  # Use new background system
                
                # Add game-like header
                pygame.draw.rect(self.screen, (40, 40, 80), (0, 0, self.WIDTH, 80))
                pygame.draw.rect(self.screen, self.SECONDARY, (0, 80, self.WIDTH, 3))
                
                # Display message in a game-style box
                message_box_width = 600
                message_box_height = 100
                message_box_x = self.WIDTH // 2 - message_box_width // 2
                message_box_y = 120
                
                # Message box with 3D effect
                pygame.draw.rect(self.screen, (60, 60, 100), 
                                (message_box_x, message_box_y, message_box_width, message_box_height), 
                                border_radius=15)
                pygame.draw.rect(self.screen, (80, 80, 120), 
                                (message_box_x, message_box_y, message_box_width, message_box_height), 
                                width=4, border_radius=15)
                
                # Message text
                message_surface = self.title_font.render(message, True, self.WHITE)
                message_rect = message_surface.get_rect(
                    center=(self.WIDTH // 2, message_box_y + message_box_height // 2)
                )
                self.screen.blit(message_surface, message_rect)
                
                # Visualize the shape in a game-like container
                if pixel_coordinates:
                    preview_title_y = 250
                    self.draw_text_centered("Shape Preview", self.subtitle_font, 
                                          (220, 220, 255), preview_title_y)
                    
                    # Draw a preview container
                    container_width = 400
                    container_height = 300
                    container_x = self.WIDTH // 2 - container_width // 2
                    container_y = preview_title_y + 30
                    
                    # Draw container background
                    pygame.draw.rect(self.screen, (50, 50, 90), 
                                    (container_x, container_y, container_width, container_height),
                                    border_radius=10)
                    pygame.draw.rect(self.screen, (100, 100, 140), 
                                    (container_x, container_y, container_width, container_height),
                                    width=3, border_radius=10)
                    
                    # Draw grid lines
                    grid_color = (70, 70, 110)
                    grid_size = 25
                    cells_x = 16  # Number of cells to show horizontally
                    cells_y = 12  # Number of cells to show vertically
                    
                    grid_width = cells_x * grid_size
                    grid_height = cells_y * grid_size
                    grid_x = self.WIDTH // 2 - grid_width // 2
                    grid_y = container_y + (container_height - grid_height) // 2
                    
                    # Draw grid lines
                    for i in range(cells_x + 1):
                        pygame.draw.line(self.screen, grid_color, 
                                        (grid_x + i * grid_size, grid_y),
                                        (grid_x + i * grid_size, grid_y + grid_height), 1)
                    
                    for j in range(cells_y + 1):
                        pygame.draw.line(self.screen, grid_color,
                                        (grid_x, grid_y + j * grid_size),
                                        (grid_x + grid_width, grid_y + j * grid_size), 1)
                    
                    # Draw shape pixels with glowing effect
                    for y, x in pixel_coordinates:
                        if 0 <= x < cells_x and 0 <= y < cells_y:
                            cell_x = grid_x + x * grid_size
                            cell_y = grid_y + y * grid_size
                            
                            # Add animated glow
                            glow_alpha = int(100 + 155 * pulse)
                            glow_surface = pygame.Surface((grid_size + 10, grid_size + 10), pygame.SRCALPHA)
                            pygame.draw.rect(glow_surface, (*self.SECONDARY[:3], glow_alpha), 
                                            (0, 0, grid_size + 10, grid_size + 10), border_radius=5)
                            self.screen.blit(glow_surface, (cell_x - 5, cell_y - 5))
                            
                            # Draw the actual pixel
                            pygame.draw.rect(self.screen, self.SECONDARY, 
                                            (cell_x, cell_y, grid_size, grid_size))
                            
                            # Add highlight
                            pygame.draw.rect(self.screen, (255, 255, 255, 100), 
                                            (cell_x, cell_y, grid_size, 3))
                
                # Instructions
                instruction_y = self.HEIGHT - 70
                instruction_color = (200, 200, 200, int(200 + 55 * pulse))
                instruction_surface = self.subtitle_font.render("Press any key to continue", True, instruction_color)
                instruction_rect = instruction_surface.get_rect(center=(self.WIDTH // 2, instruction_y))
                self.screen.blit(instruction_surface, instruction_rect)
                
                pygame.display.flip()
                
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                    elif event.type == pygame.KEYDOWN or event.type == pygame.MOUSEBUTTONDOWN:
                        running = False
            
            pygame.quit()
            sys.exit()

if __name__ == "__main__":
    interface = TrailInterface()
    interface.run() 