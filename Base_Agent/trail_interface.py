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
        self.WIDTH, self.HEIGHT = 800, 600
        
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
        self.button_font = pygame.font.Font(None, 36)
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

    def create_heart_shape(self):
        """Generate coordinates for a heart shape in a grid"""
        heart_pixels = [
            (8, 4), (9, 4), (10, 4), (11, 4),
            (7, 5), (8, 5), (9, 5), (10, 5), (11, 5), (12, 5),
            (6, 6), (7, 6), (8, 6), (9, 6), (10, 6), (11, 6), (12, 6), (13, 6),
            (5, 7), (6, 7), (7, 7), (8, 7), (9, 7), (10, 7), (11, 7), (12, 7), (13, 7), (14, 7),
            (4, 8), (5, 8), (6, 8), (7, 8), (8, 8), (9, 8), (10, 8), (11, 8), (12, 8), (13, 8), (14, 8), (15, 8),
            (4, 9), (5, 9), (6, 9), (7, 9), (8, 9), (9, 9), (10, 9), (11, 9), (12, 9), (13, 9), (14, 9), (15, 9),
            (5, 10), (6, 10), (7, 10), (8, 10), (9, 10), (10, 10), (11, 10), (12, 10), (13, 10), (14, 10),
            (6, 11), (7, 11), (8, 11), (9, 11), (10, 11), (11, 11), (12, 11), (13, 11),
            (7, 12), (8, 12), (9, 12), (10, 12), (11, 12), (12, 12),
            (8, 13), (9, 13), (10, 13), (11, 13),
            (9, 14), (10, 14)
        ]
        return heart_pixels

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
        """Function to load an image using native macOS file dialog"""
        try:
            import subprocess
            import tempfile
            import os
            
            # Create a temporary AppleScript file
            script = '''
            set theFile to choose file with prompt "Select an image:" of type {"png", "jpg", "jpeg"}
            POSIX path of theFile
            '''
            
            with tempfile.NamedTemporaryFile(suffix='.applescript', mode='w', delete=False) as f:
                f.write(script)
                script_path = f.name
            
            # Run AppleScript and get the selected file path
            try:
                result = subprocess.run(['osascript', script_path], capture_output=True, text=True)
                file_path = result.stdout.strip()
                
                # Clean up the temporary script file
                os.unlink(script_path)
                
                if file_path:
                    try:
                        # Try loading with Pygame
                        return pygame.image.load(file_path)
                    except pygame.error:
                        print("Error loading image with Pygame. Trying PIL...")
                        try:
                            # Try loading with PIL as fallback
                            from PIL import Image
                            pil_image = Image.open(file_path)
                            mode = pil_image.mode
                            size = pil_image.size
                            data = pil_image.tobytes()
                            py_image = pygame.image.fromstring(data, size, mode)
                            return py_image
                        except Exception as e:
                            print(f"Error loading image with PIL: {e}")
                            return None
            except subprocess.SubprocessError as e:
                print(f"Error running AppleScript: {e}")
                return None
                
        except Exception as e:
            print(f"Error in file dialog: {e}")
            return None

    def draw_config_option(self, title, options, selected, y_pos, spacing=40):
        """Draw a modern, futuristic configuration option with animated effects"""
        # Draw title with cute ribbon/badge effect
        title_color = self.WHITE
        title_text = title
        
        # Create ribbon background
        ribbon_width = 180
        ribbon_height = 40
        
        # Center the entire row including ribbon and options
        total_width = ribbon_width + 30  # Ribbon width plus some spacing
        
        # Calculate option button dimensions
        button_width = 200  # Base button width
        button_height = 40  # Define button height
        button_spacing = 20
        
        # Add width of option buttons to total width
        total_width += len(options) * button_width + (len(options) - 1) * button_spacing
        
        # Center the entire row horizontally
        start_x = (self.WIDTH - total_width) // 2
        
        # Position ribbon at start of the row
        ribbon_x = start_x
        ribbon_y = y_pos - ribbon_height // 2
        
        # Draw ribbon with gradient
        ribbon_rect = pygame.Rect(ribbon_x, ribbon_y, ribbon_width, ribbon_height)
        pygame.draw.rect(self.screen, self.TERTIARY, ribbon_rect, border_radius=8)
        
        # Add glass effect to ribbon
        gradient_surface = pygame.Surface((ribbon_width, ribbon_height // 2), pygame.SRCALPHA)
        for i in range(ribbon_height // 2):
            alpha = 40 - int(i * 80 / ribbon_height)
            if alpha > 0:
                pygame.draw.line(gradient_surface, (255, 255, 255, alpha), (0, i), (ribbon_width, i))
        gradient_rect = gradient_surface.get_rect(topleft=(ribbon_x, ribbon_y))
        self.screen.blit(gradient_surface, gradient_rect)
        
        # Add decorative elements to ribbon
        pygame.draw.rect(self.screen, (255, 255, 255, 100), ribbon_rect, width=2, border_radius=8)
        
        # Add triangle decorations on sides
        triangle_size = 10
        triangle_color = self.SECONDARY
        
        # Left triangle
        pygame.draw.polygon(self.screen, triangle_color, [
            (ribbon_x - triangle_size, ribbon_y + ribbon_height // 2),
            (ribbon_x, ribbon_y + ribbon_height // 4),
            (ribbon_x, ribbon_y + ribbon_height * 3 // 4)
        ])
        
        # Right triangle
        pygame.draw.polygon(self.screen, triangle_color, [
            (ribbon_x + ribbon_width + triangle_size, ribbon_y + ribbon_height // 2),
            (ribbon_x + ribbon_width, ribbon_y + ribbon_height // 4),
            (ribbon_x + ribbon_width, ribbon_y + ribbon_height * 3 // 4)
        ])
        
        # Draw title text on ribbon
        title_surface = self.config_font.render(title_text, True, self.WHITE)
        title_rect = title_surface.get_rect(center=(ribbon_x + ribbon_width // 2, ribbon_y + ribbon_height // 2))
        
        # Add shadow for depth
        shadow_surface = self.config_font.render(title_text, True, self.TEXT_SHADOW)
        shadow_rect = shadow_surface.get_rect(center=(title_rect.centerx + 1, title_rect.centery + 1))
        self.screen.blit(shadow_surface, shadow_rect)
        
        # Draw main text
        self.screen.blit(title_surface, title_rect)
        
        # Calculate option button dimensions and placement
        # Ensure buttons stay within panel bounds
        panel_width = self.WIDTH - 120  # Account for panel margins
        
        # Options start after the ribbon
        options_start_x = ribbon_x + ribbon_width + 30
        
        # Special font size adjustment for algorithm options
        is_algorithm_row = title == "Algorithm"
        
        # Find the option with the longest text to determine proper font size
        longest_option = max(options, key=len)
        longest_width = self.button_font.size(longest_option)[0]
        
        # Calculate the scale factor if text is too wide
        scale_factor = 1.0
        if longest_width > button_width - 30:  # 15px padding on each side
            scale_factor = (button_width - 30) / longest_width
            
        # Use a smaller font for options if needed
        option_font = self.button_font
        if scale_factor < 0.9:  # Only create a new font if significant scaling is needed
            # For algorithm options, use a slightly larger font than the calculation would suggest
            if is_algorithm_row:
                option_font_size = int(self.button_font.get_height() * max(scale_factor + 0.1, 0.75))
            else:
                option_font_size = int(self.button_font.get_height() * scale_factor)
            option_font = pygame.font.Font(None, max(20, option_font_size))  # Ensure minimum font size
            
        # Get mouse position for hover effects
        mouse_pos = pygame.mouse.get_pos()
        
        # Variables to track click detection
        clicked_button = None
        clicked_option = None
        
        # Draw each option as a selectable button
        for i, option in enumerate(options):
            # Calculate button position
            button_x = options_start_x + i * (button_width + button_spacing)
            button_y = y_pos - button_height // 2
            button_rect = pygame.Rect(button_x, button_y, button_width, button_height)
            
            # Check if mouse is hovering
            is_hovering = button_rect.collidepoint(mouse_pos)
            is_selected = option == selected
            
            # Determine button color based on selection/hover state
            if is_selected:
                base_color = self.SECONDARY
            else:
                base_color = self.PRIMARY
                
            if is_hovering and not is_selected:
                base_color = self.BUTTON_HOVER
            
            # Draw button with glow effect for selected or hovered
            if is_selected or is_hovering:
                # Draw glow
                glow_intensity = 3 if is_selected else 2
                for j in range(glow_intensity):
                    glow_rect = button_rect.inflate(j * 6, j * 4)
                    glow_surface = pygame.Surface((glow_rect.width, glow_rect.height), pygame.SRCALPHA)
                    glow_alpha = 120 - j * 30
                    glow_color = (*base_color, glow_alpha)
                    pygame.draw.rect(glow_surface, glow_color, glow_surface.get_rect(), border_radius=12)
                    self.screen.blit(glow_surface, glow_rect)
            
            # Draw main button
            pygame.draw.rect(self.screen, base_color, button_rect, border_radius=12)
            
            # Add glass effect
            gradient_surface = pygame.Surface((button_width, button_height // 2), pygame.SRCALPHA)
            for j in range(button_height // 2):
                alpha = 30 - int(j * 60 / button_height)
                if alpha > 0:
                    pygame.draw.line(gradient_surface, (255, 255, 255, alpha), (0, j), (button_width, j))
            
            gradient_rect = gradient_surface.get_rect(topleft=(button_x, button_y))
            self.screen.blit(gradient_surface, gradient_rect)
            
            # Add a subtle border
            border_color = (255, 255, 255, 120) if is_selected else (255, 255, 255, 70)
            pygame.draw.rect(self.screen, border_color, button_rect, width=2, border_radius=12)
            
            # Individual text sizing for algorithm options
            text_font = option_font
            if is_algorithm_row:
                # Special handling for gradient-based and cellular automata to make them slightly larger
                if option in ["Gradient-based", "Cellular Automata"]:
                    # Increased font size for these specific options 
                    specific_font_size = min(50, int(self.button_font.get_height() * 1.05))
                    text_font = pygame.font.Font(None, specific_font_size)
                    
            # Prepare text for rendering with appropriate sizing
            text_color = self.WHITE
            
            # Render text with the calculated font
            shadow_surface = text_font.render(option, True, self.TEXT_SHADOW)
            shadow_rect = shadow_surface.get_rect(center=button_rect.center)
            shadow_rect.y += 2  # Offset shadow slightly
            self.screen.blit(shadow_surface, shadow_rect)
            
            # Render main text
            text_surface = text_font.render(option, True, text_color)
            text_rect = text_surface.get_rect(center=button_rect.center)
            
            # Add a slight bounce effect when hovering or selected
            if is_hovering or is_selected:
                text_rect.y -= 1
                
            self.screen.blit(text_surface, text_rect)
            
            # If hovering, this is our potentially clicked button
            if is_hovering:
                clicked_button = button_rect
                clicked_option = option
                
        return clicked_button, clicked_option

    def configuration_menu(self):
        """Show a futuristic configuration menu with smooth animations"""
        running = True
        start_time = pygame.time.get_ticks()
        
        while running:
            current_time = pygame.time.get_ticks()
            elapsed = (current_time - start_time) / 1000  # convert to seconds
            
            # Draw the fancy background
            self.draw_background()
            
            # Add a decorative header panel
            header_height = 120
            pygame.draw.rect(self.screen, self.PANEL_COLOR, (0, 0, self.WIDTH, header_height))
            
            # Animated line under header
            line_width = int(min(self.WIDTH, (current_time - start_time) / 2))
            if line_width > 0:
                pygame.draw.rect(self.screen, self.SECONDARY, (0, header_height, line_width, 3))
            
            # Draw title with animated glow
            pulse = (math.sin(elapsed * 3) + 1) / 2  # Gentle pulse effect
            glow_intensity = int(40 + 60 * pulse)
            title_color = self.WHITE
            
            # Draw title text with glow effect - Changed from "Configuration" to "Choose a Configuration"
            self.draw_text_centered("Choose a Configuration", self.title_font, title_color, header_height // 2, glow=True)
            
            # Draw main panel for configuration options
            panel_margin = 60
            panel_width = self.WIDTH - panel_margin * 2
            panel_height = 400  # Increased height for more spacing
            panel_x = panel_margin
            panel_y = header_height + 40
            
            # Draw panel background with rounded corners
            panel_rect = pygame.Rect(panel_x, panel_y, panel_width, panel_height)
            pygame.draw.rect(self.screen, self.PANEL_COLOR, panel_rect, border_radius=15)
            
            # Add decorative elements to panel
            # Top accent line
            pygame.draw.line(self.screen, self.TERTIARY, 
                           (panel_x + 20, panel_y + 15), 
                           (panel_x + panel_width - 20, panel_y + 15), 
                           2)
            
            # Bottom accent line
            pygame.draw.line(self.screen, self.TERTIARY, 
                           (panel_x + 20, panel_y + panel_height - 15), 
                           (panel_x + panel_width - 20, panel_y + panel_height - 15), 
                           2)
            
            # Draw configuration options with increased spacing
            base_y = panel_y + 80  # Start position (increased)
            y_spacing = 70  # Increased spacing between options
            
            # Draw each configuration option
            topology_rect, new_topology = self.draw_config_option("Topology", self.topologies, 
                                                                self.selected_topology, base_y)
            
            decision_rect, new_decision = self.draw_config_option("Decision Making", self.decision_making, 
                                                                self.selected_decision, base_y + y_spacing)
            
            movement_rect, new_movement = self.draw_config_option("Movement Type", self.movement_types, 
                                                                self.selected_movement, base_y + y_spacing * 2)
            
            # Choose algorithm options based on decision making type
            current_algorithms = (self.centralized_algorithms 
                                if self.selected_decision == "Centralized" 
                                else self.distributed_algorithms)
            
            # If decision making changed, update selected algorithm to first in new list
            if new_decision and new_decision != self.selected_decision:
                self.selected_algorithm = (self.centralized_algorithms[0] 
                                         if new_decision == "Centralized" 
                                         else self.distributed_algorithms[0])
            
            algorithm_rect, new_algorithm = self.draw_config_option("Algorithm", current_algorithms, 
                                                                  self.selected_algorithm, base_y + y_spacing * 3)
            
            # Draw continue button with animated highlight - centered
            button_pulse = (math.sin(elapsed * 4) + 1) / 2
            continue_glow = self.QUATERNARY if button_pulse > 0.7 else self.PRIMARY
            # Position button at a fixed position that's visible within screen bounds 
            continue_rect = self.draw_button("Continue", self.HEIGHT - 80, 200, 50, continue_glow)
            
            # Process events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                    
                if event.type == pygame.MOUSEBUTTONDOWN:
                    # Update selections based on clicks
                    if new_topology:
                        self.selected_topology = new_topology
                    if new_decision:
                        self.selected_decision = new_decision
                    if new_movement:
                        self.selected_movement = new_movement
                    if new_algorithm:
                        self.selected_algorithm = new_algorithm
                    
                    # Handle continue button click
                    if continue_rect.collidepoint(event.pos):
                        # Play click animation
                        self.play_button_click_animation(continue_rect)
                        
                        # Return configuration dictionary
                        config = {
                            "topology": self.selected_topology,
                            "decision_making": self.selected_decision,
                            "movement_type": self.selected_movement,
                            "algorithm": self.selected_algorithm
                        }
                        print("Configuration selected:", config)  # Debug print
                        return config
            
            pygame.display.flip()
            pygame.time.Clock().tick(60)  # Limit to 60 FPS
    
    def play_button_click_animation(self, button_rect):
        """Play a quick satisfying animation when a button is clicked"""
        original_size = button_rect.size
        
        for scale in [0.95, 0.9, 0.93, 0.96, 0.98, 1.0]:
            # Clear screen and redraw background
            self.draw_background()
            
            # Calculate scaled size
            scaled_width = int(original_size[0] * scale)
            scaled_height = int(original_size[1] * scale)
            
            # Center the scaled button at the same position
            scaled_rect = pygame.Rect(
                button_rect.centerx - scaled_width // 2,
                button_rect.centery - scaled_height // 2,
                scaled_width,
                scaled_height
            )
            
            # Draw the scaled button
            pygame.draw.rect(self.screen, self.SECONDARY, scaled_rect, border_radius=15)
            
            # Add a pulse of light
            glow_surface = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            pygame.draw.circle(glow_surface, (255, 255, 255, int(50 * (1 - scale))), 
                              (button_rect.centerx, button_rect.centery), 
                              int(original_size[0] * (2 - scale)))
            self.screen.blit(glow_surface, (0, 0))
            
            # Update display
            pygame.display.flip()
            pygame.time.delay(30)

    def start_menu(self):
        """Modern, futuristic start menu with animated elements"""
        self.animation_tick = 0
        start_time = pygame.time.get_ticks()
        
        # Create animated logo elements
        logo_particles = []
        for _ in range(30):
            angle = random.uniform(0, math.pi * 2)
            speed = random.uniform(0.5, 2)
            size = random.uniform(2, 6)
            distance = random.uniform(10, 100)
            logo_particles.append({
                'angle': angle,
                'speed': speed,
                'size': size,
                'distance': distance,
                'color': random.choice([self.PRIMARY, self.SECONDARY, self.TERTIARY, self.QUATERNARY])
            })
        
        while True:
            current_time = pygame.time.get_ticks()
            elapsed = (current_time - start_time) / 1000  # Convert to seconds
            self.animation_tick += self.animation_speed
            
            # Draw animated background
            self.draw_background()
            
            # Top banner with animated gradient
            banner_height = 160
            
            # Create gradient from primary to secondary color
            for y in range(banner_height):
                progress = y / banner_height
                r = int(self.PRIMARY[0] * (1 - progress) + self.PANEL_COLOR[0] * progress)
                g = int(self.PRIMARY[1] * (1 - progress) + self.PANEL_COLOR[1] * progress)
                b = int(self.PRIMARY[2] * (1 - progress) + self.PANEL_COLOR[2] * progress)
                pygame.draw.line(self.screen, (r, g, b), (0, y), (self.WIDTH, y))
            
            # Add decorative animated accent line
            line_pos = int(50 + 10 * math.sin(elapsed * 2))
            line_alpha = int(100 + 100 * (math.sin(elapsed * 3) + 1) / 2)
            line_surface = pygame.Surface((self.WIDTH, 3), pygame.SRCALPHA)
            pygame.draw.rect(line_surface, (*self.SECONDARY, line_alpha), (0, 0, self.WIDTH, 3))
            self.screen.blit(line_surface, (0, banner_height - line_pos))
            
            # Add an animated decorative circular element
            circle_x = self.WIDTH - 80
            circle_y = 80
            circle_radius = 40 + int(5 * math.sin(elapsed * 2))
            
            # Draw outer circle
            pygame.draw.circle(self.screen, self.SECONDARY, (circle_x, circle_y), circle_radius, 3)
            
            # Draw inner circle with pulsing effect
            inner_radius = int(circle_radius * 0.7 * (0.8 + 0.2 * math.sin(elapsed * 4)))
            pygame.draw.circle(self.screen, self.TERTIARY, (circle_x, circle_y), inner_radius)
            
            # Add decorative dots inside the circle
            for p in logo_particles:
                # Update angle for rotation
                p['angle'] += elapsed * p['speed'] * 0.01
                
                # Calculate position
                px = circle_x + math.cos(p['angle']) * p['distance'] * inner_radius/circle_radius
                py = circle_y + math.sin(p['angle']) * p['distance'] * inner_radius/circle_radius
                
                # Only draw if inside the circle
                if (px - circle_x)**2 + (py - circle_y)**2 <= circle_radius**2:
                    pygame.draw.circle(self.screen, p['color'], (int(px), int(py)), int(p['size']))
            
            # Title with glow effect
            title_y = banner_height // 2
            title_glow = int(40 + 20 * math.sin(elapsed * 3))
            title_color = self.WHITE
            
            self.draw_text_centered("Programmable Matter", self.title_font, title_color, title_y, glow=True)
            
            # Subtitle with soft animation
            subtitle_y = title_y + 50
            subtitle_color = (240, 240, 255)
            subtitle_text = "Interactive Simulation"
            self.draw_text_centered(subtitle_text, self.subtitle_font, subtitle_color, subtitle_y)
            
            # Draw decorative horizontal divider
            divider_y = banner_height + 30
            divider_width = 600
            divider_x = (self.WIDTH - divider_width) // 2
            
            pygame.draw.line(self.screen, self.TERTIARY, (divider_x, divider_y), (divider_x + divider_width, divider_y), 2)
            
            # Add decorative elements to divider
            dot_spacing = divider_width // 8
            for i in range(9):
                dot_x = divider_x + i * dot_spacing
                dot_size = 4 if i % 2 == 0 else 2
                dot_color = self.SECONDARY if i % 2 == 0 else self.TERTIARY
                pygame.draw.circle(self.screen, dot_color, (dot_x, divider_y), dot_size)
            
            # Create animated button highlights
            button_pulse = (math.sin(elapsed * 3) + 1) / 2
            heart_highlight = self.QUATERNARY if button_pulse > 0.7 else self.PRIMARY
            image_highlight = self.TERTIARY if button_pulse < 0.3 else self.PRIMARY
            
            # Draw cute heart icon for heart button
            heart_icon_size = 30
            heart_icon = self.create_heart_icon(heart_icon_size)
            
            # Draw image icon for image button
            image_icon_size = 30
            image_icon = self.create_image_icon(image_icon_size)
            
            # Draw buttons with icons and modern styling
            heart_button = self.draw_button("Heart Shape", 280, 220, 50, heart_highlight)
            image_button = self.draw_button("Load Image", 360, 220, 50, image_highlight)
            
            # Add floating description text
            description_y = 480
            description_opacity = int(200 + 55 * math.sin(elapsed * 2))
            description_color = (220, 220, 255, description_opacity)
            
            # Create multiline description with a bit of animation
            descriptions = [
                "Choose Heart Shape for a cute heart pattern",
                "or Load Image to use your own custom image"
            ]
            
            for i, line in enumerate(descriptions):
                desc_surface = self.subtitle_font.render(line, True, description_color)
                desc_rect = desc_surface.get_rect(center=(self.WIDTH//2, description_y + i*40))
                
                # Apply gentle floating motion
                offset_x = math.sin(elapsed * 2 + i) * 3
                offset_y = math.cos(elapsed * 2 + i) * 2
                
                self.screen.blit(desc_surface, (desc_rect.x + offset_x, desc_rect.y + offset_y))
            
            # Process events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                    
                if event.type == pygame.MOUSEBUTTONDOWN:
                    if heart_button.collidepoint(event.pos):
                        # Play click animation
                        self.play_button_click_animation(heart_button)
                        config = self.configuration_menu()
                        return "heart", config
                    elif image_button.collidepoint(event.pos):
                        # Play click animation
                        self.play_button_click_animation(image_button)
                        # Load image first
                        player_image = None
                        while player_image is None:
                            player_image = self.load_image()
                            if player_image is None:
                                print("Please select an image to continue.")
                                continue
                        # Only proceed to configuration after image is loaded
                        config = self.configuration_menu()
                        return "image", config, player_image
            
            pygame.display.flip()
            pygame.time.Clock().tick(60)  # Limit to 60 FPS
    
    def create_heart_icon(self, size):
        """Create a cute heart icon for the heart button"""
        icon = pygame.Surface((size, size), pygame.SRCALPHA)
        
        # Draw heart
        heart_color = self.SECONDARY
        
        # Draw the heart using two circles and a triangle
        circle_radius = size // 4
        circle_y = size // 3
        
        # Left circle
        pygame.draw.circle(icon, heart_color, (circle_radius, circle_y), circle_radius)
        
        # Right circle
        pygame.draw.circle(icon, heart_color, (size - circle_radius, circle_y), circle_radius)
        
        # Triangle for bottom of heart
        points = [
            (0, circle_y),
            (size // 2, size),
            (size, circle_y)
        ]
        pygame.draw.polygon(icon, heart_color, points)
        
        return icon
    
    def create_image_icon(self, size):
        """Create an image icon for the image button"""
        icon = pygame.Surface((size, size), pygame.SRCALPHA)
        
        # Draw image icon (a rectangle with a sun/mountain design)
        frame_color = self.TERTIARY
        frame_padding = size // 10
        
        # Draw frame
        pygame.draw.rect(
            icon, 
            frame_color, 
            (frame_padding, frame_padding, size - 2*frame_padding, size - 2*frame_padding),
            2,
            border_radius=3
        )
        
        # Draw sun
        sun_color = self.QUATERNARY
        sun_radius = size // 8
        sun_x = size // 3
        sun_y = size // 3
        pygame.draw.circle(icon, sun_color, (sun_x, sun_y), sun_radius)
        
        # Draw mountains
        mountain_color = self.PRIMARY
        
        # Big mountain
        points1 = [
            (frame_padding * 2, size - frame_padding * 2),  # Bottom left
            (size // 2, frame_padding * 4),                # Top middle
            (size - frame_padding * 4, size - frame_padding * 2)  # Bottom right
        ]
        pygame.draw.polygon(icon, mountain_color, points1)
        
        # Small mountain
        points2 = [
            (size // 2, size - frame_padding * 2),         # Bottom middle
            (3 * size // 4, size // 2),                    # Top right
            (size - frame_padding * 2, size - frame_padding * 2)  # Bottom right
        ]
        pygame.draw.polygon(icon, self.SECONDARY, points2)
        
        return icon

    def run(self):
        """Main game loop with futuristic game-like UI"""
        # Show start menu and get initial choice and config
        choice, config = self.start_menu()
        
        # Initialize player image
        player_image = None
        pixel_coordinates = []
        
        if choice == "image":
            # Process the already loaded image
            player_image = self.load_image()
            pixel_coordinates = process_image(player_image)
            message = "Image Loaded Successfully!"
            
        elif choice == "heart":
            # Use the heart shape coordinates
            pixel_coordinates = self.create_heart_shape()
            message = "Heart Shape Selected!"
            print("Heart shape selected - no image required!")
        
        print(f"Number of pixels/coordinates: {len(pixel_coordinates)}")
        
        # Display the completion message with animation
        running = True
        self.animation_tick = 0
        start_time = pygame.time.get_ticks()
        
        while running:
            current_time = pygame.time.get_ticks()
            elapsed = (current_time - start_time) / 1000  # Convert to seconds
            self.animation_tick += self.animation_speed
            
            # Draw animated background
            self.draw_background()
            
            # Add futuristic header
            header_height = 100
            
            # Draw header with gradient
            for y in range(header_height):
                progress = y / header_height
                r = int(self.PRIMARY[0] * (1 - progress) + self.PANEL_COLOR[0] * progress)
                g = int(self.PRIMARY[1] * (1 - progress) + self.PANEL_COLOR[1] * progress)
                b = int(self.PRIMARY[2] * (1 - progress) + self.PANEL_COLOR[2] * progress)
                pygame.draw.line(self.screen, (r, g, b), (0, y), (self.WIDTH, y))
                
            # Animated accent line below header
            line_alpha = int(100 + 155 * (math.sin(elapsed * 3) + 1) / 2)
            line_surface = pygame.Surface((self.WIDTH, 3), pygame.SRCALPHA)
            pygame.draw.rect(line_surface, (*self.SECONDARY, line_alpha), (0, 0, self.WIDTH, 3))
            self.screen.blit(line_surface, (0, header_height))
            
            # Display message in a futuristic message box
            message_box_width = 600
            message_box_height = 90
            message_box_x = self.WIDTH // 2 - message_box_width // 2
            message_box_y = header_height + 30
            
            # Draw message box with rounded corners and gradient
            message_box_rect = pygame.Rect(message_box_x, message_box_y, message_box_width, message_box_height)
            pygame.draw.rect(self.screen, self.PANEL_COLOR, message_box_rect, border_radius=15)
            
            # Add glass effect to the message box
            gradient_surface = pygame.Surface((message_box_width, message_box_height // 2), pygame.SRCALPHA)
            for i in range(message_box_height // 2):
                alpha = 40 - int(i * 80 / message_box_height)
                if alpha > 0:
                    pygame.draw.line(gradient_surface, (255, 255, 255, alpha), 
                                   (0, i), (message_box_width, i))
            gradient_rect = gradient_surface.get_rect(topleft=(message_box_x, message_box_y))
            self.screen.blit(gradient_surface, gradient_rect)
            
            # Add a pulsing border to the message box
            border_alpha = int(100 + 155 * (math.sin(elapsed * 4) + 1) / 2)
            pygame.draw.rect(self.screen, (*self.TERTIARY, border_alpha), 
                           message_box_rect, width=3, border_radius=15)
            
            # Add decorative corner elements to the message box
            corner_size = 15
            line_width = 2
            
            # Top left corner
            pygame.draw.line(self.screen, self.SECONDARY, 
                           (message_box_x, message_box_y + corner_size), 
                           (message_box_x, message_box_y), 
                           line_width)
            pygame.draw.line(self.screen, self.SECONDARY, 
                           (message_box_x, message_box_y), 
                           (message_box_x + corner_size, message_box_y), 
                           line_width)
            
            # Top right corner
            pygame.draw.line(self.screen, self.SECONDARY, 
                           (message_box_x + message_box_width - corner_size, message_box_y), 
                           (message_box_x + message_box_width, message_box_y), 
                           line_width)
            pygame.draw.line(self.screen, self.SECONDARY, 
                           (message_box_x + message_box_width, message_box_y), 
                           (message_box_x + message_box_width, message_box_y + corner_size), 
                           line_width)
            
            # Bottom left corner
            pygame.draw.line(self.screen, self.SECONDARY, 
                           (message_box_x, message_box_y + message_box_height - corner_size), 
                           (message_box_x, message_box_y + message_box_height), 
                           line_width)
            pygame.draw.line(self.screen, self.SECONDARY, 
                           (message_box_x, message_box_y + message_box_height), 
                           (message_box_x + corner_size, message_box_y + message_box_height), 
                           line_width)
            
            # Bottom right corner
            pygame.draw.line(self.screen, self.SECONDARY, 
                           (message_box_x + message_box_width - corner_size, message_box_y + message_box_height), 
                           (message_box_x + message_box_width, message_box_y + message_box_height), 
                           line_width)
            pygame.draw.line(self.screen, self.SECONDARY, 
                           (message_box_x + message_box_width, message_box_y + message_box_height), 
                           (message_box_x + message_box_width, message_box_y + message_box_height - corner_size), 
                           line_width)
            
            # Display message with glow effect
            glow_color = self.PRIMARY if choice == "heart" else self.TERTIARY
            message_pulse = (math.sin(elapsed * 3) + 1) / 2
            message_color = self.WHITE
            
            # Add glow to message text
            message_glow_surface = self.title_font.render(message, True, glow_color)
            message_glow_rect = message_glow_surface.get_rect(
                center=(self.WIDTH // 2, message_box_y + message_box_height // 2)
            )
            
            # Apply blur effect to glow (simple implementation)
            for i in range(3):
                glow_size = int(4 + 2 * message_pulse)
                glow_surface = pygame.transform.scale(
                    message_glow_surface,
                    (message_glow_rect.width + glow_size, message_glow_rect.height + glow_size)
                )
                glow_rect = glow_surface.get_rect(
                    center=(self.WIDTH // 2, message_box_y + message_box_height // 2)
                )
                glow_surface.set_alpha(40)
                self.screen.blit(glow_surface, glow_rect)
            
            # Draw main message text
            message_surface = self.title_font.render(message, True, message_color)
            message_rect = message_surface.get_rect(
                center=(self.WIDTH // 2, message_box_y + message_box_height // 2)
            )
            self.screen.blit(message_surface, message_rect)
            
            # Draw preview section title
            preview_title_y = message_box_y + message_box_height + 40
            preview_title_text = "Pattern Preview"
            
            # Add subtle glow to title
            preview_title_glow = pygame.Surface((200, 40), pygame.SRCALPHA)
            preview_title_glow_alpha = int(40 + 20 * message_pulse)
            pygame.draw.ellipse(preview_title_glow, (*self.TERTIARY, preview_title_glow_alpha), 
                              (0, 0, 200, 40))
            preview_title_glow_rect = preview_title_glow.get_rect(
                center=(self.WIDTH // 2, preview_title_y)
            )
            self.screen.blit(preview_title_glow, preview_title_glow_rect)
            
            # Draw title text
            preview_title_surface = self.subtitle_font.render(preview_title_text, True, self.WHITE)
            preview_title_rect = preview_title_surface.get_rect(center=(self.WIDTH // 2, preview_title_y))
            self.screen.blit(preview_title_surface, preview_title_rect)
            
            # Visualize the shape in a futuristic container
            if pixel_coordinates:
                # Draw a preview container
                container_width = 400
                container_height = 300
                container_x = self.WIDTH // 2 - container_width // 2
                container_y = preview_title_y + 30
                
                # Draw container background with rounded corners
                container_rect = pygame.Rect(container_x, container_y, container_width, container_height)
                pygame.draw.rect(self.screen, self.PANEL_COLOR, container_rect, border_radius=15)
                
                # Add glass effect to the container
                gradient_surface = pygame.Surface((container_width, container_height // 3), pygame.SRCALPHA)
                for i in range(container_height // 3):
                    alpha = 30 - int(i * 90 / container_height)
                    if alpha > 0:
                        pygame.draw.line(gradient_surface, (255, 255, 255, alpha), 
                                       (0, i), (container_width, i))
                gradient_rect = gradient_surface.get_rect(topleft=(container_x, container_y))
                self.screen.blit(gradient_surface, gradient_rect)
                
                # Add a subtle border to the container
                pygame.draw.rect(self.screen, (255, 255, 255, 50), 
                               container_rect, width=2, border_radius=15)
                
                # Draw grid lines for visualization
                grid_color = (self.PRIMARY[0], self.PRIMARY[1], self.PRIMARY[2], 30)
                grid_size = 25
                cells_x = 16  # Number of cells to show horizontally
                cells_y = 12  # Number of cells to show vertically
                
                grid_width = cells_x * grid_size
                grid_height = cells_y * grid_size
                grid_x = self.WIDTH // 2 - grid_width // 2
                grid_y = container_y + (container_height - grid_height) // 2
                
                # Draw grid with rounded cells
                for i in range(cells_x):
                    for j in range(cells_y):
                        cell_x = grid_x + i * grid_size
                        cell_y = grid_y + j * grid_size
                        cell_rect = pygame.Rect(cell_x, cell_y, grid_size, grid_size)
                        pygame.draw.rect(self.screen, grid_color, cell_rect, 1, border_radius=3)
                
                # Draw shape pixels with glowing effect
                pulse = (math.sin(elapsed * 3) + 1) / 2
                for y, x in pixel_coordinates:
                    if 0 <= x < cells_x and 0 <= y < cells_y:
                        cell_x = grid_x + x * grid_size
                        cell_y = grid_y + y * grid_size
                        
                        # Choose color based on choice
                        pixel_color = self.SECONDARY if choice == "heart" else self.TERTIARY
                        
                        # Add animated glow
                        glow_alpha = int(100 + 155 * pulse)
                        glow_surface = pygame.Surface((grid_size + 10, grid_size + 10), pygame.SRCALPHA)
                        pygame.draw.rect(glow_surface, (*pixel_color, glow_alpha), 
                                       (0, 0, grid_size + 10, grid_size + 10), border_radius=8)
                        self.screen.blit(glow_surface, (cell_x - 5, cell_y - 5))
                        
                        # Draw the actual pixel with rounded corners
                        pygame.draw.rect(self.screen, pixel_color, 
                                       (cell_x, cell_y, grid_size, grid_size), border_radius=8)
                        
                        # Add highlight for 3D effect
                        pygame.draw.rect(self.screen, (255, 255, 255, 100), 
                                       (cell_x, cell_y, grid_size, grid_size // 3), border_radius=8)
            
            # Draw continue button with animation
            continue_pulse = (math.sin(elapsed * 4) + 1) / 2
            continue_color = self.QUATERNARY if continue_pulse > 0.7 else self.PRIMARY
            continue_text = "Continue to Simulation"
            continue_button = self.draw_button(continue_text, self.HEIGHT - 80, 280, 50, continue_color)
            
            # Process events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    pygame.quit()
                    sys.exit()
                elif event.type == pygame.KEYDOWN or event.type == pygame.MOUSEBUTTONDOWN:
                    # Check if continue button was clicked
                    if event.type == pygame.MOUSEBUTTONDOWN and continue_button.collidepoint(event.pos):
                        self.play_button_click_animation(continue_button)
                    running = False
        
        return choice, config

if __name__ == "__main__":
    interface = TrailInterface()
    interface.run() 