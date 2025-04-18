import pygame
import sys
from tkinter import Tk, filedialog
from PIL import Image
import numpy as np
import math

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
        self.WIDTH, self.HEIGHT = 800, 600  # Larger window for more game-like feel
        self.WHITE = (245, 245, 245)
        self.BLACK = (20, 20, 20)
        self.BLUE = (0, 102, 204)
        self.BUTTON_COLOR = (0, 122, 204)
        self.BUTTON_HOVER = (0, 180, 255)
        self.ACCENT_COLOR = (255, 80, 80)  # Accent color for highlights
        self.BACKGROUND_COLOR = (32, 32, 64)  # Darker background for game feel
        
        # Create game window
        self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
        pygame.display.set_caption("Trail Interface")
        
        # Fonts
        pygame.font.init()
        self.title_font = pygame.font.Font(None, 72)  # Larger title font
        self.button_font = pygame.font.Font(None, 36)
        self.subtitle_font = pygame.font.Font(None, 32)  # Larger subtitle font
        
        # Background pattern
        self.bg_pattern = self.create_background_pattern()
        
        # Animation variables
        self.animation_tick = 0
        self.animation_speed = 0.05

    def create_background_pattern(self):
        """Create a subtle background pattern"""
        pattern = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
        for i in range(0, self.WIDTH, 30):
            for j in range(0, self.HEIGHT, 30):
                if (i + j) % 60 == 0:
                    pygame.draw.circle(pattern, (255, 255, 255, 5), (i, j), 15)
        return pattern

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

    def draw_text_centered(self, text, font, color, y_offset):
        """Function to display text centered horizontally"""
        text_surface = font.render(text, True, color)
        text_rect = text_surface.get_rect(center=(self.WIDTH // 2, y_offset))
        self.screen.blit(text_surface, text_rect)

    def draw_button(self, text, y, width, height, color=None):
        """Function to create button centered horizontally"""
        mouse_x, mouse_y = pygame.mouse.get_pos()
        x = self.WIDTH // 2 - width // 2  # Center horizontally
        button_rect = pygame.Rect(x, y, width, height)
        
        button_color = color if color else self.BUTTON_COLOR
        hover_color = (min(button_color[0] + 40, 255), 
                      min(button_color[1] + 40, 255), 
                      min(button_color[2] + 40, 255))
        
        # Add a glow effect when hovering
        if button_rect.collidepoint(mouse_x, mouse_y):
            # Draw glow
            for i in range(3):
                glow_rect = button_rect.inflate(i*4, i*4)
                alpha = 150 - i*40
                s = pygame.Surface((glow_rect.width, glow_rect.height), pygame.SRCALPHA)
                pygame.draw.rect(s, (*hover_color[:3], alpha), s.get_rect(), border_radius=12)
                self.screen.blit(s, glow_rect)
            pygame.draw.rect(self.screen, hover_color, button_rect, border_radius=10)
        else:
            pygame.draw.rect(self.screen, button_color, button_rect, border_radius=10)

        # Add a subtle 3D effect
        pygame.draw.rect(self.screen, (255, 255, 255, 128), 
                        (x, y, width, 3), border_radius=10)  # Top highlight
        pygame.draw.rect(self.screen, (0, 0, 0, 128), 
                        (x, y + height - 3, width, 3), border_radius=10)  # Bottom shadow
        
        # Center text on button
        text_surface = self.button_font.render(text, True, self.WHITE)
        text_rect = text_surface.get_rect(center=button_rect.center)
        self.screen.blit(text_surface, text_rect)
        
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
        """Start menu function with improved game-like UI"""
        self.animation_tick = 0
        
        while True:
            self.animation_tick += self.animation_speed
            
            # Draw background
            self.screen.fill(self.BACKGROUND_COLOR)
            self.screen.blit(self.bg_pattern, (0, 0))
            
            # Add animated decorative elements
            pulse = (math.sin(self.animation_tick) + 1) / 2  # Value between 0 and 1
            
            # Top banner with animated highlight
            banner_height = 120
            pygame.draw.rect(self.screen, (40, 40, 80), (0, 0, self.WIDTH, banner_height))
            highlight_height = 5 + int(pulse * 3)
            pygame.draw.rect(self.screen, self.ACCENT_COLOR, 
                            (0, banner_height, self.WIDTH, highlight_height))
            
            # Title with animated glow
            glow_size = 2 + int(pulse * 3)
            title_shadow = self.title_font.render("Start Game", True, (100, 100, 180))
            title_shadow_rect = title_shadow.get_rect(center=(self.WIDTH//2 + glow_size, banner_height//2 + glow_size))
            self.screen.blit(title_shadow, title_shadow_rect)
            
            title_surface = self.title_font.render("Start Game", True, self.WHITE)
            title_rect = title_surface.get_rect(center=(self.WIDTH//2, banner_height//2))
            self.screen.blit(title_surface, title_rect)
            
            # Subtitle with pulsing color
            subtitle_color = (
                int(180 + 75 * pulse),
                int(180 + 75 * pulse),
                255
            )
            self.draw_text_centered("Choose Your Mode", self.subtitle_font, 
                                  subtitle_color, banner_height + 60)
            
            # Buttons - vertically stacked and centered
            btn_width, btn_height = 300, 60
            btn_spacing = 20
            first_btn_y = banner_height + 120
            
            load_img_button = self.draw_button("Load Custom Image", 
                                             first_btn_y, btn_width, btn_height)
            
            heart_shape_button = self.draw_button("Use Diamond Shape", 
                                                first_btn_y + btn_height + btn_spacing, 
                                                btn_width, btn_height, self.ACCENT_COLOR)
            
            quit_button = self.draw_button("Quit", 
                                         first_btn_y + (btn_height + btn_spacing) * 2, 
                                         btn_width, btn_height)
            
            # Decorative elements
            for i in range(5):
                star_x = self.WIDTH * (0.1 + (i * 0.2))
                star_y = self.HEIGHT * 0.85
                star_size = 15 + int(pulse * 5)
                self.draw_star(star_x, star_y, star_size, (220, 220, 100, 150))
            
            pygame.display.flip()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                
                if event.type == pygame.MOUSEBUTTONDOWN:
                    if load_img_button.collidepoint(event.pos):
                        return "image"  # Load custom image
                    if heart_shape_button.collidepoint(event.pos):
                        return "heart"  # Use heart shape
                    if quit_button.collidepoint(event.pos):
                        pygame.quit()
                        sys.exit()
    
    def draw_star(self, x, y, size, color):
        """Draw a decorative star"""
        points = []
        for i in range(10):
            angle = math.pi * 2 * i / 10
            radius = size if i % 2 == 0 else size/2
            point_x = x + radius * math.sin(angle)
            point_y = y + radius * math.cos(angle)
            points.append((point_x, point_y))
        
        pygame.draw.polygon(self.screen, color, points)

    def run(self):
        """Main game loop with improved game-like UI"""
        # Show start menu
        choice = self.start_menu()
        
        pixel_coordinates = []
        
        if choice == "image":
            # Force image selection before game starts
            player_image = None
            while player_image is None:
                player_image = self.load_image()
                if player_image is None:
                    print("Please select an image to continue.")
            
            # Process the image
            pixel_coordinates = process_image(player_image)
            message = "Image loaded successfully!"
            
        elif choice == "heart":
            # Use the heart shape coordinates
            pixel_coordinates = self.create_heart_shape()
            message = "Heart shape selected!"
            print("Heart shape selected - no image required!")
        
        print(f"Number of pixels/coordinates: {len(pixel_coordinates)}")
        
        # Display the completion message with animation
        running = True
        self.animation_tick = 0
        
        while running:
            self.animation_tick += self.animation_speed
            pulse = (math.sin(self.animation_tick) + 1) / 2
            
            # Draw background
            self.screen.fill(self.BACKGROUND_COLOR)
            self.screen.blit(self.bg_pattern, (0, 0))
            
            # Add game-like header
            pygame.draw.rect(self.screen, (40, 40, 80), (0, 0, self.WIDTH, 80))
            pygame.draw.rect(self.screen, self.ACCENT_COLOR, (0, 80, self.WIDTH, 3))
            
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
                        pygame.draw.rect(glow_surface, (*self.ACCENT_COLOR[:3], glow_alpha), 
                                        (0, 0, grid_size + 10, grid_size + 10), border_radius=5)
                        self.screen.blit(glow_surface, (cell_x - 5, cell_y - 5))
                        
                        # Draw the actual pixel
                        pygame.draw.rect(self.screen, self.ACCENT_COLOR, 
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