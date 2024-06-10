import pygame
import math
import pygame_gui
import pygame
import pygame.freetype
from pygame.sprite import Sprite
from pygame.rect import Rect
from enum import Enum
from pygame.sprite import RenderUpdates
import numpy as np

"""
TODO:
# Draw lines to edge and calc distance
# Smaller car, better track
# Calculate time per lap (save these)
# Add player class, make it possible for multiple cars
# Draw functions maybe that draws everything
# How to clean code? (break up parts), functions, classes
# Make sure its easy to change levels
# ENUM CLASS????????????????? for gamestate or something 
# Statistics: raceline, pos+speed, crashes (write them to csv?)
# PEP8 + gpt

TODO BUGS:

# How to add deepL
"""

# Window size
WINDOW_WIDTH    = 1920
WINDOW_HEIGHT   = 1080
WINDOW_SURFACE  = pygame.HWSURFACE|pygame.DOUBLEBUF|pygame.RESIZABLE

# Use enum to capture overall info
# Amount of player and rounds
# Or do we use enum only for information that is NON static

#class GameInfo(Enum):
#    Rounds = 5

class Car(pygame.sprite.Sprite):
    def __init__(self, car_image, x, y, rotations=360):
        pygame.sprite.Sprite.__init__(self)
        self.rot_img   = []
        self.min_angle = (360 / rotations) 
        car_image = pygame.image.load(car_image).convert_alpha()
        for i in range(rotations):
            rotated_image = pygame.transform.rotozoom(car_image, 360-90-(i*self.min_angle), 1)
            self.rot_img.append(rotated_image)
        self.min_angle = math.radians(self.min_angle)
        self.image       = self.rot_img[0]
        self.rect        = self.image.get_rect()
        self.rect.topleft = (x, y)
        self.mask = pygame.mask.from_surface(self.image)
        self.rect.center = (x, y)
        self.reversing = False
        self.heading   = 0
        self.speed     = 0    
        self.velocity  = pygame.math.Vector2(0, 0)
        self.position  = pygame.math.Vector2(x, y)
        self.start_pos = (x, y)

    def turn(self, angle_degrees):
        self.heading += math.radians(angle_degrees) 
        image_index = int(self.heading / self.min_angle) % len(self.rot_img)
        if (self.image != self.rot_img[ image_index ]):
            x,y = self.rect.center
            self.image = self.rot_img[ image_index ]
            self.rect  = self.image.get_rect()
            self.rect.center = (x,y)
            # need to update mask or collision will use og image
            self.mask = pygame.mask.from_surface(self.image)

    def accelerate(self, amount):
        # Add more realistic way of accelerating + a normal speed cap
        if (not self.reversing):
            if self.speed <= 10:
                self.speed += amount
        else: 
            self.speed -= amount

    def brake(self):
        # Add more realistic way of breaking
        self.speed /= 2
        if (abs(self.speed) < 0.1):
            self.speed = 0

    def reverse(self):
        # Do we need reverse?
        self.speed     = 0
        self.reversing = not self.reversing

    def update(self):
        self.velocity.from_polar((self.speed, math.degrees(self.heading)))
        self.position += self.velocity
        self.rect.center = (round(self.position[0]), round(self.position[1]))

    def reset(self):
        self.image = self.rot_img[0]
        self.position  = pygame.math.Vector2(self.start_pos[0], self.start_pos[1])
        self.velocity  = pygame.math.Vector2(0, 0)


# create one superclass i think
class Level(pygame.sprite.Sprite):
    def __init__(self, image, x, y):
        pygame.sprite.Sprite.__init__(self)
        self.image = pygame.image.load(image).convert_alpha()
        self.mask = pygame.mask.from_surface(self.image)
        self.rect = self.image.get_rect()
        self.rect.center = (x, y) 
# add move to player class maybe?

def move(black_car, keys):
    if black_car.speed != 0:
        if (keys[pygame.K_LEFT]):
            black_car.turn(-1.0)  # degrees
        if (keys[pygame.K_RIGHT]):
            black_car.turn(1.0)

#COLLISION function
BLUE = (106, 159, 181)
WHITE = (255, 255, 255)

def play_level(screen, speed, laps):
    lap_time = UIElement(
        center_position=(1840, 60),
        font_size=20,
        bg_rgb=BLUE,
        text_rgb=WHITE,
        text="Lap time",
    )

    x = UIElement(
        center_position=(1840, 90),
        font_size=20,
        bg_rgb=BLUE,
        text_rgb=WHITE,
        text="X, Y, Z",
    )

    speed = UIElement(
        center_position=(1840, 120),
        font_size=20,
        bg_rgb=BLUE,
        text_rgb=WHITE,
        text=f"Speed: {speed}",
    )

    lap_counter = UIElement(
        center_position=(1840, 150),
        font_size=20,
        bg_rgb=BLUE,
        text_rgb=WHITE,
        text=f"Laps ({laps})",
    )

    lap_time.draw(screen)
    x.draw(screen)
    speed.draw(screen)
    lap_counter.draw(screen)

def create_surface_with_text(text, font_size, text_rgb, bg_rgb):
    """ Returns surface with text written on """
    font = pygame.freetype.SysFont("Courier", font_size, bold=True)
    surface, _ = font.render(text=text, fgcolor=text_rgb, bgcolor=bg_rgb)
    return surface.convert_alpha()

class UIElement(Sprite):
    """ An user interface element that can be added to a surface """

    def __init__(self, center_position, text, font_size, bg_rgb, text_rgb, action=None):
        """
        Args:
            center_position - tuple (x, y)
            text - string of text to write
            font_size - int
            bg_rgb (background colour) - tuple (r, g, b)
            text_rgb (text colour) - tuple (r, g, b)
            action - the gamestate change associated with this button
        """
        self.image = create_surface_with_text(text=text, font_size=font_size, text_rgb=text_rgb, bg_rgb=bg_rgb)
        self.rect = self.image.get_rect(center=center_position)
        self.action = action

    def draw(self, surface):
        surface.blit(self.image, self.rect)

RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)

beam_surface = pygame.Surface((500, 500), pygame.SRCALPHA)


# easy way to get the position of the track walls
def get_walls(track, width, height):
    all_walls = []
    for x in range(0, width):
        line = []
        for j in range(0, height):
            if track.get_at((x, j)):
                line.append(True)
            else:
                line.append(False)
        all_walls.append(line)
    wall_pos = np.array([np.array(x) for x in all_walls])
    return wall_pos

# We actually want the distance at some point
def old_cast_ray(car_pos, arr, heading):
    x, y = 0, 0
    for i in range(0, 800):
        x = round(car_pos[0] + math.cos(heading) * i)
        y = round(car_pos[1] + math.sin(heading) * i)
        if arr[x, y]:
            return x, y

if __name__ == "__main__":
    pygame.init()
    window = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT), WINDOW_SURFACE)
    pygame.display.set_caption("Car sim :)")

    START_POS = 800, 200

    ### Bitmaps
    black_car = Car("assets/car1.png", 950, 100)
    background = pygame.image.load("assets/background2.png")
    track = Level("assets/track 2.png", WINDOW_WIDTH//2, WINDOW_HEIGHT//2)
    finish = Level("assets/finish2.png", 870, 100)

    car_group = pygame.sprite.Group()
    track_group = pygame.sprite.Group()
    finish_group = pygame.sprite.Group()

    #add instances to groups
    car_group.add(black_car)
    track_group.add(track)
    finish_group.add(finish)

    clock = pygame.time.Clock()
    done = False

    laps = 0

    # Get wall position (lazy way)
    wall_pos = get_walls(track.mask, WINDOW_WIDTH, WINDOW_HEIGHT)

    while not done:

        if pygame.sprite.spritecollide(track, car_group, False, pygame.sprite.collide_mask):
            black_car.reset()

        if pygame.sprite.spritecollide(finish, car_group, False, pygame.sprite.collide_mask):
            laps += 1

        for event in pygame.event.get():
            if (event.type == pygame.QUIT):
                done = True
            elif (event.type == pygame.KEYUP):
                if (event.key == pygame.K_r):  
                    black_car.reverse()
                elif (event.key == pygame.K_UP):  
                    black_car.accelerate(0.5)
                elif (event.key == pygame.K_DOWN):  
                    black_car.brake()
        
        keys = pygame.key.get_pressed()
        move(black_car, keys)

        window.blit(background, (0, 0))
        car_group.update()
        track_group.draw(window)
        car_group.draw(window)
        finish_group.draw(window)
        play_level(window, black_car.speed, laps)
        x, y = old_cast_ray(black_car.position, wall_pos, black_car.heading)
    
        pygame.draw.line(window, (255, 179, 113), [black_car.position[0], black_car.position[1]], [x, y], 5)

        pygame.display.flip()
        # set fps
        clock.tick_busy_loop(60)

    pygame.quit()


