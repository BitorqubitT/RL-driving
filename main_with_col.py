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
from utils.ui import play_level

"""
TODO:
# Clean up ui.py and main
# Reward function - calc distance on track and timer.
# Be able to spawn multiple cars
# Smaller car, better track
# Draw functions maybe that draws everything
# Make sure its easy to change levels
# Clean code + PEP8
# Statistics: save drive, raceline, pos+speed, crashes (write them to csv?)
# USE NEAT (need different setup)
# USE DQN 
# Faster time
"""

# Window size
WINDOW_WIDTH    = 1920
WINDOW_HEIGHT   = 1080
WINDOW_SURFACE  = pygame.HWSURFACE|pygame.DOUBLEBUF|pygame.RESIZABLE

BLUE = (106, 159, 181)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)

class Car(Sprite):
    def __init__(self, car_image, x, y, rotations=360):
        Sprite.__init__(self)
        self.rot_img   = []
        self.min_angle = (360 / rotations) 
        car_image = pygame.image.load(car_image).convert_alpha()
        for i in range(rotations):
            rotated_image = pygame.transform.rotozoom(car_image, 360-90-(i*self.min_angle), 1)
            self.rot_img.append(rotated_image)
        self.min_angle = math.radians(self.min_angle)
        self.image       = self.rot_img[0]
        self.rect        = self.image.get_rect()
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
        if self.speed <= 10:
                self.speed += amount
        else: 
            self.speed -= amount

    def brake(self):
        # Add more realistic way of breaking
        self.speed /= 2
        if (abs(self.speed) < 0.1):
            self.speed = 0

    def update(self):
        self.velocity.from_polar((self.speed, math.degrees(self.heading)))
        self.position += self.velocity
        self.rect.center = (round(self.position[0]), round(self.position[1]))

    def reset(self):
        self.image = self.rot_img[0]
        self.position  = pygame.math.Vector2(self.start_pos[0], self.start_pos[1])
        self.velocity  = pygame.math.Vector2(0, 0)


# create one superclass i think
class Level(Sprite):
    def __init__(self, image, x, y):
        Sprite.__init__(self)
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
def cast_ray(car_pos, arr, heading, angle_offset):
    x, y = 0, 0
    heading = heading + angle_offset
    for i in range(0, 800):
        x = round(car_pos[0] + math.cos(heading) * i)
        y = round(car_pos[1] + math.sin(heading) * i)
        if arr[x, y]:
            return x, y

if __name__ == "__main__":
    pygame.init()
    window = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT), WINDOW_SURFACE)
    pygame.display.set_caption("Car sim :)")

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
                if (event.key == pygame.K_UP):  
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

        # Cast diff rays
        ray_angles = [0, 70, -70]
        distances = []
        for i, angle in enumerate(ray_angles):
            x, y = cast_ray(black_car.position, wall_pos, black_car.heading, angle)
            distance_to_wall = math.sqrt((x - black_car.position[0]) ** 2 + (y - black_car.position[1]) ** 2)
            distances.append(round(distance_to_wall))
            pygame.draw.line(window, (255, 113, 113), [black_car.position[0], black_car.position[1]], [x, y], 5)

        play_level(window, distances[0], distances[1], distances[2], black_car.speed, laps)

        pygame.display.flip()
        # set fps
        clock.tick_busy_loop(60)

    pygame.quit()


