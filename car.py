import math
import pygame
import pygame.freetype
import random
import ast
from pygame.sprite import Sprite
import numpy as np
from utils.helper import rotate_point
from utils.helper import calculate_angle
from utils.ui import play_level
from utils.helper import is_point_on_line

class Car(Sprite):
    """
        A class to represent a car.
    """

    def __init__(self, car_image:str, x: float, y: float, player_type):
        """
        Args:
            car_image (str): location of the car image
            x (float): starting x position
            y (float): starting y position
            player_type (_type_): check if we play as ai, player or viewer
        """
        super().__init__()
        self.min_angle = math.radians(1)
        self.start_pos = (x, y)
        self.heading   = 0
        self.speed     = 0
        self.velocity  = pygame.math.Vector2(0, 0)
        self.position  = pygame.math.Vector2(x, y)
        self.player_type = player_type
        self.state = []
        self.hitbox_distances = self._load_hitbox_distances()
        self.hitwall = False
        if self.player_type != "ai":
            self.rot_img = self._load_rotated_images(car_image)
            self.image   = self.rot_img[0]
            self.rect    = self.image.get_rect()
            self.mask    = pygame.mask.from_surface(self.image)
  
    def _load_rotated_images(self, car_image: str) -> list:
        car_image = pygame.image.load(car_image).convert_alpha()
        rotated_images = []
        for i in range(360):
            rotated_image = pygame.transform.rotozoom(car_image, 360 - 90 - (i), 1)
            rotated_images.append(rotated_image)
        return rotated_images

    def _turn(self, angle_degrees) -> None:
        #TODO: check if rounding matters
        self.heading = self.heading + math.radians(angle_degrees)
        if self.player_type != "ai":
            image_index = int(self.heading / self.min_angle) % len(self.rot_img)
            if (self.image != self.rot_img[ image_index ]):
                #TODO:
                # This seems extra
                x,y = self.rect.center
                self.image = self.rot_img[ image_index ]
                self.rect  = self.image.get_rect()
                self.rect.center = (x,y)
                # need to update mask or collision will use og image
                self.mask = pygame.mask.from_surface(self.image)

    def _accelerate(self, amount) -> None:
        # Add more realistic way of accelerating + a normal speed cap
        if self.speed <= 3:
            self.speed += amount

    def _brake(self) -> None:
        self.speed = self.speed - (self.speed * 0.2)
        if (abs(self.speed) < 0.1):
            self.speed = 0

    def _cast_ray(self, arr, angle_offset) -> None:
        #TODO: SEE COMMENTS IN DISTANCE TO WALLS FUNCTIONS
        x, y = 0, 0
        heading = self.heading + angle_offset
        # TODO: find more efficient way to do this
        for i in range(0, 1600):
            x = round(self.position[0] + math.cos(heading) * i)
            y = round(self.position[1] + math.sin(heading) * i)
            # If we find a wall return x, y
            if arr[x, y]:
                return x, y

    # Efficiency
    def distance_to_walls(self, walls) -> list:
        ray_angles = [0, 1.570, 4.712, 0.524, 5.76, 3.142, 3.67, 2.618]
        distances = []
        realdis = []
        all_position = []
        #TODO: CAN USE THE SAME TACTIC HOW I USE detect wall detection
        # Check where there is overlap and stop there
        # Draw line
        # Compare overlap
        # Calculate distance between both points
        for i, angle in enumerate(ray_angles):
            x, y = self._cast_ray(walls, angle)
            all_position.append((x, y))
            # We use distance for the sensors
            distance_to_wall = math.sqrt((x - self.position[0]) ** 2 + (y - self.position[1]) ** 2)
            realdis.append(distance_to_wall)
            distances.append((1500 - distance_to_wall) / 1500)
        return distances, all_position

    # TODO: make private
    def wall_collision(self) -> bool:
        for i, dis in enumerate(self.hitbox_distances):
            if dis <= self.state[0][i]:
                self.hitwall = True
        return
    
    def _load_hitbox_distances(self) -> list:
        x, y = self.start_pos
        all_hitboxes = []
        hitboxes = np.zeros((1920,1080), dtype=bool)

        #TODO; take width and height of image
        start_x = x - 32
        start_y = y - 15

        for w in range(0, 64):
            all_hitboxes.append((start_x + w, start_y))
            all_hitboxes.append((start_x + w, start_y + 30))
            hitboxes[start_x + w, start_y] = True
            hitboxes[start_x + w, start_y + 30] = True

        for h in range(0, 30):
            all_hitboxes.append((start_x, start_y + h))
            all_hitboxes.append((start_x + 64, start_y + h))
            hitboxes[start_x, start_y + h] = True
            hitboxes[start_x + 64, start_y + h] = True

        all_distances, __ = self.distance_to_walls(hitboxes)
        all_distances_new = []
        for i in all_distances:
            all_distances_new.append(abs((i*1500) - 1500))
        #print(all_distances)
        #print(type(all_distances))
        return all_distances

    def check_checkpoint(self, checkpoints) -> bool:
        return is_point_on_line(checkpoints[0][0], checkpoints[0][1], (round(self.position[0]), round(self.position[1])))

    def update(self, walls) -> None:
        self.velocity.from_polar((self.speed, math.degrees(self.heading)))
        self.position += self.velocity
        # TODO: FIX THIS FOR HUMAN and AI
        self.state = []
        distances, __ = self.distance_to_walls(walls)
        distances.append(self.speed/3)
        self.state.append(distances)
        if self.player_type != "ai":
            self.rect.center = (round(self.position[0]), round(self.position[1]))
        self.wall_collision()


    def reset(self) -> None:
        #self.image = self.rot_img[0]
        self.position  = pygame.math.Vector2(self.start_pos[0], self.start_pos[1])
        self.velocity  = pygame.math.Vector2(0, 0)
        self.speed = 0
        self.heading = 0
        self.hitwall = False

    def action(self, keys) -> None:
        if self.player_type == "plaayer":
            if keys[pygame.K_UP]:
                self._accelerate(1.0)
            if keys[pygame.K_DOWN]:
                self._brake()
            if self.speed != 0:
                if keys[pygame.K_RIGHT]:
                    self._turn(6)
                if keys[pygame.K_LEFT]:
                    self._turn(-6)
        else:
            #if keys == 0:
              #  pass
            if keys == 0:
                self._accelerate(1.0)
            elif keys == 2:
                self._turn(6)
            elif keys == 3:
                self._brake()
            elif keys == 1:
                self._turn(-6)