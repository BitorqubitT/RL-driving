import math
import pygame
import pygame.freetype
from pygame.sprite import Sprite
import numpy as np
from utils.helper import is_point_on_line
from typing import List, Tuple, Union

class Car(Sprite):
    """
    Represents a car in the game.

    Attributes:
        start_pos (tuple): Starting position (x, y).
        heading (float): Car's heading angle in radians.
        speed (float): Car's speed.
        velocity (Vector2): Car's velocity vector.
        position (Vector2): Car's position vector.
        player_type (str): Type of player ("ai" or "human").
        state (list): Current state of the car.
        hitbox_distances (list): Distances for the car's hitbox.
        hitwall (bool): Indicates if the car has hit a wall.

    Methods:
        _load_rotated_images(car_image: str) -> list:
            Loads and returns rotated images for the car.

        _turn(angle_degrees: float) -> None:
            Turns the car by a specified angle.

        _accelerate(amount: float) -> None:
            Accelerates the car by a specified amount.

        _brake() -> None:
            Applies the brake to the car.
    """

    MAX_SPEED = 5.0
    ACCELERATION_AMOUNT = 1.0
    BRAKE_DECAY = 0.2
    ROTATION_ANGLE = 6
    RAY_LENGTH = 1800
    SCREEN_WIDTH, SCREEN_HEIGHT = 1920, 1080 

    def __init__(self, car_image:str, x: float, y: float, angle: float, player_type: str):
        super().__init__()
        self.min_angle = math.radians(1)
        self.start_pos = (x, y)
        self.heading   = angle
        self.speed     = 0.0
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
  
    def _load_rotated_images(self, car_image: str) -> List[[pygame.Surface]]:
        #car_image = pygame.image.load(car_image).convert_alpha()
        #rotated_images = []
        #for i in range(360):
        #    rotated_image = pygame.transform.rotozoom(car_image, 360 - 90 - (i), 1)
        #    rotated_images.append(rotated_image)

        car_image = pygame.image.load(car_image).convert_alpha()
        rotated_images = [pygame.transform.rotozoom(car_image, 360 - 90 - i, 1) for i in range(360)]
        return rotated_images

    def _turn(self, angle_degrees: float) -> None:
        """
        Updates the angle of the car and adjusts the image and mask accordingly.

        Args:
            angle_degrees (float): The angle in degrees to turn.

        Returns:
            None
        """
        self.heading += math.radians(angle_degrees)
        if self.player_type != "ai":
            image_index = int(self.heading / self.min_angle) % len(self.rot_img)
            if self.image != self.rot_img[ image_index ]:
                self.image = self.rot_img[image_index]
                self.rect  = self.image.get_rect()
                self.mask = pygame.mask.from_surface(self.image)
        return None

    def _accelerate(self, amount: float) -> None:
        if self.speed <= self.MAX_SPEED:
            self.speed += amount

    def _brake(self) -> None:
        self.speed -= self.speed * self.BRAKE_DECAY
        if (abs(self.speed) < 0.1):
            self.speed = 0.0

    def _cast_ray(self, arr: np.ndarray, angle_offset: float) -> Tuple[int, int]:
        """
        Casts a ray from the current position at a given angle to detect walls.

        Args:
            arr (np.ndarray): A 2D array representing the walls.
            angle_offset (float): The angle offset for the ray.

        Returns:
            tuple: The coordinates (x, y) of the first wall encountered.
        """
        x, y = 0, 0
        heading = self.heading + angle_offset
        for i in range(0, 1800):
            x = round(self.position[0] + math.cos(heading) * i)
            y = round(self.position[1] + math.sin(heading) * i)
            if 0 <= x < self.SCREEN_WIDTH and 0 <= y < self.SCREEN_HEIGHT:
                if arr[x, y]:
                    return x, y

    def distance_to_walls(self, walls: np.ndarray) -> Tuple[List[float], List[Tuple[int, int]]]:
        """
        Calculates the distance to the walls from the current position.

        Args:
            walls (list): A list representing the walls.

        Returns:
            tuple: A tuple containing a list of normalized distances and a list of wall positions.
        """
        ray_angles = [0, 1.570, 4.712, 0.524, 5.76, 3.142, 3.67, 2.618]
        distances = []
        all_position = []

        for angle in ray_angles:
            x, y = self._cast_ray(walls, angle)
            all_position.append((x, y))
            distance_to_wall = math.sqrt((x - self.position[0]) ** 2 + (y - self.position[1]) ** 2)
            # Normalise in a more robust way
            distances.append((1500 - distance_to_wall) / 1500)
        return distances, all_position

    def _wall_collision(self) -> None:
        """
        Checks for wall collisions by comparing: distance to wall, distance to hitbox.

        This function iterates through the hitbox distances and updates the
        `hitwall` attribute if a collision is detected.
        """
        for i, dis in enumerate(self.hitbox_distances):
            if dis <= self.state[0][i]:
                self.hitwall = True
                break
        return None
    
    def _load_hitbox_distances(self) -> list:
        """
        We calculate a hitbox around the car.

        Loads the hitbox distances from the starting position.

        Returns:
            list: A list of hitbox distances.
        """
        x, y = self.start_pos
        all_hitboxes = []
        hitboxes = np.zeros((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), dtype=bool)

        #TODO; take width and height of image, automate this
        start_x = x - 16
        start_y = y - 7

        for w in range(0, 32):
            all_hitboxes.append((start_x + w, start_y))
            all_hitboxes.append((start_x + w, start_y + 15))
            hitboxes[start_x + w, start_y] = True
            hitboxes[start_x + w, start_y + 15] = True
        for h in range(0, 15):
            all_hitboxes.append((start_x, start_y + h))
            all_hitboxes.append((start_x + 32, start_y + h))
            hitboxes[start_x, start_y + h] = True
            hitboxes[start_x + 32, start_y + h] = True

        all_distances, _ = self.distance_to_walls(hitboxes)
        return all_distances

    def check_checkpoint(self, checkpoint: Tuple[float, float]) -> bool:
        return is_point_on_line(checkpoint[0], checkpoint[1], (round(self.position[0]), round(self.position[1])), 6.0)
    
    def update(self, walls: np.ndarray) -> None:
        """
        Updates the state of the object, including position, velocity, and collision detection.

        Args:
            walls (list): A list representing the walls.

        Returns:
            None
        """
        self.velocity.from_polar((self.speed, math.degrees(self.heading)))
        self.position += self.velocity
        self.state = []
        distances, __ = self.distance_to_walls(walls)
        distances.append(self.speed/3)
        self.state.append(distances)
        if self.player_type != "ai":
            self.rect.center = (round(self.position[0]), round(self.position[1]))
        self._wall_collision()
        return None

    def reset(self) -> None:
        """
        Resets the object's state to its initial values.

        Returns:
            None
        """
        self.position  = pygame.math.Vector2(self.start_pos[0], self.start_pos[1])
        self.velocity  = pygame.math.Vector2(0, 0)
        self.speed = 0.0
        self.heading = 0
        self.hitwall = False
        return None

    def action(self, keys) -> None:
        """
        This function handles the actions for both player and AI types, including
        acceleration, braking, and turning.

        Args:
            keys (Union[pygame.key.ScancodeWrapper, int]): The input keys for controlling the actions.

        Returns:
            None
        """
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
            if keys == 0:
                self._accelerate(1.0)
            elif keys == 2:
                self._turn(6)
            elif keys == 3:
                self._brake()
            elif keys == 1:
                self._turn(-6)