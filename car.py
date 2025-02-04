import math
import pygame
import pygame.freetype
from pygame.sprite import Sprite
import numpy as np
from utils.helper import is_point_on_line

class Car(Sprite):

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
  
    def _load_rotated_images(self, car_image: str) -> list:
        car_image = pygame.image.load(car_image).convert_alpha()
        rotated_images = []
        for i in range(360):
            rotated_image = pygame.transform.rotozoom(car_image, 360 - 90 - (i), 1)
            rotated_images.append(rotated_image)
        return rotated_images

    def _turn(self, angle_degrees) -> None:
        """
        Updates the heading of the object and adjusts the image and mask accordingly.

        This function updates the heading of the object based on the given angle,
        and if the player type is not "ai", it updates the image and mask to reflect
        the new heading.

        Args:
            angle_degrees (float): The angle in degrees to turn.

        Returns:
            None
        """
        self.heading += math.radians(angle_degrees)
        if self.player_type != "ai":
            image_index = int(self.heading / self.min_angle) % len(self.rot_img)
            if self.image != self.rot_img[ image_index ]:
                #TODO:
                # This seems extra
                #x,y = self.rect.center
                self.image = self.rot_img[image_index]
                self.rect  = self.image.get_rect()
                #self.rect.center = (x,y)
                self.mask = pygame.mask.from_surface(self.image)
        return None

    def _accelerate(self, amount) -> None:
        # Add more realistic way of accelerating + a normal speed cap
        if self.speed <= 5.0:
            self.speed += amount

    def _brake(self) -> None:
        self.speed = self.speed - (self.speed * 0.2)
        if (abs(self.speed) < 0.1):
            self.speed = 0.0

    def _cast_ray(self, arr, angle_offset) -> None:
        """
        Casts a ray from the current position at a given angle to detect walls.

        This function casts a ray from the current position at a specified angle
        and returns the coordinates of the first wall it encounters.

        Args:
            arr (np.ndarray): A 2D array representing the walls.
            angle_offset (float): The angle offset for the ray.

        Returns:
            tuple: The coordinates (x, y) of the first wall encountered.
        """
        x, y = 0, 0
        heading = self.heading + angle_offset
        # TODO: find more efficient way to do this
        for i in range(0, 1800):
            x = round(self.position[0] + math.cos(heading) * i)
            y = round(self.position[1] + math.sin(heading) * i)
            # If we find a wall return x, y
            if arr[x, y]:
                return x, y

    def distance_to_walls(self, walls) -> tuple:
        """
        Calculates the distance to the walls from the current position using ray casting.

        Args:
            walls (list): A list representing the walls.

        Returns:
            tuple: A tuple containing a list of normalized distances and a list of wall positions.
        """
        ray_angles = [0, 1.570, 4.712, 0.524, 5.76, 3.142, 3.67, 2.618]
        distances = []
        realdis = []
        all_position = []

        for angle in ray_angles:
            x, y = self._cast_ray(walls, angle)
            all_position.append((x, y))
            # We use distance for the sensors
            distance_to_wall = math.sqrt((x - self.position[0]) ** 2 + (y - self.position[1]) ** 2)
            realdis.append(distance_to_wall)
            distances.append((1500 - distance_to_wall) / 1500)
        return distances, all_position

    # TODO: make private
    def _wall_collision(self) -> None:
        """
        Checks for wall collisions, we compare: distance to wall, distance to hitbox.

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
        We calculate a simple hitbox around the car.

        Loads the hitbox distances from the starting position.

        This function calculates the hitbox distances based on the starting position
        and updates the hitboxes array. 

        Returns:
            list: A list of hitbox distances.
        """
        x, y = self.start_pos
        all_hitboxes = []
        hitboxes = np.zeros((1920,1080), dtype=bool)

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

    def check_checkpoint(self, checkpoint) -> bool:
        return is_point_on_line(checkpoint[0], checkpoint[1], (round(self.position[0]), round(self.position[1])), 6.0)
    
    def update(self, walls) -> None:
        """
        Updates the state of the object, including position, velocity, and collision detection.

        This function updates the object's velocity and position based on its speed and heading,
        calculates the distances to walls, updates the state, and checks for collisions.

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

        This function resets the position, velocity, speed, heading, and hitwall
        attributes to their initial values.

        Args:
            None

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
        Executes actions based on the input keys.

        This function handles the actions for both player and AI types, including
        acceleration, braking, and turning.

        Args:
            keys (Union[pygame.key.ScancodeWrapper, int]): The input keys for controlling the actions.

        Returns:
            None
        """
        if self.player_type == "palayer":
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