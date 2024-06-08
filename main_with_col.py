import pygame
import math
import pygame_gui
import pygame
import pygame.freetype
from pygame.sprite import Sprite
from pygame.rect import Rect
from enum import Enum
from pygame.sprite import RenderUpdates

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

# Rewrite this whole function
def draw_beam(surface, angle, pos, track):
    c = math.cos(angle)
    s = math.sin(angle)

    #x = round(car_pos[0] + math.cos(heading) * 200)
    flip_x = c < 0
    flip_y = s < 0
    filpped_mask = track.mask
    
    # compute beam final point
    x_dest = 250 + 500 * abs(c)
    y_dest = 250 + 500 * abs(s)

    beam_surface.fill((0, 0, 0, 0))

    # draw a single beam to the beam surface based on computed final point
    # maybe dont need to draw, only create the mask.
    # of we draw on fake surface.
    pygame.draw.line(beam_surface, BLUE, (250, 250), (x_dest, y_dest))
    beam_mask = pygame.mask.from_surface(beam_surface)

    # find overlap between "global mask" and current beam mask
    offset_x = 250 - pos[0] if flip_x else pos[0] - 250
    offset_y = 250 - pos[1] if flip_y else pos[1] - 250
    hit = filpped_mask.overlap(beam_mask, (offset_x, offset_y))
    if hit is not None and (hit[0] != pos[0] or hit[1] != pos[1]):
        hx = 499 - hit[0] if flip_x else hit[0]
        hy = 499 - hit[1] if flip_y else hit[1]
        hit_pos = (hx, hy)

        pygame.draw.line(surface, BLUE, pos, hit_pos)
        pygame.draw.circle(surface, GREEN, hit_pos, 3)
        #pg.draw.circle(surface, (255, 255, 0), mouse_pos, 3)



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

    #finish = pygame.image.load("assets/finish.png")
    #track = pygame.image.load("assets/track 2.png")

    laps = 0

    while not done:

        if pygame.sprite.spritecollide(track, car_group, False, pygame.sprite.collide_mask):
            #print("awwwww")
            black_car.reset()

        # should add a check for collision with finish
        if pygame.sprite.spritecollide(finish, car_group, False, pygame.sprite.collide_mask):
            #print("we did a lap")
            laps += 1

        # prob need car position to calculate the ray
        #print(black_car.position)
        #print(track.mask)

        # can always get all position first
        # then check against a normal array
        # maybe a matrix with true or false

        """
        This is way to slow
        """
        # We actually want the distance at some point
        def old_cast_ray(car_pos, mask, heading):
            succes = False
            x, y = 0, 0
            for i in range(0, 1500):
                if succes is True:
                    return x, y
                else:
                    try:
                        x = round(car_pos[0] + math.cos(heading) * i)
                        y = round(car_pos[1] + math.sin(heading) * i)
                        print(x, y)
                        #if mask.get_at((car_pos[0] + i, car_pos[1])):
                        if mask.get_at((x, y)):
                            x = x
                            y = y
                            succes = True
                    except IndexError:
                        continue

                        
        # Want to cast a ray and find intersection somehow
        # Can create a mask if we want
        # Got to get rid of this loop
        def cast_ray(car_pos, mask, heading):
            x = round(car_pos[0] + math.cos(heading) * 200)
            y = round(car_pos[1] + math.sin(heading) * 200)
            return x, y


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
        draw_beam(window, black_car.heading, black_car.position, track)
        #xx, yy  = cast_ray(black_car.position, track.mask, black_car.heading)
        #pygame.draw.line(window, (255, 179, 113), [black_car.position[0], black_car.position[1]], [xx, yy], 5)

        pygame.display.flip()
        # set fps
        clock.tick_busy_loop(60)

    pygame.quit()


