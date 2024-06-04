import pygame
import math

"""
TODO:
# Add normal background
# Add more realistic driving behaviour
    # Maybe use real physics library for this
# Add race element (start/finish, lapcounter)
# Add multiple players?
# Split code
# Statistics
    - raceline
    - pos + speed
    - crashes 
# Save position of the car etc (replay?)
"""

# Window size
WINDOW_WIDTH    = 1920
WINDOW_HEIGHT   = 1080
WINDOW_SURFACE  = pygame.HWSURFACE|pygame.DOUBLEBUF|pygame.RESIZABLE

class Car(pygame.sprite.Sprite):
    def __init__(self, car_image, x, y, rotations=360):
        pygame.sprite.Sprite.__init__(self)
        self.rot_img   = []
        self.min_angle = (360 / rotations) 
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

class Background(pygame.sprite.Sprite):
  def __init__(self):
    pygame.sprite.Sprite.__init__(self)
    self.image = pygame.image.load('assets/background.png').convert_alpha()
    self.mask = pygame.mask.from_surface(self.image)
    self.rect        = self.image.get_rect()

if __name__ == "__main__":
    pygame.init()
    pygame.mixer.init()
    window = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT), WINDOW_SURFACE)
    pygame.display.set_caption("Car Steering")

    ### Bitmaps
    car_image  = pygame.image.load('assets/car1.png').convert_alpha()

    pygame.mouse.set_visible(False)

    #create instances of soldier and bullet
    black_car = Car(car_image, WINDOW_WIDTH//2, WINDOW_HEIGHT//2)
    background = Background()

    #create soldier and bullet groups
    car_group = pygame.sprite.Group()
    background_group = pygame.sprite.Group()

    #add instances to groups
    car_group.add(black_car)
    background_group.add(background)

    clock = pygame.time.Clock()
    done = False

    while not done:

        if pygame.sprite.spritecollide(background, car_group, False, pygame.sprite.collide_mask):
            window.fill((200, 200, 100))
        else:
            window.fill((120, 120, 120))
        
        for event in pygame.event.get():
            if (event.type == pygame.QUIT):
                done = True

            # When is this event triggered, remove>>?
            elif (event.type == pygame.VIDEORESIZE):
                WINDOW_WIDTH  = event.w
                WINDOW_HEIGHT = event.h
                window = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT), WINDOW_SURFACE)
                #background = pygame.transform.smoothscale(road_image, (WINDOW_WIDTH, WINDOW_HEIGHT))
            elif (event.type == pygame.KEYUP):
                if (event.key == pygame.K_r):  
                    black_car.reverse()
                elif (event.key == pygame.K_UP):  
                    black_car.accelerate(0.5)
                elif (event.key == pygame.K_DOWN):  
                    black_car.brake()

        # Continuous Movement keys
        # Do this for gas aswell
        keys = pygame.key.get_pressed()
        if black_car.speed != 0:
            if (keys[pygame.K_LEFT]):
                black_car.turn(-1.0)  # degrees
            if (keys[pygame.K_RIGHT]):
                black_car.turn(1.0)

        car_group.update()
        background_group.draw(window)
        car_group.draw(window)
        pygame.display.flip()

        # set fps
        clock.tick_busy_loop(60)

    pygame.quit()