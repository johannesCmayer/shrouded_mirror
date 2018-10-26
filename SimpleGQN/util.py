import random

import numpy as np
import pygame
from scipy import misc
import time


class Timer:
    def __init__(self):
        self.start_time = time.time()
        self.end_time = 0

    def stop(self):
        self.end_time = self.get_current_time()
        return self.end_time

    def get_end_time(self):
        return self.end_time

    def get_current_time(self):
        return time.time() - self.start_time


class ImgDrawer:
    def __init__(self, screen_size=(500, 500)):
        pygame.init()
        self.screen_size = screen_size
        self.screen = pygame.display.set_mode(self.screen_size)

    def draw_image(self, image, display_duration=0):
        pygame.event.pump()
        surf = pygame.surfarray.make_surface(image)
        surf = surf.convert()
        surf = pygame.transform.rotate(surf, -90)
        surf = pygame.transform.smoothscale(surf, self.screen_size)
        self.screen.blit(surf, (0, 0))
        pygame.display.flip()
        pygame.time.wait(int(display_duration * 1000))


def get_non_repeating_random_elements(num_of_elements, target):
    size = len(target)
    if num_of_elements > size:
        raise Exception("Not enough elements Available.")
    used_indecies = []
    i = 0
    for _ in range(num_of_elements):
        while i in used_indecies:
            i = random.randint(0, size - 1)
        used_indecies.append(i)
        yield target[i]


def load_images(paths, number_to_load=None):
    print('loading {} images'.format(number_to_load))
    images = []
    counter = 0
    image_get_func = None
    if number_to_load != None:
        image_get_func = lambda: get_non_repeating_random_elements(number_to_load, paths)
    else:
        image_get_func = lambda: paths
    for image in image_get_func():
        image = misc.imread(image)
        image = np.delete(image, [3], axis=2)
        images.append(image)
        counter += 1;
        if (counter % (number_to_load / 100) or counter == number_to_load):
            print('\r{}% loaded'.format(int(counter / number_to_load * 100)), end='')
    print(end='\n')
    return np.array(images)
