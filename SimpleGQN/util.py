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
    def __init__(self, screen_size=(500, 500), auto_pos_offset=(5, 10)):
        pygame.init()
        self.screen_size = screen_size
        self.screen = pygame.display.set_mode(self.screen_size)
        self.auto_pos_offset = auto_pos_offset
        self.idx_of_auto_text = 0
        pygame.display.set_mode(screen_size, pygame.RESIZABLE)

    def draw_image(self, image, display_duration=0, size=None, position=(0,0), smoothscale=True):
        surf = pygame.surfarray.make_surface(image)
        surf = surf.convert()
        surf = pygame.transform.rotate(surf, -90)
        if smoothscale:
            surf = pygame.transform.smoothscale(surf, size if size else self.screen_size)
        else:
            surf = pygame.transform.scale(surf, size if size else self.screen_size)
        self.screen.blit(surf, position)
        pygame.time.wait(int(display_duration * 1000))

    def draw_text(self, text, position=(0, 0), color=(255,0,0), size=14):
        myfont = pygame.font.SysFont('Comic Sans MS', size)
        textsurface = myfont.render(text, False, color)
        self.screen.blit(textsurface, position)

    def draw_text_auto_posself(self, text, color=(255,0,0), size=14):
        position = (self.auto_pos_offset[0], self.auto_pos_offset[1] * (self.idx_of_auto_text + 1))
        self.draw_text(text, position, color, size)
        self.idx_of_auto_text = 0

    def execute(self):
        pygame.event.pump()
        pygame.display.flip()
        self.idx_of_auto_text = 0


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


class Spinner:
    spin_chars = ['|','/','-','\\']
    def __init__(self, num_to_switch_state=2):
        self.state = 0
        self.num_to_switch_state = num_to_switch_state

    def get_current_spin_char(self):
        if self.state >= len(Spinner.spin_chars) * self.num_to_switch_state:
            self.state = 0
        return Spinner.spin_chars[self.state // self.num_to_switch_state]

    def get_spin_char(self):
        self.state += 1
        if self.state >= len(Spinner.spin_chars) * self.num_to_switch_state:
            self.state = 0
        return Spinner.spin_chars[self.state // self.num_to_switch_state]

    def print_spinner(self):
        print(get_spin_char())