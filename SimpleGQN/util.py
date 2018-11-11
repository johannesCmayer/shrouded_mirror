import random
import winsound, multiprocessing, keyboard
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

    def draw_image(self, image, display_duration=0, size=None, position=(0,0), smoothscale=False):
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
    def __init__(self, num_to_switch_state=1):
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


class AsyncKeyChecker:
    def __init__(self, key, msg=None):
        self.key = key
        self.msg = msg
        self.stop_learning_event = multiprocessing.Event()
        self.key_checker = None

    def start_checking(self):
        self.key_checker = \
            multiprocessing.Process(target=self._check_if_key_is_pressed, args=(self.stop_learning_event, self.key))
        self.key_checker.start()

    def _check_if_key_is_pressed(self, event, key_string):
        keyboard.wait(key_string)
        if self.msg:
            print(self.msg)
        winsound.Beep(220, 300)
        event.set()
        return

    def key_was_pressed(self):
        if self.stop_learning_event.is_set():
            self.key_checker.join()
            self.stop_learning_event.clear()
            return True
        else:
            return False

    def terminate(self):
        self.key_checker.terminate()

    def __enter__(self):
        self.start_checking()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.terminate()


class CharacterController:
    def __init__(self, center_pos=(0, 1.5, 0)):
        self.center_pos = np.asarray(center_pos)
        self.current_position = np.array(self.center_pos)
        self.current_y_rotation = 0
        self.prev_time = 0
        self.move_speed = 0.8
        self.rotate_speed = 0.8
        self.mouse_rotate_speed = 0.001

    @staticmethod
    def y_rot_to_quaternion(rot):
        if rot > np.pi:
            rot -= np.pi * (int(rot / np.pi))
        if rot < 0:
            rot += np.pi * (1 + int(rot / np.pi))
        return np.asarray([0, np.sin(rot), 0, np.cos(rot)])

    @property
    def current_rotation_quaternion(self):
        return self.y_rot_to_quaternion(self.current_y_rotation)

    def movement_update(self):
        delta_time = time.time() - self.prev_time
        self.prev_time = time.time()

        current_rot_x2 = self.y_rot_to_quaternion(self.current_y_rotation * 2)
        forward_vec = np.asarray([np.sin(current_rot_x2[1]), 0., np.sin(current_rot_x2[3])])
        current_rot_x2_90_deg_offset = self.y_rot_to_quaternion(self.current_y_rotation * 2 + np.pi / 2)
        right_vec = -1 * np.asarray([np.sin(current_rot_x2_90_deg_offset[1]), 0., np.sin(current_rot_x2_90_deg_offset[3])])

        keys = pygame.key.get_pressed()

        # mouse_delta = pygame.mouse.get_rel()
        # if pygame.mouse.get_focused() and not pygame.event.get_grab():
        #     pygame.event.set_grab(True)
        #     pygame.mouse.set_visible(False)
        # if keys[pygame.K_ESCAPE]:
        #     pygame.event.set_grab(False)
        #     pygame.mouse.set_visible(True)
        #
        # self.current_y_rotation += -mouse_delta[0] * self.mouse_rotate_speed * delta_time

        if keys[pygame.K_a]:
            self.current_y_rotation += self.rotate_speed * delta_time
        if keys[pygame.K_s]:
            self.current_y_rotation -= self.rotate_speed * delta_time

        if keys[pygame.K_UP]:
            self.current_position += self.move_speed * delta_time * forward_vec
        if keys[pygame.K_DOWN]:
            self.current_position += self.move_speed * delta_time * -forward_vec
        if keys[pygame.K_LEFT]:
            self.current_position += self.move_speed * delta_time * -right_vec
        if keys[pygame.K_RIGHT]:
            self.current_position += self.move_speed * delta_time * right_vec

        if keys[pygame.K_KP2]:
            self.current_position = np.array(self.center_pos)
        if keys[pygame.K_KP3]:
            self.current_y_rotation = 0