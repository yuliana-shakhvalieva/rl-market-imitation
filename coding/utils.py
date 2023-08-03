import numpy as np
import pygame
from actions import MakeRequest, Hold, StopRequest

WHITE = (255, 255, 255)
GREEN = (200, 255, 200)
RED = (232, 135, 135)
BLACK = (0, 0, 0)


class Drawer:
    def __init__(self, window_size, slice_sell, slice_buy):
        self.window_size = window_size
        self.window = pygame.display.set_mode(self.window_size)
        self.font = pygame.font.SysFont('arial', 30)
        self.columns_name = ['Продажа', 'Цена', 'Покупка']
        self.slice_sell = slice_sell
        self.slice_buy = slice_buy

        self.len_sell = len(self.slice_sell) if self.slice_sell != [] else 1
        self.len_buy = len(self.slice_buy) if self.slice_buy != [] else 1

        self.num_rows = self.len_sell + self.len_buy + 1
        self.num_cols = 3

        height = self.window_size[1] // self.num_rows
        width = self.window_size[0] // self.num_cols

        self.recursive_draw(0, 0, width, height, 0, 0)

    def recursive_draw(self, x, y, width, height, i, j):
        surf, text = self.__get_surf_and_text(width, height, i, j)

        rect = pygame.Rect([x, y, width, height])
        self.window.blit(surf, rect)
        pygame.draw.rect(self.window, BLACK, [x, y, width, height], 1)

        if text is not None:
            number_rect = text.get_rect(center=(x + width // 2, y + height // 2))
            self.window.blit(text, number_rect)

        if i == self.num_rows - 1 and j == self.num_cols - 1:
            return
        elif x < self.window_size[0] - width:
            x += width
            j += 1
            self.recursive_draw(x, y, width, height, i, j)
        else:
            x = 0
            y += height
            j = 0
            i += 1
            self.recursive_draw(x, y, width, height, i, j)

    def __get_surf_and_text(self, width, height, i, j):
        surf = pygame.Surface((width, height))
        if i == 0:  # header
            surf.fill(WHITE)
            text = self.font.render(self.columns_name[j], True, BLACK)
        elif self.slice_buy == [] and i == 1:  # slice buy is empty
            if j == 0:
                surf.fill(WHITE)
                text = None
            else:
                surf.fill(GREEN)
                text = self.font.render('-', True, BLACK)
        elif i <= self.len_buy:  # slice buy
            if j == 0:
                surf.fill(WHITE)
                text = None
            else:
                text = self.font.render(str(self.slice_buy[i - 1][j - 1]), True, BLACK)
                surf.fill(GREEN)
        elif self.slice_sell == [] and i == (1 + self.len_buy):  # slice sell is empty
            if j == 2:
                surf.fill(WHITE)
                text = None
            else:
                surf.fill(RED)
                text = self.font.render('-', True, BLACK)
        else:  # slice sell
            if j == 2:
                surf.fill(WHITE)
                text = None
            else:
                text = self.font.render(str(self.slice_sell[i - 1 - self.len_buy][j]), True, BLACK)
                surf.fill(RED)

        return surf, text


def convert_action(agent_id, pre_action):
    action = None
    action_id = np.argmax(np.abs(pre_action))
    if action_id == 0:
        price = np.abs(pre_action[1]) * 100
        volume = int(pre_action[2] * 1000)
        action = MakeRequest(agent_id, price, volume)
    elif action_id == 1:
        action = Hold(agent_id)
    elif action_id == 2:
        action = StopRequest(agent_id)
    return action


def get_options(dones):
    options = dict()
    options['agents_to_reset'] = []
    for agent_id, done in enumerate(dones):
        if done:
            options['agents_to_reset'].append(agent_id)
    return options


def minus_slice_sell(slice_sell):
    for i in range(len(slice_sell)):
        slice_sell[i][1] = - slice_sell[i][1]
    return slice_sell


def transform_slices(slice_sell, slice_buy, glass_length):
    if not slice_sell:
        slice_sell = np.zeros((glass_length, 2))
    elif len(slice_sell) < glass_length:
        addition = np.zeros((glass_length - len(slice_sell), 2))
        slice_sell = np.vstack((minus_slice_sell(slice_sell), addition))
    elif len(slice_sell) > glass_length:
        slice_sell = np.array(minus_slice_sell(slice_sell[:glass_length]))
        
    if not slice_buy:
        slice_buy = np.zeros((glass_length, 2))
    elif len(slice_buy) < glass_length:
        addition = np.zeros((glass_length - len(slice_buy), 2))
        slice_buy = np.vstack((addition, slice_buy))
    elif len(slice_buy) > glass_length:
        slice_buy = np.array(slice_buy[-glass_length:])

    return slice_sell, slice_buy


def round_price(glass_slice):
    for i in range(len(glass_slice)):
        glass_slice[i][0] = np.round(glass_slice[i][0], 2)

    return glass_slice
