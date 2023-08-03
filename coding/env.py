import random
import pygame as pygame
from actions import MakeRequest, Hold, StopRequest
from itertools import compress
import gymnasium as gym
import numpy as np

from discriminator import Discriminator
from utils import Drawer, convert_action, transform_slices
from agent import Agent

SELL_TYPE = 'sell'
BUY_TYPE = 'buy'

HOLD_REWARD = -3
STOP_NON_EXISTING_REWARD = -5
SECOND_REQUEST_REWARD = -5
STOP_EXISTING_REWARD = -1
REQUEST_REWARD = 1
NOT_ENOUGH_RESOURCE_REWARD = -5
ZERO_volume_REWARD = -5
CLOSED_REQUEST_REWARD = 5


class Glass:
    def __init__(self):
        self.agents = dict()
        self.requests_by_id = dict()
        self.num_iterations_by_id = dict()
        self.buy_requests = []
        self.sell_requests = []
        self.rewards_after_execute = None

    def register(self, agents):
        for agent in agents:
            self.agents[agent.agent_id] = agent
            self.requests_by_id[agent.agent_id] = None
            self.num_iterations_by_id[agent.agent_id] = None

        self.rewards_after_execute = [0 for _ in range(len(agents))]

    def make_action(self, request):
        if isinstance(request, MakeRequest):
            if self.requests_by_id[request.agent_id] is not None:
                self.num_iterations_by_id[request.agent_id] += 1
                self.agents[request.agent_id].num_iterations += 1
                return SECOND_REQUEST_REWARD

            elif request.volume == 0:
                return ZERO_volume_REWARD

            elif self.__resource_enough(request):
                self.__add_to_glass(request)
                return REQUEST_REWARD
            else:
                return NOT_ENOUGH_RESOURCE_REWARD

        elif isinstance(request, Hold):
            if self.requests_by_id[request.agent_id] is not None:
                self.num_iterations_by_id[request.agent_id] += 1
                self.agents[request.agent_id].num_iterations += 1
            return HOLD_REWARD

        elif isinstance(request, StopRequest):
            return self.__stop(request)

    def __resource_enough(self, request):
        if request.type == SELL_TYPE:
            if self.agents[request.agent_id].actives - request.volume < 0:
                return False
        elif request.type == BUY_TYPE:
            if self.agents[request.agent_id].money - request.price * request.volume < 0:
                return False
        return True

    def __add_to_glass(self, request):
        if request.type == SELL_TYPE:
            self.sell_requests.append(request)
        elif request.type == BUY_TYPE:
            self.buy_requests.append(request)

        self.requests_by_id[request.agent_id] = request
        self.num_iterations_by_id[request.agent_id] = 0
        self.agents[request.agent_id].num_iterations = 0
        self.agents[request.agent_id].open = True
        self.agents[request.agent_id].request = request

    def __stop(self, request):
        previous_request = self.requests_by_id[request.agent_id]
        if previous_request is not None:
            request.open = False
            self.agents[request.agent_id].open = False
            self.__clean_info(previous_request)
            return STOP_EXISTING_REWARD
        else:
            return STOP_NON_EXISTING_REWARD

    def execute(self):
        self.__shuffle_glass()
        list_requests = list(self.requests_by_id.values())
        random.shuffle(list_requests)

        for request in list_requests:
            if request is not None and request.open:
                if request.type == SELL_TYPE:
                    self.__sell(request)
                elif request.type == BUY_TYPE:
                    self.__buy(request)

    def __sell(self, sell_request):
        for buy_request in self.buy_requests:
            if buy_request.price >= sell_request.price:
                self.__trade(sell_request, buy_request)
                if not sell_request.open:
                    break

    def __buy(self, buy_request):
        for sell_request in self.sell_requests:
            if sell_request.price <= buy_request.price:
                self.__trade(buy_request, sell_request)
                if not buy_request.open:
                    break

    def __shuffle_glass(self):
        random.shuffle(self.sell_requests)
        random.shuffle(self.buy_requests)

    def __trade(self, main_request, glass_request):
        if main_request.volume == glass_request.volume:
            self.__give_close_reward(main_request.agent_id)
            self.__give_close_reward(glass_request.agent_id)
            self.__close_request(request_to_close=main_request, price=glass_request.price, mother_request=glass_request)
            self.__close_request(request_to_close=glass_request, price=glass_request.price, mother_request=main_request)
        elif main_request.volume < glass_request.volume:
            self.__give_close_reward(main_request.agent_id)
            self.__close_request(request_to_close=main_request, price=glass_request.price, mother_request=glass_request)
            glass_request.volume -= main_request.volume
        elif main_request.volume > glass_request.volume:
            self.__give_close_reward(glass_request.agent_id)
            self.__close_request(request_to_close=glass_request, price=glass_request.price, mother_request=main_request)
            main_request.volume -= glass_request.volume

    def __close_request(self, request_to_close, price, mother_request):
        if request_to_close.type == SELL_TYPE:
            self.agents[mother_request.agent_id].money -= request_to_close.volume * price
            self.agents[mother_request.agent_id].actives += request_to_close.volume
            self.__close_sell_request(request_to_close, price)
        elif request_to_close.type == BUY_TYPE:
            self.agents[mother_request.agent_id].money += request_to_close.volume * price
            self.agents[mother_request.agent_id].actives -= request_to_close.volume
            self.__close_buy_request(request_to_close, price)

    def __close_buy_request(self, buy_request, price):
        buy_request.open = False
        self.agents[buy_request.agent_id].open = False
        self.agents[buy_request.agent_id].money -= buy_request.volume * price
        self.agents[buy_request.agent_id].actives += buy_request.volume
        self.__clean_info(buy_request)

    def __close_sell_request(self, sell_request, price):
        sell_request.open = False
        self.agents[sell_request.agent_id].open = False
        self.agents[sell_request.agent_id].money += sell_request.volume * price
        self.agents[sell_request.agent_id].actives -= sell_request.volume
        self.__clean_info(sell_request)

    def __clean_info(self, request):
        self.requests_by_id[request.agent_id] = None
        self.num_iterations_by_id[request.agent_id] = None
        self.agents[request.agent_id].num_iterations = None

        if request.type == SELL_TYPE:
            open = [True for _ in range(len(self.sell_requests))]
            for i in range(len(self.sell_requests)):
                if self.sell_requests[i].agent_id == request.agent_id:
                    open[i] = False
                    break
            self.sell_requests = list(compress(self.sell_requests, open))

        elif request.type == BUY_TYPE:
            open = [True for _ in range(len(self.buy_requests))]
            for i in range(len(self.buy_requests)):
                if self.buy_requests[i].agent_id == request.agent_id:
                    open[i] = False
                    break
            self.buy_requests = list(compress(self.buy_requests, open))

    def __give_close_reward(self, agent_id):
        self.rewards_after_execute[agent_id] = CLOSED_REQUEST_REWARD / (self.num_iterations_by_id[agent_id] + 1)

    def get_two_slices(self):
        slice_buy = []
        slice_sell = []
        buy_requests = self.buy_requests
        sell_requests = self.sell_requests
        for buy_request in buy_requests:
            slice_buy.append([buy_request.price, buy_request.volume])

        for sell_request in sell_requests:
            slice_sell.append([sell_request.price, sell_request.volume])

        slice_sell.sort(key=lambda x: x[0])
        slice_buy.sort(key=lambda x: x[0], reverse=True)

        return slice_sell, slice_buy


class MarketSimulation(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 1}

    def __init__(self, agents, glass_length, window_size=(750, 500), render_mode=None):
        super(gym.Env, self).__init__()
        self.num_agents = len(agents)
        self.glass_length = glass_length
        assert self.num_agents > 1
        assert self.glass_length > 0
        self.glass = Glass()
        self.agents = agents
        self.glass.register(self.agents)
        self.render_mode = render_mode
        self.window_size = window_size
        self.window = None
        self.clock = None
        self.discriminator = Discriminator(self.glass_length)

    def step(self, pre_actions):
        rewards = []
        for pre_action in pre_actions:
            action = convert_action(*pre_action)
            rewards.append(self.glass.make_action(action))

        if self.render_mode is not None:
            self.render()
            
        glass_slice = self.__get_glass_slice()

        discriminator_reward = self.discriminator.get_reward_from_discriminator(glass_slice)

        for i in range(self.num_agents):
            rewards[i] += discriminator_reward

        self.glass.execute()

        rewards = [reward_old + reward_new for reward_old, reward_new in zip(rewards, self.glass.rewards_after_execute)]

        if self.render_mode is not None:
            self.render()

        glass_slice = self.__get_glass_slice()

        observations = self.__get_observations(glass_slice)
        dones = self.__get_dones()

        return observations, rewards, dones, {}, {}

    def __get_observations(self, glass_slice):
        observations = []
        for agent in self.agents:
            obs = [[agent.money, agent.actives]]

            if agent.open:
                obs.append([1, agent.num_iterations])
                if agent.request.type == BUY_TYPE:
                    obs.append([agent.request.price, agent.request.volume])
                elif agent.request.type == SELL_TYPE:
                    obs.append([agent.request.price, -agent.request.volume])
            else:
                obs.append([0, -1])
                obs.append([0, 0])

            observations.append(np.vstack((obs, glass_slice)))

        return observations

    def __get_glass_slice(self):
        slice_sell, slice_buy = self.glass.get_two_slices()
        slice_sell, slice_buy = transform_slices(slice_sell, slice_buy, self.glass_length)
        return np.vstack((slice_buy, slice_sell))

    def __get_dones(self):
        dones = [False for _ in range(self.num_agents)]
        for agent_id, agent in enumerate(self.agents):
            if agent.money == 0 and agent.actives == 0:
                dones[agent_id] = True
        return dones

    def render(self):
        pygame.init()
        pygame.display.init()
        pygame.display.set_caption("Биржевой стакан")
        self.clock = pygame.time.Clock()

        slice_sell, slice_buy = self.glass.get_two_slices()

        for buy in slice_buy:
            buy[0], buy[1] = np.round(buy[0], 2), buy[1]

        for sell in slice_sell:
            sell[0], sell[1] = sell[1], np.round(sell[0], 2)

        Drawer(self.window_size, slice_sell, slice_buy)

        pygame.event.pump()
        pygame.display.update()

        self.clock.tick(self.metadata["render_fps"])

    def close(self):
        if self.clock is not None:
            pygame.display.quit()
            pygame.quit()

    def reset(self, *, seed=None, options=None):
        if options == None:
            return self.__get_observations(self.__get_glass_slice())
        else:
            agents_to_reset = options['agents_to_reset']
            for agent_id in agents_to_reset:
                agent = Agent(agent_id, random.randint(0, 1_000), random.randint(50_000, 100_000))
                self.agents[agent_id] = agent
                self.glass.agents[agent_id] = agent
                self.glass.requests_by_id[agent_id] = None
                self.glass.num_iterations_by_id[agent_id] = None
            return self.__get_observations(self.__get_glass_slice())
