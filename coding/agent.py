import random


class Agent:
    def __init__(self, agent_id, actives, money):
        self.agent_id = agent_id
        self.actives = actives
        self.money = money
        self.open = False
        self.request = None
        self.num_iterations = None

    def random_action(self):
        return self.agent_id, [random.uniform(-1, 1) for _ in range(3)]

    def __str__(self):
        return f'Agent {self.agent_id}, money: {self.money}, actives: {self.actives}'
