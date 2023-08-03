import numpy as np

SELL_TYPE = 'sell'
BUY_TYPE = 'buy'


class MakeRequest:
    def __init__(self, agent_id, price, volume):
        if volume < 0:
            self.type = SELL_TYPE
        elif volume > 0:
            self.type = BUY_TYPE
        self.price = price
        self.volume = abs(volume)
        self.agent_id = agent_id
        self.open = True

    def __str__(self):
        if self.open:
            return f'Agent {self.agent_id}, price: {np.round(self.price, 2)}, {self.type} {self.volume} - open'
        else:
            return f'Agent {self.agent_id}, price: {np.round(self.price, 2)}, {self.type} {self.volume} - closed'


class Hold:
    def __init__(self, agent_id):
        self.agent_id = agent_id

    def __str__(self):
        return f'Agent {self.agent_id} - hold'


class StopRequest:
    def __init__(self, agent_id):
        self.agent_id = agent_id

    def __str__(self):
        return f'Agent {self.agent_id} - stop'
