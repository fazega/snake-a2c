import numpy as np
import variables
import operator

class SnakeEnv():
    def __init__(self):
        self.grid = np.zeros((variables.env_width, variables.env_height))
        self.snake = []
        self.fruit = None
        self.score = 0

    def init(self):
        self.score = 0
        self.grid = np.zeros((variables.env_width, variables.env_height))

        random_pos_snake = (np.random.randint(variables.env_width), np.random.randint(variables.env_height))
        self.snake = [random_pos_snake]
        self.grid[random_pos_snake[0], random_pos_snake[1]] = 1

        self.initFruit()

        return np.copy(self.grid)

    def initFruit(self):
        while True:
            random_pos_fruit = (np.random.randint(variables.env_width), np.random.randint(variables.env_height))
            if(self.grid[random_pos_fruit[0], random_pos_fruit[1]] == 0):
                break
        self.fruit = [random_pos_fruit]
        self.grid[random_pos_fruit[0], random_pos_fruit[1]] = 2

    def step(self, action):
        next_cell = None
        if(action == 0):
            next_cell = (-1,0)
        if(action == 1):
            next_cell = (1,0)
        if(action == 2):
            next_cell = (0,1)
        if(action == 3):
            next_cell = (0,-1)
        next_cell = tuple(map(operator.add, next_cell, self.snake[0]))

        # Out of bounds => lost
        if(next_cell[0] < 0 or next_cell[0] >= variables.env_width or next_cell[1] < 0 or next_cell[1] >= variables.env_height):
            # self.score += -1
            return None, 0, True

        # Eating itself => lost
        if(self.grid[next_cell[0], next_cell[1]]):
            # self.score += -1
            return None, 0, True

        reward = 0
        self.snake = [next_cell] + self.snake

        # Eating fruit => good
        if(self.grid[next_cell[0], next_cell[1]] == 2):
            reward = 1
            self.initFruit()
        self.grid[next_cell[0], next_cell[1]] = 1

        if(reward == 0):
            self.grid[self.snake[-1][0], self.snake[-1][1]] = 0
            self.snake = self.snake[:-1]

        self.score += reward
        return np.copy(self.grid), reward, False
