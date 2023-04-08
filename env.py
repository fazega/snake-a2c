"""Environment to play the game of snake.

The game consists in a grid, with a 'snake' and a 'fruit'. The snake must eat
the fruit to increase the score by 1. When eaten, the fruit is resampled
randomly somewhere on the grid, and the snake grows by one square.
The snake loses if it eats itself._
"""

import enum
import numpy as np
import operator

import variables


class Tile(enum.IntEnum):
    """Different tiles used in the game."""

    EMPTY = 0
    HEAD_SNAKE = 1
    FRUIT = 2
    TAIL_SNAKE = 3


class SnakeEnv:
    """The snake environment.

    Observation: The grid, where 0 is empty, 1 is the head of the snake,
    2 is the fruit, and 3 is the tail of the snake (see enum above).
    Reward: 1 if the snake eats the fruit, 0 otherwise.
    Action: An integer in {0, 1, 2, 3} <=> {left, right, up, down}.
    """

    def reset(self):
        """Resets the environment to a grid with one fruit and a small snake."""
        self._score = 0
        self._grid = np.zeros((variables.env_width, variables.env_height))

        random_pos_snake = (
            np.random.randint(variables.env_width),
            np.random.randint(variables.env_height),
        )
        self._snake = [random_pos_snake]
        self._grid[
            random_pos_snake[0], random_pos_snake[1]
        ] = Tile.HEAD_SNAKE.value

        self._reset_fruit()

        # The observation is the grid itself.
        return np.copy(self._grid)

    def _reset_fruit(self):
        """Resets the position of the fruit."""
        # We try to sample a position for the fruit.
        # We must keep on sampling until we find a free position.
        while True:
            random_pos_fruit = (
                np.random.randint(variables.env_width),
                np.random.randint(variables.env_height),
            )
            if (
                self._grid[random_pos_fruit[0], random_pos_fruit[1]]
                == Tile.EMPTY.value
            ):
                break
        self._fruit = [random_pos_fruit]
        self._grid[random_pos_fruit[0], random_pos_fruit[1]] = Tile.FRUIT.value

    def step(self, action: int) -> tuple[np.ndarray, float, bool]:
        """Returns the observation, reward and end of episode from an action."""
        match action:
            case 0:
                shift = (-1, 0)
            case 1:
                shift = (1, 0)
            case 2:
                shift = (0, 1)
            case 3:
                shift = (0, -1)
            case _:
                raise ValueError(
                    "The action must be between 0 and 3, included."
                )
        next_cell = tuple(map(operator.add, shift, self._snake[0]))

        # Out of bounds => lost.
        if (
            next_cell[0] < 0
            or next_cell[0] >= variables.env_width
            or next_cell[1] < 0
            or next_cell[1] >= variables.env_height
        ):
            return None, 0, True

        # Eating itself => lost.
        if self._grid[next_cell[0], next_cell[1]] == Tile.TAIL_SNAKE.value:
            return None, 0, True

        reward = 0
        self._snake = [next_cell] + self._snake

        # Eating fruit => good.
        if self._grid[next_cell[0], next_cell[1]] == Tile.FRUIT.value:
            reward = 1
            self._reset_fruit()
        self._grid[next_cell[0], next_cell[1]] = Tile.HEAD_SNAKE.value
        self._grid[self._snake[1][0], self._snake[1][1]] = Tile.TAIL_SNAKE.value

        if reward == 0:
            self._grid[
                self._snake[-1][0], self._snake[-1][1]
            ] = Tile.EMPTY.value
            self._snake = self._snake[:-1]

        self._score += reward
        return np.copy(self._grid), reward, False

    @property
    def score(self) -> float:
        return self._score
