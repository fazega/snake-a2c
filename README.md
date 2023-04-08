# Training parallel RL agents on Snake

This project implements the [A2C algorithm](https://arxiv.org/abs/1602.01783) from 2016, using Tensorflow v1. The environment used is a basic snake, with a parameterizable grid size.

## Multiprocessing

The main feature of this program is that it uses multiple processes. One process trains the weights of the networks ('learner' or 'master'), while all the others play the game to gather data with the latest weights ('actors'). They exchange information using very basic messages, transmitted with pipes. All these processes are controlled by the main, initial process which also spawns them at the beginning (see launch.py).

## Usage Example

`python3 launch.py`

Scores will be saved in a 'log_scores' files, and also printed on the terminal. Model weights will be regularly saved (after each epoch, consiting of 300 agent steps).