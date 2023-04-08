"""File defining the processes (actor, learner)."""

import itertools
from multiprocessing import Process, Queue
import os
import threading
import time

import numpy as np

from a2c import A2C
from env import SnakeEnv
import variables

MAX_IDLE_STEPS = 100


class AgentProcess(Process):

    def __init__(self, conn, id: int, n_games: int):
        """Initializes the agent process.

        Args:
            conn: The connection to receive the messages from the master.
            id: The ID of the process (determined by the master).
            n_games: The number of episodes to play until the data is returned.
                Used by the actors only.
        """
        super().__init__()
        self._conn = conn
        # Buffer for the input messages from the master. Shouldn't be much used
        # in practice.
        self._msg_queue = []

        self._n_games = n_games
        self._id = id
        self._env = SnakeEnv()

        # Seed the RNG for the process with 100*id.
        np.random.seed(self._id * 100)

    def gather_data(self):
        """Acts in the environment and produce batches of data."""
        print(f"Process {self._id} starts playing {self._n_games} games.")
        batch_values = []
        batch_states = []
        batch_actions = []
        scores = []
        for _ in range(self._n_games):
            state = self._env.reset()
            # This variable will allow us to track the last time the agent
            # scored, to avoid playing infinitely.
            last_scoring = -1
            for timestep in itertools.count():
                action = self._agent([state])
                new_state, reward, done = self._env.step(action)

                batch_states.append([state])
                batch_actions.append(action)
                # TODO(fazega) Add a proper value function and not just reward.
                batch_values.append(reward)
                # If the agent hasn't scored in MAX_IDLE_STEPS steps, we reset
                # the environment.
                if done or (timestep - last_scoring >= MAX_IDLE_STEPS):
                    break
                state = new_state
            scores.append(self._env.score)
        print(f"Process {self._id} finished playing.")
        batch = (batch_states, batch_actions, batch_values)
        self._conn.send((np.mean(scores), batch))

    def run(self):
        # First instanciate the object dealing with Tensorflow.
        self._agent = A2C(self._id)

        def treat_queue():
            """Deals with the messages sent by the master."""
            msg = self._conn.recv()
            if msg == "load":
                self._agent.load_model()
                print(f"Process {self._id} loaded the master (0) model.")

            if msg[0] == "train_with_batchs":
                # This message is only received by the master.
                assert self._id == 0
                print("Master process is training ...")
                t0 = time.time()
                self._agent.train_with_batchs(msg[1])
                self._agent.save_model()
                delta = time.time() - t0
                print(f"Master process finished training. Time : {delta}s \n")
                self._conn.send("saved")

        while True:
            if self._id != 0:
                # Only the actors gather data, not the learner.
                self.gather_data()
            treat_queue()
