"""The main object used for training."""

import random
import time
import os

import numpy as np
import tensorflow.compat.v1 as tf
# Needed as we use the v1 tf.placeholder, with is incompatible with v2 eager execution.
tf.disable_eager_execution()

import variables

N_GPU = 4
SAVE_PATH = "./models/"


class A2C:
    """An object taking care of the tensorflow related work."""

    def __init__(self, id: int):
        """Initializes the A2C object.

        Args:
            id: The ID of the process that uses this object.
        """
        self._id = id

        self._batch_states = []
        self._batch_values = []
        self._batch_actions = []
        self._batch_legal = []

        self._global_step = tf.Variable(0, name="global_step", trainable=False)

        if N_GPU > 1:
            # 1 GPU for training, n-1 for playing.
            # We distribute them evenly across the processes. For 2 GPUs, that gives
            # ID 0 gets gpu 0, ID 1 gets gpu 1, ID 3 gets gpu 0 etc.
            if(self._id <= N_GPU + 1):
                gpu_id = 0
            else:
                gpu_id = (1 + (self._id % (N_GPU-1)))
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        self._build_model()
        self._build_train_op()

        config = tf.ConfigProto()
        # Important option to make the memory allocation more dynamic.
        # Otherwise, we can get some OOM issues.
        config.gpu_options.allow_growth = True

        self._sess = tf.Session(config=config)
        self._initializer = tf.global_variables_initializer()
        self._sess.run(self._initializer)

        self._saver = tf.train.Saver()
        self._save_path = SAVE_PATH
        self._load_model()


    def load_model(self):
        """Loads the latest saved model, if any."""
        try:
            save_dir = '/'.join(self._save_path.split('/')[:-1])
            ckpt = tf.train.get_checkpoint_state(save_dir)
            load_path = ckpt.model_checkpoint_path
            self._sess.run(self._initializer)
            self._saver.restore(self._sess, load_path)
        except Exception as e:
            # Printing the exception as the catch may hide another problem.
            print(e)
            print("No saved model to load, starting a new model from scratch.")
        else:
            print("Loaded model: {}".format(load_path))



    def _build_model(self):
        """Initializes the neural network architecture, with the TF graph."""
        self._input_states = tf.placeholder(dtype = tf.float32, shape = [None, 1, variables.env_width, variables.env_height])
        input_states = tf.transpose(self._input_states, [0, 2, 3, 1])

        net = tf.layers.conv2d(input_states, 100, (4,4), activation='relu')
        net = tf.layers.batch_normalization(net)
        net = tf.layers.conv2d(net, 200, (3,3), activation='relu')
        net = tf.layers.batch_normalization(net)
        net = tf.layers.flatten(net)

        probsNet = tf.layers.dense(net, 200, tf.nn.leaky_relu)
        self._output_action_logits = tf.layers.dense(probsNet, 4, activation=None)
        self._output_action_probs = tf.nn.softmax(self._output_action_logits)

        valueNet = tf.layers.dense(net, 200, tf.nn.leaky_relu)
        self._output_value = tf.layers.dense(valueNet, 1)

    def _build_train_op(self):
        """Initializes all the TF operations needed to train (loss, gradient update, input data etc)."""
        self._actions_ph = tf.placeholder(tf.int32, (None,))
        self._value_ph = tf.placeholder(tf.float32, (None,))
        self._advantages_ph = tf.placeholder(tf.float32, (None,))

        self._cross_entropy_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=self._output_action_probs,
            labels=self._actions_ph
        )
        self._action_loss = tf.reduce_mean(tf.multiply(self._cross_entropy_loss, self._advantages_ph))

        # Add entropy if you'd like more exploration.
        # self._entropy = -tf.reduce_mean(tf.reduce_sum(self._output_action_probs*tf.log(self._output_action_probs+1e-7), axis=1))/np.log(4)
        # self._entropy = tf.where(tf.is_nan(self._entropy),0., self._entropy)
        # self._action_loss -= 0.01*self._entropy

        self._action_loss = tf.where(tf.is_nan(self._action_loss),0., self._action_loss)


        self._output_value_flatten = tf.reshape(self._output_value, (-1,))
        self._value_loss = tf.reduce_mean((self._value_ph - self._output_value_flatten)**2)
        self._value_loss = tf.where(tf.is_nan(self._value_loss),0., self._value_loss)

        # Ops for the policy network.
        learning_rate = tf.train.exponential_decay(0.0003, self._global_step, 1000, 0.92, staircase=True)
        self._optimizer_p = tf.train.AdamOptimizer(learning_rate=learning_rate)
        self._gradients_p = self._optimizer_p.compute_gradients(self._action_loss, var_list=tf.trainable_variables())
        self._clipped_gradients_p = [(tf.clip_by_norm(grad, 20.0), var) if (grad is not None) else (tf.zeros_like(var),var) for grad, var in self._gradients_p]
        self._train_p_op = self._optimizer_p.apply_gradients(self._clipped_gradients_p,  self._global_step)

        # Ops for the value network.
        self._optimizer_v = tf.train.AdamOptimizer(learning_rate=learning_rate)
        self._gradients_v = self._optimizer_v.compute_gradients(self._value_loss, var_list=tf.trainable_variables())
        self._clipped_gradients_v = [(tf.clip_by_norm(grad, 20.0), var) if grad is not None else (tf.zeros_like(var),var) for grad, var in self._gradients_v]
        self._train_v_op = self._optimizer_p.apply_gradients(self._clipped_gradients_v,  self._global_step)

    def __call__(self, state: np.ndarray) -> int:
        """Returns an action from an environment state."""
        probs = self._get_probs(state)
        return np.random.choice(4, 1, p=p)[0]

    def get_probs(self, state: np.ndarray) -> np.ndarray:
        """Returns the distribution over actions given a state."""
        probs = self._sess.run(self._output_action_probs, {
            self._input_states: [state],
        })
        probs = probs[0]
        if np.isnan(probs[0]):
            # Random policy if there is a NaN somewhete.
            probs = np.fill(shape=(len(probs),), fill_value=1 / len(probs))
        return probs

    def train_with_batchs(self, batch) -> None:
        """Applies the gradients to the weights."""
        # First, data processing on the batches to get a big list of states, actions and values.
        self._batch_states = []
        self._batch_values = []
        self._batch_actions = []

        for x in batch:
            self._batch_states += x[0]
            self._batch_actions += x[1]
            self._batch_values += x[2]
        self._batch_states = np.array(self._batch_states)
        self._batch_actions = np.array(self._batch_actions)
        self._batch_values = np.array(self._batch_values)

        batch_size = 128
        for i in range(len(self._batch_states))[::batch_size]:
            predicted_values = self._sess.run(
                self._output_value,
                {
                    self._input_states: self._batch_states[i:i+batch_size],
                }
            )
            advantages = np.reshape(np.array(self._batch_values[i:i+batch_size]) - predicted_values.T, (-1,))


            # Train the policy network.
            self._sess.run(
                self._train_p_op,
                {
                    self._input_states: self._batch_states[i:i+batch_size],
                    self._actions_ph : self._batch_actions[i:i+batch_size],
                    self._advantages_ph : advantages,
                }
            )

            # Train the value network.
            self._sess.run(
                self._train_v_op,
                {
                    self._input_states: self._batch_states[i:i+batch_size],
                    self._value_ph: self._batch_values[i:i+batch_size],
                }
            )

    def save_model(self):
        self._saver.save(self._sess, self._save_path, global_step=self._global_step)

    @property
    def train_itr(self):
        """Returns the number of gradient steps."""
        return self._sess.run(self._global_step)
