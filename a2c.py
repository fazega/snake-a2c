import numpy as np
import tensorflow as tf
import random
import time
import os
import variables


class A2C():
    def __init__(self, id):
        self.id = id

        self.batch_states = []
        self.batch_values = []
        self.batch_actions = []
        self.batch_legal = []

        self.global_step = tf.Variable(0, name="global_step", trainable=False)

        n_gpu = 4
        if(n_gpu == 1):
            self._build_model()
            self._build_train_op()
        else:
            # 1 GPU for training, n-1 for playing
            # <= variables.n_process // (2*n_gpu)
            if(self.id <= 5):
                gpu_id = 0
            else:
                gpu_id = (1 + (self.id%(n_gpu-1)))
            os.environ["CUDA_VISIBLE_DEVICES"]=str(gpu_id)
            self._build_model()
            self._build_train_op()

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        # config.log_device_placement=True
        # config.gpu_options.per_process_gpu_memory_fraction = 0.2
        self.sess = tf.Session(config=config)
        self.initializer = tf.global_variables_initializer()
        self.sess.run(self.initializer)

        self.saver = tf.train.Saver()
        self.save_path = "./trained_agents/a2c/"
        self.load_model()


    def load_model(self):
        try:
            save_dir = '/'.join(self.save_path.split('/')[:-1])
            ckpt = tf.train.get_checkpoint_state(save_dir)
            load_path = ckpt.model_checkpoint_path
            self.sess.run(self.initializer)
            self.saver.restore(self.sess, load_path)
        except Exception as e:
            print(e)
            print("No saved model to load, starting a new model from scratch.")
        else:
            print("Loaded model: {}".format(load_path))



    def _build_model(self):
        # Inputs
        self.input_states = tf.placeholder(dtype = tf.float32, shape = [None, 1, variables.env_width, variables.env_height])
        input_states = tf.transpose(self.input_states, [0, 2, 3, 1])

        net = tf.layers.conv2d(input_states, 100, (4,4), activation='relu')
        net = tf.layers.batch_normalization(net)
        net = tf.layers.conv2d(input_states, 200, (3,3), activation='relu')
        net = tf.layers.batch_normalization(net)
        net = tf.layers.flatten(net)

        probsNet = tf.contrib.layers.fully_connected(net, 200, tf.nn.leaky_relu)
        self.output_action_logits = tf.contrib.layers.fully_connected(probsNet, 4, activation_fn=None)
        self.output_action_probs = tf.nn.softmax(self.output_action_logits)

        valueNet = tf.contrib.layers.fully_connected(net, 200, tf.nn.leaky_relu)
        self.output_value = tf.contrib.layers.fully_connected(valueNet, 1)

    def _build_train_op(self):
        self.actions_ph = tf.placeholder(tf.int32, (None,))
        self.value_ph = tf.placeholder(tf.float32, (None,))
        self.advantages_ph = tf.placeholder(tf.float32, (None,))

        self.cross_entropy_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                                            logits=self.output_action_probs,
                                            labels=self.actions_ph)
        self.action_loss = tf.reduce_mean(tf.multiply(self.cross_entropy_loss, self.advantages_ph))

        # Add entropy if you'd like more exploration
        # self.entropy = -tf.reduce_mean(tf.reduce_sum(self.output_action_probs*tf.log(self.output_action_probs+1e-7), axis=1))/np.log(4)
        # self.entropy = tf.where(tf.is_nan(self.entropy),0., self.entropy)
        # self.action_loss -= 0.01*self.entropy

        self.action_loss = tf.where(tf.is_nan(self.action_loss),0., self.action_loss)


        self.output_value_flatten = tf.reshape(self.output_value, (-1,))
        self.value_loss = tf.reduce_mean((self.value_ph - self.output_value_flatten)**2)
        self.value_loss = tf.where(tf.is_nan(self.value_loss),0., self.value_loss)

        learning_rate = tf.train.exponential_decay(0.0005,
                                        self.global_step, 1000,
                                        0.95, staircase=True)
        self.optimizer_p = tf.train.AdamOptimizer(learning_rate=learning_rate)
        self.gradients_p = self.optimizer_p.compute_gradients(self.action_loss, var_list=tf.trainable_variables())
        self.clipped_gradients_p = [(tf.clip_by_norm(grad, 20.0), var) if (grad is not None) else (tf.zeros_like(var),var) for grad, var in self.gradients_p]
        self.train_p_op = self.optimizer_p.apply_gradients(self.clipped_gradients_p,  self.global_step)


        self.optimizer_v = tf.train.AdamOptimizer(learning_rate=learning_rate)
        self.gradients_v = self.optimizer_v.compute_gradients(self.value_loss, var_list=tf.trainable_variables())
        self.clipped_gradients_v = [(tf.clip_by_norm(grad, 20.0), var) if grad is not None else (tf.zeros_like(var),var) for grad, var in self.gradients_v]
        self.train_v_op = self.optimizer_p.apply_gradients(self.clipped_gradients_v,  self.global_step)

    def __call__(self, state):
        p = self.getProbs(state)
        action = np.random.choice(4, 1, p=p)[0]
        return action

    def getProbs(self, state):
        p = self.sess.run(self.output_action_probs, {
            self.input_states: [state],
        })
        p = p[0]
        if np.isnan(p[0]):
            p = (1/len(p))*np.ones((1, len(p)))[0]
        return p

    def train_with_batchs(self, batch):
        self.batch_states = []
        self.batch_values = []
        self.batch_actions = []

        for x in batch:
            self.batch_states += x[0]
            self.batch_actions += x[1]
            self.batch_values += x[2]
        self.batch_states = np.array(self.batch_states)
        self.batch_actions = np.array(self.batch_actions)
        self.batch_values = np.array(self.batch_values)

        batch_size = 7000
        for i in range(len(self.batch_states))[::batch_size]:
            v = self.sess.run(
                self.output_value,
                {
                    self.input_states: self.batch_states[i:i+batch_size],
                }
            )
            advantages_ph = np.reshape(np.array(self.batch_values[i:i+batch_size]) - v.T, (-1,))


            self.sess.run(
                self.train_p_op,
                {
                    self.input_states: self.batch_states[i:i+batch_size],
                    self.actions_ph : self.batch_actions[i:i+batch_size],
                    self.advantages_ph : advantages_ph,
                }
            )

            self.sess.run(
                self.train_v_op,
                {
                    self.input_states: self.batch_states[i:i+batch_size],
                    self.value_ph: self.batch_values[i:i+batch_size],
                }
            )

    def save_model(self):
        self.saver.save(self.sess, self.save_path, global_step=self.global_step)

    @property
    def train_itr(self):
        return self.sess.run(self.global_step)
