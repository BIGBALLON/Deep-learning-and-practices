import gym
from gym.wrappers import Monitor
import itertools
import numpy as np
import os
import random
import sys
import tensorflow as tf
from collections import deque, namedtuple

# Hyper Parameters:
GAMMA = 0.99                        # decay rate of past observations

# Epsilon
INITIAL_EPSILON = 1.0               # 0.01 # starting value of epsilon
FINAL_EPSILON = 0.1                 # 0.001 # final value of epsilon
EXPLORE_STPES = 500000              # frames over which to anneal epsilon

# replay memory
INIT_REPLAY_MEMORY_SIZE = 50000
REPLAY_MEMORY_SIZE = 300000

BATCH_SIZE = 32
FREQ_UPDATE_TARGET_Q = 10000        # Update target network every 10000 steps
TRAINING_EPISODES = 10000

MONITOR_PATH = 'breakout_videos/'
RECORD_VIDEO_EVERY = 1000

# Valid actions for breakout: ['NOOP', 'FIRE', 'RIGHT', 'LEFT']
VALID_ACTIONS = [0, 1, 2, 3]

class ObservationProcessor():
    """
    Processes a raw Atari image. Resizes it and converts it to grayscale.
    """
    def __init__(self):
        with tf.variable_scope("ob_proc"):
            self.input_state = tf.placeholder(shape=[210, 160, 3], dtype=tf.uint8)              # input image
            self.output = tf.image.rgb_to_grayscale(self.input_state)                           # rgb to grayscale
            self.output = tf.image.crop_to_bounding_box(self.output, 34, 0, 160, 160)           # crop image
            self.output = tf.image.resize_images(                                               # resize image
                self.output, [84, 84], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            self.output = tf.squeeze(self.output)                                               # remove rgb dimension

    def process(self, sess, state):
        return sess.run(self.output, { self.input_state: state })

class DQN():
    def __init__(self, scope="estimator", summaries_dir=None):
        self.scope = scope
        self.summary_writer = None
        with tf.variable_scope(scope):
            self._build_network()
            if summaries_dir:
                summary_dir = os.path.join(summaries_dir, "summaries_{}".format(scope))
                if not os.path.exists(summary_dir):
                    os.makedirs(summary_dir)
                self.summary_writer = tf.summary.FileWriter(summary_dir)

    def _build_network(self):
        self.X_pl = tf.placeholder(shape=[None, 84, 84, 4], dtype=tf.uint8, name="X")
        self.y_pl = tf.placeholder(shape=[None], dtype=tf.float32, name="y")
        self.actions_pl = tf.placeholder(shape=[None], dtype=tf.int32, name="actions")

        X = tf.to_float(self.X_pl) / 255.0

        conv1 = tf.contrib.layers.conv2d(X, 32, 8, 4, activation_fn=tf.nn.relu)
        conv2 = tf.contrib.layers.conv2d(conv1, 64, 4, 2, activation_fn=tf.nn.relu)
        conv3 = tf.contrib.layers.conv2d(conv2, 64, 3, 1, activation_fn=tf.nn.relu)

        # Fully connected layers
        flattened = tf.contrib.layers.flatten(conv3)
        fc1 = tf.contrib.layers.fully_connected(flattened, 512)
        self.predictions = tf.contrib.layers.fully_connected(fc1, len(VALID_ACTIONS))

        # Get the predictions for the chosen actions only
        batch_size = tf.shape(self.X_pl)[0]
        gather_indices = tf.range(batch_size) * tf.shape(self.predictions)[1] + self.actions_pl
        self.action_predictions = tf.gather(tf.reshape(self.predictions, [-1]), gather_indices)

        # Calcualte the loss
        self.losses = tf.squared_difference(self.y_pl, self.action_predictions)
        self.loss = tf.reduce_mean(self.losses)

        # Optimizer Parameters from original paper
        self.optimizer = tf.train.RMSPropOptimizer(0.00025, 0.99, 0.0, 1e-6)
        self.train_op = self.optimizer.minimize(self.loss, global_step=tf.contrib.framework.get_or_create_global_step())

        # Summaries for Tensorboard
        self.summaries = tf.summary.merge([
            tf.summary.scalar("loss", self.loss),
            tf.summary.histogram("loss_hist", self.losses),
            tf.summary.histogram("q_values_hist", self.predictions),
            tf.summary.scalar("max_q_value", tf.reduce_max(self.predictions))
        ])

    def predict(self, sess, s):
        return sess.run(self.predictions, { self.X_pl: s })
    def update(self, sess, s, a, y):
        feed_dict = { self.X_pl: s, self.y_pl: y, self.actions_pl: a }
        summaries, global_step, _, loss = sess.run(
            [self.summaries, tf.contrib.framework.get_global_step(), self.train_op, self.loss],feed_dict)
        if self.summary_writer:
            self.summary_writer.add_summary(summaries, global_step)
        return loss



def update_target_network(sess, behavior_Q, target_Q):

    e1_params = [t for t in tf.trainable_variables() if t.name.startswith(behavior_Q.scope)]
    e1_params = sorted(e1_params, key=lambda v: v.name)
    e2_params = [t for t in tf.trainable_variables() if t.name.startswith(target_Q.scope)]
    e2_params = sorted(e2_params, key=lambda v: v.name)

    update_ops = []
    for e1_v, e2_v in zip(e1_params, e2_params):
        op = e2_v.assign(e1_v)
        update_ops.append(op)

    sess.run(update_ops)

def make_epsilon_greedy_policy(estimator, nA):
    def policy_fn(sess, observation, epsilon):
        A = np.ones(nA, dtype=float) * epsilon / nA
        q_values = estimator.predict(sess, np.expand_dims(observation, 0))[0]
        best_action = np.argmax(q_values)
        A[best_action] += (1.0 - epsilon)
        return A
    return policy_fn

def main(_):
    # make game eviornment
    
    env = gym.envs.make("Breakout-v0")

    # Define Transition tuple
    Transition = namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])

    # The replay memory
    replay_memory = []

    # create a observation processor
    ob_proc = ObservationProcessor()

    # Behavior Network & Target Network
    behavior_Q = DQN(scope="behavior_Q", summaries_dir=r'./')
    target_Q = DQN(scope="target_Q",summaries_dir=r'./')


    # tensorflow session
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    
    epsilons = np.linspace(INITIAL_EPSILON, FINAL_EPSILON, EXPLORE_STPES)
    policy = make_epsilon_greedy_policy(
        behavior_Q,
        len(VALID_ACTIONS))

    # Populate the replay buffer
    observation = env.reset()                       # retrive first env image
    observation = ob_proc.process(sess, observation)        # process the image
    state = np.stack([observation] * 4, axis=2)     # stack the image 4 times

    total_tmp = 0
    while len(replay_memory) < INIT_REPLAY_MEMORY_SIZE:
        
        action_probs = policy(sess, state, epsilons[min(total_tmp, EXPLORE_STPES-1)])
        action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
       
        next_observation, reward, done, _ = env.step(VALID_ACTIONS[action])
        next_observation = ob_proc.process(sess, next_observation)
        next_state = np.append(state[:,:,1:], np.expand_dims(next_observation, 2), axis=2)
        replay_memory.append(Transition(state, action, reward, next_state, done))

        # Current game episode is over
        if done:
            observation = env.reset()
            observation = ob_proc.process(sess, observation)
            state = np.stack([observation] * 4, axis=2)

        # Not over yet
        else:
            state = next_state
        if len(replay_memory) % 1000 == 0:
        	print('Populate the replay buffer: %.2f%% \r' %(float(len(replay_memory)*100)/INIT_REPLAY_MEMORY_SIZE), end="")


    saver = tf.train.Saver()
    # record videos
    env = Monitor(env, directory=MONITOR_PATH, video_callable=lambda count: (count + 1) % RECORD_VIDEO_EVERY == 0, resume=True)

    # total steps
    total_t = 0
    max_reward = 0
    print('START TRAINING')
    for episode in range(TRAINING_EPISODES):

        saver.save(tf.get_default_session(), 'checkpoints')

        # Reset the environment
        observation = env.reset()
        observation = ob_proc.process(sess, observation)
        state = np.stack([observation] * 4, axis=2)
        episode_reward = 0                              # store the episode reward
        for t in itertools.count():
            # Epsilon for this time step
            epsilon = epsilons[min(total_t, EXPLORE_STPES-1)]

            # Maybe update the target estimator
            if total_t % FREQ_UPDATE_TARGET_Q == 0:
                update_target_network(sess, behavior_Q, target_Q)
                print("\nCopied model parameters to target network.")

            # Take a step
            action_probs = policy(sess, state, epsilon)
            action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
            next_state, reward, done, _ = env.step(VALID_ACTIONS[action])
            next_state = ob_proc.process(sess, next_state)
            next_state = np.append(state[:,:,1:], np.expand_dims(next_state, 2), axis=2)
            
            episode_reward += reward

            # If our replay memory is full, pop the first element
            if len(replay_memory) >= REPLAY_MEMORY_SIZE:
                replay_memory.pop(0)

            # Save transition to replay memory
            replay_memory.append(Transition(state, action, reward, next_state, done))   

            # Sample a minibatch from the replay memory
            samples = random.sample(replay_memory, BATCH_SIZE)
            states_batch, action_batch, reward_batch, next_states_batch, done_batch = map(np.array, zip(*samples))

            # Calculate q values and targets
            q_values_next = target_Q.predict(sess, next_states_batch)
            targets_batch = reward_batch + np.invert(done_batch).astype(np.float32) * GAMMA * np.amax(q_values_next, axis=1)

            # Perform gradient descent update
            states_batch = np.array(states_batch)
            loss = behavior_Q.update(sess, states_batch, action_batch, targets_batch)

            if done:
                if max_reward < episode_reward:
                    max_reward = episode_reward
                print ("Episode %d ----- reward: %d Max reward: %d" %(episode, episode_reward, max_reward))
                train_summary = tf.Summary(value=[tf.Summary.Value(tag="Episode Reward", simple_value=episode_reward)])
                behavior_Q.summary_writer.add_summary(train_summary, episode)
                behavior_Q.summary_writer.flush()
                break

            state = next_state
            total_t += 1

if __name__ == '__main__':
    tf.app.run()