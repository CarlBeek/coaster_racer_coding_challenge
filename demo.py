import cv2
import numpy as np
import random
from collections import deque
import gym
import universe
import tensorflow as tf

# hyper params:
ACTIONS = 3  # left, right, stay
KEYS = ['ArrowLeft', 'ArrowRight', 'ArrowUp']
ENV_ID = 'flashgames.CoasterRacer-v0'


# deep q network. feed in pixel data to graph session
def testGraph(inp, out, sess):

    # initialise universe/gym kak:
    env = gym.make(ENV_ID)
    env.configure(fps=5.0, remotes=1, start_timeout=15 * 60)

    # intial frame
    observation_n = env.reset()

    observation_n, reward_t, done_t, info = env.step(
        [[('KeyValue', 'ArrowUp', True)]])
    while info['n'][0]['env_status.env_state'] is None:
        observation_n, reward_t, done_t, info = env.step(
            [[('KeyValue', 'ArrowUp', True)]])
        env.render()

    observation_t = processFrame(observation_n)

    # stack frames, that is our input tensor
    inp_t = np.stack(
        (observation_t,
         observation_t,
         observation_t,
         observation_t),
        axis=2)

    previous_argmax = 0

    print(done_t)
    print(info)

    # testing time
    while(1):

        # output tensor
        out_t = out.eval(session=sess, feed_dict={inp: [inp_t]})
        # argmax function
        argmax_t = np.zeros([ACTIONS])

        argmax_t[np.argmax(out_t)] = 1

        action_t, previous_argmax = appendActions(
            observation_n, argmax_t, previous_argmax)
        observation_n, reward_t, done_t, info = env.step(action_t)
        env.render()

        while observation_n[0] is None:
            observation_n, reward_t, done_t, info = env.step(
                [[('KeyValue', 'ArrowUp', True)]])

        observation_t = processFrame(observation_n)

        inp_t1 = np.append(
            np.reshape(
                observation_t, [
                    120, 160, 1]), inp_t[
                :, :, 0:3], axis=2)

        # update our input tensor the the next frame
        inp_t = inp_t1


# crop video frame so NN is smaller and set range between 1 and 0; and
# stack-a-bitch!
def processFrame(observation_n):
    if observation_n is not None:
        obs = observation_n[0]['vision']
        # crop
        obs = cropFrame(obs)
        # downscale resolution (not sure about sizing here, was (120,160) when
        # I started but it felt like that was just truncating the colourspace)
        obs = cv2.resize(obs, (120, 160))
        # greyscale
        obs = cv2.cvtColor(obs, cv2.COLOR_BGR2GRAY)
        # Convert to float
        obs = obs.astype(np.float32)
        # scale from 1 to 255
        obs *= (1.0 / 255.0)
        # re-shape a bitch
        obs = np.reshape(obs, [120, 160])
    return obs


# crop frame to only flash portion:
def cropFrame(obs):
    # adds top = 84 and left = 18 to height and width:
    return obs[84:564, 18:658, :]


# Add appropiate actions to system
def appendActions(observation_n, argmax_t, previous_argmax):
    actions_n = ([[('KeyEvent',
                    KEYS[np.argmax(previous_argmax)],
                    False),
                   ('KeyEvent',
                    'ArrowUp',
                    True),
                   ('KeyEvent',
                    'n',
                    True),
                   ('KeyEvent',
                    KEYS[np.argmax(argmax_t)],
                    True)] for obs in observation_n])
    return actions_n, argmax_t


def main():

    sess = tf.Session()

    # restore the weights, baises and structure to the graph:
    [sess, inp, out] = restore(sess, 'CoasterRacer-dqn-100000')

    testGraph(inp, out, sess)


def restore(sess, file):

    # Variables to be restored to
    W_conv1 = tf.Variable(tf.zeros([8, 8, 4, 32]), name='W_conv1')
    b_conv1 = tf.Variable(tf.zeros([32]), name='b_conv1')

    W_conv2 = tf.Variable(tf.zeros([4, 4, 32, 64]), name='W_conv2')
    b_conv2 = tf.Variable(tf.zeros([64]), name='b_conv2')

    W_conv3 = tf.Variable(tf.zeros([3, 3, 64, 64]), name='W_conv3')
    b_conv3 = tf.Variable(tf.zeros([64]), name='b_conv3')

    W_fc4 = tf.Variable(tf.zeros([11264, 784]), name='W_fc4')
    b_fc4 = tf.Variable(tf.zeros([784]), name='b_fc4')

    W_fc5 = tf.Variable(tf.zeros([784, ACTIONS]), name='W_fc5')
    b_fc5 = tf.Variable(tf.zeros([ACTIONS]), name='b_fc5')

    # restore above weights and baises to the session:
    saver = tf.train.import_meta_graph(file + '.meta')
    saver.restore(sess, tf.train.latest_checkpoint('./'))
    sess.run(tf.global_variables_initializer())

    # Restore NN structure and input and output place holders:
    inp = tf.placeholder("float", [None, 120, 160, 4])

    conv1 = tf.nn.relu(
        tf.nn.conv2d(
            inp,
            W_conv1,
            strides=[
                1,
                4,
                4,
                1],
            padding="VALID") +
        b_conv1)

    conv2 = tf.nn.relu(
        tf.nn.conv2d(
            conv1,
            W_conv2,
            strides=[
                1,
                2,
                2,
                1],
            padding="VALID") +
        b_conv2)

    conv3 = tf.nn.relu(
        tf.nn.conv2d(
            conv2,
            W_conv3,
            strides=[
                1,
                1,
                1,
                1],
            padding="VALID") +
        b_conv3)
    # flatten conv3:
    conv3_flat = tf.reshape(conv3, [-1, 11264])

    fc4 = tf.nn.relu(tf.matmul(conv3_flat, W_fc4) + b_fc4)

    out = tf.matmul(fc4, W_fc5) + b_fc5

    return sess, inp, out


if __name__ == "__main__":
    main()
