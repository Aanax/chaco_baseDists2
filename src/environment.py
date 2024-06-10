from __future__ import division
import gym
import numpy as np
from collections import deque
from gym.spaces.box import Box
from gym.spaces import Discrete

# from skimage.color import rgb2gray
from cv2 import resize, INTER_AREA

# from skimage.transform import resize
# from scipy.misc import imresize as resize
import random


def atari_env(env_id, env_conf, args):
    """
    env_id - env name from gym
    args - args from config of run
    env_conf - specific json
    """

    mujoco_envs_list = ["InvertedPendulum-v2"]
    if env_id in mujoco_envs_list:
        print("______USING MUJOCO ENV_______")
        env = mujoco_env(env_id, env_conf, args)
        return env
    else:

        env = gym.make(env_id)
        if "NoFrameskip" in env_id:
            assert "NoFrameskip" in env.spec.id
            # idk why. for NoFrameskip frameskip is 1 (but gym autotests do it too)
            env._max_episode_steps = (
                args["Training"]["max_episode_length"] * args["Training"]["skip_rate"]
            )
            # wrapper that makes agent do nothing when env reset
            env = NoopResetEnv(env, noop_max=30)
            # kinda artificial skip for noskip envs FIXME
            env = MaxAndSkipEnv(env, skip=args.skip_rate)
        else:
            env._max_episode_steps = args["Training"]["max_episode_length"]

        # Makes done = True when life was lost!!!!!
        env = EpisodicLifeEnv(env)

        if "FIRE" in env.unwrapped.get_action_meanings():
            env = FireResetEnv(env)

        # env._max_episode_steps = args["Training"]["max_episode_length"]
        env = AtariRescale(env, env_conf)
        env = NormalizedEnv(env)
    return env


def mujoco_env(env_id, env_conf, args):
    env = gym.make(env_id)
    env._max_episode_steps = args["Training"]["max_episode_length"]

    # env = AtariRescale(env, env_conf)
    # env = NormalizedEnv(env)
    env = MujocoDiscreteWrapper(env)
    return env


class MujocoDiscreteWrapper(gym.Wrapper):
    """
    TODO inheirt from ActionWrapper
    change default action_space  (Box of shape 1) to classic Discrete
    """

    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
        self.minus = 0 - self.action_space.low[0]
        self.action_space = Discrete(
            int(self.action_space.high[0] - self.action_space.low[0])
        )

    def step(self, action):
        action = action - self.minus
        return gym.Wrapper.step(self, action)


# def process_frame(frame, conf):
#     frame = frame[conf["crop1"]:conf["crop2"] + 160, :160]
#     frame = frame.mean(2)
#     frame = frame.astype(np.float32)
#     frame *= (1.0 / 255.0)
#     frame = resize(frame, (80, conf["dimension2"]))
#     frame = resize(frame, (80, 80))
#     frame = np.reshape(frame, [1, 80, 80])
#     return frame


def process_frame(frame, conf):
    frame = frame[conf["crop1"] : conf["crop2"] + 160, :160]
    frame = resize(frame, (80, 80), interpolation=INTER_AREA)
    frame = 0.2989 * frame[:, :, 0] + 0.587 * frame[:, :, 1] + 0.114 * frame[:, :, 2]
    #     frame = np.reshape(frame, [1, 80, 80]).astype(np.float32)
    frame = np.expand_dims(frame, 0).astype(np.float32)
    return frame


class AtariRescale(gym.ObservationWrapper):
    """
    just for image showing??
    """

    def __init__(self, env, env_conf):
        gym.ObservationWrapper.__init__(self, env)
        self.observation_space = Box(0.0, 1.0, [1, 80, 80], dtype=np.uint8)
        self.conf = env_conf

    def observation(self, observation):
        return process_frame(observation, self.conf)


class NormalizedEnv(gym.ObservationWrapper):
    def __init__(self, env=None, alpha=0.9999):
        gym.ObservationWrapper.__init__(self, env)
        self.state_mean = 0
        self.state_std = 0
        self.alpha = alpha
        self.num_steps = 0

    def observation(self, observation):
        self.num_steps += 1
        self.state_mean = self.state_mean * self.alpha + observation.mean() * (
            1 - self.alpha
        )
        self.state_std = self.state_std * self.alpha + np.quantile(
            observation, 0.98
        ) * (1 - self.alpha)

        unbiased_mean = self.state_mean / (1 - pow(self.alpha, self.num_steps))
        unbiased_std = self.state_std / (1 - pow(self.alpha, self.num_steps))

        self.unbiased_mean = unbiased_mean
        self.unbiased_std = unbiased_std

        return (observation - unbiased_mean) / (unbiased_std + 1e-8)


class NoopResetEnv(gym.Wrapper):
    def __init__(self, env, noop_max=30):
        """Sample initial states by taking random number of no-ops on reset.
        No-op is assumed to be action 0.
        """
        gym.Wrapper.__init__(self, env)
        self.noop_max = noop_max
        self.override_num_noops = None
        self.noop_action = 0
        assert env.unwrapped.get_action_meanings()[0] == "NOOP"

    def reset(self, **kwargs):
        """Do no-op action for a number of steps in [1, noop_max]."""
        self.env.reset(**kwargs)
        if self.override_num_noops is not None:
            noops = self.override_num_noops
        else:
            noops = self.unwrapped.np_random.randint(
                1, self.noop_max + 1
            )  # pylint: disable=E1101
        assert noops > 0
        obs = None
        for _ in range(noops):
            obs, _, done, _ = self.env.step(self.noop_action)
            if done:
                obs = self.env.reset(**kwargs)
        return obs

    def step(self, ac):
        return self.env.step(ac)


class FireResetEnv(gym.Wrapper):
    def __init__(self, env):
        """Take action on reset for environments that are fixed until firing."""
        gym.Wrapper.__init__(self, env)
        assert env.unwrapped.get_action_meanings()[1] == "FIRE"
        assert len(env.unwrapped.get_action_meanings()) >= 3

    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        obs, _, done, _ = self.env.step(1)
        if done:
            self.env.reset(**kwargs)
        obs, _, done, _ = self.env.step(2)
        if done:
            self.env.reset(**kwargs)
        return obs

    def step(self, ac):
        return self.env.step(ac)


class EpisodicLifeEnv(gym.Wrapper):
    def __init__(self, env):
        """Make end-of-life == end-of-episode, but only reset on true game over.
        Done by DeepMind for the DQN and co. since it helps value estimation.
        """
        gym.Wrapper.__init__(self, env)
        self.lives = 0
        self.was_real_done = True

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.was_real_done = done
        # check current lives, make loss of life terminal,
        # then update lives to handle bonus lives
        lives = self.env.unwrapped.ale.lives()
        if lives < self.lives and lives > 0:
            # for Qbert sometimes we stay in lives == 0 condtion for a few frames
            # so its important to keep lives > 0, so that we only reset once
            # the environment advertises done.
            done = True
        self.lives = lives
        return obs, reward, done, self.was_real_done

    def reset(self, **kwargs):
        """Reset only when lives are exhausted.
        This way all states are still reachable even though lives are episodic,
        and the learner need not know about any of this behind-the-scenes.
        """
        if self.was_real_done:
            obs = self.env.reset(**kwargs)
        else:
            # no-op step to advance from terminal/lost life state
            obs, _, _, _ = self.env.step(0)
        self.lives = self.env.unwrapped.ale.lives()
        return obs


class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env=None, skip=4):
        """Return only every `skip`-th frame"""
        super(MaxAndSkipEnv, self).__init__(env)
        # most recent raw observations (for max pooling across time steps)
        self._obs_buffer = deque(maxlen=3)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        done = None
        for _ in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            self._obs_buffer.append(obs)
            total_reward += reward
            if done:
                break

        max_frame = np.max(np.stack(self._obs_buffer), axis=0)

        return max_frame, total_reward, done, info

    def reset(self, **kwargs):
        """Clear past frame buffer and init. to first obs. from inner env."""
        self._obs_buffer.clear()
        obs = self.env.reset(**kwargs)
        self._obs_buffer.append(obs)
        return obs


def atari_env_eval(env_id, env_conf, args):
    """
    env_id - env name from gym
    args - args from config of run
    env_conf - specific json

    same atari env but with normalizing alpha = 0.5
    """

    mujoco_envs_list = ["InvertedPendulum-v2"]
    if env_id in mujoco_envs_list:
        print("______USING MUJOCO ENV_______")
        env = mujoco_env(env_id, env_conf, args)
        return env
    else:

        env = gym.make(env_id)
        if "NoFrameskip" in env_id:
            assert "NoFrameskip" in env.spec.id
            # idk why. for NoFrameskip frameskip is 1 (but gym autotests do it too)
            env._max_episode_steps = (
                args["Training"]["max_episode_length"] * args["Training"]["skip_rate"]
            )
            # wrapper that makes agent do nothing when env reset
            env = NoopResetEnv(env, noop_max=30)
            # kinda artificial skip for noskip envs FIXME
            env = MaxAndSkipEnv(env, skip=args.skip_rate)
        else:
            env._max_episode_steps = args["Training"]["max_episode_length"]

        # Makes done = True when life was lost!!!!!
        env = EpisodicLifeEnv(env)

        if "FIRE" in env.unwrapped.get_action_meanings():
            env = FireResetEnv(env)

        # env._max_episode_steps = args["Training"]["max_episode_length"]
        env = AtariRescale(env, env_conf)
        env = NormalizedEnv(env, alpha=0.5)
    return env


# import gym
# import numpy as np
# # import universe
# from gym.spaces.box import Box
# from universe import vectorized
# from universe.wrappers import Unvectorize, Vectorize

# import cv2
# import pdb

# # Taken from https://github.com/openai/universe-starter-agent
# def create_mujoco_env(env_id):
#     env = gym.make(env_id)
#     if len(env.observation_space.shape) > 1:
#         env = Vectorize(env)
#         env = Unvectorize(env)
#     return env

# # process each frame
# def _process_frame42(frame):
#     frame = frame[34:34 + 160, :160]
#     # Resize by half, then down to 42x42 (essentially mipmapping). If
#     # we resize directly we lose pixels that, when mapped to 42x42,
#     # aren't close enough to the pixel boundary.
#     frame = cv2.resize(frame, (80, 80))
#     frame = cv2.resize(frame, (42, 42))
#     frame = frame.mean(2)
#     frame = frame.astype(np.float32)
#     frame *= (1.0 / 255.0)
#     frame = np.reshape(frame, [1, 42, 42])
#     return frame


# class AtariRescale42x42(vectorized.ObservationWrapper):

#     def __init__(self, env=None):
#         super(AtariRescale42x42, self).__init__(env)
# 	# convert the observation shape to
#         self.observation_space = Box(0.0, 1.0, [1, 42, 42])

#     def _observation(self, observation_n):
#         return [_process_frame42(observation) for observation in observation_n]


# class NormalizedEnvMujoco(vectorized.ObservationWrapper):

#     def __init__(self, env=None):
#         super(NormalizedEnvMujoco, self).__init__(env)
#         self.state_mean = 0
#         self.state_std = 0
#         self.alpha = 0.9999
#         self.num_steps = 0

#     def _observation(self, observation_n):
# 	# calculate the average of mean/std of obsesrvation
#         for observation in observation_n:
#             self.num_steps += 1
#             self.state_mean = self.state_mean * self.alpha + \
#                 observation.mean() * (1 - self.alpha)
#             self.state_std = self.state_std * self.alpha + \
#                 observation.std() * (1 - self.alpha)

#         unbiased_mean = self.state_mean / (1 - pow(self.alpha, self.num_steps))
#         unbiased_std = self.state_std / (1 - pow(self.alpha, self.num_steps))

#         return [(observation - unbiased_mean) / (unbiased_std + 1e-8) for observation in observation_n]
