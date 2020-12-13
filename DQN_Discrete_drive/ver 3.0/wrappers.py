"""
Adapted from OpenAI Baselines
https://github.com/openai/baselines/blob/master/baselines/common/atari_wrappers.py
"""
from collections import deque
import numpy as np
import gym
import copy
import cv2
cv2.ocl.setUseOpenCL(False)

def make_env(env, stack_frames=True, episodic_life=False, clip_rewards=False, scale=False):
    if episodic_life:
        env = EpisodicLifeEnv(env)

    #env = NoopResetEnv(env, noop_max=30)
    env = MaxAndSkipEnv(env, skip=4)
    #if 'FIRE' in env.unwrapped.get_action_meanings():
    #    env = FireResetEnv(env)

    env = WarpFrame(env)
    if stack_frames:
        env = FrameStack(env, 4)
    if clip_rewards:
        env = ClipRewardEnv(env)
    return env

class RewardScaler(gym.RewardWrapper):

    def reward(self, reward):
        return reward * 0.1


class ClipRewardEnv(gym.RewardWrapper):
    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)

    def reward(self, reward):
        """Bin reward to {+1, 0, -1} by its sign."""
        return np.sign(reward)


class LazyFrames(object):
    def __init__(self, frames):
        """This object ensures that common frames between the observations are only stored once.
        It exists purely to optimize memory usage which can be huge for DQN's 1M frames replay
        buffers.
        This object should only be converted to numpy array before being passed to the model.
        You'd not believe how complex the previous solution was."""
        self._frames = frames
        self._out = None

    def _force(self):
        if self._out is None:
            self._out = np.concatenate(self._frames, axis=2)
            self._frames = None
        return self._out

    def __array__(self, dtype=None):
        out = self._force()
        if dtype is not None:
            out = out.astype(dtype)
        return out

    def __len__(self):
        return len(self._force())

    def __getitem__(self, i):
        return self._force()[i]

class FrameStack(gym.Wrapper):
    def __init__(self, env, k):
        """Stack k last frames.
        Returns lazy array, which is much more memory efficient.
        See Also
        --------
        baselines.common.atari_wrappers.LazyFrames
        """
        gym.Wrapper.__init__(self, env)
        self.k = k
        self.frames1 = deque([], maxlen=k)
        self.frames2 = deque([], maxlen=k)
        shp = env.observation_space.shape
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(shp[0], shp[1], shp[2] * k), dtype=env.observation_space.dtype)

    def reset(self):
        ob = self.env.reset()
        for _ in range(self.k):
            self.frames1.append(ob['camera'])
            self.frames2.append(ob['lidar'])
        l =  self._get_ob()
        d = {}
        d['camera'] = l[0]
        d['lidar'] = l[1]
        d['state'] = ob['state']
        return d

    def step(self, action):
        ob, reward, done, info = self.env.step(action)

        self.frames1.append(ob['camera'])
        self.frames2.append(ob['lidar'])

        l = self._get_ob()
        d = {}
        d['camera'] = l[0]
        d['lidar'] = l[1]
        d['state'] = ob['state']
        return d,reward, done, info

    def _get_ob(self):
        assert len(self.frames1) == self.k
        assert len(self.frames2) == self.k
        return [LazyFrames(list(self.frames1)),LazyFrames(list(self.frames2))]


class WarpFrame(gym.ObservationWrapper):
    def __init__(self, env):
        """Warp frames to 84x84 as done in the Nature paper and later work."""
        gym.ObservationWrapper.__init__(self, env)
        self.width = 84
        self.height = 84
        self.observation_space = gym.spaces.Box(low=0, high=255,
            shape=(self.height, self.width, 1), dtype=np.uint8)

    def observation(self, frame):
        d = {}
        frame1 = frame['camera']
        frame1 = cv2.cvtColor(frame1, cv2.COLOR_RGB2GRAY)
        frame1 = cv2.resize(frame1, (self.width, self.height), interpolation=cv2.INTER_AREA)
        #cv2.imwrite("a.jpg",frame1)
        d['camera'] = frame1[:, :, None]

        frame1 = frame['lidar']
        frame1 = cv2.cvtColor(frame1, cv2.COLOR_RGB2GRAY)
        frame1 = cv2.resize(frame1, (self.width, self.height), interpolation=cv2.INTER_AREA)
        d['lidar'] = frame1[:, :, None]

        d['state'] = frame['state']
        return d


class FireResetEnv(gym.Wrapper):
    def __init__(self, env=None):
        """For environments where the user need to press FIRE for the game to start."""
        super(FireResetEnv, self).__init__(env)
        assert env.unwrapped.get_action_meanings()[1] == 'FIRE'
        assert len(env.unwrapped.get_action_meanings()) >= 3

    def step(self, action):
        return self.env.step(action)

    def reset(self):
        self.env.reset()
        obs, _, done, _ = self.env.step(1)
        if done:
            self.env.reset()
        obs, _, done, _ = self.env.step(2)
        if done:
            self.env.reset()
        return obs


class EpisodicLifeEnv(gym.Wrapper):
    def __init__(self, env=None):
        """Make end-of-life == end-of-episode, but only reset on true game over.
        Done by DeepMind for the DQN and co. since it helps value estimation.
        """
        super(EpisodicLifeEnv, self).__init__(env)
        self.lives = 0
        self.was_real_done = True
        self.was_real_reset = False

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.was_real_done = done
        # check current lives, make loss of life terminal,
        # then update lives to handle bonus lives
        lives = self.env.unwrapped.ale.lives()
        if lives < self.lives and lives > 0:
            # for Qbert somtimes we stay in lives == 0 condtion for a few frames
            # so its important to keep lives > 0, so that we only reset once
            # the environment advertises done.
            done = True
        self.lives = lives
        return obs, reward, done, info

    def reset(self):
        """Reset only when lives are exhausted.
        This way all states are still reachable even though lives are episodic,
        and the learner need not know about any of this behind-the-scenes.
        """
        if self.was_real_done:
            obs = self.env.reset()
            self.was_real_reset = True
        else:
            # no-op step to advance from terminal/lost life state
            obs, _, _, _ = self.env.step(0)
            self.was_real_reset = False
        self.lives = self.env.unwrapped.ale.lives()
        return obs


class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env=None, skip=4):
        """Return only every `skip`-th frame"""
        super(MaxAndSkipEnv, self).__init__(env)
        # most recent raw observations (for max pooling across time steps)
        self._obs_buffer1 = deque(maxlen=2)
        self._obs_buffer2 = deque(maxlen=2)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        done = None
        for _ in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            self._obs_buffer1.append(obs['camera'])
            self._obs_buffer2.append(obs['lidar'])
            total_reward += reward
            if done:
                break

        max_frame1 = np.max(np.stack(self._obs_buffer1), axis=0)
        max_frame2= np.max(np.stack(self._obs_buffer2), axis=0)
        d = {}
        d['camera'] = max_frame1
        d['lidar'] = max_frame2
        d['state'] = obs['state']
        return d, total_reward, done, info

    def reset(self):
        """Clear past frame buffer and init. to first obs. from inner env."""
        self._obs_buffer1.clear()
        self._obs_buffer2.clear()
        obs = self.env.reset()
        self._obs_buffer1.append(obs['camera'])
        self._obs_buffer2.append(obs['lidar'])
        return obs

class NoopResetEnv(gym.Wrapper):
    def __init__(self, env=None, noop_max=30):
        """Sample initial states by taking random number of no-ops on reset.
        No-op is assumed to be action 0.
        """
        super(NoopResetEnv, self).__init__(env)
        self.noop_max = noop_max
        self.override_num_noops = None
        assert env.unwrapped.get_action_meanings()[0] == 'NOOP'

    def step(self, action):
        return self.env.step(action)

    def reset(self):
        """ Do no-op action for a number of steps in [1, noop_max]."""
        self.env.reset()
        if self.override_num_noops is not None:
            noops = self.override_num_noops
        else:
            noops = np.random.randint(1, self.noop_max + 1)
        assert noops > 0
        obs = None
        for _ in range(noops):
            obs, _, done, _ = self.env.step(0)
            if done:
                obs = self.env.reset()
        return obs
