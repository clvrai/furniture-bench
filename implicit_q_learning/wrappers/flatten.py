import gym


class Flatten(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

        self.observation_space = gym.spaces.flatten_space(env.observation_space)
        self.action_space = gym.spaces.flatten_space(env.action_space)

    def observation(self, observation):
        return gym.spaces.flatten(self.env.observation_space, observation)

    def reset(self, **kwargs):
        if kwargs.get("return_info", False):
            obs, info = self.env.reset(**kwargs)
            return self.observation(obs), info
        else:
            return self.observation(self.env.reset(**kwargs))

    def step(self, action):
        action_unflatten = gym.spaces.unflatten(self.env.action_space, action)
        observation, reward, done, info = self.env.step(action_unflatten)
        return self.observation(observation), reward, done, info
