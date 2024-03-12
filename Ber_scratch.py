import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym

env = gym.make('CartPole-v1', render_mode="human")
observation, info = env.reset()

budget = 1000

if budget != 0:
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        obervation, info = env.reset()

    env.render()

    budget -= 1