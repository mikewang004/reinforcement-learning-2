import numpy as np
import matplotlib.pyplot as plt

reward_curve = np.loadtxt("data/reward_curves_epsilons{epss}_{epse}_{epsd}_num_eps_{num_episodes}_n_reps_{n_repetitions}txt")
reward_curve_mean = np.mean(reward_curve, axis=1)
plt.plot(reward_curve)
plt.plot(reward_curve_mean, marker = "o")
plt.show()
