import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

reward_curve_norm = np.loadtxt("data/experience_replay_experiment.txt")
print(reward_curve_norm.shape)
er_e = [False, True]; num_episodes = 500
plt.figure()
plt.title("Comparison of experience replay agents")
plt.xlabel("episode"); plt.ylabel("rewards")
#plt.xticks(np.linspace(0, num_episodes, reward_eval_count+1))
for j in range(0, 2):
    plt.plot(np.linspace(0, num_episodes, num_episodes+1)[:-1], reward_curve_norm[j, :], label = f"experience replay is enabled: {er_e[j]}")
plt.legend()
plt.savefig(f"plots/experience_replay_experiment.pdf")
plt.show()
