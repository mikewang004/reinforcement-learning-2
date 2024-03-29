import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter



learning_rate_1 = np.loadtxt("data/learning_rate_e-5-e-4_num_eps_500_n_reps_6.txt")
learning_rate_2 = np.loadtxt("data/learning_rate_e-4-e-3_num_eps_500_n_reps_6.txt")
lr = [1e-5, 0.5e-5, 1e-4, 0.5e-4, 1e-3, 0.5e-3] #corresponding learning rates
print(learning_rate_2.shape)

learning_rate = np.zeros([500, 6])
# Now order array in the proper way

learning_rate[:, 0] = learning_rate_1[:, 1]
learning_rate[:, 1] = learning_rate_1[:, 0]
learning_rate[:, 2] = learning_rate_2[:, 0]
learning_rate[:, 3] = learning_rate_1[:, 2]
learning_rate[:, 4] = learning_rate_2[:, 2]
learning_rate[:, 5] = learning_rate_2[:, 1]

lr = np.sort(lr)
print(lr)


for i in range(0, learning_rate.shape[1]):
    #yhat = savgol_filter(learning_rate[:, i], 500, 4)
    yhat = np.convolve(learning_rate[:, i], np.ones(100), "valid")/np.max(learning_rate[:, i])
    #plt.plot(learning_rate[:, i], label = f"learning rate = {lr[i]}")
    plt.plot(yhat, label = f"learning rate = {lr[i]}")
plt.legend()
plt.show()