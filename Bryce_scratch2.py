import numpy as np
import gymnasium as gym
import time


class Q_Learning:
    def __init__(self, env.alpha, gamma, epsilon, numberEpisodes, numberOfBins, lowerBounds, upperBounds):
    self.env = env
    self.alpha = alpha
    self.gamma = gamma
    self.epsilon = epsilon
    self.actionNumber = env.action_space.n
    self.numberEpisodes = numberEpisodes
    self.numberOfBins = numberOfBins
    self.lowerBounds = lowerBounds
    self.upperBounds = upperBounds

    #Number of bins is a 4-element array that contains the number of bins for every
    #state entry; 1 element per entry: x, dx/dt, theta, dtheta/dt

    # Storing sum of rewards in every learning episode
    self.sumRewardsEpisode = []

    # Action value function matrix
    self.Qmatrix = np.random.uniform(low = 0, high = 1, size = (numberOfBins[0],numberOfBins[1], numberOfBins[2], numberOfBins[3], self.actionNumber))


# Define function that for given state returns discreticed indeces
# Observe state vector,

#Return indeces of S in Q_sa
def returnIndexState(self, state):
    position = state[0]
    velocity = state[1]
    angle = state[2]
    angularVelocity = state[3]

    cartPositionBin = np.linspace(self.lowerBounds[0], self.upperBounds[0], self.numberOfBins[0])
    cartVelocityBin = np.linspace(self.lowerBounds[1], self.upperBounds[1], self.numberOfBins[1])
    poleAngleBin = np.linspace(self.lowerBounds[2], self.upperBounds[2], self.numberOfBins[2])
    poleAngleVelocityBin = np.linspace(self.lowerBounds[3], self.upperBounds[3], self.numberOfBins[3])

    indexPosition = np.maximum(np.digitize(state[0], cartPositionBin) - 1, 0)
    indexVelocity = np.maximum(np.digitize(state[1], cartVelocityBin) - 1, 0)
    indexAngle = np.maximum(np.digitize(state[2], poleAngleBin) - 1, 0)
    indexAngleVelocity = np.maximum(np.digitize(state[3], poleAngleVelocityBin) - 1, 0)

    return tuple([indexPosition, indexVelocity, indexAngle, indexAngleVelocity])

def selectAction(self.state, index):

    # First 500 episodes, random selection of actions are executed to ensure exploration
    if index<500:
        return np.random.choice(self.actionNumber)

    # for egreedy:
    randomNumber = np.random.random()

    # Slowly decrease epsilon
    if index > 7000:
        self.epsilon = 0.999 * self.epsilon

    if randomNumber < self.epsilon:
        return np.random.choice(self.actionNumber)
    else:
        return np.random.choice(np.where(self.Qmatrix[self.returnIndexState(state)] == np.max(self.Qmatrix[self.returnIndexState(state)]))[0])

def simulateEpisodes(self):
    for indexEpisode in range(self.episodes):
        rewardsEpisode = []
        (stateS, _) = self.env.reset
        stateS = list(stateS)

        print("Simulating episode {}".format(indexEpisode))

        done = False
        while not done:
            stateSIndex = self.returnIndexState(stateS)
            actionA = self.selectAction(stateS, indexEpisode)

            (stateSPrime, reward, done, _, _) = self.env.step(actionA)
            rewardsEpisode.append(reward)
            stateSPrime = list(stateSPrime)
            stateSPrimeIndex = self.returnIndexState(stateSPrime)
            QmaxPrime = np.max(self.Qmatrix[stateSPrimeIndex])

            if not done:
                error = reward + self.gamma * QmaxPrime - self.Qmatrix[stateSPrimeIndex + (actionA,)]
                self.Qmatrix[stateSIndex + (actionA,)] = self.Qmatrix[stateSIndex + (actionA,)] + self.alpha * error
            else:
                error = reward - self.Qmatrix[stateSPrimeIndex + (actionA,)]
                self.Qmatrix[stateSIndex + (actionA,)] = self.Qmatrix[stateSIndex + (actionA,)] + self.alpha * error
            stateS = stateSPrime
        print("Sum of rewards {}".format(sum(rewardsEpisode)))
        self.sumRewardsEpisode.append(np.sum(rewardsEpisode))

def simulateLearnedStragety(self):
    env = gym.make('CartPole-v1', render_mode="human")
    