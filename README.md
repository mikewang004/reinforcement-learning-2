Source code for our submission for the "Deep Q Learning" assignment, Reinforcement Learning, Spring semester 2024.

The model can be run via "python Experiment.py dqn [arguments]", with the arguments given as follows:
    -   policy: either "egreedy" or "softmax", sets the exploration strategy;
    -   er_enabled: boolean. Turns Experience Replay on or off;
    -   tn_enabled: boolean. Turns Target Network on or off;
    -   network_sizes: list, int. Sets the network architecture. Sets per int the amount of nodes for that layer,
    with len(network_sizes) the amount of layers; e.g. [32, 64, 32] is a network of three layers,
    of which the first and third contain 32 nodes and the second layer 64;
    -   num_episodes: int. Sets the amount of episodes per repetition;
    -   n_repetitions: int. Sets the amount of repetitions per setting;
    -   lr: float. Sets the learning rate;
    -   gamma: float. Sets the decay rate;
    -   eps_start, eps_end, eps_decay: all float. Sets respectively the epsilon start, epsilon end and epsilon decay value
    for the egreedy exploration strategy;
    -   temp: float. Sets the temperature for the softmax strategy.
    -   save_txt, save_plot: boolean. Saves a "reward_curves.txt" respectively a "reward_curve.pdf" when True. Default set to False.


For example "python Experiment.py dqn 'softmax' False False" runs the model with the Softmax strategy and both Experience Replay
and Target Network turned off. All other parameters can be overwritten via the command prompt but are more easily tuned inside "Experiment.py".

The file "Experiment.py" should be enough to reproduce all results from our report; if this is not the case, we include also a file 
"Experiment-3.py" which contains all code we have used to generate the plots in our report. Running this takes very long 
but should produce most plots as-is. 

