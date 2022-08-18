import matplotlib
import numpy as np
from matplotlib import pyplot as plt
import pickle as pk


def plot():
    with open(f'evaluation_selfplay', 'rb') as file:
        eval = pk.load(file)
    eval = np.array(eval)
    eval = eval[:100]
    matplotlib.rcParams['figure.dpi'] = 300
    plt.plot(np.arange(1, 101), eval[:, 0], label='random')
    plt.plot(np.arange(1, 101), eval[:, 1], label='weak_rule_based')
    plt.plot(np.arange(1, 101), eval[:, 2], label='strong_rule_based')
    plt.title('training by self play')
    plt.xlabel('iterations')
    plt.ylabel('wins to games played ratio, over 100 games')
    plt.legend()
    plt.savefig('plots/self_play')
    plt.show()


    eval = []
    for rule_based in list(['random', 'strong_rule_based', 'weak_rule_based']):
        with open(f'evaluation_{rule_based}', 'rb') as file:
            eval.append(pk.load(file))

    max_iter = 60
    for i, _ in enumerate(eval):
        eval[i] = np.array(eval[i])[:max_iter]
    eval = np.array(eval).T
    matplotlib.rcParams['figure.dpi'] = 300
    plt.plot(np.arange(1, max_iter+1), eval[:, 0], label='random')
    plt.plot(np.arange(1, max_iter+1), eval[:, 1], label='weak_rule_based')
    plt.plot(np.arange(1, max_iter+1), eval[:, 2], label='strong_rule_based')
    plt.title('training against rule based')
    plt.xlabel('iterations')
    plt.ylabel('wins to games played ratio, over 100 games')
    plt.legend()
    plt.savefig('plots/rule_based')
    plt.show()
