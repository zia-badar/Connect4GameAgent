import copy

import torch
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

from environement import C4Env
from policy import Policy

def generate_interaction(pi, env, horizon, gamma=0.99, _lambda=0.95):

    current_state, current_player_turn = env.get_current_state()

    timestamp = 0
    states = []
    actions = []
    values = []
    rewards = []
    terminals = []
    while len(states) < horizon:
        success = False
        while not success:
            action = pi.act(current_state)
            success, _, reward, ended = env.step(action)

        value = pi.value(current_state)
        states.append(current_state)
        actions.append(action)
        values.append(value)
        rewards.append(reward)              # reward belongs to the state from which it was generated as a consequence of performing action.

        terminals.append(ended)

        if ended:
            env.reset()     # skip terminal state because no action or reward exists for it

        current_state, current_player_turn = env.get_current_state()
        timestamp += 1


    values.append(pi.value(current_state))
    states = torch.cat(states).reshape(((horizon,) + C4Env.board_size))
    actions = torch.cat(actions)
    values = torch.cat(values).squeeze(-1).detach()
    rewards = torch.tensor(rewards)

    # https://arxiv.org/abs/1506.02438
    # http://incompleteideas.net/book/RLbook2020.pdf#page=309
    terminals = torch.tensor(terminals)
    advantages = torch.empty((horizon + 1)).cuda()
    deltas = torch.empty(horizon).cuda()
    advantages[-1] = 0
    for t in reversed(range(T)):
        deltas[t] = rewards[t] + gamma*values[t+1]*(not terminals[t]) - values[t]
        advantages[t] = deltas[t] + gamma*_lambda*(advantages[t+1] * (not terminals[t]))

    vtargets = values[:-1] + advantages[:-1]
    advantages = advantages[:-1]

    return states, actions, vtargets, advantages

def evaluate(pi):
    with torch.no_grad():
        env = C4Env(rule_based_opponent_step=True)
        total_rewards = 0
        total = 100
        for _ in range(total):
            is_ended = False
            while not is_ended:
                s, _ = env.get_current_state()
                a = pi.act(s)

                success, _, r, e = env.step(a)
                if success:
                    is_ended = e
            total_rewards += r
            env.reset()

        print(f'score: {total_rewards/total}')


if __name__ == '__main__':

    pi = Policy().cuda()
    adam = torch.optim.Adam(list(pi.parameters()), lr=2.5e-4)


    epsilon = 0.2
    c1 = 0.1
    c2 = 0.01
    N = 8      # actors
    T = 256     # timestamp per actor
    M = 32      # SGD batch size
    k = 4       # epochs per iteration

    environments = []
    for _ in range(N):
        environments.append(C4Env(rule_based_opponent_step=True))

    for iteration in tqdm(range(10000)):
        pi_old = copy.deepcopy(pi)


        states, actions, vtargets, advantages = ([], [], [], [])
        for i in range(N):
            ss, acs, vtgs, advs = generate_interaction(pi_old, environments[i], T)
            states.append(ss)
            actions.append(acs)
            vtargets.append(vtgs)
            advantages.append(advs)


        dataset = TensorDataset(torch.cat(states), torch.cat(actions), torch.cat(vtargets), torch.cat(advantages))
        dataloader = DataLoader(dataset, batch_size=M, shuffle=True)


        total_loss = 0
        for epochs in range(k):
            for states, actions, vtargets, advantages in dataloader:
                adam.zero_grad()

                vthetas = pi.value(states)

                rtheta = pi.prob(actions, states) / pi_old.prob(actions, states)
                l_clip = torch.minimum(rtheta*advantages, torch.clip(rtheta, 1 - epsilon, 1 + epsilon)*advantages)
                l_vf = (vthetas - vtargets)**2
                entropy = pi.entropy(states)

                loss = -torch.mean(l_clip - c1*l_vf + c2*entropy)
                loss.backward()
                total_loss += loss.item()

                adam.step()

        print(f'total loss: {total_loss}')
        evaluate(pi)

