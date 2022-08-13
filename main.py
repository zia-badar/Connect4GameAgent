import copy

import torch
import torchvision.models
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

from environement import C4Env
from policy import Policy

def generate_interaction(pi, env, max_timestamp=100):

    current_state, current_player_turn, current_ended = env.get_current_state()

    timestamp = 0
    states = []
    actions = []
    values = []
    rewards = []
    terminals = []
    while timestamp < max_timestamp:
        success = False
        while not success:
            if current_player_turn == 0:
                action = pi.act(current_state)
            else:
                action = torch.randint(0, C4Env.board_size[1], (1,)).cuda()

            success, _, reward, _ = env.interact(action)

        value = pi.value(current_state)
        states.append(current_state)
        actions.append(action)
        values.append(value)
        rewards.append(reward)              # reward belongs to the state from which it was generated as a consequence of performing action.

        _, _, _, current_ended = env.interact(action, True)
        terminals.append(current_ended)

        if current_ended:
            env.reset()     # skip terminal state because no action or reward exists for it

        current_state, current_player_turn, _ = env.get_current_state()
        timestamp += 1


    values.append(pi.value(current_state))
    states = torch.cat(states).reshape(((max_timestamp, ) + C4Env.board_size))
    actions = torch.cat(actions)
    values = torch.cat(values).squeeze(-1)
    rewards = torch.tensor(rewards)
    terminals = torch.tensor(terminals)
    return states, actions, values, rewards, terminals

# https://arxiv.org/abs/1506.02438
# http://incompleteideas.net/book/RLbook2020.pdf#page=309
def compute_advantage(pi, rewards, values, terminal, gamma=0.1, _lambda=0.3):
    T = len(rewards)
    advantages = torch.empty((T+1)).cuda()
    deltas = torch.empty(T).cuda()
    advantages[-1] = 0
    for t in reversed(range(T)):
        deltas[t] = rewards[t] + gamma*values[t+1]*(not terminal[t]) - values[t]
        advantages[t] = deltas[t] + gamma*_lambda*(advantages[t+1] * (not terminal[t]))

    return values[:-1] + advantages[:-1], advantages[:-1]

def evaluate(pi):
    with torch.no_grad():
        env = C4Env()
        total_rewards = 0
        total = 100
        dist = torch.zeros(C4Env.board_size[1])
        for _ in tqdm(range(total)):
            is_ended = False
            while not is_ended:
                s, p, e = env.get_current_state()
                if p == 0:
                    a = pi.act(s)
                else:
                    a = torch.randint(0, C4Env.board_size[1], (1,)).cuda()

                success, _, _, _e = env.interact(a)
                if success:
                    if _e:
                        dist[a] += 1
                    _, s, r, is_ended = env.interact(a, True)
            total_rewards += r
            env.reset()

        print(f'score: {total_rewards/total}, {dist}')


if __name__ == '__main__':

    batch = 32
    action_shape = (6,)


    pi = Policy(C4Env.board_size, (C4Env.board_size[1], )).cuda()
    pi_old = copy.deepcopy(pi)
    adam = torch.optim.Adam(list(pi.parameters()))


    epsilon = 0.2
    c1 = 0.1
    c2 = 0.1
    N = 10      # actors
    T = 256     # timestamp per actor
    M = 64      # SGD batch size
    k = 4       # epochs per iteration

    environments = []
    for _ in range(N):
        environments.append(C4Env())

    bar = tqdm(range(1000))
    for iteration in bar:
        pi_before_update = copy.deepcopy(pi)


        states, actions, vtargets, advantages = ([], [], [], [])
        for i in range(N):
            ss, acs, vs, rs, ts = generate_interaction(pi, environments[i], T)
            vtgs, advs = compute_advantage(pi, rs, vs.detach(), ts)
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

        pi_old = pi_before_update
        bar.set_description(f'total loss: {total_loss}')
        evaluate(pi)

