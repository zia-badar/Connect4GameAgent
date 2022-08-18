import copy
import pickle as pk

import torch
from torch.utils.data import TensorDataset, DataLoader

from environement import C4Env
from policy import Policy

def generate_interaction(pi_s, env, horizon, gamma=0.99, _lambda=0.95):

    current_state, current_player_turn = env.get_current_state()

    states = [[], []]
    actions = [[], []]
    values = [[], []]
    rewards = [[], []]
    terminals = [[], []]
    while len(states[0]) < horizon + 1 or len(states[1]) < horizon + 1:         # +1 for extra value for advantage compuatation
        success = False
        while not success:
            action = pi_s[current_player_turn].act(current_state)
            success, _, reward, ended = env.step(action)

        value = pi_s[current_player_turn].value(current_state)
        states[current_player_turn].append(current_state)
        actions[current_player_turn].append(action)
        values[current_player_turn].append(value)
        rewards[current_player_turn].append(reward)              # reward belongs to the state from which it was generated as a consequence of performing action.
        terminals[current_player_turn].append(ended)

        if len(terminals[(current_player_turn + 1) % 2]) and not terminals[(current_player_turn + 1) % 2][-1]:         # if last state of opposite player belongs to the same episode set ended and reward to it.
            terminals[(current_player_turn + 1) % 2][-1] = ended
            rewards[(current_player_turn + 1) % 2][-1] -= reward        # -ve reward because of opposite player

        if ended:
            env.reset()     # skip terminal state because no action or reward exists for it

        current_state, current_player_turn = env.get_current_state()


    vtargets, advantages = [None, None], [None, None]
    for player_turn in list([0, 1]):
        states[player_turn] = torch.cat(states[player_turn][:horizon]).reshape(((horizon,) + C4Env.board_size))
        actions[player_turn] = torch.cat(actions[player_turn][:horizon])
        values[player_turn] = torch.cat(values[player_turn][:(horizon+1)]).squeeze(-1).detach()
        rewards[player_turn] = torch.tensor(rewards[player_turn][:horizon])

        # https://arxiv.org/abs/1506.02438
        # http://incompleteideas.net/book/RLbook2020.pdf#page=309
        terminals[player_turn] = torch.tensor(terminals[player_turn][:horizon])
        advan = torch.empty((horizon + 1)).cuda()
        deltas = torch.empty(horizon).cuda()
        advan[-1] = 0
        for t in reversed(range(T)):
            deltas[t] = rewards[player_turn][t] + gamma*values[player_turn][t+1]*(not terminals[player_turn][t]) - values[player_turn][t]
            advan[t] = deltas[t] + gamma*_lambda*(advan[t+1] * (not terminals[player_turn][t]))

        vtargets[player_turn] = values[player_turn][:-1] + advan[:-1]
        advantages[player_turn] = advan[:-1]

    return torch.stack(states), torch.stack(actions), torch.stack(vtargets), torch.stack(advantages)

def evaluate(pi):
    opponents = list(['random', 'weak_rule_based', 'strong_rule_based'])
    win_to_loss_ratio = [0, 0, 0]

    with torch.no_grad():
        for i, opponent in enumerate(opponents):
            env = C4Env(rule_based_opponent_type=opponent)
            total_wins = 0
            total = 100
            for _ in range(total):
                is_ended = False
                while not is_ended:
                    s, _ = env.get_current_state()
                    a = pi.act(s)

                    success, _, r, e = env.step(a)
                    if success:
                        is_ended = e
                total_wins += (r == 1)
                total -= (r == 0)       # draw
                env.reset()

            win_to_loss_ratio[i] = (total_wins / total)

    return win_to_loss_ratio

if __name__ == '__main__':

    pi_s = (Policy().cuda(), Policy().cuda())
    adams = (torch.optim.Adam(list(pi_s[0].parameters()), lr=2.5e-4), torch.optim.Adam(list(pi_s[1].parameters()), lr=2.5e-4) )

    epsilon = 0.2
    c1 = 0.1
    c2 = 0.01
    N = 8      # actors
    T = 256     # timestamp per actor
    M = 32      # SGD batch size
    k = 4       # epochs per iteration

    environments = []
    for _ in range(N):
        environments.append(C4Env())

    evaluations = []
    for iteration in range(1, 10000):
        pi_old_s = (copy.deepcopy(pi_s[0]), copy.deepcopy(pi_s[1]))


        states, actions, vtargets, advantages = ([], [], [], [])
        for i in range(N):
            ss, acs, vtgs, advs = generate_interaction(pi_old_s, environments[i], T)
            states.append(ss)
            actions.append(acs)
            vtargets.append(vtgs)
            advantages.append(advs)

        states = torch.stack(states).permute(0, 2, 1, 3, 4).reshape(N*T, 2, C4Env.board_size[0], C4Env.board_size[1])
        actions = torch.stack(actions).permute(0, 2, 1).reshape(N*T, 2)
        vtargets = torch.stack(vtargets).permute(0, 2, 1).reshape(N*T, 2)
        advantages = torch.stack(advantages).permute(0, 2, 1).reshape(N*T, 2)

        dataset = TensorDataset(states, actions, vtargets, advantages)
        dataloader = DataLoader(dataset, batch_size=M, shuffle=True)


        for epochs in range(k):
            for states, actions, vtargets, advantages in dataloader:
                for player_turn in list([0, 1]):
                    _adam = adams[player_turn]

                    _adam.zero_grad()

                    _states = states[:, player_turn, :, :]
                    _actions = actions[:, player_turn]
                    _vtargets = vtargets[:, player_turn]
                    _advantages = advantages[:, player_turn]
                    _pi = pi_s[player_turn]
                    _pi_old = pi_old_s[player_turn]

                    vthetas = _pi.value(_states)

                    rtheta = _pi.prob(_actions, _states) / _pi_old.prob(_actions, _states)
                    l_clip = torch.minimum(rtheta*_advantages, torch.clip(rtheta, 1 - epsilon, 1 + epsilon)*_advantages)
                    l_vf = (vthetas - _vtargets)**2
                    entropy = _pi.entropy(_states)

                    loss = -torch.mean(l_clip - c1*l_vf + c2*entropy)
                    loss.backward()

                    _adam.step()

        evaluations.append(evaluate(pi_s[0]))
        print(f'evaluation: {evaluations[-1]}')

        if iteration % 100 == 0:
            with open(f'evaluation_selfplay', 'wb') as file:
                pk.dump(evaluations, file, pk.HIGHEST_PROTOCOL)