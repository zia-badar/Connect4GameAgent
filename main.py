import torch

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
                action = pi.act(current_state[None])
            else:
                action = torch.randint(0, C4Env.board_size[1], (1,))

            success, _, reward, _ = env.interact(action)

        value = pi.value(current_state[None])
        states.append(current_state[None])
        actions.append(action)
        values.append(value)
        rewards.append(reward)              # reward belongs to the state from which it was generated as a consequence of performing action.

        _, _, _, current_ended = env.interact(action, True)
        terminals.append(current_ended)

        if current_ended:
            env.reset()     # skip terminal state because no action or reward exists for it

        current_state, current_player_turn, _ = env.get_current_state()
        timestamp += 1


    values.append(pi.value(current_state[None]))
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
    advantages = torch.empty((T+1))
    deltas = torch.empty(T)
    advantages[:-1] = 0
    for t in reversed(range(T)):
        deltas[t] = rewards[t] + gamma*values[t+1]*(not terminal[t]) - values[t]
        advantages[t] = deltas[t] + gamma*_lambda*(advantages[t+1] * (not terminal[t]))

    return values[:-1] + advantages[:-1], advantages[:-1]

if __name__ == '__main__':

    batch = 32
    state_shape = (10, 10)
    action_shape = (6,)


    pi = Policy(C4Env.board_size, (C4Env.board_size[1], ))
    pi_old = Policy(C4Env.board_size, (C4Env.board_size[1], ))
    env = C4Env()
    adam = torch.optim.Adam(pi.paramters())


    epsilon = 0.2
    c1 = 0.1
    c2 = 0.1

    for i in range(1000):
        adam.zero_grad()

        states, actions, values, rewards, terminals = generate_interaction(pi, env, 256)
        vtarget, advantages = compute_advantage(pi, rewards, values, terminals)
        vtheta = values[:-1]

        rtheta = pi.prob(actions, states) / pi_old.prob(actions, states)
        l_clip = torch.minimum(rtheta*advantages, torch.clip(rtheta, 1 - epsilon, 1 + epsilon)*advantages)
        l_vf = (vtheta - vtarget)**2
        entropy = pi.entropy(states)

        loss = torch.mean(l_clip - c1*l_vf + c2*entropy)
        if torch.any(torch.isinf(torch.tensor([-torch.inf]))).item():
            pass

        print(loss)

        loss.backward()
        adam.step()


