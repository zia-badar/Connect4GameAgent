import gym
import torch
from gym import spaces
import numpy as np
from enum import IntEnum

class position_occupied_type(IntEnum):
    EMPTY = 0
    PLAYER_1 = 1
    PLAYER_2 = 2

class env(gym.Env):
    # rows = 10
    # cols = 10
    # rows = 5
    # cols = 5
    # rows = 4
    # cols = 4
    rows = 10
    cols = 10
    size = 4
    starting_player = 0  # starting player
    reward_player = 0
    observation_space = spaces.Box(0, 2, np.array((rows, cols, 1)), dtype=np.int32)
    action_space = spaces.Discrete(cols)

    def __init__(self):
        self.reset()

    def interact(self, action, s=False):
        if not s:
            possible, reward = self.occupy_column(action.item(), self.player_turn+1, checking=True)
            reward = reward * (-1 if self.player_turn != self.reward_player else 1)
            return possible, None, reward, None
        else:
            return self.step(action.item())

    def step(self, action):
        def is_board_full():
            return np.argwhere(self.board[0] != position_occupied_type.EMPTY).size == self.cols

        assert self.action_space.contains(action), 'invalid action, not from action space'
        assert not(is_board_full()), 'board is already full'

        # _, backup_reward = self.occupy_column(action, ((self.player_turn + 1) % 2) + 1, True)

        possible, reward = self.occupy_column(action, self.player_turn+1)
        game_ended = True if reward or is_board_full() else False

        if possible:
            # if reward == 0:
            #     reward = backup_reward / (self.cols * 10)
            reward = reward * (-1 if self.player_turn != self.reward_player else 1)
            self.player_turn = (self.player_turn + 1) % 2


        return possible, torch.tensor(self.board, dtype=torch.float).cuda(), reward, game_ended,

    def occupy_column(self, column, type, checking=False):

        directions = np.array([[0, 1], [1, 1], [1, 0], [1, -1], [0, -1], [-1, -1], [-1, 0], [-1, 1]])

        def in_bounds(p):
            return p[0] >= 0 and p[0] < self.rows and p[1] >=0 and p[1] < self.cols

        def connection_len(center):
            lengths = np.zeros((3, 3))
            for d in directions:
                for l in range(1, self.size):
                    p = center + l*d
                    if not(in_bounds(p)) or self.board[tuple(p)] != type:
                        break
                    lengths[tuple(d+1)] += 1

                    if lengths[tuple(d+1)] + 1 == self.size:
                        return lengths

            return lengths

        def connection_possible(lengths):
            for d in directions[0:4]:               # horizontal, vertical, 2 diagonals
                opposite_d = -d
                if lengths[tuple(d+1)] + lengths[tuple(opposite_d+1)] + 1 >= self.size:
                    return True

            return False;

        def reward(center):
            return 1 if connection_possible(connection_len(center)) else 0

        p = np.array([0, column])
        r = np.array([1, 0])
        for _ in range(self.rows):
            if self.board[tuple(p)] != position_occupied_type.EMPTY:
                top = p - r
                if in_bounds(top):
                    if not(checking):
                        self.board[tuple(top)] = type
                    return True, reward(top)
                else:
                    return False, 0
            p = p + r

        top = p - r
        if not (checking):
            self.board[tuple(top)] = type
        return True, reward(top)

    def get_current_state(self):
        return torch.tensor(self.board, dtype=torch.float).cuda(), self.player_turn, None

    def reset(self):
        self.player_turn = self.starting_player
        self.board = np.zeros((self.rows, self.cols), dtype=np.int32)

        return np.expand_dims(self.board, -1)           # 1 dimension in end for pixel color dimensions, required by policy

    def close(self):
        print('closing connect n environment')


