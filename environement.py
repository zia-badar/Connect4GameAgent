import torch


class C4Env():
    board_size = (8, 7)
    length = 4

    def __init__(self, rule_based_opponent_type=None):
        self.reset()
        self.rule_based_opponent_type = rule_based_opponent_type

    def step(self, action):
        possible, reward, ended = self.outcome(action)

        if not possible:
            return False, None, None, None

        self.do_it(action)
        if self.rule_based_opponent_type != None and not ended:
            if self.rule_based_opponent_type == 'random':
                opponent_action, opponent_reward, opponent_ended = self.random_opponent()
            elif self.rule_based_opponent_type == 'weak_rule_based':
                opponent_action, opponent_reward, opponent_ended = self.weak_rule_based_opponent()
            elif self.rule_based_opponent_type == 'strong_rule_based':
                opponent_action, opponent_reward, opponent_ended = self.strong_rule_based_opponent()

            self.do_it(opponent_action)
            ended = opponent_ended
            reward += opponent_reward

        return True, self.board, reward if self.rule_based_opponent_type != None else abs(reward), ended    # when environment donot simulates any opponent reward is always +ve, because its the player turn who is interacting with the environment.

    def random_opponent(self):
        possible = False
        while not possible:
            action = torch.randint(0, C4Env.board_size[1], (1,)).cuda()
            possible, reward, ended = self.outcome(action)

        return action, reward, ended

    def weak_rule_based_opponent(self):
        max_length = -1
        for action in torch.arange(0, C4Env.board_size[1]):
            possible, reward, ended, lengths = self.outcome(action, return_length=True)
            if possible and lengths.sum().item() > max_length:
                best_action = action
                best_reward = reward
                best_ended = ended
                max_length = lengths.sum().item()

        return best_action, best_reward, best_ended

    def strong_rule_based_opponent(self):
        winning_action = None                                       # take winning move
        for action in torch.arange(0, C4Env.board_size[1]):
            possible, reward, ended = self.outcome(action)
            if possible and ended:
                winning_action = action
                break

        if winning_action != None:
            return (winning_action,) + self.outcome(winning_action)[1:]

        loss_avoiding_action = None                                 # avoid lossing move
        self.player_turn = (self.player_turn + 1) % 2
        for action in torch.arange(0, C4Env.board_size[1]):
            possible, reward, ended = self.outcome(action)
            if possible and ended:
                loss_avoiding_action = action
                break
        self.player_turn = (self.player_turn + 1) % 2

        if loss_avoiding_action != None:
            return (loss_avoiding_action,) + self.outcome(loss_avoiding_action)[1:]

        return self.weak_rule_based_opponent()                      # take best possible move

    def outcome(self, action, return_length=False):
        col = action.item()
        if not(self.top[col] < self.board_size[0]):          # action column is full
            return (False, None, None ) + ((None, ) if return_length else ())

        update_r, update_c = self.top[col], col
        player_indicator = self.player_turn + 1

        # down, up, left, right, down-left, up-right, down-right, up-left,
        directions = torch.tensor([[[-1, 0], [1, 0]], [[0, -1], [0, 1]], [[-1, -1], [1, 1]], [[-1, 1], [1, -1]]] ).cuda()       # |4x2x2|
        positions = torch.tensor([update_r, update_c]).cuda().reshape(1, 1, 1, -1) + directions.unsqueeze(-2) * torch.arange(1, C4Env.length).cuda().reshape(1, 1, -1, 1).cuda()        # |4x2x3x2|
        valid_positions = torch.logical_and(torch.logical_and(positions[:, :, :, 0] >= 0, positions[:, :, :, 0] < C4Env.board_size[0]), torch.logical_and(positions[:, :, :, 1] >= 0, positions[:, :, :, 1] < C4Env.board_size[1])).cuda()  # |4x2x3|
        d0 = torch.where(valid_positions, positions[:, :, :, 0], torch.zeros(1, dtype=torch.long).cuda())
        d1 = torch.where(valid_positions, positions[:, :, :, 1], torch.zeros(1, dtype=torch.long).cuda())
        board_values = torch.where(valid_positions, self.board[d0, d1], torch.zeros(1, dtype=torch.float).cuda())
        a = (board_values == player_indicator).to(torch.int)
        b = torch.cat([a, torch.zeros_like(a[:, :, :1])], dim=-1)           # padding with zeros to compute length
        lengths = torch.argmin(b, dim=-1).cuda()
        ended = torch.any(torch.ge(torch.sum(lengths, dim=1), C4Env.length-1)).item()

        draw = True
        for c, v in enumerate(self.top):
            draw &= (v == self.board_size[0]) if c != update_c else (v == (self.board_size[0]-1))
        ended |= draw
        reward = (-1) ** (self.player_turn) if ended and not draw else 0

        return (True, reward, ended) + ((lengths, ) if return_length else ())

    def do_it(self, action):
        col = action.item()
        update_r, update_c = self.top[col], col
        player_indicator = self.player_turn + 1

        self.board = torch.clone(self.board)  # torch autograd requires input not changed for grad computation
        self.board[update_r, update_c] = player_indicator
        self.top[update_c] += 1
        self.player_turn = (self.player_turn + 1) % 2

    def get_current_state(self):
        return self.board, self.player_turn

    def reset(self):
        self.board = torch.zeros(C4Env.board_size).cuda()
        self.top = [0] * C4Env.board_size[1]
        self.player_turn = 0