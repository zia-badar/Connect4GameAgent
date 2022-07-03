import torch


class C4Env():
    board_size = (8, 7)
    length = 4

    def __init__(self):
        self.reset()

    def interact(self, action, step=False):
        col = action.item()
        if not(self.top[col] < self.board_size[0]):          # action column is full
            return False, None, None, None

        update_r, update_c = self.top[col], col
        player_indicator = self.player_turn + 1

        # down, left, right, down-left, down-right, up-left, up-right
        directions = torch.tensor([[-1, 0], [0, -1], [0, 1], [-1, -1], [-1, 1], [1, -1], [1, 1]])
        diff = directions[:, None] * torch.arange(1, C4Env.length)[:, None]
        positions = torch.tensor([update_r, update_c]) + diff
        valid_positions = torch.logical_and(torch.logical_and(positions[:, :, 0] >= 0, positions[:, :, 0] < C4Env.board_size[0]), torch.logical_and(positions[:, :, 1] >= 0, positions[:, :, 1] < C4Env.board_size[1]))
        valid_directions = directions[torch.argwhere(torch.all(valid_positions, dim=1))[:, 0]]

        diff = valid_directions[:, None] * torch.arange(1, C4Env.length)[:, None]
        positions = torch.tensor([update_r, update_c]) + diff
        ended = torch.any(torch.all(self.board[positions[:, :, 0], positions[:, :, 1]] == player_indicator, dim=1)).item()
        reward = (-1) ** (self.player_turn) if ended else 0

        if step:
            self.board = torch.clone(self.board).detach()                # torch autograd requires input not changed for grad computation
            self.board[update_r, update_c] = player_indicator
            self.top[update_c] += 1
            self.player_turn = (self.player_turn + 1) % 2
            self.ended = ended

        return True, self.board, reward, ended

    def get_current_state(self):
        return self.board, self.player_turn, self.ended

    def reset(self):
        self.board = torch.zeros(C4Env.board_size, requires_grad=False)
        self.top = [0] * C4Env.board_size[1]
        self.player_turn = 0
        self.ended = False