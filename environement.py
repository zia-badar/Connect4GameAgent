import torch


class C4Env():
    board_size = (10, 10)
    length = 4

    def __init__(self):
        self.reset()

    def interact(self, action, step=False):
        col = action.item()
        if not(self.top[col] < self.board_size[0]):          # action column is full
            return False, None, None, None

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

        if step:
            self.board = torch.clone(self.board)                # torch autograd requires input not changed for grad computation
            self.board[update_r, update_c] = player_indicator
            self.top[update_c] += 1
            self.player_turn = (self.player_turn + 1) % 2
            self.ended = ended

        return True, self.board, reward, ended

    def get_current_state(self):
        return self.board, self.player_turn, self.ended

    def reset(self):
        self.board = torch.zeros(C4Env.board_size).cuda()
        self.top = [0] * C4Env.board_size[1]
        self.player_turn = 0
        self.ended = False