import numpy as np
from typing import NewType, Any

# Define a new type to store board state
BoardState = NewType('BoardState', np.ndarray)

class State:
    r"""State of the Tic Tac Board.
    Args:
        boardsize: int = square matrix dimensions
    """
    def __init__(self, boardsize: int):
        self.boardsize = boardsize
        self.board = np.zeros((boardsize, boardsize), dtype=int)
        self.to_move = 'X'
        self.utility = 0

    def get_board_state(self) -> BoardState:
        return BoardState(self.board)

    def update_board_state(self, row: int, col: int, val: int) -> None:
        assert col <= self.boardsize - 1
        assert row <= self.boardsize - 1
        assert val in (0, -1, 1)
        self.board[row, col] = val

    @staticmethod
    def decode_board(val):
        # converts int vals to strings
        converter = {0: '', 1: 'X', -1: 'O'}
        return converter[val]

    def print_board_state(self):
        curr_state = self.get_board_state().flatten().tolist()
        curr_state = np.array(list(map(State.decode_board, curr_state)))
        curr_state = curr_state.reshape((self.boardsize, self.boardsize))
        print(curr_state)

    # Needs to be updated to check for all 1s or -1s in rows, columns or diagonals

    def is_goal_state(self) -> tuple[bool, int]:
        winning_sum = 0
        curr_state = self.get_board_state()
        row_sums = np.sum(curr_state, axis=0)
        col_sums = np.sum(curr_state, axis=1)
        winner_sum_x = self.boardsize * 1
        winner_sum_o = self.boardsize * -1

        trace = np.trace(curr_state)
        diag = np.diagonal(curr_state)
        diag_sum = np.sum(diag)
        flip_diag = np.diagonal(np.fliplr(curr_state))
        flip_sum = np.sum(flip_diag)

        if winner_sum_x in row_sums or winner_sum_x in col_sums or winner_sum_x == diag_sum or winner_sum_x == flip_sum:
            return True, 1
        if winner_sum_o in row_sums or winner_sum_o in col_sums or winner_sum_o == diag_sum or winner_sum_o == flip_sum:
            return True, -1
        if not np.any(curr_state == 0):
            return True, 0
        return False, 0

    @staticmethod
    def is_terminal_state(curr_state, boardsize) -> tuple[bool, int]:
        winning_sum = 0
        goal_state = False
        row_sums = np.sum(curr_state, axis=0)
        col_sums = np.sum(curr_state, axis=1)
        winner_sum_x = boardsize * 1
        winner_sum_o = -winner_sum_x

        # diag value and flipdiag value sum fiag and sum flipdiag
        trace = int(np.trace(curr_state))
        flip_diag = np.diagonal(np.fliplr(curr_state))
        flip_sum = int(np.sum(flip_diag))

        if winner_sum_x in row_sums:
            goal_state=True
            winning_sum = winner_sum_x

        elif winner_sum_o in row_sums:
            goal_state = True
            winning_sum = winner_sum_o

        elif winner_sum_x in col_sums:
            goal_state = True
            winning_sum = winner_sum_x

        elif winner_sum_o in col_sums:
            goal_state = True
            winning_sum = winner_sum_o

        elif winner_sum_o == trace:
            goal_state = True
            winning_sum = winner_sum_x

        elif winner_sum_x == trace:
            goal_state = True
            winning_sum = winner_sum_x

        elif winner_sum_o == trace:

            goal_state = True
            winning_sum = winner_sum_o
        elif winner_sum_o == flip_sum:
            goal_state = True
            winning_sum = winner_sum_o
        elif winner_sum_x == flip_sum:
            goal_state = True
            winning_sum = winner_sum_x
        elif np.any(list(curr_state == 0)):
            goal_state = False
            winning_sum = 0
        else:
            goal_state = True
            winning_sum = 0
        return goal_state, winning_sum


    def utility(self, is_agent_x: bool):
        goal, winner = self.is_goal_state()
        if goal:
            if winner == 1:
                return 1 if is_agent_x else -1
            elif winner == -1:
                return -1 if is_agent_x else 1
        return 0

