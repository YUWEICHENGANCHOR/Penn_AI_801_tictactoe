import time
import numpy as np
from tttagent import TicTacToeAgent
class RandomAgent(TicTacToeAgent):
    def __init__(self, name: str, boardsize: int, is_agent_x: bool):
        self.name = name
        self.boardsize = boardsize
        self.is_agent_x = is_agent_x

    def get_next_move(self, state) -> tuple[float, int, int, int]:
        start_time = time.time()
        board = state.get_board_state()
        empty_positions = np.argwhere(board == 0)
        if len(empty_positions) == 0:
            raise ValueError("No more moves possible.")
        move = empty_positions[np.random.choice(len(empty_positions))]
        row, col = move
        value = -1
        if self.is_agent_x:
            value = 1
        end_time = time.time()
        time_taken = end_time - start_time
        return (time_taken, row, col, value)