import time
import numpy as np
from copy import deepcopy
from tttagent import TicTacToeAgent
from tttstate import State


class AlphaBetaPruning(TicTacToeAgent):
    def __init__(self, name: str, boardsize: int, is_agent_x: bool):
        self.name = name
        self.boardsize = boardsize
        self.is_agent_x = is_agent_x

    def initialize(self):
        pass

    def get_next_move(self, state) -> tuple[float, int, int, int]:
        start_time = time.perf_counter()
        _, best_move = self.minimax(state, -np.inf, np.inf, True, 2, 1 if self.is_agent_x else -1,
                                    -1 if self.is_agent_x else 1)
        end_time = time.perf_counter()
        move_time = end_time - start_time
        return (move_time, best_move[0], best_move[1], 1 if self.is_agent_x else -1)

    def minimax(self, state, alpha, beta, maximizing, depth, maxp, minp):
        if depth == 0 or state.is_goal_state()[0]:
            return self.utility_of_state(state), None

        rows_left, columns_left = np.where(state.get_board_state() == 0)
        if rows_left.shape[0] == 0:
            return self.utility_of_state(state), None

        if maximizing:
            max_eval = -np.inf
            best_move = None
            for i in range(rows_left.shape[0]):
                next_state = deepcopy(state)
                next_state.update_board_state(rows_left[i], columns_left[i], maxp)
                eval, _ = self.minimax(next_state, alpha, beta, False, depth - 1, maxp, minp)
                if eval > max_eval:
                    max_eval = eval
                    best_move = (rows_left[i], columns_left[i])
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break
            return max_eval, best_move
        else:
            min_eval = np.inf
            best_move = None
            for i in range(rows_left.shape[0]):
                next_state = deepcopy(state)
                next_state.update_board_state(rows_left[i], columns_left[i], minp)
                eval, _ = self.minimax(next_state, alpha, beta, True, depth - 1, maxp, minp)
                if eval < min_eval:
                    min_eval = eval
                    best_move = (rows_left[i], columns_left[i])
                beta = min(beta, eval)
                if beta <= alpha:
                    break
            return min_eval, best_move

    def utility_of_state(self, state):
        state_copy = state.get_board_state().flatten()
        heuristic = 0
        for line in self.winning_lines():
            maxp = minp = 0
            for index in line:
                if state_copy[index] == 1:
                    maxp += 1
                elif state_copy[index] == -1:
                    minp += 1
            heuristic += self.heuristic_table()[maxp][minp]
        return heuristic

    def winning_lines(self):
        return [
            [0, 1, 2, 3, 4],
            [5, 6, 7, 8, 9],
            [10, 11, 12, 13, 14],
            [15, 16, 17, 18, 19],
            [20, 21, 22, 23, 24],
            [0, 5, 10, 15, 20],
            [1, 6, 11, 16, 21],
            [2, 7, 12, 17, 22],
            [3, 8, 13, 18, 23],
            [4, 9, 14, 19, 24],
            [0, 6, 12, 18, 24],
            [4, 8, 12, 16, 20]
        ]

    def heuristic_table(self):
        heuristic_table = np.zeros((6, 6))
        for index in range(6):
            heuristic_table[index][0] = 10 ** index
            heuristic_table[0][index] = -10 ** index
        return heuristic_table


class AStarAgent(TicTacToeAgent):
    def __init__(self, name: str, boardsize: int, is_agent_x: bool):
        self.name = name
        self.boardsize = boardsize
        self.is_agent_x = is_agent_x

    def initialize(self):
        pass

    def get_next_move(self, state) -> tuple[float, int, int, int]:
        start_time = time.perf_counter()
        best_move = self.a_star_search(state)
        end_time = time.perf_counter()
        move_time = end_time - start_time
        return (move_time,) + best_move

    def a_star_search(self, state):
        def heuristic(state):
            # Simple heuristic: count the difference between the number of 'X' and 'O'
            return np.sum(state.get_board_state())

        frontier = []
        heapq.heappush(frontier, (0, state))
        came_from = {}
        cost_so_far = {}
        came_from[state] = None
        cost_so_far[state] = 0

        while frontier:
            _, current = heapq.heappop(frontier)

            if current.is_goal_state()[0]:
                break

            for move in self.get_legal_moves(current):
                new_state = self.result(current, move)
                new_cost = cost_so_far[current] + 1
                if new_state not in cost_so_far or new_cost < cost_so_far[new_state]:
                    cost_so_far[new_state] = new_cost
                    priority = new_cost + heuristic(new_state)
                    heapq.heappush(frontier, (priority, new_state))
                    came_from[new_state] = (current, move)

        # Reconstruct the path to the goal
        current = state
        while came_from[current][0] != state:
            current = came_from[current][0]
        return came_from[current][1]

    def get_legal_moves(self, state):
        return list(zip(*np.where(state.get_board_state() == 0)))

    def result(self, state, move):
        new_state = State(self.boardsize)
        new_state.board = state.board.copy()
        new_state.update_board_state(move[0], move[1], 1 if self.is_agent_x else -1)
        return new_state
