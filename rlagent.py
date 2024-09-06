import time
import numpy as np
import pickle
from tttagent import TicTacToeAgent
from tttstate import State
from collections import defaultdict
from copy import deepcopy

class RLAgent(TicTacToeAgent):
    def __init__(self, name: str, boardsize: int, is_agent_x: bool):
        self.name = name
        self.boardsize = boardsize
        self.is_agent_x = is_agent_x
        self.actions = np.arange(0, self.boardsize ** 2)
        self.Q = defaultdict(lambda: np.zeros(self.boardsize**2))
        if self.is_agent_x:
            self.player1 = self.name
            self.player2 = 'Other'
        else:
            self.player2 = self.name
            self.player1 = 'Other'
        self.current_player = self.player1
        self.other_player = self.player2
        self.epsilon = 0.2
        self.gamma = 0.90
        self.alpha = 0.3

    """
    This method will initialize the learning process
    """
    def initialize(self):
        filename = 'rl-model-' + str(self.boardsize)
        # initialize the state from file if available
        try:
            print(f"Loading model for {self.name}")
            self.Q = pickle.load(open(filename + '.pkl', "rb"))
        except (OSError, IOError) as e:
            print(f"Starting learning for {self.name}")
            board_state = self.reset()
            self.learn(board_state)
            qtable = {}
            for key in self.Q:
                qtable[key] = self.Q[key]
            with open(filename + '.pkl', 'wb') as fhandle:
                pickle.dump(qtable, fhandle, protocol=pickle.HIGHEST_PROTOCOL)

    def reset(self):
        if self.is_agent_x:
            self.player1 = self.name
            self.player2 = 'Other'
        else:
            self.player2 = self.name
            self.player1 = 'Other'
        self.current_player = self.player1
        self.other_player = self.player2
        self.first_move_played = False
        if self.is_agent_x:
            self.rl_agent_turn = True
        else:
            self.rl_agent_turn = False
        board_state = State(self.boardsize)
        return board_state

    def get_state_key(self, state):
        board_state = state.get_board_state()
        return tuple(board_state.reshape(self.boardsize ** 2))

    def find_action_index(self, row, col, value):
        actions = np.zeros((self.boardsize, self.boardsize))
        actions[row, col] = value
        return np.argmax(np.ravel((actions)))

    def find_action(self, index):
        '''
        Finds the row, column and value given an index
        Returns:
        '''
        actions = np.ravel(np.zeros((self.boardsize, self.boardsize)))
        actions[index] = 10
        actions = actions.reshape((self.boardsize, self.boardsize))
        row, col = np.argwhere(actions == np.max(actions)).flatten()
        if self.is_agent_x:
            value = 1
        else:
            value = -1
        return row, col, value

    def available_moves(self, state):
        empty_positions = np.argwhere(state.get_board_state() == 0)
        moves = np.zeros(len(empty_positions), dtype=int)
        i = 0
        for rows, columns in empty_positions:
            moves[i] = rows * self.boardsize + columns
            i += 1
        return moves


    def get_random_move(self, board_state, instantiated=False):
        if not instantiated:
            state = board_state.get_board_state()
        else:
            state = board_state
        empty_positions = np.argwhere(state == 0)
        if len(empty_positions) == 0:
            raise ValueError("No more moves possible.")
        move = empty_positions[np.random.choice(len(empty_positions))]
        row, col = move
        value = -1
        if self.is_agent_x:
            value = 1
        return row, col, value

    def get_rl_move(self, state):
        # random or best depending on epsilon
        if self.current_player == self.player1:
            value =1
        else:
            value = -1

        if np.random.uniform(0, 1) < self.epsilon:
            row, col, _ = self.get_random_move(state)
            index = self.find_action_index(row, col, value)
        else:
            index, row, col, _ = self.get_best_move(state)
        return index, row, col, value

    def get_best_move(self, state):
        index = 0
        state_key = self.get_state_key(state)
        moves = self.available_moves(state)
        if state_key not in self.Q:
            for move in moves:
                self.Q[state_key][move] = 0.0
        index, row, col, value = self.get_q_move(state_key, moves)
        return index, row, col, value

    def get_q_move(self, state_key, moves):
        index = 0
        ref_q_value = 0
        if self.current_player == self.player1:
            ref_q_value = np.max(self.Q[state_key][moves])
        else:
            ref_q_value = np.min(self.Q[state_key][moves])

        if list(self.Q[state_key][moves]).count(ref_q_value) > 1:

            best_moves = [move for move in moves if self.Q[state_key][move] == ref_q_value]
            index = best_moves[np.random.choice(len(best_moves))]
        else:
            if self.current_player==self.player1:
                index =  moves[np.argmax(self.Q[state_key][moves])]
            else:
                index = moves[np.argmin(self.Q[state_key][moves])]
        row, col, value = self.find_action(index)
        return index, row, col, value

    def switch_players(self):
        if self.current_player == self.player1:
            self.current_player = self.player2
            self.other_player = self.player1
        else:
            self.current_player = self.player1
            self.other_player = self.player2

    def learn(self,board_state):
        n_episodes = 10000
        for i in range(n_episodes):
            print(f"Episode {i}")
            current_action_index = 0
            board_state = State(self.boardsize)
            is_terminal_state, winner = State.is_terminal_state(board_state.get_board_state(), self.boardsize)

            while is_terminal_state is False:
                current_action_index, row, column, value = self.get_rl_move(board_state)
                current_state_key = self.get_state_key(board_state)
                next_board_state = deepcopy(board_state)
                next_board_state.update_board_state(row, column, value)
                next_state_key = self.get_state_key(next_board_state)
                reward = self.get_reward(next_board_state)
                next_board_over, winner = State.is_terminal_state(next_board_state.get_board_state(), self.boardsize)
                if next_board_over:
                    expected = reward
                else:
                    next_Qs = self.Q[next_state_key]
                    if self.current_player == self.player1:
                        expected = reward + (self.gamma * min(next_Qs))
                    elif self.current_player == self.player2:
                        expected = reward + (self.gamma * max(next_Qs))
                change = self.alpha * (expected - self.Q[current_state_key][current_action_index])
                self.Q[current_state_key][current_action_index] += change
                board_state = deepcopy(next_board_state)
                #board_state.print_board_state()
                self.switch_players()
                is_terminal_state, winner = State.is_terminal_state(board_state.get_board_state(), self.boardsize)

    def get_reward(self, istate):
        boardsize = self.boardsize
        is_terminal_state, winner = State.is_terminal_state(istate.get_board_state(), boardsize)
        if is_terminal_state:
            if winner == self.boardsize:
                return 1
            elif winner == -self.boardsize:
                return -1
            else:
                return 0.5
        else:
            return 0.0

    def get_next_move(self, state) -> tuple[float, int, int, int]:
        start_time = time.perf_counter()
        state_key = self.get_state_key(state)
        if state_key in self.Q:
            index, row, column, value = self.get_best_move(state)
        else:
            row, column, value = self.get_random_move(state)
        end_time = time.perf_counter()
        execution_time = end_time - start_time
        return tuple([execution_time, row, column, value])





