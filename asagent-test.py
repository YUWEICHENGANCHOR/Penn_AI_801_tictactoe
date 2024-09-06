import time
from tttstate import State
from asagent import AlphaBetaPruning
from rlagent import RLAgent
from randomagent import RandomAgent
from tttmetrics import Metrics


def main():
    board_size = 5
    print('Initializing ... test')
    # initialize the state
    state = State(board_size)
    # initialize the agents
    agent_x = AlphaBetaPruning('alpha_beta', board_size, True)
    agent_o = RandomAgent('rlagent', board_size, False)
    start_time = time.time()
    agent_x.initialize()
    agent_o.initialize()
    # Initialize lists to store execution times
    agent_x_time = []
    agent_o_time = []
    is_goal_state, winner = state.is_goal_state()
    state.print_board_state()

    while is_goal_state is False:
        exec_time_x, row, column, value = agent_x.get_next_move(state)
        agent_x_time.append(exec_time_x)
        state.update_board_state(row, column, value)
        is_goal_state, winner = state.is_goal_state()
        state.print_board_state()
        if (is_goal_state):
            break
        exec_time_o, row, column, value = agent_o.get_next_move(state)
        agent_o_time.append(exec_time_o)
        state.update_board_state(row, column, value)
        is_goal_state, winner = state.is_goal_state()
        state.print_board_state()
        if (is_goal_state):
            break
    end_time = time.time()
    game_time = end_time - start_time
    avg_agent_x_time = sum(agent_x_time) / len(agent_x_time)
    avg_agent_o_time = sum(agent_o_time) / len(agent_o_time)
    winner_name = "Draw" if winner == 0 else "Agent X" if winner == board_size else "Agent O"

    Metrics.save_record(board_size, game_time, agent_x.name, agent_o.name, avg_agent_x_time, avg_agent_o_time,
                        winner_name)
    print(f"Game time: {game_time}")
    print(f"Winner: {winner_name}")


if __name__ == '__main__':
    main()
