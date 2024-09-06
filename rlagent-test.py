import time
import matplotlib.pyplot as plt
from collections import defaultdict
from tttstate import State
from rlagent import RLAgent
from randomagent import RandomAgent
from asagent import AlphaBetaPruning
from tttmetrics import Metrics

def main():
    board_size = 5
    agent_o = RLAgent('RLAgent', board_size, False)
    agent_x = AlphaBetaPruning('AlphaBetaAgent', board_size, True)
    agent_x.initialize()
    agent_o.initialize()
    n_games = 100
    win_stats = defaultdict(lambda: 0)
    win_stats['Draw'] = 0
    for game in range(n_games):
        print(f"Game: {game}")
        state = State(board_size)
        start_time = time.time()
        agent_x_time = []
        agent_o_time = []
        is_goal_state, winning_sum = State.is_terminal_state(state.get_board_state(), board_size)
        #state.print_board_state()
        while is_goal_state is False:
            #print('Agent X')
            exec_time_x, row, column, value = agent_x.get_next_move(state)
            agent_x_time.append(exec_time_x)
            state.update_board_state(row, column, value)
            is_goal_state, winning_sum = State.is_terminal_state(state.get_board_state(), board_size)
            #state.print_board_state()
            if (is_goal_state):
                winner_name = get_winner_name(agent_x.is_agent_x, agent_x.name, agent_o.name, winning_sum, board_size)
                win_stats[winner_name] += 1
                break
            #print('Agent O')
            exec_time_o, row, column, value = agent_o.get_next_move(state)
            agent_o_time.append(exec_time_o)
            state.update_board_state(row, column, value)
            is_goal_state, winning_sum = State.is_terminal_state(state.get_board_state(), board_size)
            #state.print_board_state()
            if (is_goal_state):
                winner_name = get_winner_name(agent_x.is_agent_x, agent_x.name, agent_o.name, winning_sum, board_size)
                win_stats[winner_name] += 1
                break
        end_time = time.time()
        game_time = end_time - start_time
        avg_agent_x_time = sum(agent_x_time) / len(agent_x_time)
        avg_agent_o_time = sum(agent_o_time) / len(agent_o_time)

        Metrics.save_record(board_size, game_time, agent_x.name, agent_o.name, avg_agent_x_time, avg_agent_o_time,
                             winner_name)
    print(win_stats)
    fig, ax = plt.subplots()
    categories = list(win_stats.keys())
    counts = list(win_stats.values())
    bar_labels = categories
    ax.bar(categories, counts, label=bar_labels)
    ax.set_ylabel('Count')
    ax.set_title('Tic Tac Toe Result for '+ str(n_games)+' game between '+ agent_x.name + ' and '+ agent_o.name)
    plt.savefig(agent_x.name+'_'+agent_o.name+'_'+str(board_size)+'_'+str(n_games)+'.png')

def get_winner_name(is_agent_x, agent_x_name, agent_o_name, winning_sum, board_size):
    if is_agent_x:
        winner_name = "Draw" if winning_sum == 0 else agent_x_name if winning_sum == board_size else agent_o_name
    else:
        winner_name = "Draw" if winning_sum == 0 else agent_o_name if winning_sum == board_size else agent_x_name

    return winner_name

if __name__ == '__main__':
    main()
