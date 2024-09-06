import pandas as pd
import os

class Metrics:
    @staticmethod
    def save_record(board_size: int, game_time: int, agent_x_name: str, agent_o_name: str, agent_x_time: int,
                    agent_o_time: int, winner: str):
        r"""

        Args:
            board_size: The gird size as board_size x board_size square
            game_time: Time taken to complete the game
            agent_x_name: Name of Agent X that mentions the algorithm used
            agent_o_name: Name of Agent O that mentions the algorithm used
            agent_x_time: Average time taken by Agent X for each move
            agent_o_time: Average time taken by Agent O for each move
            winner: Agent name that has won the game

        Returns:

        """
        record = {
                    'board_size': board_size,
                    'game_time': game_time,
                    'agent_x_name': agent_x_name,
                    'agent_o_name': agent_o_name,
                    'agent_x_time': agent_x_time,
                    'agent_o_time': agent_o_time,
                    'winner': winner
                }
        filename = 'game_metrics.csv'
        if os.path.exists(filename):
            df = pd.read_csv(filename)
            df = pd.concat([df, pd.DataFrame([record])], ignore_index=True)

        else:
            df = pd.DataFrame([record])

        df.to_csv(filename, index=False)
