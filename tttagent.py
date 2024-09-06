class TicTacToeAgent:
    r"""Base class for all Tic Tac Toe Agents.

    Args:
        name: a name of type str
    """
    def __init__(self, name: str, boardsize:int):
        self.name = name
        self.boardsize = boardsize

    def initialize(self):
        pass

    def get_next_move(self, state) -> tuple[float, int, int, int]:
        pass
