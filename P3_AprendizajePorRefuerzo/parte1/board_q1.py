import numpy as np


class board_q1():
    def __init__(self, initState, boardInit):
        self.listName = ['Y','L','O','F'] # Yo, Libre, Ocupado, Final
        self.listSuccessorStates = []
        self.listNextStates = []

        self.board = boardInit.copy()

        self.boardCopy = boardInit.copy()

        self.board[0, 3] = self.listName[3]
        self.board[1, 1] = self.listName[2]
        self.board[initState[0], initState[1]] = self.listName[0]
        self.currentState = initState


    def getListNextStates(self, mypiece):

        self.listNextStates = []

        if str(self.board[mypiece[0]][mypiece[1]]) == 'Y':
            listPotencialNextStates = [
                [mypiece[0] + 1, mypiece[1]], [mypiece[0] - 1, mypiece[1]],
                [mypiece[0], mypiece[1] - 1], [mypiece[0], mypiece[1] + 1]
            ]
            for coord in listPotencialNextStates:
                if -1 < coord[0] < 3 and -1 < coord[1] < 4:
                    if not self.board[coord[0]][coord[1]] == self.listName[2]: self.listNextStates.append(coord)

    def isEndPosition(self, mypiece):
        return self.board[mypiece[0]][mypiece[1]] == self.listName[3]

    def print_board(self):
        print("*" * 15)
        for row in self.board:
            row_str = "|"
            for cell in row:
                if cell == -1:
                    row_str += "   |"  # Celda vacía
                else:
                    row_str += f" {cell} |"  # Mostrar contenido de la celda
            print(row_str)
        print("*" * 15)

    def move(self, currentState, action, actions, drunked = False):
        dict_move = {
            'T': [-1, 0],
            'B': [1, 0],
            'L': [0, -1],
            'R': [0, 1],
        }

        act = actions[action]
        if drunked and np.random.random() > 0.99:
            del dict_move[act]
            act = np.random.choice(list(dict_move.keys()))
            action = actions.index(act)

        move_values = dict_move[act]
        nextState = [currentState[0] + move_values[0], currentState[1] + move_values[1]]

        if 0 <= nextState[0] < self.board.shape[0] and 0 <= nextState[1] < self.board.shape[1]:
            if not self.board[nextState[0], nextState[1]] == self.listName[2]:
                self.board[currentState[0], currentState[1]] = self.boardCopy[
                    currentState[0], currentState[1]]
                if not self.board[nextState[0], nextState[1]] == self.listName[3]: self.board[nextState[0], nextState[1]] = self.listName[0]
                self.currentState = nextState
                return nextState, action

        return currentState, action





