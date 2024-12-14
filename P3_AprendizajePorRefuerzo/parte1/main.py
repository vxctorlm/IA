import time

import board_q1

import numpy as np
import sys

class problem():

    def __init__(self, initState, initBoard):
        self.initialBoard = initBoard
        self.board_q1 = board_q1.board_q1(initState, initBoard)

        self.listNextStates = []
        self.listVisitedStates = []
        self.currentState = self.board_q1.currentState

    def getCurrentState(self):
        return self.currentState

    def getListNextStates(self, mypiece):
        self.board_q1.getListNextStates(mypiece)
        self.listNextStates = self.board_q1.listNextStates.copy()
        return self.listNextStates

    def isEnd(self, mypiece):
        return self.board_q1.isEndPosition(mypiece)

    def qlearning(self, alpha, gamma, epsilon, actions = ['T', 'B', 'L', 'R']):
        startState = self.currentState
        Q_table = np.zeros((12, 4)) # 3 rows x 4 columns = 12 , 4 actions
        Q_table_old = Q_table.copy()
        reward = 0

        max_iterations = 150
        number_of_iterations = 0
        end = False

        while not end:

            Q_table_old = Q_table.copy()
            currentState = startState
            self.board_q1 = board_q1.board_q1(startState,self.initialBoard)



            while not self.isEnd(currentState):
                idx = currentState[0] * self.board_q1.board.shape[1] + currentState[1] # i * columns numbers + j
                randomValue = np.random.random()
                if randomValue < epsilon: action = np.random.randint(0, len(actions)) # Exploración
                else: action = np.argmax(Q_table[idx]) # Explotación

                nextState, action = self.board_q1.move(currentState, action, actions, drunked = True)
                if nextState == currentState: continue

                next_idx = nextState[0] * self.board_q1.board.shape[1] + nextState[1]

                Q_table[idx, action] += alpha * (
                        self.board_q1.boardCopy[nextState[0], nextState[1]]
                        + gamma * np.max(Q_table[next_idx])
                        - Q_table[idx, action]
                )

                currentState = nextState

            if np.abs(Q_table - Q_table_old).mean() < 1e-4:
                end = True

            number_of_iterations += 1



        return Q_table, reward

    def reconstructPath(self, currentState, qTable):
        end = False
        acciones = [(-1,0), (1,0), (0,-1), (0,1)]
        path = []
        while not end:
            path.append(currentState)
            idx_current = currentState[0] * self.board_q1.board.shape[1] + currentState[1]
            idx_max = np.argmax(qTable[idx_current])

            if qTable[idx_current][idx_max] == 0:
                end = True

            currentState = (acciones[idx_max][0] + currentState[0], acciones[idx_max][1] + currentState[1])

        return path



    def print_Q_table(self, Q_table, actions):
        print("Q-learning table: [Top, Bottom, Left, Right]")
        for row in range(self.board_q1.board.shape[0]):
            for col in range(self.board_q1.board.shape[1]):
                state_idx = row * self.board_q1.board.shape[1] + col
                action_values = [f"{Q_table[state_idx, a]:.2f}" for a in range(len(actions))]
                print(f"({row},{col}): {action_values}")

if __name__ == "__main__":

    if len(sys.argv) < 1:
        sys.exit(1)

    initBoard1 = np.full((3, 4), -1, dtype=object)
    initBoard1[0,3] = 100

    initBoard2 = np.array([
        [-3,-2, -1, 100],
        [-4, -1, -2, -1],
        [-5, -4, -3, -2]
    ]).astype(object)

    problem = problem([2,0], initBoard2)
    alpha = 0.1
    gamma = 0.9
    epsilon = 0.1
    actions = ['T', 'B', 'L', 'R']

    start_time = time.time()
    q_table, reward = problem.qlearning(alpha, gamma, epsilon, actions)
    elapsed_time = time.time() - start_time
    print(f"El algoritmo ha tardado {elapsed_time}")
    problem.print_Q_table(q_table, actions)
    print(f"{problem.reconstructPath([2,0], q_table)}")





