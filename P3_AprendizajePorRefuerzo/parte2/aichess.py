#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  8 11:22:03 2022

@author: ignasi
"""
import queue

import chess
import numpy as np
import sys

from itertools import permutations

import random
class Aichess():

    """
    A class to represent the game of chess.

    ...

     Attributes:
    -----------
    chess : Chess
        Represents the current state of the chess game.
    listNextStates : list
        A list to store the next possible states in the game.
    listVisitedStates : list
        A list to store visited states during search.
    pathToTarget : list
        A list to store the path to the target state.
    currentStateW : list
        Represents the current state of the white pieces.
    depthMax : int
        The maximum depth to search in the game tree.
    checkMate : bool
        A boolean indicating whether the game is in a checkmate state.
    dictVisitedStates : dict
        A dictionary to keep track of visited states and their depths.
    dictPath : dict
        A dictionary to reconstruct the path during search.

    Methods:
    --------
    getCurrentState() -> list
        Returns the current state for the whites.

    getListNextStatesW(myState) -> list
        Retrieves a list of possible next states for the white pieces.

    isSameState(a, b) -> bool
        Checks whether two states are the same.

    isVisited(mystate) -> bool
        Checks if a state has been visited.

    isCheckMate(mystate) -> bool
        Checks if a state represents a checkmate.

    DepthFirstSearch(currentState, depth) -> bool
        Depth-first search algorithm.

    worthExploring(state, depth) -> bool
        Checks if a state is worth exploring during search using the optimised DFS algorithm.

    DepthFirstSearchOptimized(currentState, depth) -> bool
        Optimized depth-first search algorithm.

    reconstructPath(state, depth) -> None
        Reconstructs the path to the target state. Updates pathToTarget attribute.

    canviarEstat(start, to) -> None
        Moves a piece from one state to another.

    movePieces(start, depthStart, to, depthTo) -> None
        Moves all pieces between states.

    BreadthFirstSearch(currentState, depth) -> None
        Breadth-first search algorithm.

    h(state) -> int
        Calculates a heuristic value for a state using Manhattan distance.

    AStarSearch(currentState) 
        A* search algorithm -> To be implemented by you

    translate(s) -> tuple
        Translates traditional chess coordinates to list indices.

    """

    def __init__(self, TA, myinit=True, actions = []):

        if myinit:
            self.chess = chess.Chess(TA, True)
        else:
            self.chess = chess.Chess([], False)

        self.listNextStates = []
        self.listVisitedStates = []
        self.pathToTarget = []
        self.currentStateW = self.chess.boardSim.currentStateW;
        self.depthMax = 8
        self.checkMate = False

        # Prepare a dictionary to control the visited state and at which
        # depth they were found
        self.dictVisitedStates = {}
        # Dictionary to reconstruct the BFS path
        self.dictPath = {}

        self.qTable = {} # 64 posiciones possibles por pieza (4096 acciones 2 Piezas), 8 + 4 acciones possibles
        self.listActions = actions


    def getCurrentState(self):
    
        return self.myCurrentStateW

    def getListNextStatesW(self, myState):

        self.chess.boardSim.getListNextStatesW(myState)
        self.listNextStates = self.chess.boardSim.listNextStates.copy()

        return self.listNextStates

    def isSameState(self, a, b):

        isSameState1 = True
        # a and b are lists
        for k in range(len(a)):

            if a[k] not in b:
                isSameState1 = False

        isSameState2 = True
        # a and b are lists
        for k in range(len(b)):

            if b[k] not in a:
                isSameState2 = False

        isSameState = isSameState1 and isSameState2
        return isSameState

    def isVisited(self, mystate):

        if (len(self.listVisitedStates) > 0):
            perm_state = list(permutations(mystate))

            isVisited = False
            for j in range(len(perm_state)):

                for k in range(len(self.listVisitedStates)):

                    if self.isSameState(list(perm_state[j]), self.listVisitedStates[k]):
                        isVisited = True

            return isVisited
        else:
            return False


    def isCheckMate(self, mystate):
        
        # list of possible check mate states
        listCheckMateStates = [[[0,0,2],[2,4,6]],[[0,1,2],[2,4,6]],[[0,2,2],[2,4,6]],[[0,6,2],[2,4,6]],[[0,7,2],[2,4,6]]]

        # Check all state permuations and if they coincide with a list of CheckMates
        for permState in list(permutations(mystate)):
            if list(permState) in listCheckMateStates:
                return True

        return False

    def DepthFirstSearch(self, currentState, depth):
        
        # We visited the node, therefore we add it to the list
        # In DF, when we add a node to the list of visited, and when we have
        # visited all noes, we eliminate it from the list of visited ones
        self.listVisitedStates.append(currentState)

        # is it checkmate?
        if self.isCheckMate(currentState):
            self.pathToTarget.append(currentState)
            return True
        
        if depth + 1 <= self.depthMax:
            for son in self.getListNextStatesW(currentState):
                
                if not self.isVisited(son):
                    # in the state son, the first piece is the one just moved
                    # We check the position of currentState
                    # matched by the piece moved
                    if son[0][2] == currentState[0][2]:
                        fitxaMoguda = 0
                    else:
                        fitxaMoguda = 1

                    # we move the piece to the new position
                    self.chess.moveSim(currentState[fitxaMoguda],son[0])
                    # We call again the method with the son, 
                    # increasing depth
                    if self.DepthFirstSearch(son,depth+1):
                        #If the method returns True, this means that there has
                        # been a checkmate
                        # We ad the state to the list pathToTarget
                        self.pathToTarget.insert(0,currentState)
                        return True
                    # we reset the board to the previous state
                    self.chess.moveSim(son[0],currentState[fitxaMoguda])

        # We eliminate the node from the list of visited nodes
        # since we explored all successors
        self.listVisitedStates.remove(currentState)

    def worthExploring(self, state, depth):
        
        # First of all, we check that the depth is bigger than depthMax
        if depth > self.depthMax: return False
        visited = False
        # check if the state has been visited
        for perm in list(permutations(state)):
            permStr = str(perm)
            if permStr in list(self.dictVisitedStates.keys()):
                visited = True
                # If there state has been visited at a epth bigger than 
                # the current one, we are interestted in visiting it again
                if depth < self.dictVisitedStates[perm]:
                    # We update the depth associated to the state
                    self.dictVisitedStates[permStr] = depth
                    return True
        # Whenever not visited, we add it to the dictionary 
        # at the current depth
        if not visited:
            permStr = str(state)
            self.dictVisitedStates[permStr] = depth
            return True

    def DepthFirstSearchOptimized(self, currentState, depth):
        # is it checkmate?
        if self.isCheckMate(currentState):
            self.pathToTarget.append(currentState)
            return True

        for son in self.getListNextStatesW(currentState):
            if self.worthExploring(son,depth+1):
                
                # in state 'son', the first piece is the one just moved
                # we check which position of currentstate matche
                # the piece just moved
                if son[0][2] == currentState[0][2]:
                    fitxaMoguda = 0
                else:
                    fitxaMoguda = 1

                # we move the piece to the novel position
                self.chess.moveSim(currentState[fitxaMoguda], son[0])
                # we call the method with the son again, increasing depth
                if self.DepthFirstSearchOptimized(son, depth + 1):
                    # If the method returns true, this means there was a checkmate
                    # we add the state to the list pathToTarget
                    self.pathToTarget.insert(0, currentState)
                    return True
                # we return the board to its previous state
                self.chess.moveSim(son[0], currentState[fitxaMoguda])

    def canviarEstat(self, start, to):
        # We check which piece has been moved from one state to the next
        if start[0] == to[0]:
            fitxaMogudaStart=1
            fitxaMogudaTo = 1
        elif start[0] == to[1]:
            fitxaMogudaStart = 1
            fitxaMogudaTo = 0
        elif start[1] == to[0]:
            fitxaMogudaStart = 0
            fitxaMogudaTo = 1
        else:
            fitxaMogudaStart = 0
            fitxaMogudaTo = 0
        # move the piece changed
        self.chess.moveSim(start[fitxaMogudaStart], to[fitxaMogudaTo])

    def movePieces(self, start, depthStart, to, depthTo):
        
        # To move from one state to the next for BFS we will need to find
        # the state in common, and then move until the node 'to'
        moveList = []
        # We want that the depths are equal to find a common ancestor
        nodeTo = to
        nodeStart = start
        # if the depth of the node To is larger than that of start, 
        # we pick the ancesters of the node until being at the same
        # depth
        while(depthTo > depthStart):
            moveList.insert(0,to)
            nodeTo = self.dictPath[str(nodeTo)][0]
            depthTo-=1
        # Analogous to the previous case, but we trace back the ancestors
        #until the node 'start'
        while(depthStart > depthTo):
            ancestreStart = self.dictPath[str(nodeStart)][0]
            # We move the piece the the parerent state of nodeStart
            self.canviarEstat(nodeStart, ancestreStart)
            nodeStart = ancestreStart
            depthStart -= 1

        moveList.insert(0,nodeTo)
        # We seek for common node
        while nodeStart != nodeTo:
            ancestreStart = self.dictPath[str(nodeStart)][0]
            # Move the piece the the parerent state of nodeStart
            self.canviarEstat(nodeStart,ancestreStart)
            # pick the parent of nodeTo
            nodeTo = self.dictPath[str(nodeTo)][0]
            # store in the list
            moveList.insert(0,nodeTo)
            nodeStart = ancestreStart
        # Move the pieces from the node in common
        # until the node 'to'
        for i in range(len(moveList)):
            if i < len(moveList) - 1:
                self.canviarEstat(moveList[i],moveList[i+1])


    def BreadthFirstSearch(self, currentState, depth):
        """
        Check mate from currentStateW
        """
        BFSQueue = queue.Queue()
        # The node root has no parent, thus we add None, and -1, which would be the depth of the 'parent node'
        self.dictPath[str(currentState)] = (None, -1)
        depthCurrentState = 0
        BFSQueue.put(currentState)
        self.listVisitedStates.append(currentState)
        # iterate until there is no more candidate nodes
        while BFSQueue.qsize() > 0:
            # Find the optimal configuration
            
            node = BFSQueue.get()
            depthNode = self.dictPath[str(node)][1] + 1
            if depthNode > self.depthMax:
                break
            # If it not the root node, we move the pieces from the previous to the current state
            if depthNode > 0:
                self.movePieces(currentState, depthCurrentState, node, depthNode)

            if self.isCheckMate(node):
                # Si és checkmate, construïm el camí que hem trobat més òptim
                self.reconstructPath(node, depthNode)
                break

            for son in self.getListNextStatesW(node):
                if not self.isVisited(son):
                    self.listVisitedStates.append(son)
                    BFSQueue.put(son)
                    self.dictPath[str(son)] = (node, depthNode)
            currentState = node
            depthCurrentState = depthNode


    def h(self,state):
        
        if state[0][2] == 2:
            posicioRei = state[1]
            posicioTorre = state[0]
        else:
            posicioRei = state[0]
            posicioTorre = state[1]
        # With the king we wish to reach configuration (2,4), calculate Manhattan distance
        fila = abs(posicioRei[0] - 2)
        columna = abs(posicioRei[1]-4)
        # Pick the minimum for the row and column, this is when the king has to move in diagonal
        # We calculate the difference between row an colum, to calculate the remaining movements
        # which it shoudl go going straight        
        hRei = min(fila, columna) + abs(fila-columna)
        # with the tower we have 3 different cases
        if posicioTorre[0] == 0 and (posicioTorre[1] < 3 or posicioTorre[1] > 5):
            hTorre = 0
        elif posicioTorre[0] != 0 and posicioTorre[1] >= 3 and posicioTorre[1] <= 5:
            hTorre = 2
        else:
            hTorre = 1
        # In our case, the heuristics is the real cost of movements
        return hRei + hTorre

     
    def AStarSearch(self, currentState):
        
        # Utilizamos el dictPath para obtener el g(n) de los nodos (g(n)=depth)
        self.dictPath[str(currentState)] = (None, 0) 
        
        # Inicializamos la Frontera
        frontera = []
        self.listVisitedStates.append(currentState) # Añadimos en alcanzados currentState

        gCurrentState = 0 # Inicializamos el coste g(n) del currentState
        # Añadimos en la frontera el currentState y su f(n) = g(n) + h(n) 
        frontera.append((self.h(currentState),currentState)) 
        
        # Mientras la frontera no este vacia
        while len(frontera) > 0:
            # Expandimos nodo con f(n) mínimo de la frontera
            fn, nodo = min(frontera, key=lambda x: x[0]) 
            frontera.remove((fn, nodo)) # Eliminamos el nodo de la frontera

            # Sacamos el valor g(n) de los sucesores del nodo
            gnode = self.dictPath[str(nodo)][1]
            # Si g(n) > 0 -> no es el estado inicial (root)
            if gnode > 0: 
                 # Movemos las piezas desde el currentState al nodo
                 self.movePieces(currentState, gCurrentState, nodo, gnode)

            # Comprovamos si el nodo realiza jaque mate
            if self.isCheckMate(nodo):
                # Imprimimos la profundidad para obtener el estado objetivo
                print(f"Profundidad mínima para obtener estado objetivo {str(nodo)}: {gnode}")	
                # Reconstruimos el camino
                self.reconstructPath(nodo, gnode)
                return None
                

            # Expandimos los hijos/sucesores del nodo minimo 
            for son in self.getListNextStatesW(nodo):
                # Obtenemos h(n) del hijo/sucesores
                hson = self.h(son)
                gson = gnode +1
                # Si el hijo no esta en alcanzado o si su camino es mas eficiente
                if (not self.isVisited(son) and str(son) not in self.dictPath) or (str(son) in self.dictPath and gson < self.dictPath[str(son)][1]):
                    # Guardamos el alcanzado el hijo 
                    self.listVisitedStates.append(son)
                    
                    # Guardamos su g(n)
                    self.dictPath[str(son)] = (nodo, gson)

                    # Añadimos el hijo a la frontera con su f(n)
                    frontera.append(((hson + gson), son))
            
            # Guardamos el nodo, en currentState para la próxima iteración
            currentState = nodo
            # Guardamos g(n) del nodo, en gCurrentState para la próxima iteración
            gCurrentState = gnode
        
        return None

    def getPieceState(self, state, pieceType):
        for piece in state:
            if piece[2] == pieceType:
                return piece
        return None  # Retorna None si no encuentra la pieza

    def stateToString(self, whiteState):
        """
        Convert the white pieces' state to a string representation.

        Input:
        - whiteState (list): List representing the state of white pieces.

        Returns:
        - stringState (str): String representation of the white pieces' state.
        """
        wkState = self.getPieceState(whiteState, 6)
        wrState = self.getPieceState(whiteState, 2)
        stringState = str(wkState[0]) + "," + str(wkState[1]) + ","
        if wrState is not None:
            stringState += str(wrState[0]) + "," + str(wrState[1])

        return stringState

    def reconstructPath(self, initialState):
        """
        Reconstruct the path of moves based on the initial state using Q-values.

        Input:
        - initialState (list): Initial state of the chessboard. eg [[7, 0, 2], [7, 4, 6]]

        Returns:
        - path (list): List of states representing the sequence of moves.
        """
        currentState = initialState
        currentString = self.stateToString(initialState)
        checkMate = False
        self.chess.board.print_board()

        # Add the initial state to the path
        path = [initialState]


        while not checkMate:
            currentDict = self.qTable[currentString]
            maxQ = -100000 # -float('inf') mejor opcion
            maxState = None

            # Check which is the next state with the highest Q-value
            for stateString, qValue in currentDict.items():
                if maxQ < qValue:
                    maxQ = qValue
                    maxState = stateString

            state = self.stringToState(maxState)
            # When we get it, add it to the path
            path.append(state)
            movement = self.getMovement(currentState, state)
            # Make the corresponding movement
            print(f"Current State: {currentState}, movement: {movement}")
            self.chess.move(movement[0], movement[1])
            self.chess.board.print_board()
            currentString = maxState
            currentState = state

            # When it gets to checkmate, the execution is over
            if self.isCheckMate(currentState):
                checkMate = True

        print("Sequence of moves: ", path)

    def getListActions(self):
        return self.listActions

    def newBoardSim(self, states):
        TA = np.zeros((8,8))
        for state in states:
            TA[state[0], state[1]] = state[2]

        self.chess.newBoardSim(TA)



    def stringToState(self, stringWhiteState):
        """
        Convert a string representation of white pieces' state to a list.

        Input:
        - stringWhiteState (str): String representation of the white pieces' state.

        Returns:
        - whiteState (list): List representing the state of white pieces.
        """
        whiteState = []
        whiteState.append([int(stringWhiteState[0]), int(stringWhiteState[2]), 6])
        if len(stringWhiteState) > 4:
            whiteState.append([int(stringWhiteState[4]), int(stringWhiteState[6]), 2])

        return whiteState

    def getListNextStatesWFixed(self, currentState):
        return self.chess.boardSim.getFixedListNextStatesW(currentState)

    def qlearning(self, startState, alpha, gamma, epsilon):
        currentState = startState
        max_iterations = 1000
        number_of_iterations = 0

        while number_of_iterations < max_iterations:
            self.newBoardSim(currentState)

            while not self.isCheckMate(currentState):

                if currentState[0][2] == 6:
                    currentState[0], currentState[1] = currentState[1], currentState[0]

                currentString = self.stateToString(currentState)

                if currentString not in self.qTable:
                    self.qTable[currentString] = {}

                actions = self.getListNextStatesW(currentState)

                randomValue = np.random.random()
                if randomValue < epsilon:
                    nextState = random.choice(actions)
                else:
                    nextState = max(
                        actions,
                        key=lambda action: self.qTable[currentString].get(self.stateToString(action), 0)
                    )

                if self.isCheckMate(nextState):
                    reward_function = 100

                else:
                    reward_function = -1

                if nextState[0][2] == 6: nextState[0], nextState[1] = nextState[1], nextState[0]

                nextString = self.stateToString(nextState)
                if nextString not in self.qTable[currentString]:
                    self.qTable[currentString][nextString] = 0

                if nextString not in self.qTable:
                    self.qTable[nextString] = {}


                self.qTable[currentString][nextString] = (
                        (1 - alpha) * self.qTable[currentString][nextString]
                        + alpha * (reward_function + gamma * max(self.qTable[nextString].values(), default=0))
                )

                if currentState[0] == nextState[0]: self.chess.moveSim(currentState[1], nextState[1])
                elif currentState[1] == nextState[1]: self.chess.moveSim(currentState[0], nextState[0])

                currentState = nextState

            currentState = startState
            number_of_iterations += 1
            print(number_of_iterations)

    def getMovement(self, state, nextState):
        # Given a state and a successor state, return the postiion of the piece that has been moved in both states
        pieceState = None
        pieceNextState = None
        for piece in state:
            if piece not in nextState:
                movedPiece = piece[2]
                pieceNext = self.getPieceState(nextState, movedPiece)
                if pieceNext != None:
                    pieceState = piece
                    pieceNextState = pieceNext
                    break

        return [pieceState, pieceNextState]

    def print_q_table_max(self, q_table):

        print("Estado actual         | Estado con mayor Q     | Valor Q máximo")

        for current_state, actions in q_table.items():
            if actions:  # Si hay acciones disponibles para este estado
                max_next_state = max(actions, key=actions.get)  # Estado con el mayor Q-value
                max_value = actions[max_next_state]  # Máximo Q-value
                print(
                    f"{str(self.stringToState(current_state)):<20} | "
                    f"{str(self.stringToState(max_next_state)):<20} | "
                    f"{max_value:.2f}"
                )
            else:
                print(
                    f"{str(self.stringToState(current_state)):<20} | {'N/A':<20} | {'N/A'}"
                )


def translate(s):
    """
    Translates traditional board coordinates of chess into list indices
    """

    try:
        row = int(s[0])
        col = s[1]
        if row < 1 or row > 8:
            print(s[0] + "is not in the range from 1 - 8")
            return None
        if col < 'a' or col > 'h':
            print(s[1] + "is not in the range from a - h")
            return None
        dict = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4, 'f': 5, 'g': 6, 'h': 7}
        return (8 - row, dict[col])
    except:
        print(s + "is not in the format '[number][letter]'")
        return None



if __name__ == "__main__":

    if len(sys.argv) < 1:
        sys.exit(1)

    # intiialize board
    TA = np.zeros((8, 8))
    # load initial state
    # white pieces

    
    TA[7][0] = 2
    #TA[7][4] = 6
    TA[7][7] = 6
    TA[0][4] = 12
    
    # initialise bord
    print("stating AI chess... ")
    aichess = Aichess(TA, True)
    currentState = aichess.chess.board.currentStateW.copy()
    print("printing board")
    aichess.chess.boardSim.print_board()

    # get list of next states for current state
    print("current State",currentState,"\n")
    alpha = 0.3
    gamma = 0.8
    epsilon = 0.2
    aichess.qlearning(currentState, alpha, gamma, epsilon)

    aichess.reconstructPath(currentState)




