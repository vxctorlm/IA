#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  8 11:22:03 2022

@author: ignasi
"""
import copy
import math
from turtledemo.forest import doit1

import chess
import board
import numpy as np
import sys
import queue
from typing import List

RawStateType = List[List[List[int]]]

from itertools import permutations


class Aichess():
    """
    A class to represent the game of chess.

    ...

    Attributes:
    -----------
    chess : Chess
        represents the chess game

    Methods:
    --------
    startGame(pos:stup) -> None
        Promotes a pawn that has reached the other side to another, or the same, piece

    """

    def __init__(self, TA, myinit=True):

        if myinit:
            self.chess = chess.Chess(TA, True)
        else:
            self.chess = chess.Chess([], False)

        self.listNextStates = []
        self.listVisitedStates = []
        self.listVisitedSituations = []
        self.pathToTarget = []
        self.currentStateW = self.chess.boardSim.currentStateW;
        self.depthMax = 8;
        self.checkMate = False

    def copyState(self, state):

        copyState = []
        for piece in state:
            copyState.append(piece.copy())
        return copyState

    def isVisitedSituation(self, color, mystate):

        if (len(self.listVisitedSituations) > 0):
            perm_state = list(permutations(mystate))

            isVisited = False
            for j in range(len(perm_state)):

                for k in range(len(self.listVisitedSituations)):
                    if self.isSameState(list(perm_state[j]), self.listVisitedSituations.__getitem__(k)[1]) and color == \
                            self.listVisitedSituations.__getitem__(k)[0]:
                        isVisited = True

            return isVisited
        else:
            return False


    def getCurrentStateW(self):

        return self.myCurrentStateW

    def getListNextStatesW(self, myState):

        self.chess.boardSim.getListNextStatesW(myState)
        self.listNextStates = self.chess.boardSim.listNextStates.copy()

        return self.listNextStates

    def getListNextStatesB(self, myState):
        self.chess.boardSim.getListNextStatesB(myState)
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

    def isWatchedBk(self, currentState):

        self.newBoardSim(currentState)

        bkPosition = self.getPieceState(currentState, 12)[0:2]
        wkState = self.getPieceState(currentState, 6)
        wrState = self.getPieceState(currentState, 2)

        # Si les negres maten el rei blanc, no és una configuració correcta
        if wkState == None:
            return False
        # Mirem les possibles posicions del rei blanc i mirem si en alguna pot "matar" al rei negre
        for wkPosition in self.getNextPositions(wkState):
            if bkPosition == wkPosition:
                # Tindríem un checkMate
                return True
        if wrState != None:
            # Mirem les possibles posicions de la torre blanca i mirem si en alguna pot "matar" al rei negre
            for wrPosition in self.getNextPositions(wrState):
                if bkPosition == wrPosition:
                    return True
        return False



    def isWatchedWk(self, currentState):
        self.newBoardSim(currentState)

        wkPosition = self.getPieceState(currentState, 6)[0:2]
        bkState = self.getPieceState(currentState, 12)
        brState = self.getPieceState(currentState, 8)

        # If whites kill the black king , it is not a correct configuration
        if bkState == None:
            return False
        # We check all possible positions for the black king, and chck if in any of them it may kill the white king
        for bkPosition in self.getNextPositions(bkState):
            if wkPosition == bkPosition:
                # That would be checkMate
                return True
        if brState != None:
            # We check the possible positions of the black tower, and we chck if in any o them it may killt he white king
            for brPosition in self.getNextPositions(brState):
                if wkPosition == brPosition:
                    return True

        return False



    def newBoardSim(self, listStates):
        # We create a  new boardSim
        TA = np.zeros((8, 8))
        for state in listStates:
            TA[state[0]][state[1]] = state[2]

        self.chess.newBoardSim(TA)

    def getPieceState(self, state, piece):
        pieceState = None
        for i in state:
            if i[2] == piece:
                pieceState = i

                break
        return pieceState

    def getCurrentState(self):
        listStates = []
        for i in self.chess.board.currentStateW:
            listStates.append(i)
        for j in self.chess.board.currentStateB:
            listStates.append(j)
        return listStates

    def getNextPositions(self, state):
        # Given a state, we check the next possible states
        # From these, we return a list with position, i.e., [row, column]
        if state == None:
            return None
        if state[2] > 6:
            nextStates = self.getListNextStatesB([state])
        else:
            nextStates = self.getListNextStatesW([state])
        nextPositions = []
        for i in nextStates:
            nextPositions.append(i[0][0:2])
        return nextPositions

    def getWhiteState(self, currentState):
        whiteState = []
        wkState = self.getPieceState(currentState, 6)
        whiteState.append(wkState)
        wrState = self.getPieceState(currentState, 2)
        if wrState != None:
            whiteState.append(wrState)
        return whiteState

    def getBlackState(self, currentState):
        blackState = []
        bkState = self.getPieceState(currentState, 12)
        blackState.append(bkState)
        brState = self.getPieceState(currentState, 8)
        if brState != None:
            blackState.append(brState)
        return blackState

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

    def heuristica(self, currentState, color):
        #In this method, we calculate the heuristics for both the whites and black ones
        #The value calculated here is for the whites,
        # but finally from verything, as a function of the color parameter, we multiply the result by -1
        value = 0

        bkState = self.getPieceState(currentState, 12)
        wkState = self.getPieceState(currentState, 6)
        wrState = self.getPieceState(currentState, 2)
        brState = self.getPieceState(currentState, 8)
        if bkState is None:
            return float('inf')
        if wkState is None:
            return -float('inf')
        filaBk = bkState[0]
        columnaBk = bkState[1]
        filaWk = wkState[0]
        columnaWk = wkState[1]

        if wrState != None:
            filaWr = wrState[0]
            columnaWr = wrState[1]
        if brState != None:
            filaBr = brState[0]
            columnaBr = brState[1]

        # We check if they killed the black tower
        if brState == None:
            value += 50
            fila = abs(filaBk - filaWk)
            columna = abs(columnaWk - columnaBk)
            distReis = min(fila, columna) + abs(fila - columna)
            if distReis >= 3 and wrState != None:
                filaR = abs(filaBk - filaWr)
                columnaR = abs(columnaWr - columnaBk)
                value += (min(filaR, columnaR) + abs(filaR - columnaR))/10
            # If we are white white, the closer our king from the oponent, the better
            # we substract 7 to the distance between the two kings, since the max distance they can be at in a board is 7 moves
            value += (7 - distReis)
            # If they black king is against a wall, we prioritize him to be at a corner, precisely to corner him
            if bkState[0] == 0 or bkState[0] == 7 or bkState[1] == 0 or bkState[1] == 7:
                value += (abs(filaBk - 3.5) + abs(columnaBk - 3.5)) * 10
            #If not, we will only prioritize that he approahces the wall, to be able to approach the check mate
            else:
                value += (max(abs(filaBk - 3.5), abs(columnaBk - 3.5))) * 10

        # They killed the black tower. Within this method, we consider the same conditions than in the previous condition
        # Within this method we consider the same conditions than in the previous section, but now with reversed values.
        if wrState == None:
            value += -50
            fila = abs(filaBk - filaWk)
            columna = abs(columnaWk - columnaBk)
            distReis = min(fila, columna) + abs(fila - columna)

            if distReis >= 3 and brState != None:
                filaR = abs(filaWk - filaBr)
                columnaR = abs(columnaBr - columnaWk)
                value -= (min(filaR, columnaR) + abs(filaR - columnaR)) / 10
            # If we are white, the close we have our king from the oponent, the better
            # If we substract 7 to the distance between both kings, as this is the max distance they can be at in a chess board
            value += (-7 + distReis)

            if wkState[0] == 0 or wkState[0] == 7 or wkState[1] == 0 or wkState[1] == 7:
                value -= (abs(filaWk - 3.5) + abs(columnaWk - 3.5)) * 10
            else:
                value -= (max(abs(filaWk - 3.5), abs(columnaWk - 3.5))) * 10

        # We are checking blacks
        if self.isWatchedBk(currentState):
            value += 20

        # We are checking whites
        if self.isWatchedWk(currentState):
            value += -20

        # If black, values are negative, otherwise positive
        if not color:
            value = (-1) * value

        return value

    def isCheckMate(self, currentState, toMove):
        """
        Args:
            currentState: Current state
            toMove: bool with white true or black false

        Returns: boolean of check mate or tie:

        """
        if toMove:
            check = self.isWatchedBk(currentState)
        else: check = self.isWatchedWk(currentState)
        if not check:
            self.newBoardSim(currentState)
            return False
        return check

    def mergeStates(self, move, opponentPieces):
        result = []
        opponent_positions = {(piece[0], piece[1]): piece for piece in opponentPieces}
        for piece in move:
            if (piece[0], piece[1]) in opponent_positions:
                # Capturar pieza del oponente
                del opponent_positions[(piece[0], piece[1])]
            result.append(piece)
        result.extend(opponent_positions.values())  # Añadir piezas no capturadas
        return result

    def minimax(self, state, depth, maximizingPlayer):
        # Simular un nuevo tablero
        self.newBoardSim(state)

        # Evaluar si hay jaque mate
        if(self.isCheckMate(state, maximizingPlayer) or depth == 0):
            return state, self.heuristica(state, maximizingPlayer)

        # Maximizing player (blanco)
        if maximizingPlayer:
            maxEval = -float('inf')
            maxState = None

            # Generar todos los movimientos posibles para el jugador blanco
            for move in self.getListNextStatesW(self.getWhiteState(state)):
                fstate = self.mergeStates(move,self.getBlackState(state))
                if self.isVisited(fstate):  # Filtrar estados ya visitados
                    continue

                check= self.isCheckMate(fstate, False)
                if check:
                    continue
                # Llamada recursiva con el jugador contrario (minimizing)


                _, eval = self.minimax(fstate, depth - 1, False)

                # Actualizar el mejor estado y evaluación
                if eval > maxEval:
                    maxEval = eval
                    maxState = fstate

            return maxState, maxEval
        # Minimizing player (Black)

        else:
            minEval = float('inf')
            minState = None

            for move in self.getListNextStatesB(self.getBlackState(state)):
                fstate = self.mergeStates(move,self.getWhiteState(state))
                if self.isVisited(fstate):  # Filtrar estados ya visitados
                    continue

                check= self.isCheckMate(fstate, True)
                if check:

                    continue

                _, eval = self.minimax(fstate, depth - 1, True)

                if eval < minEval:
                    minEval = eval
                    minState = fstate

            return minState, minEval

    def minimaxGame(self, depthWhite, depthBlack):

        currentState = self.getCurrentState()
        color = True
        it = 0
        # Mientras nadie haga checkMate o no se haya superado el limite de iteraciones, entramos al bucle
        while (not self.isCheckMate(currentState,color)) and (
                it < 100):
            it += 1
            print(it)
            currentState = self.getCurrentState()


            # Turno de jugar los blancos
            if color:
                newState, _ = self.minimax(currentState, depthWhite, color)

                print("White's Turn")
            else:
                newState, _ = self.minimax(currentState, depthBlack, color)

                print("Black's Turn")

            # Para controlar tablas. Si un jugador no tiene un mejor movimiento que quedarse quieto, es empate.
            if newState == None:

                if self.isCheckMate(currentState,not color):
                    print("Mate")
                    it = 100
                    continue
                print("TABLAS")
                return

            pc = self.getMovement(currentState, newState)  # Cogemos la pieza que se ha movido
            self.chess.move(pc[0], pc[1])  # Movemos el tablero
            color = not color  # Cambiamos el turno
            currentState = self.getCurrentState()
            self.chess.board.print_board()

        if it < 100:
            print("Check Mate State")
            self.chess.board.print_board()
        else:
            print("Maximo de iteracions superado")

    def alphaBeta(self, state, depth, alpha, beta, maximizingPlayer):
        # Simulate a new board
        self.newBoardSim(state)

        # Check if the game is over or depth is 0
        if self.isCheckMate(state, maximizingPlayer) or depth == 0:
            return state, self.heuristica(state, maximizingPlayer)

        if maximizingPlayer:
            maxEval = -float('inf')
            maxState = None

            # Generate all possible moves for white
            for move in self.getListNextStatesW(self.getWhiteState(state)):
                fstate = self.mergeStates(move, self.getBlackState(state))
                if self.isVisited(fstate):
                    continue

                check = self.isCheckMate(fstate, False)
                if check:
                    continue

                _, eval = self.alphaBeta(fstate, depth - 1, alpha, beta, False)

                if eval > maxEval:
                    maxEval = eval
                    maxState = fstate

                alpha = max(alpha, eval)
                if beta <= alpha:
                    break  # Beta cutoff

            return maxState, maxEval
        else:
            minEval = float('inf')
            minState = None

            # Generate all possible moves for black
            for move in self.getListNextStatesB(self.getBlackState(state)):
                fstate = self.mergeStates(move, self.getWhiteState(state))
                if self.isVisited(fstate):
                    continue

                check = self.isCheckMate(fstate, True)
                if check:
                    continue

                _, eval = self.alphaBeta(fstate, depth - 1, alpha, beta, True)

                if eval < minEval:
                    minEval = eval
                    minState = fstate

                beta = min(beta, eval)
                if beta <= alpha:
                    break  # Alpha cutoff

            return minState, minEval

    def alphaBetaPoda(self, depthWhite, depthBlack):
        currentState = self.getCurrentState()
        color = True
        it = 0

        # Main game loop
        while not self.isCheckMate(currentState, color) and it < 100:
            it += 1
            print(it)

            if color:
                newState, _ = self.alphaBeta(currentState, depthWhite, -float('inf'), float('inf'), True)
                print("White's Turn")
            else:
                newState, _ = self.alphaBeta(currentState, depthBlack, -float('inf'), float('inf'), False)
                print("Black's Turn")

            # Check for stalemate
            if newState is None:
                if self.isCheckMate(currentState, not color):
                    print("Mate")
                    it = 100
                    continue
                print("TABLAS")
                return

            pc = self.getMovement(currentState, newState)
            self.chess.move(pc[0], pc[1])
            color = not color
            currentState = self.getCurrentState()
            self.chess.board.print_board()

        if it < 100:
            print("Check Mate State")
            self.chess.board.print_board()
        else:
            print("Maximum iterations exceeded")

    def maxexp_value(self, state, depth):
        self.newBoardSim(state)
        if self.isCheckMate(state, True) or depth == 0:
            return None, self.heuristica(state, True)

        maxEval = -float('inf')
        bestState = None

        for move in self.getListNextStatesW(self.getWhiteState(state)):
            fstate = self.mergeStates(move, self.getBlackState(state))
            if self.isVisited(fstate):
                continue

            check = self.isCheckMate(fstate, False)
            if check:
                continue

            _, eval = self.minexp_value(fstate, depth - 1)

            if eval > maxEval:
                maxEval = eval
                bestState = fstate

        return bestState, maxEval

    def minexp_value(self, state, depth):
        self.newBoardSim(state)
        if self.isCheckMate(state, False) or depth == 0:
            return None, self.heuristica(state, False)

        values = []
        possibleStates = []

        for move in self.getListNextStatesB(self.getBlackState(state)):
            fstate = self.mergeStates(move, self.getWhiteState(state))
            if self.isVisited(fstate):
                continue

            if self.isCheckMate(fstate, True):  # Skip states where White wins
                continue

            _, eval = self.maxexp_value(fstate, depth - 1)
            values.append(eval)
            possibleStates.append(fstate)

        if not values:  # No valid moves for Black
            return None, float('inf')

        # Calculate the expected value
        expectedValue = self.calculateValue(values)

        # Select a plausible state for Black (e.g., the first valid state)
        bestState = possibleStates[0] if possibleStates else None

        return bestState, expectedValue

    def expectimaxGame(self, depthWhite, depthBlack):
        currentState = self.getCurrentState()
        color = True  # Blanco comienza
        iteration = 0

        while not self.isCheckMate(currentState, color) and iteration < 100:
            iteration += 1
            print(f"Iteration: {iteration}")

            if color:  # Turno de los blancos
                newState, _ = self.maxexp_value(currentState, depthWhite)
                print("White's Turn")
            else:  # Turno de los negros
                newState, _ = self.minexp_value(currentState, depthBlack)
                print("Black's Turn")

            if newState is None:
                print("Stalemate")
                return

            pieceMove = self.getMovement(currentState, newState)
            self.chess.move(pieceMove[0], pieceMove[1])
            # Append the state along with the player who visited it
            self.listVisitedStates.append((newState, color))
            color = not color  # Cambiar turno
            currentState = self.getCurrentState()
            self.chess.board.print_board()

        if iteration < 100:
            print("Checkmate")
        else:
            print("Maximum iterations reached")

    def mitjana(self, values):
        sum = 0
        N = len(values)
        for i in range(N):
            sum += values[i]

        return sum / N

    def desviacio(self, values, mitjana):
        sum = 0
        N = len(values)

        for i in range(N):
            sum += pow(values[i] - mitjana, 2)

        return pow(sum / N, 1 / 2)

    def calculateValue(self, values):

        if len(values) == 0:
            return 0
        mitjana = self.mitjana(values)
        desviacio = self.desviacio(values, mitjana)
        # If deviation is 0, we cannot standardize values, since they are all equal, thus probability willbe equiprobable
        if desviacio == 0:
            # We return another value
            return values[0]

        esperanca = 0
        sum = 0
        N = len(values)
        for i in range(N):
            #Normalize value, with mean and deviation - zcore
            normalizedValues = (values[i] - mitjana) / desviacio
            # make the values positive with function e^(-x), in which x is the standardized value
            positiveValue = pow(1 / math.e, normalizedValues)
            # Here we calculate the expected value, which in the end will be expected value/sum
            # Our positiveValue/sum represent the probabilities for each value
            # The larger this value, the more likely
            esperanca += positiveValue * values[i]
            sum += positiveValue

        return esperanca / sum




if __name__ == "__main__":
    #   if len(sys.argv) < 2:
    #       sys.exit(usage())

    # intiialize board
    TA = np.zeros((8, 8))

    #Configuració inicial del taulell
    TA[7][0] = 2
    TA[7][4] = 6
    TA[0][7] = 8
    TA[0][4] = 12

    # initialise board
    print("stating AI chess... ")
    aichess = Aichess(TA, True)



    print("printing board")
    aichess.chess.boardSim.print_board()

  # Run exercise 1
    print("MinimaxGame")
    aichess.minimaxGame(2,2)
    aichess = Aichess(TA, True)
  #excercise 2
    print("AlphaBetaGame")
    aichess.alphaBetaPoda(2,2)
    aichess = Aichess(TA, True)
  #excercise 3
    print("expectimaxGame")
    aichess.expectimaxGame(2,2)


  # Add code to save results and continue with other exercises
  #to answer of to inform we modified slightly the algorithms to achieve half alphabeta and minmax, etc...,
  #here is only the algorithms implemented a minimax an alphabeta and an expectation. I recommend using depth of 2, 2 to get faster results.
  #But it works with diferents depth.



