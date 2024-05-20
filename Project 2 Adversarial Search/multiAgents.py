# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        oldFood = currentGameState.getFood().asList()
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood().asList()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        if len(newFood) == len(oldFood): 
            distance = 999999
            for food in newFood:
                distance = min(distance, manhattanDistance(food, newPos))
        else:
            distance = 0
            
        """"print(distance)"""
        score = -distance
        """"print(score)"""
        for ghost in newGhostStates:
            ghostPos = manhattanDistance(ghost.getPosition(), newPos)
            """"penalize score based on ghost position"""
            score -= 2 ** (5 - ghostPos)
        return score

        """return successorGameState.getScore()"""

def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        score, action = self.minimax(gameState, 0, 0)
        return action
    
        util.raiseNotDefined()
    
    def minimax(self, state, index, depth):
        
        if depth == self.depth or state.isWin() or state.isLose():
            return state.getScore(), "noMove"   #score and action

        if index == 0:
            return self.maxFunc(state, 0, depth)

        else:
            return self.minFunc(state, index, depth)
    
    def maxFunc(self, state, index, depth):
        
        score = float('-inf')


        for move in state.getLegalActions(index):
            successor = state.generateSuccessor(index, move)
            successor_index = index + 1
            successor_depth = depth

            if successor_index == state.getNumAgents():
                successor_index = 0
                successor_depth += 1

            curr_val, curr_action = self.minimax(successor, successor_index, successor_depth)

            if curr_val > score:
                score = curr_val
                action = move

        return score, action       #score and action
    
    def minFunc(self, state, index, depth):
        
        score = float('inf')

        for move in state.getLegalActions(index):
            successor = state.generateSuccessor(index, move)
            successor_index = index + 1
            successor_depth = depth

            if successor_index == state.getNumAgents():
                successor_index = 0
                successor_depth += 1

            curr_val, curr_action = self.minimax(successor, successor_index, successor_depth)

            if curr_val < score:
                score = curr_val
                action = move

        return score, action         #score and action
    

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        score, action = self.alphaBeta(gameState, 0, 0, float('-inf'), float('inf'))
        return action
    
        util.raiseNotDefined()
    
    def alphaBeta(self, state, index, depth, alpha, beta):
        
        if depth == self.depth or state.isWin() or state.isLose():
            return state.getScore(), "noMove"  #score and action

        if index == 0:
            return self.maxFunc(state, 0, depth, alpha, beta)

        else:
            return self.minFunc(state, index, depth, alpha, beta)
    
    def maxFunc(self, state, index, depth, alpha, beta):
        
        v = float('-inf')

        for move in state.getLegalActions(index):
            successor = state.generateSuccessor(index, move)
            successor_index = index + 1
            successor_depth = depth

            if successor_index == state.getNumAgents():
                successor_index = 0
                successor_depth += 1

            curr_val, curr_action = self.alphaBeta(successor, successor_index, successor_depth, alpha, beta)

            if curr_val > v:
                v = curr_val
                action = move
            
            alpha = max(alpha, v)
            if v > beta:
                return v, action

        return v, action       #score and action
    
    def minFunc(self, state, index, depth, alpha, beta):
        
        v = float('inf')

        for move in state.getLegalActions(index):
            successor = state.generateSuccessor(index, move)
            successor_index = index + 1
            successor_depth = depth

            if successor_index == state.getNumAgents():
                successor_index = 0
                successor_depth += 1

            curr_val, curr_action = self.alphaBeta(successor, successor_index, successor_depth, alpha, beta)

            if curr_val < v:
                v = curr_val
                action = move
                
            beta = min(beta, v)
            if v < alpha:
                return v, action

        return v, action         #score and action

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        score, action = self.expectiMax(gameState, 0, 0)
        return action
    
        util.raiseNotDefined()
    
    def expectiMax(self, gameState, index, depth):
        
        if depth == self.depth or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState),"noMove"   #score and action

        if index == 0:
            return self.maxFunc(gameState, 0, depth)

        else:
            return self.expecFunc(gameState, index, depth)
    
    def maxFunc(self, state, index, depth):
        
        v = float('-inf')


        for move in state.getLegalActions(index):
            successor = state.generateSuccessor(index, move)
            successor_index = index + 1
            successor_depth = depth

            if successor_index == state.getNumAgents():
                successor_index = 0
                successor_depth += 1

            curr_val, curr_action = self.expectiMax(successor, successor_index, successor_depth)

            if curr_val > v:
                v = curr_val
                action = move
            

        return v, action       #score and action
    
    def expecFunc(self, state, index, depth):
        
        legalMoves = state.getLegalActions(index)
        expec_value = 0
        expec_action = ""

        for move in state.getLegalActions(index):
            successor = state.generateSuccessor(index, move)
            successor_index = index + 1
            successor_depth = depth

            if successor_index == state.getNumAgents():
                successor_index = 0
                successor_depth += 1

            curr_val, curr_action = self.expectiMax(successor, successor_index, successor_depth)

            expec_value += curr_val/len(legalMoves)
                

        return expec_value, expec_action         #score and action

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    
    Pacman aims to eat all the food capsules quickly.
    It calculates the manhattan distance to all the capsules 
    from the current position of pacman and targets to eat them.
    If the food capsule is not eaten, it will add it to the list to eat them.
    
    """
    "*** YOUR CODE HERE ***"
    
    position = list(currentGameState.getPacmanPosition())
    
    capsulePosition = currentGameState.getCapsules()
    ghostStates = currentGameState.getGhostStates()
    
    foodList = currentGameState.getFood().asList() 
    dist = []  
       
    for food in foodList:
        pacmanfoodDist = manhattanDistance(food, position)
        dist.append(-1 * pacmanfoodDist)
        
    if not dist:
        dist.append(0)
        
    result = max(dist) + currentGameState.getScore()

    return result
    util.raiseNotDefined()


# Abbreviation
better = betterEvaluationFunction
