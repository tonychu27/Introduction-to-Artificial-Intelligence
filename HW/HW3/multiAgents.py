from util import manhattanDistance
from game import Directions
import random
import util
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
        # Collect legal moves and child states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(
            gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(
            len(scores)) if scores[index] == bestScore]
        # Pick randomly among the best
        chosenIndex = random.choice(bestIndices)

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        The evaluation function takes in the current and proposed child
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.
        """
        # Useful information you can extract from a GameState (pacman.py)
        childGameState = currentGameState.getPacmanNextState(action)
        newPos = childGameState.getPacmanPosition()
        newFood = childGameState.getFood()
        newGhostStates = childGameState.getGhostStates()
        newScaredTimes = [
            ghostState.scaredTimer for ghostState in newGhostStates]

        minGhostDistance = min(
            [manhattanDistance(newPos, state.getPosition()) for state in newGhostStates])

        scoreDiff = childGameState.getScore() - currentGameState.getScore()

        pos = currentGameState.getPacmanPosition()
        nearestFoodDistance = min([manhattanDistance(pos, food)
                                  for food in currentGameState.getFood().asList()])
        newFoodsDistances = [manhattanDistance(
            newPos, food) for food in newFood.asList()]
        newNearestFoodDistance = 0 if not newFoodsDistances else min(
            newFoodsDistances)
        isFoodNearer = nearestFoodDistance - newNearestFoodDistance

        direction = currentGameState.getPacmanState().getDirection()
        if minGhostDistance <= 1 or action == Directions.STOP:
            return 0
        if scoreDiff > 0:
            return 8
        elif isFoodNearer > 0:
            return 4
        elif action == direction:
            return 2
        else:
            return 1


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

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (Part 1)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.getNextState(agentIndex, action):
        Returns the child game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """

        # Begin your code (Part 1)
        # raise NotImplementedError("To be implemented")
        """
        There are two functions in part 1, the first is isTerminal which is used to determine
        the game is finish or not, and the second is minimaxSearch which is used to implement
        the Minimax Search algorithm.

        In the minimaxSearch function, I used a recursive way to implement it, starting from 
        the leaves to the root. When curDepth is 0 then return the evaluationFunction of the state,
        otherwise, keep doing recursion and find the optimal solution.

        And there are several parameters in the function.
        curGameState: The current game state
        number: How many agents are there
        curDepth: The current depth that we explored
        index: 0 means that the player is pacman so we will find the maximum evaluationFunction, using
        MAX and nextAction to store the optimal value and the next state,and when index is not equal 
        to 0, it means that the player is ghost so we will find the minimum evaluationFunction, using 
        MIN and nextAction to store the optimal value and the next state. The nextAction get from
        a for loop which is used to iterate each possible action.

        """
        def isTerminal(state):
            if state.isWin() or state.isLose():
                return True
            return False

        def minimaxSearch(curGameState, number, curDepth, index):
            if curDepth == 0:
                return self.evaluationFunction(curGameState), 0
            else:
                nextAction = 0
                actionList = curGameState.getLegalActions(index)
                if index == 0:
                    MAX = float("-inf")
                    for act in actionList:
                        nextState = curGameState.getNextState(index, act)
                        if isTerminal(nextState):
                            cmp = self.evaluationFunction(nextState)
                        else:
                            cmp, _ = minimaxSearch(
                                nextState, number, curDepth-1, (index + 1) % number)
                        if MAX < cmp:
                            MAX = cmp
                            nextAction = act
                    return MAX, nextAction
                else:
                    MIN = float("inf")
                    for act in actionList:
                        nextState = curGameState.getNextState(index, act)
                        if isTerminal(nextState):
                            cmp = self.evaluationFunction(nextState)
                        else:
                            cmp, _ = minimaxSearch(
                                nextState, number, curDepth-1, (index + 1) % number)
                        if MIN > cmp:
                            MIN = cmp
                            nextAction = act
                    return MIN, nextAction
        _, act = minimaxSearch(gameState, gameState.getNumAgents(
        ), gameState.getNumAgents()*self.depth, 0)
        return act
        # End your code (Part 1)


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (Part 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        # Begin your code (Part 2)
        # raise NotImplementedError("To be implemented")
        """
        This part is very similar to part 1, the differences are I added two 
        parameter, alpha and beta.
        alpha: The best option so far from any MAX node to the root
        beta: The best option so far from any MIN node to the root

        Based on Alpha-Beta Pruning, when MAX > beta at MAX node, we will return MAX immediately
        or we will return max(alpha, MAX). When alpha < MIN at MIN node, we will return MIN 
        immediately or we will return min(beta, min) 
        """
        def isTerminal(state):
            if state.isWin() or state.isLose():
                return True
            return False

        def AlphaBetaPruning(curGameState, number, curDepth, index, alpha, beta):
            if curDepth == 0:
                return self.evaluationFunction(curGameState), 0
            else:
                nextAction = 0
                actionList = curGameState.getLegalActions(index)
                if index == 0:
                    MAX = float("-inf")
                    for act in actionList:
                        nextState = curGameState.getNextState(index, act)
                        if isTerminal(nextState):
                            cmp = self.evaluationFunction(nextState)
                        else:
                            cmp, _ = AlphaBetaPruning(
                                nextState, number, curDepth-1, (index + 1) % number, alpha, beta)
                        if MAX < cmp:
                            MAX = cmp
                            nextAction = act
                        if MAX > beta:
                            return MAX, act
                        alpha = max(alpha, MAX)
                    return MAX, nextAction
                else:
                    MIN = float("inf")
                    for act in actionList:
                        nextState = curGameState.getNextState(index, act)
                        if isTerminal(nextState):
                            cmp = self.evaluationFunction(nextState)
                        else:
                            cmp, _ = AlphaBetaPruning(
                                nextState, number, curDepth-1, (index + 1) % number, alpha, beta)
                        if MIN > cmp:
                            MIN = cmp
                            nextAction = act
                        if MIN < alpha:
                            return MIN, act
                        beta = min(beta, MIN)
                    return MIN, nextAction
        _, act = AlphaBetaPruning(gameState, gameState.getNumAgents(), gameState.getNumAgents()*self.depth, 0,
                                  float('-inf'), float('inf'))
        return act
        # End your code (Part 2)


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (Part 3)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        # Begin your code (Part 3)
        # raise NotImplementedError("To be implemented")
        """
        This part is same as part 1 when index=0
        When index=1, I declared a variable called "sum" to add all the cmp,
        and then return sum divide by number of legal action.

        """
        def isTerminal(state):
            if state.isWin() or state.isLose():
                return True
            return False

        def expectimaxSearch(curGameState, number, curDepth, index):
            if curDepth == 0:
                return self.evaluationFunction(curGameState), 0
            else:
                nextAction = 0
                actionList = curGameState.getLegalActions(index)
                if index == 0:
                    MAX = float("-inf")
                    for act in actionList:
                        nextState = curGameState.getNextState(index, act)
                        if isTerminal(nextState):
                            cmp = self.evaluationFunction(nextState)
                        else:
                            cmp, _ = expectimaxSearch(
                                nextState, number, curDepth-1, (index + 1) % number)
                        if MAX < cmp:
                            MAX = cmp
                            nextAction = act
                    return MAX, nextAction
                else:
                    sum = 0
                    for act in actionList:
                        nextState = curGameState.getNextState(index, act)
                        if isTerminal(nextState):
                            cmp = self.evaluationFunction(nextState)
                        else:
                            cmp, _ = expectimaxSearch(
                                nextState, number, curDepth-1, (index + 1) % number)
                        sum += cmp
                    return sum/len(actionList), nextAction
        _, act = expectimaxSearch(gameState, gameState.getNumAgents(
        ), gameState.getNumAgents()*self.depth, 0)
        return act
        # End your code (Part 3)


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (Part 4).
    """
    # Begin your code (Part 4)
    # raise NotImplementedError("To be implemented")
    """
    I get some useful information from the current game state first.
    And I return evalution by some rule with some information

    If pacman is too close to ghost (distance between them is smaller than 3)
    then return -40*minGhostDist means that let pacman go to eat ghost
    And then I will get the distance between pacman and all the foods, if the food
    is farther from the pacman's current position, then it will have lower 
    evalution value, after doing the for loop, add the evalution value to current 
    score and return it.
    """
    currentScore = currentGameState.getScore()
    pos = currentGameState.getPacmanPosition()
    foodPos = currentGameState.getFood().asList()
    minGhostDist = min([manhattanDistance(pos, ghost)
                       for ghost in currentGameState.getGhostPositions()])
    evalution = -0.4*minGhostDist
    if minGhostDist < 3:
        return - 40 * minGhostDist
    for food in foodPos:
        evalution -= 0.05*manhattanDistance(pos, food)
    evalution += currentScore

    return evalution
    # End your code (Part 4)


# Abbreviation
better = betterEvaluationFunction
