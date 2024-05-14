import numpy as np
import pickle

# Default actions in GridWorld environments
default_action_set = [(1, 0), (-1, 0), (0, -1), (0, 1)]  # R, L, D, U

TERMINATE_ACTION = (0, 0)


class TDLambdaAgent(object):
    """
    Implements tabular TD(lambda) with the general TD error
    """

    def __init__(self, max_row, max_col):
        self.action_set = default_action_set
        self.option_set = []
        self.default_max_actions = len(self.action_set)  # will stay fixed
        self.max_actions = len(self.action_set)  # can increase

        self.V = np.zeros((max_row, max_col))
        self.e = np.zeros((max_row, max_col))
        self.states_rc = [(r, c) for r in range(max_row) for c in range(max_col)]
        self.last_state, self.last_action = -1, -1
        self.steps = 0
        self.max_row, self.max_col = max_row, max_col

    def start(self, state):
        # Saving the state as last_state
        self.last_state = state[0]
        # Getting the cartesian form of the states
        row, col = self.states_rc[state[0]]
        # set policy
        ca = self.random_uniform(row, col)
        action = self.action_set[ca]
        self.last_action = ca
        # Updating steps
        self.steps = 1
        return self.last_action

    def step(self, reward, state):
        """
        Arguments: reward: floting point, state: integer
        Returns: action: list with two integers [row, col]
        """
        current_state = self.states_rc[state[0]]
        # crow: row of current state
        crow = current_state[0]
        # ccol: col of current state
        ccol = current_state[1]

        # Getting the coordinates of the last state
        lrow, lcol = self.states_rc[self.last_state]
        # la: integer representation of last action
        la = self.last_action

        # UWT Procedure
        rho = 1.0
        if current_state in self.subgoals:  # beta = 1
            target = reward + self.get_stopping_bonus(current_state)
            delta = target - self.V[lrow][lcol]
            self.e[current_state] += 1  # add the gradient (replacing trace)
            self.e *= rho  # IS ratio
            self.V += self.alpha * delta * self.e
        else:  # beta = 0
            target = reward + (self.discount) * (self.V[crow][ccol])
            delta = target - self.V[lrow][lcol]
            self.e[current_state] += 1  # add the gradient (replacing trace)
            self.e *= rho  # IS ratio
            self.V += self.alpha * delta * self.e
            self.e *= self.discount * self.lmbda

        # delta = target - self.V[lrow][lcol]
        # self.UWT(self.V, self.e, current_state, self.alpha, delta, rho, self.discount, self.lmbda)

        # Update Q value
        # target = reward + (self.discount)*(self.V[crow][ccol])
        # Update: New Estimate = Old Estimate + StepSize[Target - Old Estimate]
        # self.V[lrow][lcol] += self.alpha*(target - self.V[lrow][lcol])

        # Choose action
        ca = self.random_uniform(crow, ccol)
        action = self.action_set[ca]
        self.last_state = self.states_rc.index(current_state)
        self.last_action = ca
        self.steps += 1
        return self.last_action

    def end(self, reward):
        """
        Arguments: reward: floating point
        Returns: Nothing
        """
        lrow, lcol = self.states_rc[self.last_state]
        la = self.last_action
        # We know that the agent has transitioned in the terminal state
        # for which all action values are 0
        target = reward + 0
        self.V[lrow][lcol] += self.alpha * (target - self.V[lrow][lcol])
        # Resetting last_state and last_action for the next episode
        self.last_state, self.last_action = -1, -1
        return

    # def UWT(self, current_state, reward, delta):
    #     lrow, lcol = self.states_rc[self.last_state]
    #     e[current_state] += 1 # add the gradient (replacing trace)
    #     e *= 1.0 # IS ratio
    #     target = reward + (self.discount)*(self.V[current_state[0]][current_state[1]])
    #     self.V[lrow][lcol] += self.alpha*delta
    #     e *= self.discount*self.lmbda(1-beta)

    def cleanup(self):
        self.V = np.zeros((self.max_row, self.max_col))
        self.last_state, self.last_action = -1, -1
        self.steps = 0
        return

    def random_uniform(self, row, col):
        ca = np.random.randint(self.max_actions)
        return ca

    # Getter and Setter functions
    def set_alpha(self, alpha):
        self.alpha = alpha

    def set_epsilon(self, epsilon):
        self.epsilon = epsilon

    def set_discount(self, discount):
        self.discount = discount

    def set_lmbda(self, lmbda):
        self.lmbda = lmbda

    def set_dimension(self, dimension):
        max_rows, max_cols = int(dimension[0]), int(dimension[1])
        self.__init__(self, max_rows, max_cols)

    def set_subgoals(self, subgoals):
        self.subgoals = subgoals

    def get_steps(self):
        return self.steps

    def get_V(self):
        return self.V

    def set_V(self, V):
        self.V = V

    def get_stopping_bonus(self, state):
        if state in self.subgoals:
            return 10.0
        else:
            return 0.0

    def message(self, in_message):
        print("Invalid agent message: " + in_message)
        exit()
