import numpy as np
import pickle

# Default actions in GridWorld environments
default_action_set = [(1,0), (-1,0), (0,-1), (0,1)] # R, L, D, U

TERMINATE_ACTION = (0,0)

class SarsaLmbdaAgent(object):
    """
    Implements tabular Sarsa(lambda) with the normal TD error
    """

    def __init__(self, max_row, max_col):
        self.action_set = default_action_set
        self.option_set = []
        self.default_max_actions = len(self.action_set) # will stay fixed
        self.max_actions = len(self.action_set) # can increase

        self.Q = np.zeros((max_row, max_col, self.default_max_actions))
        self.e = np.zeros((max_row, max_col))
        self.states_rc = [(r, c) for r in range(max_row)
                          for c in range(max_col)]
        self.initial_alpha = None

        self.last_state, self.last_action = -1, -1
        self.steps = 0
        self.max_row, self.max_col = max_row, max_col
        
        self.name = "Sarsa(lambda)"

    def start(self, state):
        # Saving the state as last_state
        self.last_state = state[0]
        # Getting the cartesian form of the states
        row, col = self.states_rc[state[0]]
        # set policy
        ca = self.epsilongreedy(row,col)
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

        # Choose action
        ca = self.epsilongreedy(crow,ccol)
        action = self.action_set[ca]

        # UWT Procedure
        # rho = 1.0
        # if current_state in self.subgoals: # beta = 1
        #     # target = reward + self.get_stopping_bonus(current_state)
        #     # delta = target - self.V[lrow][lcol]
        #     self.e[current_state] += 1 # add the gradient (replacing trace)
        #     self.e *= rho # IS ratio
        #     self.V += self.alpha*self.delta*self.e
        # else: # beta = 0
        #     # target = reward + (self.discount)*(self.V[crow][ccol])
        #     # delta = target - self.V[lrow][lcol]
        #     self.e[current_state] += 1 # add the gradient (replacing trace)
        #     self.e *= rho # IS ratio
        #     self.V += self.alpha*self.delta*self.e
        #     self.e *= self.discount*self.lmbda


        # Update Q value
        target = reward + (self.discount)*(self.Q[crow][ccol][ca])
        # Update: New Estimate = Old Estimate + StepSize[Target - Old Estimate]
        self.Q[lrow][lcol][la] += self.alpha*(target - self.Q[lrow][lcol][la])

        # Update last state and last action
        self.last_state = self.states_rc.index(current_state)
        self.last_action = ca
        self.steps += 1
        return self.last_action
    
    # def UWT()
        


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
        self.Q[lrow][lcol][la] += self.alpha*(target - self.Q[lrow][lcol][la])
        # Resetting last_state and last_action for the next episode
        self.last_state, self.last_action = -1, -1
        return

    def cleanup(self):
        self.set_alpha(self.initial_alpha)
        self.Q = np.zeros((self.max_row, self.max_col, self.default_max_actions))
        self.last_state, self.last_action = -1, -1
        self.steps = 0
        return

    def epsilongreedy(self, row, col):
        if np.random.uniform() < self.epsilon:
            ca = np.random.randint(self.max_actions)
        else:
            q = self.Q[row][col]
            # Breaking ties randomly
            ca = np.random.choice(np.flatnonzero(q == q.max()))
        return ca
    
    def uniform_random(self):
        return np.random.randint(self.max_actions)

    # Getter and Setter functions
    def set_alpha(self, alpha):
        self.initial_alpha = alpha
        self.alpha = alpha

    def set_epsilon(self, epsilon):
        self.epsilon = epsilon

    def set_delta(self, delta): # gets delta from TD(lambda)
        self.delta = delta

    def set_discount(self, discount):
        self.discount = discount

    def set_lmbda(self, lmbda):
        self.lmbda = lmbda

    def set_action(self, action):
        self.action = action

    def set_discount(self, discount):
        self.discount = discount

    def set_dimension(self, dimension):
        max_rows, max_cols = int(dimension[0]), int(dimension[1])
        self.__init__(self, max_rows, max_cols)

    def set_subgoals(self, subgoals):
        self.subgoals = subgoals

    def get_steps(self):
        return self.steps

    def add_terminate_action(self):
        self.action_set.append(TERMINATE_ACTION)
        self.default_max_actions = len(self.action_set)
        self.max_actions = len(self.action_set)
        self.Q = np.zeros((self.max_row, self.max_col,
                           self.default_max_actions))
    def get_Q(self):
        return self.Q

    def set_Q(self, Q):
        self.Q = Q

    def get_policy(self):
        pi = np.zeros((len(self.states_rc,)), dtype=int)

        for idx, state in enumerate(self.states_rc):
            row, col = state[0], state[1]
            q = self.Q[row][col]
            # Taking last max to break ties inorder to prefer Terminate action
            ca = np.flatnonzero(q == q.max())[-1]
            pi[idx] = ca # each state will have related optimal action idx

        return pi

    def message(self, in_message):
        print("Invalid agent message: " + in_message)
        exit()

class SarsaLambdaFeatAtt(SarsaLmbdaAgent):
    """
    Implements Sarsa(lambda) with the feature-based TD error.
    Starting with the one subgoal case.
    here Q represents the value followed by the policy for the hallway feature
    """
    def __init__(self, max_row, max_col, subgoals):
        super().__init__(max_row, max_col)
        self.subgoals = subgoals
        self.subgoal_vfs = []

        for subgoal in self.subgoals:
            # Make the value function and eligibility trace for each subgoal
            self.subgoal_vfs.append({"V": np.zeros((max_row, max_col)),
                                     "Q": np.zeros((max_row, max_col, self.default_max_actions)),
                                     "e": np.zeros((max_row, max_col)),
                                     "e_sa": np.zeros((max_row, max_col, self.default_max_actions)),
                                     })
            
        self.name = "Sarsa(lambda)_FeatAtt"

    
    def start(self, state):
        self.last_state = state[0]
        row, col = self.states_rc[state[0]]
        ca = self.uniform_random()
        # ca = self.epsilongreedy(row,col)
        action = self.action_set[ca]
        self.last_action = ca
        self.steps = 1
        return self.last_action
        
    def step(self, reward, state):
        current_state = self.states_rc[state[0]]
        crow = current_state[0]
        ccol = current_state[1]
        lrow, lcol = self.states_rc[self.last_state]
        la = self.last_action
        ca = self.uniform_random()
        # ca = self.epsilongreedy(crow,ccol)
        action = self.action_set[ca]
        # rho = 1.0

        for subgoal in self.subgoals:
            subgoal_vf = self.subgoal_vfs[self.subgoals.index(subgoal)]["V"]
            subgoal_q = self.subgoal_vfs[self.subgoals.index(subgoal)]["Q"]
            subgoal_trace = self.subgoal_vfs[self.subgoals.index(subgoal)]["e"]
            subgoal_trace_sa = self.subgoal_vfs[self.subgoals.index(subgoal)]["e_sa"]
            rho = self.imp_samp(crow, ccol, ca)[self.subgoals.index(subgoal)]
            z = self.get_stopping_bonus(current_state)

            # Technically z should be defined such that z(subgoal) > v_subgoal(subgoal),
            # which causes beta to be 1 at the subgoal.
            if current_state in self.subgoals: # beta = 1
            # if z >= max(self.Q[crow][ccol]): # beta = 1
                delta_i_v = reward + z - subgoal_vf[lrow][lcol]
                delta_i_q = reward + z - subgoal_q[lrow][lcol][la]
                subgoal_trace[lrow][lcol] += 1
                subgoal_trace_sa[lrow][lcol][la] += 1
                subgoal_trace *= rho
                subgoal_vf += self.alpha*delta_i_v*subgoal_trace
                subgoal_q += self.alpha*delta_i_q*subgoal_trace_sa
            else: # beta = 0
                delta_i_v = reward + (self.discount)*(subgoal_vf[crow][ccol]) - subgoal_vf[lrow][lcol]
                delta_i_q = reward + (self.discount)*(subgoal_q[crow][ccol][ca]) - subgoal_q[lrow][lcol][la]
                subgoal_trace[lrow][lcol] += 1
                subgoal_trace_sa[lrow][lcol][la] += 1
                subgoal_trace *= rho
                subgoal_vf += self.alpha*delta_i_v*subgoal_trace
                subgoal_q += self.alpha*delta_i_q*subgoal_trace_sa
                subgoal_trace *= self.discount*self.lmbda

        # Update Q value - Figure 2 doesnt use this. A regular sarsa(lambda) agent would do this though
        target = reward + (self.discount)*(self.Q[crow][ccol][ca])
        self.Q[lrow][lcol][la] += self.alpha*(target - self.Q[lrow][lcol][la])

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

        for subgoal in self.subgoals:
            subgoal_vf = self.subgoal_vfs[self.subgoals.index(subgoal)]["V"]
            subgoal_q = self.subgoal_vfs[self.subgoals.index(subgoal)]["Q"]
            subgoal_trace = self.subgoal_vfs[self.subgoals.index(subgoal)]["e"]
            subgoal_trace_sa = self.subgoal_vfs[self.subgoals.index(subgoal)]["e_sa"]
            rho = self.imp_samp(lrow, lcol, la)
            z = 0.0 # no stopping bonus at the terminal state
            delta_i_v = reward + z - subgoal_vf[lrow][lcol]
            delta_i_q = reward + z - subgoal_q[lrow][lcol][la]
            subgoal_trace[lrow][lcol] += 1
            subgoal_trace *= rho
            subgoal_vf += self.alpha*delta_i_v*subgoal_trace
            subgoal_q += self.alpha*delta_i_q*subgoal_trace_sa
        
        target = reward + 0
        self.Q[lrow][lcol][la] += self.alpha*(target - self.Q[lrow][lcol][la])
        # Resetting last_state and last_action for the next episode
        self.last_state, self.last_action = -1, -1

    def cleanup(self):
        self.set_alpha(self.initial_alpha)
        self.Q = np.zeros((self.max_row, self.max_col, self.default_max_actions))
        self.subgoal_vfs = []
        for subgoal in self.subgoals:
            self.subgoal_vfs.append({"V": np.zeros((self.max_row, self.max_col)),
                                     "Q": np.zeros((self.max_row, self.max_col, self.default_max_actions)),
                                     "e": np.zeros((self.max_row, self.max_col)),
                                     "e_sa": np.zeros((self.max_row, self.max_col, self.default_max_actions)),
                                     })
        self.last_state, self.last_action = -1, -1
        self.steps = 0
        return

    def imp_samp(self, row, col, action):
        """
        Returns the importance sampling ratios for a given (s,a) for each subgoal's policy
        Assumes behaviour is uniform random
        """
        rhos = []

        # mu = 1/self.max_actions

        # # mu is the e-greedy policy wrt self.Q
        if action == np.argmax(self.Q[row][col]):
            mu = 1.0 - self.epsilon + self.epsilon/self.max_actions
        else:
            mu = self.epsilon/self.max_actions

        for subgoal in self.subgoals:
            # subgoal_vf = self.subgoal_vfs[self.subgoals.index(subgoal)]["V"]
            subgoal_q = self.subgoal_vfs[self.subgoals.index(subgoal)]["Q"]
            if action == np.argmax(subgoal_q[row][col]): # TODO check this - prob wrong policy
                pi = 1.0
            else:
                pi = 0.0
            rhos.append(pi/mu)
        return rhos
    
    def get_stopping_bonus(self, state):
        #TODO: fix this with eigenoptions?
        if state in self.subgoals:
            return 1.0 # in tabular case this is the same as adding \bar{w}^i
        else:
            return 0.0
        
        
    def get_v_subgoals(self, state):
        v_subgoals = []
        for subgoal in self.subgoals:
            v_subgoals.append(self.subgoal_vfs[self.subgoals.index(subgoal)]["V"][state[0]][state[1]])
        return v_subgoals
    
    def get_q_subgoals(self, state):
        q_subgoals = []
        for subgoal in self.subgoals:
            q_subgoals.append(self.subgoal_vfs[self.subgoals.index(subgoal)]["Q"][state[0]][state[1]])
        return q_subgoals
