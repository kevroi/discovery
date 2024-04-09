import numpy as np
import pickle
from utils.plot_utils import to_env_list
import matplotlib.pyplot as plt


# Default actions in GridWorld environment
default_action_set = [(0, 1), (0, -1), (1, 0), (-1, 0)] # R, L, D, U

TERMINATE_ACTION = (0,0)

def initialize_env(env_type):

    if env_type == "Grid":
        return GridEnvironment()
    elif env_type == "4-Room":
        return RoomEnvironment()
    elif env_type == "I-Maze":
        return I_MazeEnvironment()
    else:
        print("Invalid Environment: " + env_type)
        exit()

# returns relevant environment information
def parse_env(env_grid):
    max_row = len(env_grid)
    max_col = len(env_grid[0])
    obstacles = []
    lava = []
    phi_1 = []
    phi_2 = []
    start_state = (-1,-1)
    goal_state = (-1,-1)
    for r in range(max_row):
        for c in range(max_col):
            if env_grid[r][c] == 'X':
                obstacles.append((r,c))
            elif env_grid[r][c] == 'L':
                lava.append((r,c))
            elif env_grid[r][c] == 'S':
                start_state = (r,c)
            elif env_grid[r][c] == 'G':
                goal_state = (r,c)
            elif env_grid[r][c] == '1':
                phi_1.append((r,c))
            elif env_grid[r][c] == '2':
                phi_1.append((r,c))
                phi_2.append((r,c))
            elif env_grid[r][c] == '3':
                phi_2.append((r,c))

    return max_row, max_col, start_state, goal_state, obstacles, lava, phi_1, phi_2

class BaseEnvironment(object):

    def __init__(self, max_row, max_col, start_state,
                 goal_state, obstacle_vector = [], lava_vector = [],
                 reward_vector = None, phi_1=None, phi_2=None):
        states_rc = [(r, c) for r in range(max_row) for c in range(max_col)]
        self.states_rc = states_rc # all possible states (r,c)
        self.max_row, self.max_col = max_row, max_col

        # use exploring starts if start_state is None
        self.start_state = start_state 
        self.goal_state = goal_state

        self.action_set = default_action_set[:]
        self.default_max_actions = len(self.action_set) # will stay fixed
        self.max_actions = len(self.action_set) # can increase if we add more options

        self.obstacle_vector = obstacle_vector
        self.lava_vector = lava_vector
        self.phi_1 = phi_1
        self.phi_2 = phi_2

        if reward_vector is None:
            # Reward is 0.0 everywhere, and 1.0 in goal state
            self.reward_vector = np.zeros((len(self.states_rc))) * 0.0
            try: # It is possible that there's no goal state e.g. (-1,-1)
                goal_idx = self.states_rc.index(goal_state)
                self.reward_vector[goal_idx] = 1.0
            except ValueError:
                pass

            for obs_state in obstacle_vector:
                obs_idx = self.states_rc.index(obs_state)
                self.reward_vector[obs_idx] = float('-inf')

        else:
            self.reward_vector = reward_vector

        self.current_state = None

    def start(self):
        # exploring starts
        if self.start_state == (-1,-1):
            valid_state_idx = [idx for idx, state in enumerate(self.states_rc) if state not in self.obstacle_vector]
            start_state_int = np.random.choice(valid_state_idx)
            #start_state_int = np.random.randint(len(self.states_rc))
        # start state is specified
        else:
            start_state_int = self.states_rc.index(self.start_state)
        self.current_state = np.asarray([start_state_int])

        # Returning a copy of the current state
        return np.copy(self.current_state)

    def step(self, action):
        if not action < self.max_actions:
            print ("Invalid action taken!!")
            print ("action: ", action)
            print ("current_state", self.current_state)

        action = self.action_set[action]
        # print("state_rc", self.states_rc[self.current_state[0]], "action", action)

        # if terminate action
        if action == TERMINATE_ACTION:
            self.current_state = None
            result = {"reward": 0, "state": None, "isTerminal": True}
            return result

        else:
            s = self.current_state[0]

            # Getting the coordinate representation of the state
            s = self.states_rc[s]
            nr = min(self.max_row - 1, max(0, s[0] + action[0]))
            nc = min(self.max_col - 1, max(0, s[1] + action[1]))
            ns = (nr, nc) 


            # if s or ns is an obstacle, don't move
            if s in self.obstacle_vector or ns in self.obstacle_vector:
                # Going back to the integer representation
                s = self.states_rc.index(s)
                ns = s #self.states_rc.index(ns) # same state
                reward = 0 #- 0.001 # small step penalty
            else:
                # Going back to the integer representation
                s = self.states_rc.index(s)
                ns = self.states_rc.index(ns) # next state
                reward = self.reward_vector[ns] - self.reward_vector[s] #- 0.001 # small step penalty

            # check lava
            if self.states_rc[ns] in self.lava_vector:
                reward = -1.0
        
            self.current_state[0] = ns
        # print(reward)
        # check terminal
        if self.goal_state != (-1,-1) and \
            self.states_rc.index(self.goal_state) == self.current_state[0]:

            self.current_state = None
            result = {"reward": 1.0, "state": None, "isTerminal": True}
            return result

        else:
            result = {"reward": reward, "state": self.current_state,
                      "isTerminal": False}
            return result

    def cleanup(self):
        self.current_state = None
        return

    # Getter and Setter functions
    def set_start_state(self, start_state):
        self.start_state = start_state

    def set_goal_state(self, goal_state):
        self.goal_state = goal_state
    def add_terminate_action(self):
        self.action_set.append(TERMINATE_ACTION)
        self.max_actions = len(self.action_set)

    def get_grid_dimension(self):
        return self.max_row, self.max_col

    def get_default_max_actions(self):
        return self.default_max_actions

    def set_current_state(self, current_state):
        self.current_state = np.asarray([current_state])

    def set_eigen_purpose(self, eigen_purpose):
        self.reward_vector = eigen_purpose


class GridEnvironment(BaseEnvironment):
    def __init__(self):
        grid_env = to_env_list('environments/env_layouts/gridenv.txt')
        max_row, max_col, start_state, goal_state, obstacles, lava, _, _ = parse_env(grid_env)
        self.max_row = max_row
        self.max_col = max_col
        self.name = "grid"

        BaseEnvironment.__init__(self, max_row, max_col, start_state,
                 goal_state, obstacles, lava)


class RoomEnvironment(BaseEnvironment):
    def __init__(self):
        room_env = to_env_list('environments/env_layouts/room.txt')
        max_row, max_col, start_state, goal_state, obstacles, lava, _, _ = parse_env(room_env)
        self.max_row = max_row
        self.max_col = max_col
        self.name = "rooms"

        BaseEnvironment.__init__(self, max_row, max_col, start_state,
                 goal_state, obstacles, lava)



class I_MazeEnvironment(BaseEnvironment):
    def __init__(self):
        I_maze_env = to_env_list('environments/env_layouts/imaze.txt')
        max_row, max_col, start_state, goal_state, obstacles, lava, _, _ = parse_env(I_maze_env)
        self.max_row = max_row
        self.max_col = max_col
        self.name = "IMaze"

        BaseEnvironment.__init__(self, max_row, max_col, start_state,
                 goal_state, obstacles, lava)


class HallEnvironment(BaseEnvironment):
    def __init__(self):
        room_env = to_env_list('environments/env_layouts/hall.txt')
        max_row, max_col, start_state, goal_state, obstacles, lava, _, _ = parse_env(room_env)
        self.max_row = max_row
        self.max_col = max_col
        self.name = "hall"

        BaseEnvironment.__init__(self, max_row, max_col, start_state,
                 goal_state, obstacles, lava)


class ToyEnvironment(BaseEnvironment):
    def __init__(self, name):
        room_env = to_env_list(f"environments/env_layouts/{name}.txt")
        max_row, max_col, start_state, goal_state, obstacles, lava, _, _ = parse_env(room_env)
        self.max_row = max_row
        self.max_col = max_col
        self.name = name

        BaseEnvironment.__init__(self, max_row, max_col, start_state,
                 goal_state, obstacles, lava)
        

class phi1Environment(BaseEnvironment):
    def __init__(self, name):
        room_env = to_env_list(f"environments/env_layouts/{name}.txt")
        max_row, max_col, start_state, goal_state, obstacles, lava, phi_1, phi_2 = parse_env(room_env)
        self.max_row = max_row
        self.max_col = max_col
        self.name = name
        self.ph1 = phi_1

        BaseEnvironment.__init__(self, max_row, max_col, start_state,
                 goal_state, obstacles, lava, phi_1=phi_1)

    def step(self, action):
        if not action < self.max_actions:
            print ("Invalid action taken!!")
            print ("action: ", action)
            print ("current_state", self.current_state)

        action = self.action_set[action]
        # print("state_rc", self.states_rc[self.current_state[0]], "action", action)

        # if terminate action
        if action == TERMINATE_ACTION:
            self.current_state = None
            result = {"reward": 0, "state": None, "isTerminal": True}
            return result

        else:
            s = self.current_state[0]

            # Getting the coordinate representation of the state
            s = self.states_rc[s]
            nr = min(self.max_row - 1, max(0, s[0] + action[0]))
            nc = min(self.max_col - 1, max(0, s[1] + action[1]))
            ns = (nr, nc) 

            # Check if the next state is in phi_1
            if ns in self.phi_1:
                ns = self.phi_1[1]  # Assuming all states in phi_1 are treated as the same state

            # if s or ns is an obstacle, don't move
            if s in self.obstacle_vector or ns in self.obstacle_vector:
                # Going back to the integer representation
                s = self.states_rc.index(s)
                ns = s #self.states_rc.index(ns) # same state
                reward = 0 #- 0.001 # small step penalty
            else:
                # Going back to the integer representation
                s = self.states_rc.index(s)
                ns = self.states_rc.index(ns) # next state
                reward = self.reward_vector[ns] - self.reward_vector[s] #- 0.001 # small step penalty

            # check lava
            if self.states_rc[ns] in self.lava_vector:
                reward = -1.0

            self.current_state[0] = ns
        # print(reward)
        # check terminal
        if self.goal_state != (-1,-1) and \
            self.states_rc.index(self.goal_state) == self.current_state[0]:

            self.current_state = None
            result = {"reward": 1.0, "state": None, "isTerminal": True}
            return result

        else:
            result = {"reward": reward, "state": self.current_state,
                    "isTerminal": False}
            return result


class phi2Environment(BaseEnvironment):
    def __init__(self, name):
        room_env = to_env_list(f"environments/env_layouts/{name}.txt")
        max_row, max_col, start_state, goal_state, obstacles, lava, phi_1, phi_2 = parse_env(room_env)
        self.max_row = max_row
        self.max_col = max_col
        self.name = name
        self.ph1 = phi_1

        BaseEnvironment.__init__(self, max_row, max_col, start_state,
                 goal_state, obstacles, lava, phi_2=phi_2)

    def step(self, action):
        if not action < self.max_actions:
            print ("Invalid action taken!!")
            print ("action: ", action)
            print ("current_state", self.current_state)

        action = self.action_set[action]
        # print("state_rc", self.states_rc[self.current_state[0]], "action", action)

        # if terminate action
        if action == TERMINATE_ACTION:
            self.current_state = None
            result = {"reward": 0, "state": None, "isTerminal": True}
            return result

        else:
            s = self.current_state[0]

            # Getting the coordinate representation of the state
            s = self.states_rc[s]
            nr = min(self.max_row - 1, max(0, s[0] + action[0]))
            nc = min(self.max_col - 1, max(0, s[1] + action[1]))
            ns = (nr, nc) 

            # Check if the next state is in phi_1
            if ns in self.phi_2:
                ns = self.phi_2[1]  # first state in phi_2 is the bottleneck state

            # if s or ns is an obstacle, don't move
            if s in self.obstacle_vector or ns in self.obstacle_vector:
                # Going back to the integer representation
                s = self.states_rc.index(s)
                ns = s #self.states_rc.index(ns) # same state
                reward = 0 #- 0.001 # small step penalty
            else:
                # Going back to the integer representation
                s = self.states_rc.index(s)
                ns = self.states_rc.index(ns) # next state
                reward = self.reward_vector[ns] - self.reward_vector[s] #- 0.001 # small step penalty

            # check lava
            if self.states_rc[ns] in self.lava_vector:
                reward = -1.0

            self.current_state[0] = ns
        # print(reward)
        # check terminal
        if self.goal_state != (-1,-1) and \
            self.states_rc.index(self.goal_state) == self.current_state[0]:

            self.current_state = None
            result = {"reward": 1.0, "state": None, "isTerminal": True}
            return result

        else:
            result = {"reward": reward, "state": self.current_state,
                    "isTerminal": False}
            return result

def plot_gridworld(file_path, hallways = None):
    with open(file_path, 'r') as file:
        grid_data = file.read().splitlines()

    rows = len(grid_data)
    cols = len(grid_data[0])

    grid = [[cell for cell in row] for row in grid_data]

    for row in range(rows):
        for col in range(cols):
            if grid[row][col] == 'S':
                start_x, start_y = col, row
            elif grid[row][col] == 'G':
                goal_x, goal_y = col, row

    if hallways:
        for hallway in hallways:
            grid[hallway[0]][hallway[1]] = 'H'
            # for row in range(hallway[0], hallway[1] + 1):
            #     for col in range(hallway[2], hallway[3] + 1):
            #         grid[row][col] = 'H'
    

    plt.figure(figsize=(cols, rows))
    plt.imshow([[ colour_from_letter(cell) for cell in row] for row in grid], cmap='gray', interpolation='none', aspect='equal')

    # # Mark the start and goal positions
    # plt.plot(start_x, start_y, 'bs', markersize=40)
    # plt.plot(goal_x, goal_y, 'gs', markersize=40)

    # Customize grid lines
    plt.xticks(range(cols), [])
    plt.yticks(range(rows), [])
    
    # Draw black grid lines
    for i in range(cols + 1):
        plt.axvline(x=i - 0.5, color='#45474B', linewidth=2)
    for i in range(rows + 1):
        plt.axhline(y=i - 0.5, color='#45474B', linewidth=2)

    plt.gca().set_facecolor('white')
    plt.show()

def colour_from_letter(letter):
    if letter == 'X':
        return [114,154,159]
    elif letter == 'S':
        return [75,162,185]
    elif letter == 'G':
        return [173,226,181]
    elif letter == 'H':
        return [219,92,84]
    elif letter =='L':
        return [252,223,141]
    else:
        return [255,255,255]

if __name__ == "__main__":
    file_path = "environments/env_layouts/2room.txt"  # Replace with the path to your text file
    plot_gridworld(file_path)
