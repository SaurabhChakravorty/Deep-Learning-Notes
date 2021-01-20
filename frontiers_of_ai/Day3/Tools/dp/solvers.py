import numpy as np


class DPsolver:

    def __init__(self, gridworld, gamma, iterations):
        self.gridworld = gridworld
        self.gamma = gamma
        self.iterations = iterations
        self.cntr = 0
        size = gridworld.size
        self.q_table = np.zeros((4, size[0], size[1]), dtype=np.float32)
        self.path = []
    
    def step(self):
        rows, cols = self.gridworld.size
        for r in range(rows):
            for c in range(cols):
                for act in range(4):
                    reward = self.gridworld.reward((r, c), act)
                    cell_next = self.gridworld.transition((r, c), act)
                    r2, c2 = cell_next
                    self.q_table[act, r, c] = reward + self.gamma * np.max(self.q_table[:, r2, c2])
        self.cntr += 1
    
    def trajectory(self):
        self.gridworld.reset()
        sum_rewards = 0
        itr = 0
        while not self.gridworld.in_terminal() and itr < 20:
            r, c = self.gridworld.current_cell
            action = np.argmax(self.q_table[:, r, c])
            self.gridworld.transition((r, c), action)
            sum_rewards += self.gridworld.reward((r, c), action)
            self.path.append((r, c))
            itr += 1
        return sum_rewards

    def is_learning_finished(self):
        return self.cntr > self.iterations
