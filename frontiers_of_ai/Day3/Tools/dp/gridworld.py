import numpy as np


class GridWorld:

    def __init__(self, size, start_cell, obstacles, terminating_state):
        self.size = size
        self.start = start_cell
        self.obstacles = obstacles
        self.termin = terminating_state
        self.current_cell = self.start
    
    def reset(self, start=None):  # we can use random too
        if start is None:
            self.current_cell = self.start
        else:
            self.current_cell = start
    
    def transition(self, cell, action):
        r, c = cell
        if cell == self.termin:
            self.current_cell = (r, c)
            return (r, c)
        if action == 0:  # left
            c_next = max(0, c - 1)
            r_next = r
        elif action == 1:  # up
            c_next = c
            r_next = max(0, r - 1)
        elif action == 2:  # right
            c_next = min(self.size[1]-1, c + 1)
            r_next = r
        elif action == 3:  # down
            c_next = c
            r_next = min(self.size[0]-1, r + 1)
        if (r_next, c_next) in self.obstacles:
            self.current_cell = (r, c)
            return (r, c)
        self.current_cell = (r_next, c_next)
        return (r_next, c_next)

    def reward(self, cell, action):
        if cell == self.termin:
            return 0
        return -1
    
    def in_terminal(self):
        return self.current_cell == self.termin
