from solvers import QLearning, DPsolver
from gridworld import GridWorld
import display


# grid world parameters
size = (6, 6)
start_cell = (0, 0)
obstacles = [(3, 3)]
terminating_state = (3, 5)
# q learning parameters
gamma = 0.9
alpha = 0.1
episodes = 1000

gw = GridWorld(size, start_cell, obstacles, terminating_state)
ql = QLearning(gw, gamma, alpha, episodes)
dp = DPsolver(gw, gamma, episodes)

while not dp.is_learning_finished():
    dp.step()
    #print(ql.sum_rewards[-1])

sum_rewards = dp.trajectory()
print(sum_rewards)
display.plot_learning_curve(ql)
