from matplotlib import pyplot as plt
from matplotlib import cm
from gridworld import GridWorld
from solvers import DPsolver
from pygame.locals import *
import pygame
import numpy as np


def plot_learning_curve(ql):
    values = ql.sum_rewards
    x = list(range(len(values)))
    y = values
    plt.plot(x, y, 'ro')
    plt.show()


def draw_rectangle(bckg):
    rect = pygame.Rect(20, 20, 500, 500)
    pygame.draw.rect(bckg, (0, 0, 0), rect)


def do_learning_cycle(ql):
    ql.path = []
    if not ql.is_learning_finished():
        ql.step()


def trajectory(ql):
    ql.trajectory()


class Button:
    
    def __init__(self, bckg, position, width, label):
        self.bckg = bckg
        self.position = position
        self.width = width
        self.label = label
    
    def __draw_button(self):
        x, y = self.position
        rect = pygame.Rect(x, y, self.width, 30)
        pygame.draw.rect(self.bckg, (143, 142, 140), rect)
        default_font = pygame.font.get_default_font()
        font = pygame.font.Font(default_font, 8)
        text = font.render(self.label, True, [0, 0, 0])
        text_rect = text.get_rect()
        text_rect.center = (x + self.width // 2, y + 15)
        self.bckg.blit(text, text_rect)
    
    def redraw(self):
        self.__draw_button()
    
    def register(self, action):
        self.action_when_pushed = action
    
    def operate_if_pushed(self, params):
        push = np.any(np.array(pygame.mouse.get_pressed()))
        if push:
            m_pos = pygame.mouse.get_pos()
            x, y = m_pos
            x0, y0 = self.position
            if x0 < x < x0 + self.width and y0 < y < y0 + 30:
                self.action_when_pushed(params)
                pygame.time.wait(500)


class Cell:

    def __init__(self, bckg, position, size, q_vals):
        self.bckg = bckg
        self.position = position
        self.size = size
        self.q_vals = (q_vals - np.min(q_vals)) / (np.sum(q_vals - np.min(q_vals)) + 1e-5) * 100
        # create color map
        x = np.linspace(0.0, 1.0, 100)
        self.cmap = cm.get_cmap(plt.get_cmap('summer'))(x)[:, :3]
    
    def get_color(self, q_value):
        return self.cmap[int(q_value)] * 255
    
    def draw(self):
        sx, sy = self.size[0], self.size[1]
        # draw shapes
        ul = self.position
        ur = (self.position[0] + sx, self.position[1])
        dr = (self.position[0] + sx, self.position[1] + sy)
        dl = (self.position[0], self.position[1] + sy)
        cc = (self.position[0] + sx // 2, self.position[1] + sy // 2)
        pygame.draw.polygon(self.bckg, self.get_color(self.q_vals[0]), [ul, dl, cc])
        pygame.draw.polygon(self.bckg, self.get_color(self.q_vals[1]), [ul, ur, cc])
        pygame.draw.polygon(self.bckg, self.get_color(self.q_vals[2]), [ur, dr, cc])
        pygame.draw.polygon(self.bckg, self.get_color(self.q_vals[3]), [dr, dl, cc])
        pygame.draw.line(self.bckg, [0, 0, 0], ul, dr, 2)
        pygame.draw.line(self.bckg, [0, 0, 0], ur, dl, 2)


def draw_cells(bckg, ql):
    def draw_rect(r, c, color):
        x = c * (sx+5) + x0
        y = r * (sy+5) + y0
        rect = pygame.Rect(x, y, sx, sy)
        pygame.draw.rect(bckg, color, rect)

    _, h, w = ql.q_table.shape
    sx = (500 - (w + 1) * 5) // w
    sy = (500 - (h + 1) * 5) // h
    size = (sx, sy)
    x0, y0 = 25, 25
    y = y0
    for r in range(h):
        x = x0
        for c in range(w):
            Cell(bckg, (x, y), size, ql.q_table[:, r, c]).draw()
            x += sx + 5
        y += sy + 5
    # draw obstacles
    for obs in ql.gridworld.obstacles:
        r, c = obs
        draw_rect(r, c, (140, 140, 140))
    # draw terminal:
    r, c = ql.gridworld.termin
    draw_rect(r, c, (49, 124, 245))
    # draw trajectory if exists
    for pp in ql.path:
        r, c = pp
        x = c * (sx+5) + x0 + sx // 2
        y = r * (sy+5) + y0 + sy // 2
        pygame.draw.circle(bckg, [255, 0, 0], (x, y), min(sx, sy) // 4)


def simulate_learning():
    # Initialise screen
    pygame.init()
    screen = pygame.display.set_mode((800, 600))
    pygame.display.set_caption('DP example')

    # Fill background
    background = pygame.Surface(screen.get_size())
    background = background.convert()
    background.fill((250, 250, 250))

    button_do = Button(background, (600, 150), 100, 'DO')
    button_do.register(do_learning_cycle)

    button_tra = Button(background, (600, 300), 100, 'TRA')
    button_tra.register(trajectory)

    # grid world parameters
    size = (6, 6)
    start_cell = (0, 0)
    obstacles = [(3, 3), (1, 1)]
    terminating_state = (5, 3)
    # dp parameters
    gamma = 0.9

    gw = GridWorld(size, start_cell, obstacles, terminating_state)
    ql = dp = DPsolver(gw, gamma, 15)

    # Event loop
    while not ql.is_learning_finished():
        for event in pygame.event.get():
            if event.type == QUIT:
                return
        draw_rectangle(background)
        draw_cells(background, ql)
        button_do.redraw()
        button_do.operate_if_pushed(ql)

        button_tra.redraw()
        button_tra.operate_if_pushed(ql)

        screen.blit(background, (0, 0))
        pygame.display.flip()
        pygame.time.wait(10)
        background.fill((250,250,250))
    #plot_learning_curve(ql)

simulate_learning()
