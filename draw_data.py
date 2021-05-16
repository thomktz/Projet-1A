# %%
data = np.load("data/data.npy")
labels = np.load("data/labels.npy")
# %%

from gui import characters, radius, color, draw_limit, window_width, window_height, title_size
import pygame
from latex_2_img import latex_to_img
import numpy as np
import random

to_draw = [12,13,14,15]

min_y = np.inf
max_y = -np.inf
min_x = np.inf
max_x = -np.inf
pygame.init()
screen = pygame.display.set_mode((window_width,window_height))
screen.fill("white")
draw_on = False
color = "black"
pygame.font.init()
myfont = pygame.font.SysFont('Garamond', 30)

def pilImageToSurface(pilImage):
    return pygame.image.fromstring(
        pilImage.tobytes(), pilImage.size, pilImage.mode).convert()

def draw_line(srf, color, start, end):
    dx = end[0]-start[0]
    dy = end[1]-start[1]
    distance = max(abs(dx), abs(dy))
    for i in range(distance):
        x = int( start[0]+float(i)/distance*dx)
        y = int( start[1]+float(i)/distance*dy)
        pygame.draw.circle(srf, color, (x, y), radius)


def init_window(screen):
    global min_y, max_y, min_x, max_x
    screen.fill("white")
        
    pygame.draw.line(screen, "black", (draw_limit, 0), (draw_limit, window_height), width=3)
    pygame.draw.line(screen, "black", (0, title_size), (window_width, title_size), width=3)
    textsurface = myfont.render('Draw', True, "black")
    tx, ty = myfont.size("Draw")
    screen.blit(textsurface, (window_width//4-tx//2, title_size//2-ty//2))
    textsurface = myfont.render('Objective', True, "black")
    tx, ty = myfont.size("Objective")
    screen.blit(textsurface, (3*window_width//4-tx//2, title_size//2-ty//2))
    min_y = np.inf
    max_y = -np.inf
    min_x = np.inf
    max_x = -np.inf



textsurface = myfont.render('Press Enter to start', True, "black")
tx, ty = myfont.size("Press Enter to start")
screen.blit(textsurface, (window_width//2, (window_height-title_size)//2))

try:
    init_window(screen, flush)
    while True:
        e = pygame.event.wait()
        if e.type == pygame.QUIT:
            raise StopIteration
        if e.type == pygame.MOUSEBUTTONDOWN and e.pos[0] < draw_limit:
            x, y = e.pos
            if x > max_x:
                max_x = x
            elif x < min_x:
                min_x = x
            if y > max_y:
                max_y = y
            elif y < min_y:
                min_y = y
            pygame.draw.circle(screen, color, e.pos, radius)
            draw_on = True
        if e.type == pygame.MOUSEBUTTONUP:
            draw_on = False
        elif e.type == pygame.MOUSEMOTION:
            if draw_on and e.pos[0] < draw_limit and e.pos[1] > title_size:
                x, y = e.pos
                if x > max_x:
                    max_x = x
                elif x < min_x:
                    min_x = x
                if y > max_y:
                    max_y = y
                elif y < min_y:
                    min_y = y
                pygame.draw.circle(screen, color, e.pos, radius)
                draw_line(screen, color, e.pos, last_pos)
            last_pos = e.pos
        elif e.type == pygame.KEYDOWN and e.key == pygame.K_RETURN:
            i = to_draw[random.randint(0,len(to_draw)-1)]
            char = characters[i]
            pygameSurface = pilImageToSurface(latex_to_img(char))
            init_window(screen)
            screen.blit(pygameSurface, pygameSurface.get_rect(center = ((draw_limit+window_width)//2, window_height//2)))

# %%
