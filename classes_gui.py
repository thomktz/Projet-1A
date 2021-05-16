# %%
import pygame
import matplotlib.pyplot as plt
import numpy as np
from image_treatment import padding, scaling
import pickle
from latex_2_img import latex_to_img

characters = ["a",
              "\sum ",
              "\\forall ",
              "\exists ",
              "\int",
              "\mathbb{R}",
              "\in ",
              ",",
              "x",
              "\geq ",
              "<",
              "\leq ",
              "=",
              "i",
              "0",
              "1"]

def list_str(list):
    return [str(e) for e in list]

class Symbol():
    def __init__(self, i, height,y, parent = None):
        self.parent = parent
        self.height = height         #-1 : indice, 0 : normal, 1 : exposant
        self.y = y
        self.base_character = characters[i]
        self.indices = []
        self.exposants = []
    
    def __str__(self):
        if len(self.indices) > 0:
            if len(self.exposants) > 0:
                return self.base_character + "_{" + "".join(list_str(self.indices))+ "}^{" + "".join(list_str(self.exposants)) + "}"
            else:
                return self.base_character + "_{" + "".join(list_str(self.indices))+ "}"
        else:
            if len(self.exposants) > 0:
                return self.base_character + "^{" + "".join(list_str(self.exposants)) + "}"
            else:
                return self.base_character
                

window_width = 900
window_height = 500
draw_limit = window_width // 2
title_size = window_height - window_width//2
index_min_dist = window_height//10
radius = 5

min_y = np.inf
max_y = -np.inf
min_x = np.inf
max_x = -np.inf

model = pickle.load(open("models/sklearn_MLP.pkl", "rb"))

pygame.init()
screen = pygame.display.set_mode((window_width,window_height))
screen.fill("white")
pygame.display.flip()
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

def update_extremums(e):
    global min_y, max_y, min_x, max_x
    x, y = e.pos
    if x > max_x:
        max_x = x
    elif x < min_x:
        min_x = x
    if y > max_y:
        max_y = y
    elif y < min_y:
        min_y = y

def reset_extremums():
    global min_y, max_y, min_x, max_x
    min_y = np.inf
    max_y = -np.inf
    min_x = np.inf
    max_x = -np.inf

def init_window(screen, flush, reset_extrems = True):
    global min_y, max_y, min_x, max_x
    if flush:
        screen.fill("white")
    pygame.draw.line(screen, "black", (draw_limit, 0), (draw_limit, window_height), width=3)
    pygame.draw.line(screen, "black", (0, title_size), (window_width, title_size), width=3)
    textsurface = myfont.render('Input drawing', True, "black")
    tx, ty = myfont.size("Input drawing")
    screen.blit(textsurface, (window_width//4-tx//2, title_size//2-ty//2))
    textsurface = myfont.render('Output', True, "black")
    tx, ty = myfont.size("Output")
    screen.blit(textsurface, (3*window_width//4-tx//2, title_size//2-ty//2))
    if reset_extrems:
        reset_extremums()
    pygame.display.flip()
    
def predict_screen(screen):
    global min_y, max_y, min_x, max_x
    array = 255-pygame.surfarray.array3d(screen)[min_x-radius:max_x+radius, min_y-radius:max_y+radius].swapaxes(0,1)
    transformed = padding(scaling([array]))
    arg = np.argmax(model.predict_proba(transformed.reshape((1,-1))))
    return arg
    
def move_drawing(screen, margin = radius):
    global min_y, max_y, min_x, max_x
    Z = pygame.surfarray.array3d(screen)[min_x-radius:max_x+radius, title_size+2:]
    surf = pygame.surfarray.make_surface(Z)
    init_window(screen, flush=True, reset_extrems=False)
    screen.blit(surf, [margin, title_size+2, max_x-min_x, window_height-(title_size+2)])
    pygame.display.flip()

def draw_latex(screen, symbols):
    string = "".join(list_str(symbols))
    pygameSurface = pilImageToSurface(latex_to_img(string))
    screen.blit(pygameSurface, pygameSurface.get_rect(center = ((draw_limit+window_width)//2, window_height//2)))
    pygame.display.flip()
    reset_extremums()


continuer = True

symbols = []

try:
    init_window(screen, True)
    while continuer:
        e = pygame.event.wait()
        if e.type == pygame.QUIT:
            raise StopIteration
        elif e.type == pygame.MOUSEBUTTONDOWN and e.pos[0] < draw_limit:
            update_extremums(e)
            pygame.draw.circle(screen, color, e.pos, radius)
            pygame.display.flip()
            draw_on = True
        elif e.type == pygame.MOUSEBUTTONUP:
            draw_on = False
        elif e.type == pygame.MOUSEMOTION:
            if draw_on and e.pos[0] < draw_limit and e.pos[1] > title_size:
                update_extremums(e)
                pygame.draw.circle(screen, color, e.pos, radius)
                draw_line(screen, color, e.pos, last_pos)
                pygame.display.flip()
            last_pos = e.pos
        elif e.type == pygame.KEYDOWN and e.key == pygame.K_RETURN:
            arg = predict_screen(screen)
            detected = Symbol(arg, None, (max_y+min_y)//2)
            if symbols != [] and detected.y < symbols[-1].y - index_min_dist: # detected est un exposant de symbols[-1]
                detected.parent = symbols[-1]
                detected.height = 1
                symbols[-1].exposants.append(detected)
                            
            elif symbols != [] and detected.y > symbols[-1].y + index_min_dist: # detected est un indice de symbols[-1]
                detected.parent = symbols[-1]
                detected.height = -1
                symbols[-1].indices.append(detected)
                        
            else: #detected est un caractère normal
                detected.height = 0
                symbols.append(detected)
                move_drawing(screen)
                
            draw_latex(screen, symbols)
except StopIteration:
    pygame.display.quit()
    pygame.quit()
    pass
# %%
