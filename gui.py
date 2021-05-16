# %%
import pygame
import matplotlib.pyplot as plt
import numpy as np
from image_treatment import padding, scaling
import pickle
from latex_2_img import latex_to_img

characters = ["a",
              "\sum_"+"{...}"+"^"+"{...}"+"...",
              "\forall",
              "\exists",
              "\int_{...}^{...}...",
              "\mathbb{R}",
              "\in",
              ",",
              "x",
              ">=",
              "<",
              "<="]
special = [1, 4]


def print_character(real_str, temp_str, i, bounds, upper, exit):
    if exit:
        if bounds is None:
            if bounds["upper"] == "" and bounds["lower"] == "":
                
            
    if not i in special and bounds is None:
        real_str += characters[i]
        temp_str += characters[i]
        return None
        
    else:
        if bounds is not None:
            if upper:
                bounds["upper"] +=characters[i]
                if temp_str[-2] == "...": #dans le cas ou c'est le premier caractère de la borne
                    temp_str[-7:-4] = bounds["upper"]
            else:
                bounds["lower"] += characters[i]
        if i == 1:
            temp_str += characters[1]
            bounds = {"char": i,"upper": "", "lower": ""}
            return bounds
            
            


window_width = 900
window_height = 500
draw_limit = window_width // 2
title_size = window_height - window_width//2
index_min_dist = window_height//10
radius = 5

model = pickle.load(open("models/sklearn_MLP.pkl", "rb"))

pygame.init()
screen = pygame.display.set_mode((window_width,window_height))
screen.fill("white")
draw_on = False
color = "black"
pygame.font.init()
myfont = pygame.font.SysFont('Garamond', 30)

img = []

min_y = np.inf
max_y = -np.inf
min_x = np.inf
max_x = -np.inf


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


def init_window(screen, flush):
    global min_y, max_y, min_x, max_x
    if flush:
        screen.fill("white")
    else:
        
    pygame.draw.line(screen, "black", (draw_limit, 0), (draw_limit, window_height), width=3)
    pygame.draw.line(screen, "black", (0, title_size), (window_width, title_size), width=3)
    textsurface = myfont.render('Input drawing', True, "black")
    tx, ty = myfont.size("Input drawing")
    screen.blit(textsurface, (window_width//4-tx//2, title_size//2-ty//2))
    textsurface = myfont.render('Output', True, "black")
    tx, ty = myfont.size("Output")
    screen.blit(textsurface, (3*window_width//4-tx//2, title_size//2-ty//2))
    min_y = np.inf
    max_y = -np.inf
    min_x = np.inf
    max_x = -np.inf
    
    
tex_str = []
temp_str = []
flush = True
bounds = None

try:
    init_window(screen, flush)
    while True:
        lastEvent = None
        e = pygame.event.wait()
        if e.type == pygame.QUIT:
            raise StopIteration
        if e.type == pygame.MOUSEBUTTONDOWN and e.pos[0] < draw_limit:
            lastEvent = None
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
            lastEvent = None
            draw_on = False
        if e.type == pygame.MOUSEMOTION:
            if draw_on and e.pos[0] < draw_limit and e.pos[1] > title_size:
                lastEvent = None
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
        if e.type == pygame.KEYDOWN and e.key == pygame.K_RETURN:
            if lastEvent == pygame.K_RETURN:
                flush = True
                init_window(screen, flush=flush)
                if arg ==1:
                    tex_str = tex_str[:-16] + " "
                    
            else:
                lastEvent = pygame.K_RETURN
                
                img = 255-pygame.surfarray.array3d(screen)[min_x-radius:max_x+radius, min_y-radius:max_y+radius].swapaxes(0,1)
                transformed = padding(scaling([img]))
                arg = np.argmax(model.predict_proba(transformed.reshape((1,-1))))
                tex = characters[arg]
                tex_str += tex
                if arg in special or bounds is not None:
                    flush = False
                    if (max_y+min_y)//2) > y + index_min_dist: #indice du bas
                        bounds = print_character(tex_str, temp_str, arg, bounds, False, False)
                        pygameSurface = pilImageToSurface(latex_to_img(''.join(temp_str)))
                        init_window(screen, flush=flush)
                        screen.blit(pygameSurface, pygameSurface.get_rect(center = ((draw_limit+window_width)//2, window_height//2)))
                        pygame.display.flip()
                    elif (max_y+min_y)//2) < y - index_min_dist: #indice du haut
                        bounds = print_character(tex_str temp_str, arg, bounds, True, False)
                        pygameSurface = pilImageToSurface(latex_to_img(''.join(temp_str)))
                        init_window(screen, flush=flush)
                        screen.blit(pygameSurface, pygameSurface.get_rect(center = ((draw_limit+window_width)//2, window_height//2)))
                        pygame.display.flip()
                    else:
                        bounds = print_character(tex_str, temp_str, arg, bounds, True)
                        pygameSurface = pilImageToSurface(latex_to_img(''.join(text_str)))
                        flush = True
                        init_window(screen, flush=flush)
                        screen.blit(pygameSurface, pygameSurface.get_rect(center = ((draw_limit+window_width)//2, window_height//2)))
                        pygame.display.flip()

                pygameSurface = pilImageToSurface(latex_to_img(''.join(tex_str)))
                init_window(screen, flush=flush)
                screen.blit(pygameSurface, pygameSurface.get_rect(center = ((draw_limit+window_width)//2, window_height//2)))
                pygame.display.flip()
                last_y = (max_y+min_y)//2)

        pygame.display.flip()

except StopIteration:
    pygame.display.quit()
    pygame.quit()
    pass

# %%
clf.predict_proba([X_test[0]])
# %%
