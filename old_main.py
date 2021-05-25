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

special = [1, 4]



def print_character(i, bounds, upper, exit, y):
    global temp_str, tex_str, old_temp_str, old_tex_str
    old_temp_str = temp_str
    old_tex_str = tex_str
    if exit:
        if bounds is None:
            temp_str = characters[i]
        elif bounds["upper"] == "..." and bounds["lower"] != "...":
            temp_str = characters[bounds["char"]] + "_{" +bounds["lower"] + "}"
        elif bounds["lower"] == "..." and bounds["upper"] != "...":
            temp_str = characters[bounds["char"]] + "^{" +bounds["upper"] + "}"
        else:
             temp_str = characters[bounds["char"]] + "_{" +bounds["lower"] + "}^{" + bounds["upper"] + "}"
            
            
    if not i in special and bounds is None:
        tex_str += characters[i]
        temp_str = ""
        return None
        
    else:
        if bounds is not None:
            if upper:
                if bounds["upper"] == "...":
                    bounds["upper"] = characters[i]
                else:
                    bounds["upper"] += characters[i]
            else:
                if bounds["lower"] == "...":
                    bounds["lower"] = characters[i]
                else:
                    bounds["lower"] += characters[i]
            temp_str = characters[bounds["char"]] + "_{" +bounds["lower"] + "}^{" + bounds["upper"] + "}"
            #print(temp_str)
            return bounds
        if i == 1:
            temp_str += characters[1]
            bounds = {"char": i,"upper": "...", "lower": "...", "y" : y}
            temp_str = characters[bounds["char"]] + "_{" +bounds["lower"] + "}^{" + bounds["upper"] + "}"
            return bounds
            
            


window_width = 900
window_height = 500
draw_limit = window_width // 2
title_size = window_height - window_width//2
index_min_dist = window_height//5
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
    
    
tex_str = ""
old_tex_str = ""
temp_str = ""
old_temp_str = ""
flush = True
bounds = None

try:
    init_window(screen, flush)
    lastEvent = None
    while True:
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
            print(lastEvent, e.type)
            if lastEvent == pygame.K_RETURN:
                bounds = print_character(arg, bounds, True, True, (max_y+min_y)//2)
                tex_str += temp_str
                temp_str = ""
                flush = True
                pygameSurface = pilImageToSurface(latex_to_img(tex_str))
                init_window(screen, flush=flush)
                screen.blit(pygameSurface, pygameSurface.get_rect(center = ((draw_limit+window_width)//2, window_height//2)))

                    
            else:
                lastEvent = pygame.K_RETURN
                
                img = 255-pygame.surfarray.array3d(screen)[min_x-radius:max_x+radius, min_y-radius:max_y+radius].swapaxes(0,1)
                transformed = padding(scaling([img]))
                arg = np.argmax(model.predict_proba(transformed.reshape((1,-1))))
                
                print(bounds)
                print(arg, special)
                if bounds is not None:
                    flush = False
                    print((max_y+min_y)//2, bounds["y"], index_min_dist )
                    if (max_y+min_y)//2 > bounds["y"] + index_min_dist: #indice du bas
                        bounds = print_character(arg, bounds, False, False, (max_y+min_y)//2)
                        pygameSurface = pilImageToSurface(latex_to_img(tex_str+temp_str))
                        init_window(screen, flush=flush)
                        screen.blit(pygameSurface, pygameSurface.get_rect(center = ((draw_limit+window_width)//2, window_height//2)))
                        pygame.display.flip()
                    elif (max_y+min_y)//2 < bounds["y"] - index_min_dist: #indice du haut
                        bounds = print_character(arg, bounds, True, False, (max_y+min_y)//2)
                        pygameSurface = pilImageToSurface(latex_to_img(tex_str+temp_str))
                        init_window(screen, flush=flush)
                        screen.blit(pygameSurface, pygameSurface.get_rect(center = ((draw_limit+window_width)//2, window_height//2)))
                        pygame.display.flip()
                    else:
                        bounds = print_character(arg, bounds, True, True, (max_y+min_y)//2)
                        tex_str += temp_str
                        temp_str = ""
                        pygameSurface = pilImageToSurface(latex_to_img(tex_str))
                        flush = True
                        init_window(screen, flush=flush)
                        screen.blit(pygameSurface, pygameSurface.get_rect(center = ((draw_limit+window_width)//2, window_height//2)))
                        pygame.display.flip()
                
                elif arg in special:
                    bounds = None
                    flush = False
                    temp_str = ""
                    bounds = print_character(arg, bounds, True, False, (max_y+min_y)//2)
                    #print("temp_str :", temp_str)
                    pygameSurface = pilImageToSurface(latex_to_img(tex_str+temp_str))
                    init_window(screen, flush=flush)
                    screen.blit(pygameSurface, pygameSurface.get_rect(center = ((draw_limit+window_width)//2, window_height//2)))
                    pygame.display.flip()
                else:
                    bounds = print_character(arg, bounds, True, False, (max_y+min_y)//2)
                    pygameSurface = pilImageToSurface(latex_to_img(tex_str))
                    init_window(screen, flush=flush)
                    screen.blit(pygameSurface, pygameSurface.get_rect(center = ((draw_limit+window_width)//2, window_height//2)))
                    pygame.display.flip()
                    last_y = (max_y+min_y)//2
        elif e.type == pygame.KEYDOWN and e.key == pygame.K_BACKSPACE:
            if bounds is not None and arg not in special:
                temp_str = old_temp_str
                pygameSurface = pilImageToSurface(latex_to_img(temp_str))
                init_window(screen, flush=flush)
                screen.blit(pygameSurface, pygameSurface.get_rect(center = ((draw_limit+window_width)//2, window_height//2)))
                pygame.display.flip()
            elif bounds is not None:
                temp_str = ""
                tex_str = old_tex_str
                pygameSurface = pilImageToSurface(latex_to_img(tex_str))
                init_window(screen, flush=flush)
                screen.blit(pygameSurface, pygameSurface.get_rect(center = ((draw_limit+window_width)//2, window_height//2)))
                pygame.display.flip()       
        pygame.display.flip()

except StopIteration:
    pygame.display.quit()
    pygame.quit()
    pass


# %%
