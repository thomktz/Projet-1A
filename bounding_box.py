"""
Ce programme sert à generer l'animation des bounding boxes du readme
"""
# %%
import pygame
import matplotlib.pyplot as plt
import numpy as np
from image_treatment import padding, scaling
import pickle
from latex_2_img import latex_to_img
import time


                

window_width = 900
window_height = 500
draw_limit = window_width // 2
title_size = window_height - window_width//2
index_min_dist = window_height//10
radius = 5

time_threshold = 0.7
distance_threshold = window_height//10 #entre 2 squares

min_y = np.inf
max_y = -np.inf
min_x = np.inf
max_x = -np.inf

points = []


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
    if x < min_x:
        min_x = x
    if y > max_y:
        max_y = y
    if y < min_y:
        min_y = y

def reset_extremums():
    global min_y, max_y, min_x, max_x
    min_y = np.inf
    max_y = -np.inf
    min_x = np.inf
    max_x = -np.inf

def erase_drawing(screen):
    pygame.draw.rect(screen, "white", pygame.Rect(0,title_size+3,draw_limit, window_height-draw_limit))

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

        

if __name__ == '__main__':

    model = pickle.load(open("models/sklearn_MLP.pkl", "rb"))

    old_rect = [np.inf,-np.inf,np.inf,-np.inf]
    
    latex = True
    pygame.init()
    screen = pygame.display.set_mode((window_width,window_height))
    screen.fill("white")
    pygame.display.flip()
    draw_on = False
    color = "black"
    pygame.font.init()
    myfont = pygame.font.SysFont('Garamond', 30)

    continuer = True
    symbols = []
    has_drawn = False
    end_time = time.time()
    frames = 0
    
    try:
        init_window(screen, True)
        while continuer:
            #print(time.time()-end_time)   
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

                    draw_line(screen, "black", last_pos, e.pos)
                    pygame.draw.circle(screen, color, e.pos, radius)
                    draw_line(screen, color, e.pos, last_pos)
                    old_min_x, old_max_x, old_min_y, old_max_y = old_rect
                    pygame.draw.rect(screen, "white", (old_min_x-radius, old_min_y-radius, old_max_x-old_min_x+2*radius, old_max_y-old_min_y+2*radius), 3)
                    pygame.draw.rect(screen, "red", (min_x-radius, min_y-radius, max_x-min_x+2*radius, max_y-min_y+2*radius), 3)
                    old_rect = (min_x, max_x, min_y,max_y)
                    pygame.display.flip()
                    pygame.image.save(screen, f"frames/frame_{str(frames).zfill(4)}.jpeg")
                    frames += 1
                last_pos = e.pos
    except StopIteration:
        pygame.display.quit()
        pygame.quit()
        pass
    # %%

import cv2
import numpy as np
import glob
 
img_array = []
for filename in glob.glob('frames_bb/*.jpeg'):
    img = cv2.imread(filename)
    height, width, layers = img.shape
    size = (width,height)
    img_array.append(img)
 
 
out = cv2.VideoWriter('bounding_boxes.avi',cv2.VideoWriter_fourcc(*'DIVX'), 60, size)
 
for i in range(len(img_array)):
    out.write(img_array[i])
out.release()
# %%
