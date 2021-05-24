# %%
from main_v3 import *
import random

# Si premier lancement :
# data = np.empty((0,32,32))
# labels = np.empty((0))


data = np.load("data/data.npy")
labels = np.load("data/labels.npy")


#indexes_to_draw = [i for i in range(14)]
indexes_to_draw = [0,13]
characters_to_draw = [characters[i] for i in indexes_to_draw]

min_y = np.inf
max_y = -np.inf
min_x = np.inf
max_x = -np.inf

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

def init_window_draw(screen):
    global min_y, max_y, min_x, max_x
    screen.fill("white")
    pygame.draw.line(screen, "black", (draw_limit, 0), (draw_limit, window_height), width=3)
    pygame.draw.line(screen, "black", (0, title_size), (window_width, title_size), width=3)
    textsurface = myfont.render('Drawing', True, "black")
    tx, ty = myfont.size("Drawing")
    screen.blit(textsurface, (window_width//4-tx//2, title_size//2-ty//2))
    textsurface = myfont.render('Target', True, "black")
    tx, ty = myfont.size("Target")
    screen.blit(textsurface, (3*window_width//4-tx//2, title_size//2-ty//2))
    reset_extremums()
    pygame.display.flip()

def extract_symbol(screen):
    global min_y, max_y, min_x, max_x
    #print(min_y, max_y, min_x, max_x, radius)
    array = 255-pygame.surfarray.array3d(screen)[min_x-radius:max_x+radius, min_y-radius:max_y+radius].swapaxes(0,1)
    #plt.imshow(array)
    #plt.show()
    transformed = padding(scaling([array]))
    return transformed

def update_db(symbol, i, save = True):
    global labels, data
    labels = np.concatenate((labels, [i]), axis = 0)
    data = np.concatenate((data, symbol), axis = 0)
    idx = np.random.permutation(len(data))
    x,y = data[idx], labels[idx]
    if save:
        np.save("data/data.npy", x)
        np.save("data/labels.npy", y)
    

if __name__ == '__main__':

    pygame.init()
    screen = pygame.display.set_mode((window_width,window_height))
    screen.fill("white")
    pygame.display.flip()
    draw_on = False
    color = "black"
    pygame.font.init()
    myfont = pygame.font.SysFont('Garamond', 30)

    continuer = True
    nb_added = 0

    try:
        init_window_draw(screen)
        textsurface = myfont.render('Press Enter to start', True, "black")
        tx, ty = myfont.size("Press Enter to start")
        screen.blit(textsurface, (window_width//2, window_height//2))
        pygame.display.flip()
        
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
                update_extremums(e)
                draw_on = False
            elif e.type == pygame.MOUSEMOTION:
                if draw_on and e.pos[0] < draw_limit and e.pos[1] > title_size:
                    update_extremums(e)
                    pygame.draw.circle(screen, color, e.pos, radius)
                    draw_line(screen, color, e.pos, last_pos)
                    pygame.display.flip()
                last_pos = e.pos
            elif e.type == pygame.KEYDOWN and e.key == pygame.K_RETURN:
                if nb_added == 0:
                    init_window_draw(screen)
                    nb_added =+1
                    rd = indexes_to_draw[random.randint(0,len(indexes_to_draw)-1)]
                    draw_latex(screen, [Symbol(rd,0,[0,0,0,0],None)])
                else:
                    print(nb_added)
                    nb_added =+1
                    update_db(extract_symbol(screen), rd)
                    init_window_draw(screen)
                    rd = indexes_to_draw[random.randint(0,len(indexes_to_draw)-1)]
                    draw_latex(screen, [Symbol(rd,0,[0,0,0,0],None)])
            elif e.type == pygame.KEYDOWN and e.key == pygame.K_BACKSPACE:
                reset_extremums()
                init_window_draw(screen)
                rd = indexes_to_draw[random.randint(0,len(indexes_to_draw)-1)]
                draw_latex(screen, [Symbol(rd,0,[0,0,0,0],None)])
    except StopIteration:
        pygame.display.quit()
        pygame.quit()
        pass
# %%
