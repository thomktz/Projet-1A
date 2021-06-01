# %%
import pygame
import matplotlib.pyplot as plt
import numpy as np
from image_treatment import padding, scaling
import pickle
from latex_2_img import latex_to_img
from CNN_model import load_model
import cv2
import time
import torch


# On veut pouvoir switcher entre les deux modèles facilement
models = {"sklearn" : {"model" :pickle.load(open("models/sklearn_MLP.pkl", "rb"))}
          ,"pytorch" :  {"model" : load_model("models/CNN_1.pth")}}
models["sklearn"]["predict"] = lambda x : models["sklearn"]["model"].predict_proba(x.reshape((1,-1)))
models["pytorch"]["predict"] = lambda x : models["pytorch"]["model"](torch.Tensor(x).unsqueeze(0)).detach().numpy()[0]

# Liste des caractères sur lesquels le réseau est entrainé
characters = ["a",
              "\sum ",
              "\\forall ",
              "\exists ",
              "\int ",
              "\mathbb{R}",
              "\in ",
              ",",
              "x",
              "\geq ",
              "\leq ",
              "=",
              "i",
              "n"]


def list_str(list):
    """
    Renvoie la liste des strings des elements d'une liste
    """
    return [str(e) for e in list]

class Symbol():
    """
    Classe principale sur laquelle repose le programme
    """
    def __init__(self, i, height,rect, parent = None):
        miny, maxy, minx, maxx = rect
        self.parent = parent
        self.height = height         #-1 : indice, 0 : normal, 1 : exposant
        self.y = (miny+maxy)//2      # Sera utilisé pour déterminer la hauteur
        self.base_character = characters[i]
        self.indices = []            # Liste d'objets de la classe Symbol
        self.exposants = []          # Liste d'objets de la classe Symbol
        self.rect = rect
        self.last_addition = [None]  # Une pile qui sert à pouvoir supprimer les caractères dans l'ordre
    
    def __str__(self):
        """
        Fonction récursive qui renvoie le code LateX d'un symbole, avec indices et exposants 
        """
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



# Parametres de fenêtre
window_width = 900
window_height = 500
draw_limit = window_width // 2                # Délimite la partie dans laquelle on peut dessiner (gauche)
title_size = window_height - window_width//2  # Délimite la partie dans laquelle on peut dessiner (haut)
radius = 5                                    # Rayon du trait du dessin

# Parametres de détection
index_min_dist = window_height//10     # Distance minimale verticale entre un caractere et son indice/exposant
time_threshold = 0.7                   # Temps à partir duquel on considère que le dessin est fini
distance_threshold = window_height//10 # [plus utilisé] Distance minimale entre 2 traits pour considerer qu'ils sont de deux dessins différents

min_y = np.inf
max_y = -np.inf
min_x = np.inf
max_x = -np.inf


def distance_from_rect(rect, p):
    """
    Renvoie la distance entre un point et un rectangle
    N'est plus utilisée dans la version finale
    """
    x,y = p
    min_y, max_y, min_x, max_x = rect
    dx = max(min_x - x, 0, x - max_x)
    dy = max(min_y - y, 0, y - max_y)
    return np.sqrt(dx*dx + dy*dy)


def pilImageToSurface(pilImage):
    """
    Prend l'image PIL LaTeX générée par matplotlib et la coupe et la traite pour être affichable avec Pygame
    """
    image = 255- np.array(pilImage)                      # On inverse l'image pour avoir le bord blanc égal à 0 et les caractères noirs égaux à 255
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  
    x, y, w, h = cv2.boundingRect(image)                 # On trouve le plus petit rectangle non nul contenant les caractères
    newImg = np.swapaxes((255-image[y:y+h, x:x+w]),0,1)  # On coupe l'image et on l'inverse une fois de plus pour retrouver les couleurs de base
    return pygame.surfarray.make_surface(newImg)


def draw_line(srf, color, start, end):
    """
    Dessine une ligne continue entre deux points
    """
    dx = end[0]-start[0]
    dy = end[1]-start[1]
    distance = max(abs(dx), abs(dy))
    for i in range(distance):
        x = int( start[0]+float(i)/distance*dx)
        y = int( start[1]+float(i)/distance*dy)
        pygame.draw.circle(srf, color, (x, y), radius)


def update_extremums(e):
    """
    Met à jour les extremums de la bounding box.
    Est appelée à chaque mouvement de souris.
    """
    global min_y, max_y, min_x, max_x
    x, y = e.pos
    if x > max_x:
        max_x = x
    if x < min_x:  # On n'utilise pas 'elif' pour ne pas avoir de problème en cas de trait monotone. Il n'y a aucun ralentissement
        min_x = x
    if y > max_y:
        max_y = y
    if y < min_y:  
        min_y = y


def reset_extremums():
    """
    Remet à zéro les extremums de la bounding box
    Est appelée quand un caractère est reconnu
    """
    global min_y, max_y, min_x, max_x
    min_y = np.inf
    max_y = -np.inf
    min_x = np.inf
    max_x = -np.inf


def init_window(screen, flush, reset_extrems = True):
    """
    (re)Dessine l'interface graphique (texte, traits de délimitation)
    """
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
    

def predict_screen(screen, predict, debug = False):
    """
    Entrées : Ecran, fonction de detection
    Prend l'écran, en extrait le dessin, le transforme en (32,32) et prédit ce qui s'y trouve
    Renvoie l'indice du caractère reconnu dans la liste
    """
    global min_y, max_y, min_x, max_x
    if debug:
        start = time.time()
    array = 255-pygame.surfarray.array3d(screen)[min_x-radius:max_x+radius, min_y-radius:max_y+radius].swapaxes(0,1) # On inverse les couleurs : Il parait plus logique d'avoir la partie "vide"
    transformed = padding(scaling([array])) # On transforme la zone en forme (32,32)                                 # du dessin égale à 0 et la partie remplie égale à 255
    pred = predict(transformed) # On prédit la zone (32,32)                                                          # Le même traitement a été fait lors de l'apprentissage
    if debug:
        print("prediction : ",pred)
        print("temps pour prédire : ", time.time()-start)
    arg = np.argmax(pred)
    return arg
    
    
def move_drawing(screen, margin = radius):
    """
    Bouge le dessin vers la gauche jusqu'à ce qu'il touche l'écran
    """
    global min_y, max_y, min_x, max_x
    Z = pygame.surfarray.array3d(screen)[min_x-radius:max_x+radius, title_size+2:]
    surf = pygame.surfarray.make_surface(Z)
    init_window(screen, flush=True, reset_extrems=False)
    screen.blit(surf, [margin, title_size+2, max_x-min_x, window_height-(title_size+2)])
    pygame.display.flip()


def draw_latex(screen, symbols, font = None):
    """
    Affiche en LaTeX sur l'écran la liste symbols
    
    Cette fonction à été source de nombreux problèmes car il y avait une incompatibilité entre l'execution du code dans un notebook (avec les # %% de Visual Studio Code)
    et l'execution normale. L'execution en Notebook marchait très bien mais pour une raison étrange (comportement différent de matplotlib) les images apparaissaient en enorme et de manière
    décalée dans l'éxecution normale. Il fallut alors faire un traitement plus fort à l'image de sortie de matplotlib (cf pilImageToSurface et le code dans latex_2_img.py)
    """
    if symbols != []:
        string = "".join(list_str(symbols))
        image = latex_to_img(string)
        pygameSurface = pilImageToSurface(image)
        a,b,c,d = pygameSurface.get_rect()
        pygame.draw.rect(screen, "white", pygame.Rect(draw_limit+3, title_size+3, window_width-draw_limit, window_height-title_size))
        screen.blit(pygameSurface, ((window_width+draw_limit)//2-c//2, (window_height+title_size)//2-d//2,c,d))
        pygame.display.flip()
        reset_extremums()
    else:
        pygame.draw.rect(screen, "white", pygame.Rect(draw_limit+3, title_size+3, window_width-draw_limit, window_height-title_size))
        pygame.display.flip()
        reset_extremums()
    if font is not None: # Permet l'execution de cette fonction dans draw_data.py. Dans draw_data.py, python ne savait pas qui était "myfont". Pour ne pas modifier tout main.py, 
        textsurface = font.render('Status : Done', True, "red")                                                                             # cette solution à été choisie
        tx, ty = font.size("Status : Done...")
    else:
        textsurface = myfont.render('Status : Done', True, "red")
        tx, ty = myfont.size("Status : Done...")
    screen.blit(textsurface, (window_width-tx-10, title_size+ty//2))
    pygame.display.flip()
    #print("Done", time.time())
   

def update_symbols(screen, detected, latex = True):
    """
    Modifie la liste 'symbols' avec le symbole 'detected', en fonction de sa position y
    On peut choisir d'afficher l'image compilée en LaTeX ou le string grâce au booléen 'latex' pour aller plus vite
    """
    if symbols != [] and detected.y < symbols[-1].y - index_min_dist: # detected est un exposant de symbols[-1]
        detected.parent = symbols[-1]
        detected.height = 1
        symbols[-1].exposants.append(detected)
        symbols[-1].last_addition.append(1)
                    
    elif symbols != [] and detected.y > symbols[-1].y + index_min_dist: # detected est un indice de symbols[-1]
        detected.parent = symbols[-1]
        detected.height = -1
        symbols[-1].indices.append(detected)
        symbols[-1].last_addition.append(-1)
                
    else: #detected est un caractère normal
        detected.height = 0
        symbols.append(detected)
        move_drawing(screen)
    if latex:
        textsurface = myfont.render('Status : Drawing '+ detected.base_character, True, "red")
        tx, ty = myfont.size('Status : Drawing '+ detected.base_character)
        pygame.draw.rect(screen, "white", pygame.Rect(draw_limit+3, title_size+3, window_width-draw_limit, 2*ty))
        screen.blit(textsurface, (window_width-tx-10, title_size+ty//2))
        pygame.display.flip()
        draw_latex(screen, symbols)
    else:
        out="".join(list_str(symbols))
        textsurface = myfont.render(out, True, "black")
        tx, ty = myfont.size(out)
        pygame.draw.rect(screen, "white", pygame.Rect(draw_limit+3, title_size+3, window_width-draw_limit, window_height-title_size))
        screen.blit(textsurface, (3*window_width//4-tx//2, window_height//2-ty//2))
        pygame.display.flip()
        reset_extremums()
        
def delete_last_symbol(screen, latex):
    """
    Permet de supprimer le dernier symbole de la pile (ou son dernier symbole en indice ou exposant)
    Peut être appelée plusieurs fois d'affilée grâce à la pile Symbol.last_addition pour supprimer dans l'ordre
    """
    global symbols
    if symbols[-1].last_addition[-1] == 1: # Si le dernier symbole ajouté à symbols[-1] est un exposant
        symbols[-1].exposants.pop(-1)
        symbols[-1].last_addition.pop(-1)
    elif symbols[-1].last_addition[-1] == -1: # Si le dernier symbole ajouté à symbols[-1] est un indice
        symbols[-1].indices.pop(-1)
        symbols[-1].last_addition.pop(-1)
    else: # Si symbols[-1] n'a ni exposant ni indice
        symbols.pop(-1)
        pygame.draw.rect(screen, "white", pygame.Rect(0, title_size+3, window_width-draw_limit-3, window_height-title_size))
    if latex:
        draw_latex(screen, symbols)
    else:
        out="".join(list_str(symbols))
        textsurface = myfont.render(out, True, "black")
        tx, ty = myfont.size(out)
        pygame.draw.rect(screen, "white", pygame.Rect(0, title_size+3, window_width-draw_limit-3, window_height-title_size))
        pygame.draw.rect(screen, "white", pygame.Rect(draw_limit+3, title_size+3, window_width-draw_limit, title_size))
        screen.blit(textsurface, (3*window_width//4-tx//2, window_height//2-ty//2))
        pygame.display.flip()
        reset_extremums()
        

if __name__ == '__main__':
    """
    Boucle principale, pour ne pas être lancée quand on utilise les fonctions de main.py dans d'autres fonctions comme draw_latex.py
    """

    recording = False # Permet d'enregistrer les frames de l'écran
    frames = 0        # Pour dénombrer les frames enregistrées
    latex = True      # Affiche les images compilées ou simplement le string
    model_type = "pytorch"  # Permet de selectionner le modèle ("pytorch" ou "sklearn")
    predict = models[model_type]["predict"]
    pygame.init()
    screen = pygame.display.set_mode((window_width,window_height))
    screen.fill("white")
    pygame.display.flip()
    draw_on = False # Nous ne sommes pas entrain de dessiner
    color = "black"
    pygame.font.init()
    myfont = pygame.font.SysFont('Garamond', 30)

    continuer = True
    symbols = []
    has_drawn = False # On n'a pas encore dessiné. On en a besoin pour savoir si on n'a pas dessiné depuis un certain temps. Sans ce paramêtre, le programme 
                      # essayerait de prédire un écran vide dès lors qu'on ne ferait rien pendant plus de 0.7s
    end_time = time.time()
    
    try:
        init_window(screen, True)
        while continuer:
            if has_drawn and time.time() - end_time > time_threshold: # Si on a déjà dessiné, et qu'on a arrêté de dessiner depuis plus de 'time_threshold' secondes :
                has_drawn = False
                #print(time.time() - end_time, time_threshold)                  # Prints qui servaient à savoir quelle partie de programme prenait du temps
                #print("Predicting screen ")                                    # C'est finalement la partie matplotlib qui est chronophage
                t1 = time.time()
                arg = predict_screen(screen, predict, debug = False)
                #print("Done", time.time()-t1)                                  # La prédiction durait entre 0.01 et 0.02s
                detected = Symbol(arg, None, [min_y, max_y, min_x, max_x])
                #print("Updating", time.time())
                update_symbols(screen, detected, latex = latex)
            
            if recording:
                pygame.image.save(screen, f"frames_main/frame_{str(frames).zfill(4)}.jpeg")
            frames += 1
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    raise StopIteration
                elif e.type == pygame.MOUSEBUTTONDOWN and e.pos[0] < draw_limit:
                    has_drawn = False
                    update_extremums(e)
                    pygame.draw.circle(screen, color, e.pos, radius)
                    pygame.display.flip()
                    draw_on = True
                elif e.type == pygame.MOUSEBUTTONUP:
                    has_drawn = True
                    end_time = time.time()
                    draw_on = False
                    
                elif e.type == pygame.MOUSEMOTION:
                    if draw_on and e.pos[0] < draw_limit and e.pos[1] > title_size:
                        update_extremums(e)
                        pygame.draw.circle(screen, color, e.pos, radius)
                        draw_line(screen, color, e.pos, last_pos)
                        pygame.display.flip()
                    last_pos = e.pos
                elif e.type == pygame.KEYDOWN and e.key == pygame.K_BACKSPACE:
                    delete_last_symbol(screen, latex)
                    
    except StopIteration:
        
        pygame.display.quit()
        pygame.quit()
        pass
    # %%

def create_video():
    """
    Sert à créer la vidéo (puis le .gif) à l'aide les frames enregistrées dans un lancement précedant du main avec 'recording = True'
    """
    import cv2
    import numpy as np
    import glob
    
    img_array = []
    for filename in glob.glob('frames_main/*.jpeg'):
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width,height)
        img_array.append(img)
    
    
    out = cv2.VideoWriter('main.avi',cv2.VideoWriter_fourcc(*'DIVX'), 100, size)
    
    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()
# %%
