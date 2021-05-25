# Projet-1A

Thomas KIENTZ  
Pierre ROUILLARD

# L'idée

La première idée de projet vient du constat qu'il est fastidieux pour celui qui ne sait pas coder en LaTeX d'écrire des formules mathématiques sur ordinateur. Nous avions alors pensé à faire de l' *Object Detection* sur des images de formules mathématiques.

# Les données

Il a fallu trouver un Dataset qui n'avait pas trop de classes (pour du *proof of concept*) mais qui s'appliquait parfaitement à notre problematique. Nous avons alors créé notre propre Dataset à l'aide d'une tablette graphique pour dessiner des caractères mathématiques.

![batch_0](https://user-images.githubusercontent.com/60552083/119355651-d25f0380-bca5-11eb-9d8d-2c9ee0dec7c0.jpg)

Puis, après traitement, les images sont mises dans un fichier `data.npy` et les labels dans `labels.npy`

![batch_1](https://user-images.githubusercontent.com/60552083/119355798-f91d3a00-bca5-11eb-9807-62a75cd988cc.PNG)

# Le classifier

Il faut, pour detecter des objets, pouvoir reconnaitre les objets. Pour cela, nous avons essayé deux types de réseaux :

### Le MLP

Le *MLP* (ou *Multi-Layer Perceptron*) est le réseau neuronal le plus classique possible : Il est uniquement constitué de couches de réseaux denses, et de fonctions d'activation. A l'aide de `sklearn`, on peut en entrainer un très simplement en quelques lignes, cf [sklearn_model.py](https://github.com/thomktz/Projet-1A/blob/main/sklearn_model.py)

### Réseau convolutionnel

Le réseau neuronal convolutionnel est grandement utilisé pour tout type de données 2D (ou plus) car il permet d'exploiter la structure, i.e. quels points (pixels, ici) sont proches de quels autres dans cet espace.

[La structure et la fonction forward de notre CNN sont définis dans ces lignes](https://github.com/thomktz/Projet-1A/blob/ffe490b3460205f07d11ebe2575f33fa40d3da9f/CNN_model.py#L56-L83)  

On utilisera finalement le réseau convolutionnel dans la version finale. La première version, présentée juste après utilisait encore le MLP.
Les classes à reconnaitre sont :

```python
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
```


# La première version

Dans cette idée d'*Object Detection*, il nous faut d'abord proposer des "boites" (*bounding boxes*) qui peuvent contenir des caractères pour ensuite classifier ce que l'on detecte. On applique pour cela la `SelectiveSearchSegmentation` de `OpenCV`.
Ainsi à partir de notre formule de base :
![batch2](https://github.com/PierreRlld/pORJ/blob/main/formule.jpg)
On obtient :
![batch3](https://github.com/PierreRlld/pORJ/blob/main/formule%20SS.png)
On remarque cependant d'innombrables boxes se superposant sur un même caractère. L'objectif est d'en réduire le nombre le plus possible.
On va donc pour cela implémenter la méthode *NMS* pour `Non Maximum Suppression`.

Entrée : 

- Ensemble des boxes fournies par la *Selective Search* (liste B) et chacune possède un "score"  
- Liste F vide  
- Seuil N  
         
Principe : 
- La box de score le plus élevé est ajoutée dans la liste F et est prise comme boxe de référence.
- On calcule le rapport *IoU* (*Intersection over Union*) de cette box avec toutes les autres boxes.
- Toutes celles dont la valeur *IoU* dépasse le seuil fixé sont retirées de la liste B.
- Répéter jusqu'à ce que la liste B soit vide.

On obtient alors en sortie :

![batch4](https://github.com/PierreRlld/pORJ/blob/main/r%C3%A9sultat.png)

Le résultat est loin d'être parfait, et necessite d'écrire sur une tablette pour avoir une photo assez nette et pour avoir toujours la même epaisseur de trait. Une simple capture d'écran d'une tablette après avoir écrit est inefficace car on exploite pas le fait d'avoir l'évolution du tracé.

# La deuxième version (`old_main.py`)

L'idée phare de cette nouvelle version est l'exploitation du mouvement de la souris (ou du stylet de la tablette) pour detecter les *bounding boxes*. Il faut donc une interface graphique sur laquelle on peut dessiner. La partie graphique du projet est assurée par `pygame` et `matplotlib`.
Pour creer les *bounding boxes*, on applique à chaque mouvement de souris, quand le bouton est enfoncé, 

```python
if x > max_x:
    max_x = x
if x < min_x:
    min_x = x
if y > max_y:
    max_y = y
if y < min_y:
    min_y = y
```
On ne peut pas remplacer un `if` par un `elif` pour gagner en rapidité car un trait strictement monotone en x ou en y laissera un min ou un max à sa valeur initiale, qui est ±∞  
En animation, et en prenant compte de la taille du trait :

![bounding_boxes](https://user-images.githubusercontent.com/60552083/119487647-f2e89580-bd59-11eb-9da9-c8b5d6dba21d.gif)

Pour valider la selection, il suffit d'appuyer sur *Entrée* et, en fonction de la valeur de `latex` (`True` ou `False`), le programme interprête le dessin et montre la prédiction sous forme de code LaTeX ou comme image compilée de code LaTeX. Si le symbole est en indice ou en exposant (selon une distance des centres par rapport au dernier), alors le dernier reste à l'écran, sinon l'écran est réinitialisé.

Les problèmes de cette version sont :
1. Il faut appuyer sur *Entrée* entre chaque caractère
2. la façon dont le code latex est géré (à l'aide d'un `string` temporaire, un `string` final et un dictionnaire des exposants et indices pour les caractères "spéciaux", qui sont dans une liste à part) est très sale, limitée et rend le code illisible, impossible à améliorer et à développer.
3. La façon dont le dessin reste ou disparait de l'écran n'est pas très naturelle

# La version finale (`main.py`)

Cette version tente de résoudre les problèmes mentionnés plus haut. La grande "innovation" de cette version est la classe `Symbol` :

```python
class Symbol():
    def __init__(self, i, height, rect, parent = None):
        miny, maxy, minx, maxx = rect
        self.parent = parent
        self.height = height         #-1 : indice, 0 : normal, 1 : exposant
        self.y = (miny+maxy)//2
        self.base_character = characters[i]
        self.indices = []
        self.exposants = []
        self.rect = rect
        self.last_addition = None
    
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
```

avec

```python
def list_str(list):
    return [str(e) for e in list]
```
Cette structure en "arbres" (`Symbol` ~ `Noeud(caractère, exposants, indices)` avec `exposants` et `indices` des Noeuds) permet d'ajouter et de retirer des exposants et les indices dans l'odre qu'on le souhaite, très facilement. Pour sortir le `string` LaTeX, il suffit d'appeler `str(symbol)` qui appelle recursivement `str(exposants[i])` et `str(indices[i])` pour tout i et reconstruit le string final. On se contente pour cette version d'une hauteur de l'arbre de 1, i.e. racine et une feuille de chaque coté car il est difficile de savoir si un symbole est un indice d'un exposant, un exposant d'un indice ou un symbole normal à la hauteur 0.  

Cette classe règle le problèmre numéro 2, et rend le code un peu plus lisible.

Pour règler le numéro 3, c'est maintenant un peu plus simple : Si `symbol.hauteur == 0`, il est balayé vers la gauche jusqu'au bord de l'écran. Sinon, rien ne bouge

Enfin, pour le problème numéro 1, deux critères ont été envisagés
- Premièrement, le temps entre deux traits. Si on dépasse une certaine valeur de différence de temps depuis notre dernier trait, on évalue automatiquement. Avec une souris, une valeur de *0,7s* est recommandée, avec un stylo et sur tablette on pourrait descendre à *0.5s*
- Ensuite, la distance entre un nouveau trait et la fin de l'ancien. Si cette valeur dépasse une valeur fixée en pixels, alors on évalue automatiquement. Cette méthode à l'inconvénient de ne pas être compatible avec la nouvelle méthode qui règle le problème numéro 3.  
On utilise alors uniquement le premier.

![main](https://user-images.githubusercontent.com/60552083/119516823-5c2ad180-bd77-11eb-9172-6e9a1bd23307.gif)
>La vidéo est un peu trompeuse. Les frames sont enregistrées à chaque tour de boucle et donc pas pendant le chargement. Il y a un temps de chargement plus grand entre la fin du dessin et l'apparition du code LaTeX. Il y a plus de détails sur cela dans les remarques finales.

On peut appuyer sur *Retour* pour effacer le dernier symbole, qu'il soit indice, exposant ou normal.

# La nouvelle base de données (`draw_data.py`)

Le programme étant actuellement utilisé sur ordinateur avec une fonction bien précise qui genère les dessins toujours de la même façon, il y a une grande perte d'efficacité à ne pas avoir un dataset généré au même endroit, i.e. dans notre programme.

Alors nous avons créé un programme qui fait l'inverse de `main.py`, c'est à dire qui nous montre un symbole et qui nous fait le dessiner. Tout est automatiquement enregistré à chaque ajout. Les symboles sont affichés au hasard pour ne pas écrire toujours les symboles de la même manière. On appuie sur *Entrée* pour valider, et on peut appuyer sur *Retour* pour annuler.

![draw_data](https://user-images.githubusercontent.com/60552083/119520386-8af67700-bd7a-11eb-9a2b-fe2d7acf02c0.gif)

