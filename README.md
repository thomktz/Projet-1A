# Overview

First-year project at ENSAE Paris  
Grade : 20/20

# Projet-1A

Projet informatique de 1ère année de  

Thomas KIENTZ  
Pierre ROUILLARD  


<img src="https://user-images.githubusercontent.com/60552083/119563408-cbb8b500-bda7-11eb-8b11-c94cd89e7226.png" alt="alt text" width="400" height="400">


# L'idée

La première idée de projet vient du constat qu'il est fastidieux pour celui qui ne sait pas coder en LaTeX d'écrire des formules mathématiques sur ordinateur. Nous avions alors pensé à faire de l' *Object Detection* sur des images de formules mathématiques.

# Les données

Il a fallu trouver un Dataset qui n'avait pas trop de classes (pour du *proof of concept*) mais qui s'appliquait parfaitement à notre problèmatique. Nous avons alors créé notre propre Dataset à l'aide d'une tablette graphique pour dessiner des caractères mathématiques.

![batch_0](https://user-images.githubusercontent.com/60552083/119355651-d25f0380-bca5-11eb-9d8d-2c9ee0dec7c0.jpg)

Puis, après [traitement](decoupeur.py), les images sont mises dans un fichier `data.npy` et les [labels](create_labels.py) dans `labels.npy`

![batch_1](https://user-images.githubusercontent.com/60552083/119355798-f91d3a00-bca5-11eb-9807-62a75cd988cc.PNG)

# Le classifier

Il faut, pour detecter des objets, pouvoir reconnaitre les objets. Pour cela, nous avons essayé deux types de réseaux :

### Le MLP

Le *MLP* (ou *Multi-Layer Perceptron*) est le réseau neuronal le plus classique possible : Il est uniquement constitué de couches de réseaux denses, et de fonctions d'activation. A l'aide de `sklearn`, on peut en entrainer un très simplement en quelques lignes, cf [sklearn_model.py](https://github.com/thomktz/Projet-1A/blob/main/sklearn_model.py)

### Réseau convolutionnel

Le réseau neuronal convolutionnel est grandement utilisé pour tout type de données 2D (ou plus) car il permet d'exploiter la structure, i.e. quels points (pixels, ici) sont proches de quels autres dans cet espace.

[La structure et la fonction forward de notre CNN sont définis dans ces lignes](CNN_model.py#L59-L80)  

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
On obtient, dans [ce notebook](https://github.com/thomktz/Projet-1A/blob/0ceee86e52f29779da767fae25ca12f78a6223f8/Version_1%20(Selective%20search).ipynb) :
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
Pour créer les *bounding boxes*, on applique à chaque mouvement de souris, quand le bouton est enfoncé, 

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

![186546175_804711137147772_5325915247756031820_n](https://user-images.githubusercontent.com/60552083/119552830-429b8100-bd9b-11eb-8217-7f3fe9f7c03e.jpg)

On définit alors un exposant comme un symbole dont la position moyenne en `y` est telle que `y < symbols[-1].y - min_dist` et un indice `y > symbols[-1].y + min_dist`. (Dans `pygame`, le coin (0,0) est en **haut** à gauche). `symbols` est la liste des symboles de hauteur `0`  

Déterminons la complexité d'un appel de `str(symbol)` :

Soit `symbol` un objet de la classe `Symbol`, de hauteur `n`

```python
C_n = C(str(symbol)) = O(1) + sum( [ C(str(symbol.exposants[i])) for i in range(len(symbol.exposants)) ] )
                            + sum( [ C(str(symbol.indices[i])) for i in range(len(symbol.indices)) ] )
```
Or, 
```python
sum( [ C(str(symbol.exposants[i])) for i in range(len(symbol.exposants)) ] ) = O(len(symbol.exposants)) * C_(n-1)

sum( [ C(str(symbol.indices[i])) for i in range(len(symbol.indices)) ] ) = O(len(symbol.indices)) * C_(n-1)
```
D'où, en notant `m` le nombre moyen d'indices et d'exposants,

![batch4](https://github.com/PierreRlld/pORJ/blob/main/complexite.png)

Cette classe règle le problème numéro 2, et rend le code un peu plus lisible.

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

# Remarques finales

### Le problème du temps

Il y a en réalité un temps assez long entre la fin du dessin et l'affichage LaTeX, de l'ordre de *2 à 3s*. Il y a les *0.7s* d'attente pour savoir si c'est bien la fin du caractère, puis *2s* d'attente. L'entiereté de ce temps (entre *98%* et *99%*) est passée à compiler et à afficher le code LaTeX, car la detection dure en moyenne *0.02s*

Pour génerer des images LaTeX, on utilise `matplotlib`. Toutes les autres alternatives qui ont été essayées n'ont pas marché sur nos machines, malgré plusieurs installations de LaTeX, beaucoup de packages python et de documentations obscures. Vu le nombre limité de caractères, il serait probablement plus rapide de recoder un LaTeX simpliste en python qui peut lire du code et sortir une image. C'est le point sur lequel nous avons le plus perdu de temps, sans succès. Enormément de solutions semblent uniquement marcher sur Linux ou comptent sur des executables introuvables ou ininstallables sous Windows (pnglatex, tex2pix, latex2png, ...)

### Test unitaire

Nous avons réalisé un [test unitaire](test_unitaire.py) sur l'appel de `str` sur un objet de la classe `Symbol`

```python
class TestSymbol(unittest.TestCase):
    def test_exposant(self):
        symb = Symbol(0,0,[0,0,0,0], None)
        symb.exposants.append(Symbol(1,0,[0,0,0,0], None))
        out = str(symb)
        self.assertEqual(out,str(characters[0]) + "^{" + str(characters[1]) + "}")
    def test_indice(self):
        symb = Symbol(0,0,[0,0,0,0], None)
        symb.indices.append(Symbol(1,0,[0,0,0,0], None))
        out = str(symb)
        self.assertEqual(out,str(characters[0]) + "_{" + str(characters[1]) + "}")
    def test_both(self):
        symb = Symbol(0,0,[0,0,0,0], None)
        symb.indices.append(Symbol(1,0,[0,0,0,0], None))
        symb.exposants.append(Symbol(2,0,[0,0,0,0], None))
        out = str(symb)
        self.assertEqual(out,str(characters[0]) + "_{" + str(characters[1]) +"}^{" + str(characters[2]) + "}")
    def test_neither(self):
        symb = Symbol(0,0,[0,0,0,0], None)
        out = str(symb)
        self.assertEqual(out,str(characters[0]))


if __name__ == '__main__':
    unittest.main()
```

On obtient le résultat suivant :

![test_unit](https://user-images.githubusercontent.com/60552083/119562103-41bc1c80-bda6-11eb-8fd1-ef82b9089a45.PNG)

### Perspectives

- Une évolution interessante du projet serait de développer l'interface graphique et de la rendre plus ergonomique et l'adapter pour la tablette. Nous ne serions plus obligés de bouger constamment les dessins et donc nous pourrions remettre le critère de distance entre deux dessins pour plus de rapidité.  
- Un moteur LaTeX fonctionnel serait nécessaire à cela, et il faudrait soit réussir à installer une version compatible et exportable de LaTeX, soit coder un interprêteur LaTeX simplifé pour nos besoins.
- Récolter un Dataset contenant plus de classes, et des structures nouvelles telles que les fractions ou les coefficient binomiaux.

### Instructions pour le code

Pour executer `main.py`, il faut s'assurer d'avoir `pygame` d'installé, ainsi que `torch`, `sklearn`, quelques autre libraries standards et d'avoir `cuda` et un GPU. Il est possible de faire tourner le code sans GPU mais il faudra modifier quelques lignes dans `main.py` et dans `CNN_model.py`.

Ensuite, executer `main.py` devrait immédiatement ouvrir la fenêtre `pygame`.

Le code est rédigé sur *Visual Studio Code*, et utilise abondément la fonctionnalité de *Notebook* intégrée, grâce aux `# %%`.

Le modèle entrainé reconnait très bien l'écriture de celui qui a fait le dataset, mais il faudra peut-être rajouter quelques points dans le dataset et fine-tuner pour reconnaitre une écriture différente
