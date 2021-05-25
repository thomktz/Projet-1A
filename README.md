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

# La deuxième version

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
On ne peut pas remplacer un `if` par un `elif` pour gagner en rapidité car un trait monotone en x ou en y laissera un min ou un max à sa valeur initiale, qui est ±∞  
En animation, et en prenant compte de la taille du trait :

![bounding_boxes](https://user-images.githubusercontent.com/60552083/119487647-f2e89580-bd59-11eb-9da9-c8b5d6dba21d.gif)



