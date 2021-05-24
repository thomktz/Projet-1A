# Projet-1A

Thomas KIENTZ
Pierre ROUILLARD

## L'idée

La première idée de projet vient du constat qu'il est fastidieux pour celui qui ne sait pas coder en LaTeX d'écrire des formules mathématiques sur ordinateur. Nous avions alors pensé à faire de l' *Object Detection* sur des images de formules mathématiques.

## Les données

Il a fallu trouver un Dataset qui n'avait pas trop de classes (pour du *proof of concept*) mais qui s'appliquait parfaitement à notre problematique. Nous avons alors créé notre propre Dataset à l'aide d'une tablette graphique pour dessiner des caractères mathématiques.

![batch_0](https://user-images.githubusercontent.com/60552083/119355651-d25f0380-bca5-11eb-9d8d-2c9ee0dec7c0.jpg)

Puis, après traitement, les images sont mises dans un fichier `data.npy` et les labels dans `labels.npy`

![batch_1](https://user-images.githubusercontent.com/60552083/119355798-f91d3a00-bca5-11eb-9807-62a75cd988cc.PNG)

## La première version

Dans cette idée d'*Object Detection*, il nous faut d'abord proposer des "boites" (*bounding boxes*) qui peuvent contenir des caractères pour ensuite classifier ce que l'on detecte. On applique pour cela la `SelectiveSearchSegmentation` de `OpenCV`

