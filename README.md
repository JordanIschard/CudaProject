# CudaProject

## Introduction

Le projet **CUDA** est réalisé en binôme. IL est ici composée de *Mathieu Guilbert* et de *Jordan Ischard*.

Nous avons choisi de faire la convolution **laplacian of gaussian** avec un **grayscale** ainsi que la convolution **à déterminer** avec un **à déterminer**. Ces convolutions seront fait respectivement par Jordan Ischard et Mathieu Guilbert.

---

## 1ère convolution : **Laplacian of Gaussian**

Cette convolution utilise la matrice suivante :
|   |   |   |   |   |
|---|---|---|---|---|
| 0 | 0 |-1 | 0 | 0 |
| 0 |-1 |-2 |-1 | 0 |
|-1 |-2 |16 |-2 |-1 |
| 0 |-1 |-2 |-1 | 0 |
| 0 | 0 |-1 | 0 | 0 |
|   |   |   |   |   |

Il est nécessaire d'effectuer un filtre **grayscale** sur l'image avant l'utilisation de cette convolution. D'où son rajout dans le programme. La première version va consister à effectuer cette convolution sur *CPU*, ensuite nous passerons sur *GPU* sans utilisation de mémoire partagée. Nous ajouterons la mémoire partagée pour voir le gain en temps d'exécution et enfin si nous avons le temps on utilisera les streams.

### Programme sur CPU

Cette version va simplement effectuer un **grayscale** et ensuite la convolution désirée. Sur cette partie aucun problème n'a été remarqué.

### Programme sur GPU

#### Première version : Sans mémoire partagée

Cette version va simplement effectuer un **grayscale** et ensuite la convolution désirée. La gain et important mais il reste un gros soucis lié au données entre le **grayscale** et la covolution qui sont obligé d'être réimportées sur le GPU.

Cette partie m'a bloqué sur un point que je ne pensais pas compliqués. Lorsque que l'on récupère les données d'une image en noir et blanc la taille de celle-ci est `rows*cols` mais par contre pour une image en couleur c'est `3* rows*cols`.

Un autre point est le lien à faire entre **grayscale** et la convolution.


#### Deuxième version : Avec mémoire partagée

#### Troisième version : Avec streams

### Résumé des résultats

| Version | Nom de l'image | Dimensions | Nombre de threads | Nombre de streams | Temps d'exécution (millisecondes) | Gain
| :--: | :--: | :--: | :--: | :--: | :--: | :--: |
| *CPU* | `gray_building.jpg` | 1023 x 1024 | X | X | 41 | X |
| *CPU* | `color_house.jpg` | 1920 x 1279 | X | X | 88 | X |
| *GPU V1* | `gray_building.jpg` | 1023 x 1024 | 16 x 16 | X | ? | ? |
| *GPU V1* | `color_house.jpg` | 1920 x 1279 | 16 x 16 | X | ? | ? |
| *GPU V1* | `gray_building.jpg` | 1023 x 1024 | 32 x 32 | X | ? | ? |
| *GPU V1* | `color_house.jpg` | 1920 x 1279 | 32 x 32 | X | ? | ? |
| *GPU V1* | `gray_building.jpg` | 1023 x 1024 | 64 x 64 | X | ? | ? |
| *GPU V1* | `color_house.jpg` | 1920 x 1279 | 64 x 64 | X | ? | ? |
| *GPU V2* | `gray_building.jpg` | 1023 x 1024 | 16 x 16 | X | ? | ? |
| *GPU V2* | `color_house.jpg` | 1920 x 1279 | 16 x 16 | X | ? | ? |
| *GPU V2* | `gray_building.jpg` | 1023 x 1024 | 32 x 32 | X | ? | ? |
| *GPU V2* | `color_house.jpg` | 1920 x 1279 | 32 x 32 | X | ? | ? |
| *GPU V2* | `gray_building.jpg` | 1023 x 1024 | 64 x 64 | X | ? | ? |
| *GPU V2* | `color_house.jpg` | 1920 x 1279 | 64 x 64 | X | ? | ? |
| *GPU V3* | `gray_building.jpg` | 1023 x 1024 | 16 x 16 | 2 | ? | ? |
| *GPU V3* | `color_house.jpg` | 1920 x 1279 | 16 x 16 | 2 | ? | ? |
| *GPU V3* | `gray_building.jpg` | 1023 x 1024 | 32 x 32 | 2 | ? | ? |
| *GPU V3* | `color_house.jpg` | 1920 x 1279 | 32 x 32 | 2 | ? | ? |
| *GPU V3* | `gray_building.jpg` | 1023 x 1024 | 64 x 64 | 2 | ? | ? |
| *GPU V3* | `color_house.jpg` | 1920 x 1279 | 64 x 64 | 2 | ? | ? |

