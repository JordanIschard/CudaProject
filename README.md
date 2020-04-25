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

Il est nécessaire d'effectuer un filtre **grayscale** sur l'image avant l'utilisation de cette convolution. D'où son rajout dans le programme. La première version va consister à effectuer cette convolution sur *CPU*, ensuite nous passerons sur *GPU* sans utilisation de mémoire partagée. Nous ajouterons la mémoire partagée pour voir le gain en temps d'exécution et enfin si nous avons le temps on utilisera les streams.

### Programme sur CPU

Cette version va simplement effectuer un **grayscale** et ensuite la convolution désirée. Sur cette partie aucun problème n'a été remarqué.

| Version | Nom de l'image | Dimensions | Temps d'exécution (millisecondes) |
| :--: | :--: | :--: | :--: |
| *CPU* | `color_building.jpg` | 853 x 1280 | 28 |
| *CPU* | `color_house.jpg` | 1920 x 1279 | 63 |

### Programme sur GPU

#### Première version : Sans mémoire partagée

Cette version va effectuer un **grayscale** et ensuite la convolution désirée. La gain et important mais il reste un gros soucis lié au données entre le **grayscale** et la covolution qui sont obligé d'être réimportées sur le GPU.

Cette partie m'a bloqué sur un point que je ne pensais pas compliqués. Lorsque que l'on récupère les données d'une image en noir et blanc la taille de celle-ci est `rows*cols` mais par contre pour une image en couleur c'est `3* rows*cols`.

| Version | Nom de l'image | Dimensions | Nombre de threads | Temps d'exécution (millisecondes) | Gain sur la dernière version
| :--: | :--: | :--: | :--: | :--: | :--: |
| *GPU V1* | `color_building.jpg` | 853 x 1280 | 16 x 16 | 0.40 | x70 |
| *GPU V1* | `color_house.jpg` | 1920 x 1279 | 16 x 16 | 0.75 | x84 |
| *GPU V1* | `color_building.jpg` | 853 x 1280 | 32 x 32 | 0.33 | x84 |
| *GPU V1* | `color_house.jpg` | 1920 x 1279 | 32 x 32 | 0.60 | x105 |

#### Deuxième version : Avec mémoire partagée

Afin d'avoir un meilleur gain, nous avons ajouter une mémoire partagée. Les données de l'image initiale sont convertie pour donner une image en noir et blanc. Dans la version précédente on perdait du temps à récupérer les données intermédiaire. Pour palier à cela on mais les données en noir et blanc dans la mémoire partagée.

Un problème est survenu via l'utilisation de la mémoire partagée. En effet, les indices étant un peu modifié on avait un effet cadriage sur l'image de sortie.

| Version | Nom de l'image | Dimensions | Nombre de threads | Temps d'exécution (millisecondes) | Gain sur la dernière version
| :--: | :--: | :--: | :--: | :--: | :--: |
| *GPU V2* | `color_building.jpg` | 853 x 1280 | 16 x 16 | 0.28 | x1.43 |
| *GPU V2* | `color_house.jpg` | 1920 x 1279 | 16 x 16 | 0.51 | x1.47 |
| *GPU V2* | `color_building.jpg` | 853 x 1280 | 32 x 32 | 0.26 | x1.27 |
| *GPU V2* | `color_house.jpg` | 1920 x 1279 | 32 x 32 | 0.48 | x1.25 |

#### Troisième version : Avec streams

| Version | Nom de l'image | Dimensions | Nombre de threads | Nombre de streams | Temps d'exécution (millisecondes) | Gain sur la dernière version
| :--: | :--: | :--: | :--: | :--: | :--: | :--: |
| *GPU V3* | `color_building.jpg` | 853 x 1280 | 16 x 16 | 2 | ? | ? |
| *GPU V3* | `color_house.jpg` | 1920 x 1279 | 16 x 16 | 2 | ? | ? |
| *GPU V3* | `color_building.jpg` | 853 x 1280 | 32 x 32 | 2 | ? | ? |
| *GPU V3* | `color_house.jpg` | 1920 x 1279 | 32 x 32 | 2 | ? | ? |

### Résumé des résultats

| Version | Nombre de threads optimale | Nombre de streams optimale | Temps d'exécution en millisecondes (moyen sur les images testées) | Gain sur la dernière version (moyen)
| :--: | :--: | :--: | :--: | :--: |
| *GPU V1* | 32 x 32 | X | 0.46 | x97 |
| *GPU V2* | 32 x 32 | X | 0.37 | x1.26 |
| *GPU V3* | 32 x 32 | ? | ? | ? |
