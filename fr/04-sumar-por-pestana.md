# Les tenseurs à partir de zéro (4) : additionner par onglet

*Des totaux, des moyennes et le meilleur jour de la semaine, tirés d'un tenseur en une seule ligne, et la règle qui ne fait jamais défaut : l'axe que tu additionnes disparaît.*

*Les tenseurs à partir de zéro · leçon A, session 4 sur 8*

*Scénario : les personnages et l'histoire sont inventés. Les données techniques, historiques et les chiffres réels indiquent leur source.*

**Lundi 23 novembre, 19 h 40. Réserve de la succursale Centre.**

Le rideau métallique est déjà à mi-hauteur. Ça sent le carton neuf et le café réchauffé. Mateo a posé son ordinateur portable sur un carton de cahiers à lignes, et doña Lupe, debout derrière lui, n'a pas enlevé son pull.

« Il reste trente jours », dit-elle. « Monsieur Ortega m'a encore appelée. Il veut “quelque chose de concret”. »

« J'y suis presque, ma tante. »

« C'est ce que tu as dit lundi dernier. Et celui d'avant. »

Mateo ne répond pas. Depuis une semaine, il découpe le tenseur de la papeterie en tranches : un jour par-ci, un produit par-là. Il a vu des choses bizarres. Le vendredi de Nord, mou comme un mardi. Les sacs à dos, immobiles. Mais ce sont des morceaux. Et avec des morceaux, on ne convainc pas une banque.

« J'ai besoin de tout voir ensemble », dit-il. « Pas un jour. Pas un produit. La semaine entière de chaque succursale, en un seul nombre. »

« Et ça, l'ordinateur le fait ? »

« Il le fait en une ligne. Si je lui dis bien quoi regrouper. »

Doña Lupe s'assoit sur un pot de peinture retourné.

« Je te le dis tout de suite, mon petit : Nord est sur l'avenue, à côté de l'arrêt de bus. Par cette porte-là, il entre deux fois plus de monde que par celle-ci. Si elle vend moins qu'ici, je veux savoir combien de moins. »

Mateo pose les doigts sur le clavier. Jusqu'ici, il a appris à **extraire des morceaux** du tenseur. Aujourd'hui, c'est l'inverse : **résumer**. Regrouper beaucoup de nombres en un seul, selon l'axe qu'il choisit.

Dehors, un bus passe et le rideau vibre.

« Allez, vas-y », dit doña Lupe. « Fais parler ton tenseur. »

*C'est la dernière session de la leçon A. Tu vas faire la même chose que Mateo, étape par étape.*

## Révision pour commencer

Réponds sans regarder la session 3. Les réponses se trouvent dans les solutions, à la fin.

1. Quelle forme a `papeleria[:, 4]` et que représente-t-il ?
2. Quelle différence de forme y a-t-il entre `papeleria[0]` et `papeleria[0:1]` ?
3. Comment extrais-tu les ventes de sacs à dos des deux succursales pour tous les jours ?

> **Le code de cette session** se trouve dans le dépôt public du cours, dans le dossier `sesion-04` : [github.com/HabilMX/curso-tensores](https://github.com/HabilMX/curso-tensores/tree/main/sesion-04). Écris-le toi-même d'abord, car c'est ainsi qu'on apprend ; le dépôt sert à comparer quand quelque chose ne donne pas le même résultat chez toi. Chaque programme est accompagné de la sortie qu'il doit te donner.

## Objectifs de cette session

À la fin, tu seras capable de faire ces quatre choses. À côté de chacune se trouve l'exercice qui le démontre.

| # | À la fin, tu peux… | Démontré par |
|---|---|---|
| 1 | **Prédire** la forme qui reste après avoir additionné un tenseur selon un ou plusieurs axes, avant d'exécuter le code | Exercice 14 |
| 2 | **Calculer** des totaux et des moyennes par axe avec `sum` et `mean`, et **corriger** l'erreur de `mean` avec des entiers | Exercices 15 et 17 |
| 3 | **Trouver** le plus grand ou le plus petit selon un axe, et **à quelle position** il se trouve, avec `max`, `min`, `argmax` et `argmin` | Exercices 16 et 17 |
| 4 | **Vérifier** n'importe quel résultat obtenu avec des tenseurs par rapport au même calcul fait avec `for` | Exercices 15, 16 et 17 |

---

## Ce dont tu as besoin

- L'environnement virtuel de la session 2, avec PyTorch installé, activé.
- Un fichier `papeleria.py` avec les données de doña Lupe. Si tu l'as déjà depuis la session 3, utilise-le. Sinon, copie-le ainsi :

```python
import torch

# eje 0 = sucursal (0 Centro, 1 Norte)
# eje 1 = día      (0 lunes ... 6 domingo)
# eje 2 = producto (0 cuadernos, 1 plumas, 2 mochilas)
centro = [
    [1200, 800, 500],    # lunes
    [1500, 900, 700],    # martes
    [1000, 700, 500],    # miércoles
    [1300, 900, 600],    # jueves
    [2000, 1200, 700],   # viernes
    [1200, 800, 400],    # sábado
    [800, 500, 300],     # domingo
]
norte = [
    [300, 200, 100],
    [400, 250, 150],
    [350, 200, 100],
    [300, 150, 100],
    [500, 300, 200],
    [300, 200, 100],
    [200, 100, 100],
]
papeleria = torch.tensor([centro, norte])
```

Chaque étape d'aujourd'hui est un nouveau fichier dans le même dossier, qui commence par `from papeleria import ...`. Ainsi, tu ne recopies pas les données à chaque fois.

---

## Pourquoi c'est important

Un tenseur garde le détail. Chaque case, chaque jour, chaque produit. Mais le détail n'accuse personne et ne sauve personne : ce qui sert, ce sont les **réponses**. Combien chaque succursale a-t-elle vendu ? Quel jour vend-on le plus ? Combien rentre-t-il en moyenne par jour ?

Ce soir-là, Mateo se rend compte d'une chose : les trois questions sont la même opération. **Regrouper beaucoup de nombres en un seul, selon une direction.** Dans Excel, c'est la colonne de totaux au bout de chaque ligne, ou la ligne de totaux en bas. Dans un tenseur, cela s'appelle **réduire**, parce qu'après l'opération il reste moins de nombres.

Et ce n'est pas une astuce de papeterie. Les systèmes d'intelligence artificielle réduisent tout le temps. Si une banque a les achats de 1 000 clients sur 30 jours, avec 5 données par achat (forme `(1000, 30, 5)`, comme dans la session 1), et qu'elle fait la moyenne sur les jours, elle obtient la façon dont chaque client achète d'habitude : un tableau de `(1000, 5)`. Un achat qui s'éloigne beaucoup de cette moyenne est celui qu'il vaut la peine d'examiner. Exactement ce que monsieur Ortega est en train de faire, à l'autre bout du fil, avec le terminal de Nord.

Dans les magasins, il se passe la même chose en grand. En 2020, un concours international de prévision appelé M5 a demandé de deviner les ventes quotidiennes de 10 magasins d'une chaîne de grande distribution américaine (données publiques du concours M5 de Kaggle) dans trois États des États-Unis : 42 840 séries de ventes organisées par niveaux (source : [M5 Forecasting - Accuracy, Kaggle](https://www.kaggle.com/c/m5-forecasting-accuracy) ; résultats dans l'*International Journal of Forecasting*, [doi:10.1016/j.ijforecast.2021.11.013](https://doi.org/10.1016/j.ijforecast.2021.11.013)). Additionner selon un axe, comme tu vas le faire aujourd'hui, est la façon de passer d'un niveau à celui du dessus.

## Le cas : la papeterie de doña Lupe, encore une fois

Le tenseur `papeleria` a la forme `(2, 7, 3)` : 2 succursales, 7 jours, 3 produits. Ce sont les 42 nombres de toujours. Ce soir, Mateo va leur poser les questions qui empêchent vraiment doña Lupe de dormir.

### Étape 1 — Le total de tout

Mateo commence par le plus simple, pour s'échauffer : combien les deux succursales ont-elles vendu ensemble sur toute la semaine ? Crée `paso1.py` :

```python
from papeleria import papeleria

total = papeleria.sum()
print(total)
print(total.shape)
print(total.item())
```

Sortie :

```
tensor(23100)
torch.Size([])
23100
```

`sum()` sans rien à l'intérieur regroupe **les 42 nombres** en un seul : 23 100 pesos. Regarde la forme : `torch.Size([])`, vide. C'est un scalaire, comme le `()` de la session 1, et `.item()` te le donne sous forme de nombre Python ordinaire. Le compte est bon : 18 500 pour Centre plus 4 600 pour Nord.

« Vingt-trois mille cent ? » Doña Lupe fait la moue. « Ça, je le sais déjà, mon petit. Ça, mon cahier me le dit. »

Elle a raison. Un seul nombre cache tout. Elle veut savoir **combien chaque succursale a vendu**, **chaque jour**, **chaque produit**. Pour cela, tu dis à `sum` selon quel axe regrouper.

### Étape 2 — Un onglet : additionner en largeur et en longueur

Avant de s'attaquer au cube entier, Mateo fait ce que ferait n'importe quel détective pressé mais sans envie de se tromper : tester en terrain connu. Un seul onglet, celui de Centre. Un tableau de 7 jours × 3 produits. Avec un tableau, il y a deux façons d'additionner, et tu les connais par Excel :

```
             Cuad.  Plumas  Moch.   │ total del día
   Lun       1200    800    500     │  2500
   Mar       1500    900    700     │  3100
   ...                              │  ...
   Dom        800    500    300     │  1600
   ─────────────────────────────────┘
   total     9000   5800   3700
   del producto
```

- **En largeur** (la colonne de droite) : tu regroupes les 3 produits de chaque jour. Il reste 7 nombres, un par jour.
- **En longueur** (la ligne du bas) : tu regroupes les 7 jours de chaque produit. Il reste 3 nombres, un par produit.

Dans PyTorch, tu indiques lequel avec `dim=`, qui veut dire **dimension**, c'est-à-dire axe. Rappelle-toi comment sont numérotés les axes de l'onglet : l'axe 0, ce sont les jours (les lignes), et l'axe 1, les produits (les colonnes).

Crée `paso2.py` :

```python
from papeleria import papeleria, centro

pestana_centro = papeleria[0]          # la pestaña del Centro: 7 días × 3 productos
print(pestana_centro.shape)

por_dia = pestana_centro.sum(dim=1)    # suma a lo largo del eje 1 (productos)
print(por_dia)
print(por_dia.shape)

# Lo mismo con for, como ya sabes hacerlo
por_dia_con_for = []
for renglon in centro:
    por_dia_con_for.append(sum(renglon))
print(por_dia_con_for)

assert por_dia.shape == (7,), "Esperaba un número por día: forma (7,)"
assert por_dia.tolist() == por_dia_con_for, "La suma con tensores no coincide con la del for"
print("Por día: cuadra")
```

Sortie :

```
torch.Size([7, 3])
tensor([2500, 3100, 2200, 2800, 3900, 2400, 1600])
torch.Size([7])
[2500, 3100, 2200, 2800, 3900, 2400, 1600]
Por día: cuadra
```

« Ces nombres, je les connais par cœur », dit doña Lupe. « Lundi deux mille cinq cents, mardi trois mille cent… »

Exactement. C'est la liste `por_dia` qu'elle a dictée à Mateo dans la session 1. Le tenseur et le `for` donnent la même chose. Premier test réussi : l'outil ne ment pas en terrain connu.

Regarde maintenant la forme, qui est l'essentiel d'aujourd'hui. Tu es entré avec `(7, 3)`. Tu as additionné avec `dim=1`. Tu es sorti avec `(7,)`. **Le 3, qui était l'axe 1, a disparu.** Logique : tu as regroupé les trois produits en un seul, donc il n'y a plus d'« axe des produits ».

Un nouvel outil dans ce code : `.tolist()`. C'est un cousin de `.item()` : `.item()` convertit un tenseur d'un seul nombre en nombre Python, et `.tolist()` convertit un tenseur de plusieurs nombres en liste Python. On l'utilise pour que le `assert` puisse comparer une liste à une liste.

Maintenant, en longueur. Dans le terminal Python, avec `from papeleria import papeleria` :

```
>>> papeleria[0].sum(dim=0)
tensor([9000, 5800, 3700])
>>> papeleria[0].sum(dim=0).shape
torch.Size([3])
```

Tu es entré avec `(7, 3)`, tu as additionné avec `dim=0` et tu es sorti avec `(3,)` : **le 7, qui était l'axe 0, a disparu.** Tu as regroupé les sept jours et il est resté un total par produit. (Dans l'exercice 15, tu vérifieras quelque chose de semblable avec un `for`.)

Mateo reste à regarder les deux formes. `(7,)`. `(3,)`. Et il écrit sur un post-it, qu'il colle sur le couvercle de son ordinateur, la règle de toute la session. Apprends-la telle quelle :

> **L'axe que tu réduis disparaît de la forme.** Les autres restent, dans le même ordre.

Une astuce pour ne pas t'embrouiller : `dim=` ne dit pas **ce qui te reste**, il dit **ce que tu regroupes**. `dim=1` ne veut pas dire « donne-moi les produits » ; ça veut dire « regroupe les produits ». C'est pour ça que les produits disparaissent.

« Autrement dit, ce que tu regroupes s'en va », résume doña Lupe. « Comme la prime de fin d'année. »

### Étape 3 — Le cube complet : trois façons d'additionner

Maintenant, oui, le cube complet. `(2, 7, 3)` : trois axes, donc il y a trois façons d'en additionner un seul. Mateo n'exécute encore rien. D'abord, il prend un stylo — de ceux que vend sa tante — et prédit chaque forme sur une serviette en papier. Fais-le aussi, avec la règle : barre le nombre de l'axe que tu regroupes.

| Tu regroupes | Quel axe | Forme qui reste | Ce que ça signifie |
|---|---|---|---|
| les succursales | `dim=0` | ~~2~~, 7, 3 → `(7, 3)` | chaque jour et produit, avec les deux succursales ensemble |
| les jours | `dim=1` | 2, ~~7~~, 3 → `(2, 3)` | chaque succursale et produit, sur toute la semaine |
| les produits | `dim=2` | 2, 7, ~~3~~ → `(2, 7)` | chaque succursale et jour, avec tous les produits |

Vérifie-le maintenant. Crée `paso3.py` :

```python
from papeleria import papeleria

sin_sucursal = papeleria.sum(dim=0)   # junta las dos pestañas
print(sin_sucursal.shape)
print(sin_sucursal)

sin_dia = papeleria.sum(dim=1)        # junta los 7 días
print(sin_dia.shape)
print(sin_dia)

sin_producto = papeleria.sum(dim=2)   # junta los 3 productos
print(sin_producto.shape)
print(sin_producto)
```

Sortie :

```
torch.Size([7, 3])
tensor([[1500, 1000,  600],
        [1900, 1150,  850],
        [1350,  900,  600],
        [1600, 1050,  700],
        [2500, 1500,  900],
        [1500, 1000,  500],
        [1000,  600,  400]])
torch.Size([2, 3])
tensor([[9000, 5800, 3700],
        [2350, 1400,  850]])
torch.Size([2, 7])
tensor([[2500, 3100, 2200, 2800, 3900, 2400, 1600],
        [ 600,  800,  650,  550, 1000,  600,  400]])
```

Lis-les comme doña Lupe :

- **`dim=0`** pose un onglet sur l'autre et additionne case par case. Le lundi, en cahiers, Centre a vendu 1 200 et Nord 300 : ensemble, 1 500. C'est la première case. Les succursales ont disparu : c'est comme si doña Lupe n'avait qu'un seul grand magasin.
- **`dim=1`** regroupe la semaine. La ligne du haut, c'est Centre : 9 000 en cahiers, 5 800 en stylos, 3 700 en sacs à dos, les mêmes qu'à l'étape 2. Celle du bas, c'est Nord. Les jours ont disparu.
- **`dim=2`** regroupe les produits. La ligne du haut est le `por_dia` de Centre que tu connais déjà ; celle du bas, celui de Nord. Les produits ont disparu.

Les trois prédictions de la serviette étaient justes. C'est l'avantage de la règle : **tu connais la forme avant d'exécuter le code.** Et si la forme qui sort n'est pas celle que tu as prédite, tu sais que tu as regroupé le mauvais axe.

Mais Mateo ne regarde plus les formes. Il regarde la ligne du bas de `dim=1`. *2350, 1400, 850.* La semaine entière de Nord, produit par produit. Elle n'atteint même pas ce que Centre vend en cahiers seuls.

Il lui manque un calcul. Celui qu'il est venu faire.

### Étape 4 — Regrouper deux axes à la fois : le total de chaque succursale

La question de doña Lupe, celle de la porte par laquelle entre deux fois plus de monde : combien chaque succursale a-t-elle vendu ? Pour cela, il faut regrouper **les jours et les produits** : deux axes à la fois. `sum` accepte un tuple d'axes. Crée `paso4.py` :

```python
from papeleria import papeleria, centro, norte

por_sucursal = papeleria.sum(dim=(1, 2))   # junta días y productos
print(por_sucursal.shape)
print(por_sucursal)

por_sucursal_con_for = []
for pestana in [centro, norte]:
    total = 0
    for renglon in pestana:
        for venta in renglon:
            total = total + venta
    por_sucursal_con_for.append(total)
print(por_sucursal_con_for)

assert por_sucursal.shape == (2,), "Esperaba un número por sucursal: forma (2,)"
assert por_sucursal.tolist() == por_sucursal_con_for, "No coincide con el for"
assert por_sucursal[0].item() == 18500, "El Centro debería sumar 18,500"
print("Por sucursal: cuadra")
```

Sortie :

```
torch.Size([2])
tensor([18500,  4600])
[18500, 4600]
Por sucursal: cuadra
```

18 500 pour Centre et 4 600 pour Nord.

Les nombres étaient dans le tableau depuis la session 1. Mais on ne les avait jamais mis comme ça, l'un à côté de l'autre, avec la porte de Nord en tête.

Doña Lupe s'approche de l'écran. Elle ne dit rien pendant un long moment.

« Quatre fois », dit-elle enfin. « Centre a vendu à peu près quatre fois ce que Nord a vendu ; Nord, environ un quart. Et par sa porte, il entre deux fois plus de monde. »

« Ma tante, ça ne dit pas pourquoi. Ça dit seulement combien. »

« Mmh. » Elle regarde déjà son téléphone. « Et qui fait la caisse de Nord tous les soirs, Mateo ? »

Il connaît la réponse. Tout le monde la connaît. Fer.

La règle, pendant ce temps, reste la même : 2, ~~7~~, ~~3~~ → `(2,)`. Tu as barré deux nombres, deux axes ont disparu.

Le `for` a eu besoin de **trois boucles**, l'une dans l'autre. Le tenseur, d'une ligne. Et ce n'est pas seulement plus court à écrire : avec de grands tableaux, le `for` de Python peut prendre beaucoup plus de temps que le `sum` de PyTorch. Combien de plus dépend de ton ordinateur, de la version et de la taille, alors mesure-le sur ta machine avant de citer un chiffre.

**Attention si tu le fais en deux étapes.** Mateo, les mains froides, veut refaire le calcul d'une autre manière, au cas où : additionner d'abord un axe, puis l'autre. Mais attention : **après la première somme, les axes sont renumérotés.** Regarde cette tentative. Crée `paso4c.py` :

```python
from papeleria import papeleria

paso_a = papeleria.sum(dim=1)     # quito los días: queda (2, 3)
print(paso_a.shape)
paso_b = paso_a.sum(dim=2)        # ¿quito los productos?
```

Sortie :

```
torch.Size([2, 3])
Traceback (most recent call last):
  File ".../paso4c.py", line 5, in <module>
    paso_b = paso_a.sum(dim=2)        # ¿quito los productos?
             ^^^^^^^^^^^^^^^^^
IndexError: Dimension out of range (expected to be in range of [-2, 1], but got 2)
```

Lis-le mot à mot : « dimension hors limites : j'attendais quelque chose entre -2 et 1, et tu m'as donné 2 ». Après avoir enlevé les jours, `paso_a` a la forme `(2, 3)` : seulement deux axes, le 0 et le 1. Les produits ne sont plus l'axe 2 ; ce sont maintenant l'axe 1. (Les nombres négatifs comptent les axes à partir de la fin ; pour l'instant, tu n'en as pas besoin.)

La solution simple : fais les deux choses en un seul appel, `dim=(1, 2)`. Là, les nombres se réfèrent toujours à la forme d'origine.

Mateo revient à l'appel en une ligne. 4 600. Le nombre n'a pas bougé.

À l'autre bout de la réserve, doña Lupe compose un numéro. Elle laisse sonner. Raccroche. Recompose.

### Étape 5 — La moyenne quotidienne, et l'erreur que tu vas forcément voir

« Calcule-moi combien vend Nord un jour normal », dit doña Lupe sans lâcher le téléphone. « Pour le lui dire en face. »

Autrement dit : combien vend-on **en moyenne** par jour dans chaque succursale ? La moyenne se calcule avec `mean` (en anglais, « moyenne »). Ça fonctionne comme `sum`, avec `dim=`. D'abord, tu calcules les ventes de chaque jour (en regroupant les produits), puis tu fais la moyenne des 7 jours. Crée `paso5a.py` :

```python
from papeleria import papeleria

ventas_diarias = papeleria.sum(dim=2)       # (2, 7): sucursal × día
print(ventas_diarias.dtype)
promedio_diario = ventas_diarias.mean(dim=1)
```

Sortie :

```
torch.int64
Traceback (most recent call last):
  File ".../paso5a.py", line 5, in <module>
    promedio_diario = ventas_diarias.mean(dim=1)
                      ^^^^^^^^^^^^^^^^^^^^^^^^^^
RuntimeError: mean(): could not infer output dtype. Input dtype must be either a floating point or complex dtype. Got: Long
```

Ça plante.

« Et maintenant, quoi ? » demande doña Lupe.

« Rien de grave, ma tante. Il se plaint que je lui ai donné des entiers. »

« Comme tout le monde. Moi aussi, j'aime les comptes ronds. »

Lis-le lentement, traduit :

- *mean(): could not infer output dtype* : « `mean` n'a pas su de quel type faire le résultat ».
- *Input dtype must be either a floating point or complex dtype* : « ce qui entre doit être en décimaux » (*floating point*, ce sont les nombres à virgule ; *complex*, une autre sorte de nombres que tu n'utiliseras pas ici).
- *Got: Long* : « et tu m'as donné `Long` ». `Long` est le nom que PyTorch donne en interne à `int64`, le type d'entiers que tu as vu dans la session 2. C'est pour ça qu'on a affiché le `dtype` avant : `torch.int64`.

Pourquoi refuse-t-il ? Parce qu'une moyenne ne tombe presque jamais juste : 18 500 divisé par 7 donne 2 642,857…, et un tenseur d'entiers n'a nulle part où ranger le « ,857 ». Au lieu de le couper en cachette, PyTorch s'arrête et te prévient. Mateo apprécie. Dans cette histoire, il y a déjà trop de choses qui se passent en cachette.

Il y a deux façons de corriger.

**Correction 1 : convertir en décimaux avec `.float()`.** Crée `paso5b.py` :

```python
from papeleria import papeleria, centro

ventas_diarias = papeleria.sum(dim=2)       # (2, 7): sucursal × día
decimales = ventas_diarias.float()
print(decimales.dtype)
print(decimales)

promedio_diario = decimales.mean(dim=1)     # quita el eje de los días
print(promedio_diario.shape)
print(promedio_diario)

# Con for: el total del Centro entre 7 días
total_centro = 0
for renglon in centro:
    total_centro = total_centro + sum(renglon)
print(total_centro / 7)

assert promedio_diario.shape == (2,), "Esperaba un promedio por sucursal"
assert round(promedio_diario[0].item(), 2) == round(total_centro / 7, 2), "No coincide con el for"
print("Promedio diario: cuadra")
```

Sortie :

```
torch.float32
tensor([[2500., 3100., 2200., 2800., 3900., 2400., 1600.],
        [ 600.,  800.,  650.,  550., 1000.,  600.,  400.]])
torch.Size([2])
tensor([2642.8572,  657.1429])
2642.8571428571427
Promedio diario: cuadra
```

`.float()` ne change pas les nombres, seulement la façon dont ils sont stockés : remarque le point après chacun (`2500.`). Ce sont maintenant des `float32`, des décimaux. Centre vend en moyenne 2 642,86 $ par jour et Nord, 657,14 $.

« Six cent cinquante-sept pesos », répète doña Lupe. « Avec cet emplacement. Un stand de chewing-gums vend autant. »

Deux détails dans ce code :

- La forme : `(2, 7)` avec `dim=1` a donné `(2,)`. Les jours ont disparu, comme toujours.
- Le `assert` compare **en arrondissant à deux décimales**, parce que le tenseur dit 2642.8572 et le `for` 2642.8571428571427. Aucun des deux n'a tort : le tenseur utilise `float32` et Python divise avec un `float` de 64 bits. Les deux sont des approximations binaires, mais `float32` conserve moins de précision. Sans arrondir, le `assert` échouerait alors que le calcul est bon. Attention : on arrondit seulement pour comparer ou afficher ; pour une vraie comptabilité, stocke les montants en centimes sous forme d'entiers ou utilise l'arithmétique décimale, pas `float32`.

**Correction 2 : créer le tenseur en décimaux dès le départ.** Si les données de doña Lupe arrivaient avec des décimales, le tenseur naîtrait en décimaux et `mean` fonctionnerait directement (pour de l'argent exact, mieux vaut des centimes entiers). Crée `paso5c.py` :

```python
import torch

con_centavos = torch.tensor([2500.0, 3100.5, 2200.0])
print(con_centavos.dtype)
print(con_centavos.mean())
```

Sortie :

```
torch.float32
tensor(2600.1667)
```

Il suffit que les nombres portent un point décimal pour que PyTorch choisisse `float32` tout seul. Pour la papeterie, qui existe déjà en entiers, le plus pratique est la correction 1 : `.float()` juste avant le `mean`.

### Étape 6 — Le meilleur jour : `max` et `argmax`

Mateo a un soupçon différent de celui de sa tante, et il veut le tester avant de dire quoi que ce soit. Si quelqu'un cachait des ventes n'importe comment, Nord se comporterait bizarrement : de bons jours là où il ne faut pas, des jours morts au hasard. Alors il pose la dernière question de la soirée : quel jour vend-on le plus dans chaque succursale ? Ici, **deux** réponses t'intéressent : **combien** on a vendu ce jour-là (la valeur) et **quel jour** c'était (la position). Crée `paso6.py` :

```python
from papeleria import papeleria

ventas_diarias = papeleria.sum(dim=2)       # (2, 7): sucursal × día
resultado = ventas_diarias.max(dim=1)
print(resultado)
```

Sortie :

```
torch.return_types.max(
values=tensor([3900, 1000]),
indices=tensor([4, 4]))
```

Ça a l'air bizarre, mais c'est simple. Quand tu donnes `dim=` à `max`, il ne renvoie pas un tenseur, mais **une petite boîte avec deux choses** :

- `values` (valeurs) : la vente la plus haute de chaque succursale. 3 900 à Centre, 1 000 à Nord.
- `indices` (indices) : **à quelle position** de l'axe se trouvait cette vente. 4 et 4.

Et quelle est la position 4 de l'axe des jours ? Lundi 0, mardi 1, mercredi 2, jeudi 3… **vendredi**. Les deux succursales vendent le plus le vendredi.

Comme avec un tuple Python, tu peux extraire les deux choses d'un coup, chacune dans sa variable. Et avec une liste de noms, tu traduis l'indice en mots. Crée `paso6b.py` :

```python
from papeleria import papeleria

dias = ["lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo"]
sucursales = ["Centro", "Norte"]

ventas_diarias = papeleria.sum(dim=2)
valores, indices = ventas_diarias.max(dim=1)
print(valores)
print(indices)

for s in range(2):
    dia = dias[indices[s].item()]
    print(sucursales[s], "vendió más el", dia, ":", valores[s].item())
```

Sortie :

```
tensor([3900, 1000])
tensor([4, 4])
Centro vendió más el viernes : 3900
Norte vendió más el viernes : 1000
```

La règle de la forme vaut aussi ici : tu es entré avec `(2, 7)`, tu as regroupé les jours avec `dim=1`, et `valores` comme `indices` ont la forme `(2,)` : un par succursale.

Mateo s'adosse à sa chaise. Le même jour pour les deux. Nord ne se comporte pas comme un magasin livré à lui-même. Il se comporte comme Centre, mais en miniature. Comme si, à chaque jour, de façon à peu près égale, il manquait un morceau.

Ça ne ressemble pas à quelqu'un qui prend de l'argent dans la caisse un jour sur deux. Ça ressemble à autre chose. Mais à quoi ? Mateo n'a pas encore de quoi le découvrir, alors il ne dit rien.

Dans l'exercice 16, tu vérifieras un résultat de ce genre par rapport à un `for` : c'est plus long, mais tu sais déjà l'écrire.

**Ses cousins.** Si tu veux seulement la position, sans la valeur, utilise `argmax` (« l'argument du maximum », c'est-à-dire où il se trouve). Si tu veux le plus bas, `min` et `argmin` fonctionnent de la même façon. Et si tu enlèves le `dim`, `max` cherche dans tout le tenseur et renvoie un seul nombre. Crée `paso6c.py` :

```python
from papeleria import papeleria

ventas_diarias = papeleria.sum(dim=2)
print(ventas_diarias.argmax(dim=1))
print(ventas_diarias.max())
print(ventas_diarias.min(dim=1))
```

Sortie :

```
tensor([4, 4])
tensor(3900)
torch.return_types.min(
values=tensor([1600,  400]),
indices=tensor([6, 6]))
```

Le vendredi est le meilleur jour des deux succursales ; 3 900 est la vente quotidienne la plus haute de tout le tenseur ; et le jour le plus faible des deux est le 6, le dimanche. Même schéma, encore une fois. Nord copie Centre jusque dans ses mauvais jours.

Une différence qui va te faire gagner du temps : `sum` et `mean` acceptent un tuple d'axes, comme `dim=(1, 2)` ; **`max` et `min` non**, un seul à la fois. Tu le verras dans le tableau des erreurs.

**Pour quand tu en auras besoin : `keepdim`.** Si tu ajoutes `keepdim=True` à n'importe laquelle de ces fonctions, l'axe que tu regroupes ne disparaît pas, il reste avec une taille de 1 : `papeleria.sum(dim=2, keepdim=True)` donne la forme `(2, 7, 1)` au lieu de `(2, 7)`. Aujourd'hui, tu n'en as pas besoin ; dans la leçon B, tu verras quand c'est utile.

### Tout dans un tableau

Il est presque neuf heures. Mateo recopie sur une seule feuille tout ce qu'il a obtenu ce soir, pour que doña Lupe puisse l'apporter à monsieur Ortega. Chaque question, une ligne :

| Question de doña Lupe | Code | Forme |
|---|---|---|
| Combien ai-je vendu au total ? | `papeleria.sum()` | `()` |
| Combien chaque succursale a-t-elle vendu ? | `papeleria.sum(dim=(1, 2))` | `(2,)` |
| Combien chaque succursale a-t-elle vendu chaque jour ? | `papeleria.sum(dim=2)` | `(2, 7)` |
| Combien chaque succursale a-t-elle vendu de chaque produit ? | `papeleria.sum(dim=1)` | `(2, 3)` |
| Combien ai-je vendu de chaque produit chaque jour, succursales regroupées ? | `papeleria.sum(dim=0)` | `(7, 3)` |
| Combien est-ce que je vends en moyenne par jour dans chaque succursale ? | `papeleria.sum(dim=2).float().mean(dim=1)` | `(2,)` |
| Quel jour est-ce que je vends le plus dans chaque succursale ? | `papeleria.sum(dim=2).argmax(dim=1)` | `(2,)` |

---

## PyTorch et TensorFlow : la même somme, sous un autre nom

Avant de refermer son ordinateur, Mateo veut être sûr que le 4 600 n'est pas un caprice de PyTorch. Il lance le même calcul dans TensorFlow, l'autre grand outil. Là, les fonctions s'appellent `reduce_sum` et `reduce_mean` (on y voit le mot « réduire »), et l'axe s'appelle `axis` au lieu de `dim`. Ce bloc est à lire, pas à exécuter : le cours n'installe pas TensorFlow.

```python
import tensorflow as tf
from papeleria import centro, norte

papeleria_tf = tf.constant([centro, norte])
print(tf.reduce_sum(papeleria_tf, axis=(1, 2)))
print(tf.reduce_mean(tf.reduce_sum(papeleria_tf, axis=2), axis=1))
```

Sortie (TensorFlow 2.21) :

```
tf.Tensor([18500  4600], shape=(2,), dtype=int32)
tf.Tensor([2642  657], shape=(2,), dtype=int32)
```

La somme est identique : 4 600 dans les deux outils. Mais regarde la moyenne : **2642 et 657, sans décimales et sans aucun avertissement**. TensorFlow conserve le type entier et fait une division entière : il tronque la partie fractionnaire ; PyTorch a préféré planter. Une erreur, au moins, te prévient ; un nombre faussé que personne ne remarque, c'est pire. Si tu utilises TensorFlow, convertis en décimaux avant de faire la moyenne :

```python
promedios_tf = tf.reduce_mean(
    tf.cast(tf.reduce_sum(papeleria_tf, axis=2), tf.float32),
    axis=1,
)
print(promedios_tf)
```

Sortie (TensorFlow 2.21) :

```
tf.Tensor([2642.8572  657.1429], shape=(2,), dtype=float32)
```

Mateo le note en le soulignant deux fois. À trente jours d'une échéance, un nombre qui change en silence est la dernière chose dont on a besoin.

---

👉 **Si tu viens de la gestion :** ce que Mateo a fait ce soir-là, tu le fais avec un tableau croisé dynamique. Quand tu mets « succursale » en lignes et « somme des ventes » en valeurs, Excel regroupe les jours et les produits et te laisse un nombre par succursale : c'est `papeleria.sum(dim=(1, 2))`. Si tu mets « jour » en lignes, il regroupe succursales et produits : `papeleria.sum(dim=(0, 2))`. La règle d'aujourd'hui vaut aussi là-bas : ce que tu ne mets pas dans le tableau croisé dynamique disparaît du rapport. Et la leçon de l'étape 5 aussi : avant de présenter une moyenne, vérifie que ton outil n'a pas coupé les centimes. Et une de plus, que le tableau croisé dynamique ne te dira pas : un total bas te dit **combien** il manque, jamais **qui** est coupable.

---

## Exercices de la session 4

Les solutions se trouvent à la fin de cette livraison. Essaie d'abord ; le `assert` te dit si tu as réussi. Toujours dans le même ordre : **d'abord la forme, ensuite la valeur.**

**Exercice 14 — Prédis la forme.** Écris sur papier quelle forme reste dans chaque cas et ce que cela signifie pour doña Lupe. Ensuite, transforme chaque prédiction en `assert` et exécute-le. Le premier est résolu :

```python
from papeleria import papeleria

assert papeleria.sum(dim=0).shape == (7, 3)          # ejemplo resuelto
assert papeleria.sum(dim=1).shape == ____
assert papeleria.sum(dim=2).shape == ____
assert papeleria.sum(dim=(0, 1)).shape == ____
assert papeleria.sum(dim=(0, 2)).shape == ____
assert papeleria.sum(dim=(0, 1, 2)).shape == ____
print("Ejercicio 14 superado")
```

Question bonus : de quelle autre façon peux-tu écrire `papeleria.sum(dim=(0, 1, 2))` ?

**Exercice 15 — La moyenne par produit (à moitié faite).** Doña Lupe veut savoir combien elle vend **en moyenne par jour** de chaque produit dans chaque succursale. Remplis les deux trous :

Le préfixe `f` devant des guillemets crée une **f-string** : elle permet d'insérer une valeur entre accolades dans un texte. Ici, elle fera en sorte que le message du `assert` affiche la forme réelle si quelque chose échoue.

```python
from papeleria import papeleria, norte

promedio = papeleria.____().mean(dim=____)
print(promedio)

assert promedio.shape == (2, 3), f"Esperaba (sucursal, producto) = (2, 3) y salió {tuple(promedio.shape)}"

suma_plumas_norte = 0
for renglon in norte:
    suma_plumas_norte = suma_plumas_norte + renglon[1]

assert round(promedio[1, 1].item(), 2) == round(suma_plumas_norte / 7, 2), "No coincide con el for"
print("Ejercicio 15 superado")
```

Regarde le message du premier `assert` : si tu te trompes d'axe, il te dit quelle forme tu as obtenue. Cette forme te dit quel axe tu as regroupé par erreur.

**Exercice 16 — Le meilleur et le pire de la semaine.** Écris le code toi-même. Pour chaque question : un `assert` de la forme, puis la valeur.

a) Quel est le jour le plus faible de chaque succursale, et combien a-t-on vendu ce jour-là ? Vérifie la réponse de Nord avec un `for`.
b) Quel est le produit vedette (celui qui s'est le plus vendu dans la semaine) de chaque succursale ?
c) En regroupant les deux succursales, quel jour a-t-on le plus vendu, et combien ? Le résultat doit être 4 900.

Indice : pour chaque question, décide d'abord quels axes tu regroupes pour la somme, puis selon quel axe tu cherches.

**Exercice 17 — Tes notes.** Reprends les notes de l'exercice 4 de la session 1. Si tu n'as pas gardé les tiennes, utilise celles-ci (élève × matière × partiel) :

```python
parciales = [
    [[8, 9, 9], [7, 8, 9], [9, 9, 10], [6, 7, 8]],
    [[10, 9, 10], [9, 9, 9], [8, 8, 9], [9, 10, 10]],
    [[7, 7, 8], [8, 8, 8], [6, 7, 7], [8, 9, 9]],
]
```

Convertis-les en tenseur et calcule, chacun avec son `assert` de forme :

a) La moyenne générale de chaque élève.
b) La moyenne de chaque élève dans chaque matière.
c) Quel élève a la meilleure moyenne.
d) Quelle est la matière la plus difficile du groupe (celle qui a la moyenne la plus basse, en regroupant élèves et partiels).

Vérifie la question a) du premier élève avec un `for`. Si tu as utilisé tes propres données avec un autre ordre d'axes, tes `dim` seront différents de ceux de la solution, et c'est normal : ce qui compte, c'est que la forme que tu prédis soit celle qui sort.

---

## Erreurs courantes (et comment t'en rendre compte)

| Erreur | Ce que l'on observe | Comment t'en rendre compte | Ce qu'il faut faire |
|---|---|---|---|
| Croire que `dim` dit ce qui te reste | Tu veux un total par jour et tu écris `dim=1`, en pensant « l'axe des jours » | La forme sort en `(2, 3)` au lieu de `(2, 7)` | `dim` dit ce que tu **regroupes**. Pour un total par jour, regroupe les autres axes |
| Faire la moyenne d'entiers | `RuntimeError: mean(): could not infer output dtype. ... Got: Long` | Le `dtype` indique `torch.int64` | `.float()` avant le `mean`, ou des données avec un point décimal |
| Additionner en deux étapes avec l'ancienne numérotation | `IndexError: Dimension out of range (expected to be in range of [-2, 1], but got 2)` | Tu as demandé un axe qui n'existe plus | Après une réduction, les axes sont renumérotés. Mieux vaut un seul appel : `dim=(1, 2)` |
| Comparer un tenseur à une liste | `assert por_dia == [2500, 3100, ...]` lève `AssertionError` alors que les nombres sont bons | Le `assert` plante avec des nombres qui coïncident à l'œil | Convertis d'abord : `por_dia.tolist() == [...]` |
| Comparer des décimaux sans arrondir | `2642.857177734375 == 18500 / 7` donne `False` | Le `assert` plante à cause du dernier chiffre | Arrondis les deux côtés : `round(x, 2)` |
| Donner un tuple à `max` | `TypeError: max() received an invalid combination of arguments - got (dim=tuple, ), ...` | Tu as écrit `max(dim=(1, 2))` | `max` et `min` n'acceptent qu'un seul axe. Additionne ou réduis d'abord l'un, puis cherche dans l'autre |

---

## Autoévaluation de la leçon A

Tu as terminé la première leçon. Avant de passer à la leçon B, vérifie que tu sais faire chaque chose **sans regarder les sessions**. Si l'une d'elles te coûte, retourne à la session indiquée et refais son exercice.

**Session 1 — Un nombre, une liste, un tableau, un cube**
- [ ] Je dis combien de dimensions a un ensemble de données et ce que représente chacune.
- [ ] J'écris ces données sous forme de listes imbriquées Python et je désigne un nombre par ses indices.
- [ ] Je calcule la forme manuellement et je dis combien il y a de nombres au total, en la multipliant.
- [ ] Je reconnais une liste irrégulière et j'explique pourquoi elle ne peut pas être un tenseur.
- [ ] J'écris un `assert` avec un message en espagnol pour vérifier mon propre travail.

**Session 2 — Ton premier tenseur**
- [ ] Je convertis une liste Python en tenseur avec `torch.tensor`.
- [ ] Je lis `shape`, `ndim`, `dtype` et `numel()` d'un tenseur et j'explique ce que dit chacun.
- [ ] Je sais quel `dtype` PyTorch choisit pour les entiers et pour les décimaux.

**Session 3 — Les axes ont un nom**
- [ ] J'extrais un nombre avec `papeleria[1, 4, 2]` et je le convertis en nombre Python avec `.item()`.
- [ ] J'extrais une tranche complète selon n'importe quel axe, comme `papeleria[:, 4]` ou `papeleria[:, :, 2]`, et je donne sa forme avant de l'exécuter.

**Session 4 — Additionner par onglet**
- [ ] Je prédis la forme après avoir réduit un ou plusieurs axes.
- [ ] Je calcule des totaux et des moyennes par axe avec `sum` et `mean`, et je corrige l'erreur de `mean` avec des entiers.
- [ ] Je trouve le plus grand, le plus petit et leur position avec `max`, `min`, `argmax` et `argmin`.
- [ ] Je vérifie un résultat obtenu avec des tenseurs par rapport au même calcul fait avec `for`.

## Révision de la leçon A

Dix questions sur les quatre sessions. Réponds-y sans rien regarder ; les réponses se trouvent dans les solutions.

1. Quelle est la règle pour savoir combien de dimensions a un ensemble de données ?
2. Si doña Lupe garde 4 semaines de ventes, par succursale, jour et produit, quelle forme a le tenseur et combien de nombres contient-il ?
3. Que fait `assert` quand la condition est vraie, et que fait-il quand elle est fausse ?
4. Que se passe-t-il si tu donnes à `torch.tensor` la liste `[[1, 2, 3], [4, 5]]` ?
5. Que t'indiquent `papeleria.ndim`, `papeleria.numel()` et `papeleria.dtype` ?
6. Quel `dtype` a `torch.tensor([1, 2, 3])` ? Et `torch.tensor([1.5, 2.0])` ?
7. Quel nombre donne `papeleria[1, 4, 2]` et à quoi sert d'y ajouter `.item()` ?
8. Quelle forme a `papeleria[:, :, 2]` et que représente-t-il ?
9. Quelle forme laisse `papeleria.sum(dim=0)` et que signifie-t-elle pour doña Lupe ?
10. Que renvoie `max(dim=1)` que `sum(dim=1)` ne renvoie pas ?

---

## Révision pour la prochaine session

La session 5 commence par ces trois questions. Réponds-y sans regarder la leçon :

1. Si tu additionnes `papeleria` avec `dim=1`, quelle forme reste-t-il et que signifie-t-elle ?
2. Pourquoi `mean` échoue-t-il avec des entiers et comment le corriges-tu ?
3. Comment vérifies-tu que ta somme avec des tenseurs est correcte ?

🔁 **La semaine prochaine** commence la leçon B, **Les tenseurs au travail**. Mateo va devoir regarder plus loin qu'une semaine en arrière : des mois entiers. Et pour les voir, il va devoir changer la forme du tenseur sans changer ses données : les mêmes nombres, rangés dans d'autres boîtes. Tu liras aussi, mot à mot, l'erreur qui apparaît quand la nouvelle forme ne rentre pas.

Avant de baisser le rideau, doña Lupe montre son téléphone à Mateo. Elle a demandé à Fer le relevé de caisse de la semaine depuis quatre heures. Ensuite sont venus les appels : ceux de l'après-midi, et ceux de ce soir. Aucun rappel. Mateo lui demande d'attendre, que les nombres disent combien et pas qui. Elle range son téléphone sans lui répondre.

Fer n'a pas répondu au téléphone de tout l'après-midi.

---

## Solutions (ne les regarde pas avant d'avoir essayé)

**Révision pour commencer (de la session 3).**

1. `papeleria[:, 4]` a la forme `(2, 3)` : les trois produits du vendredi dans chaque succursale.
2. `papeleria[0]` a la forme `(7, 3)` : l'onglet de Centre, sans l'axe des succursales. `papeleria[0:1]` a la forme `(1, 7, 3)` : une tranche qui conserve l'axe des succursales, avec une seule succursale à l'intérieur.
3. `papeleria[:, :, 2]` : toutes les succursales, tous les jours, le produit 2 (sacs à dos). Forme `(2, 7)`.

**Exercice 14.**

```python
assert papeleria.sum(dim=0).shape == (7, 3)          # cada día y producto, sucursales juntas
assert papeleria.sum(dim=1).shape == (2, 3)          # cada sucursal y producto, en la semana
assert papeleria.sum(dim=2).shape == (2, 7)          # cada sucursal y día, productos juntos
assert papeleria.sum(dim=(0, 1)).shape == (3,)       # cada producto, en todo
assert papeleria.sum(dim=(0, 2)).shape == (7,)       # cada día, en todo
assert papeleria.sum(dim=(0, 1, 2)).shape == ()      # un solo número: el total
```

Si tu affiches aussi les valeurs :

```
tensor([11350,  7200,  4550])
tensor([3100, 3900, 2850, 3350, 4900, 3000, 2000])
tensor(23100)
Ejercicio 14 superado
```

Question bonus : `papeleria.sum(dim=(0, 1, 2))` revient au même que `papeleria.sum()`. Regrouper tous les axes, c'est tout regrouper : 23 100.

**Exercice 15.** Les trous sont `float` et `1` : tu convertis en décimaux et tu regroupes les jours.

```python
promedio = papeleria.float().mean(dim=1)
```

Sortie :

```
tensor([[1285.7142,  828.5714,  528.5714],
        [ 335.7143,  200.0000,  121.4286]])
Ejercicio 15 superado
```

Nord vend en moyenne exactement 200 $ de stylos par jour (1 400 divisé par 7). Si tu as mis `dim=0`, le message du `assert` t'a dit :

```
AssertionError: Esperaba (sucursal, producto) = (2, 3) y salió (7, 3)
```

Le `(7, 3)` te trahit : ce sont les succursales qui ont disparu, pas les jours.

**Exercice 16.**

```python
from papeleria import papeleria, norte

dias = ["lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo"]
productos = ["cuadernos", "plumas", "mochilas"]

# a) Día más flojo: junto productos (2, 7) y busco el mínimo a lo largo de los días
valores, indices = papeleria.sum(dim=2).min(dim=1)
assert valores.shape == (2,), "Esperaba un valor por sucursal"
assert valores.tolist() == [1600, 400], "Los mínimos deberían ser 1,600 y 400"
assert indices.tolist() == [6, 6], "El domingo debería ser el día más flojo"
print(valores, indices)

peor_dia = 0
for d in range(7):
    if sum(norte[d]) < sum(norte[peor_dia]):
        peor_dia = d
assert indices[1].item() == peor_dia, "No coincide con el for"

# b) Producto estrella: junto días (2, 3) y busco la posición del máximo en productos
estrella = papeleria.sum(dim=1).argmax(dim=1)
assert estrella.shape == (2,), "Esperaba un producto por sucursal"
assert estrella.tolist() == [0, 0], "Los cuadernos deberían ser el producto estrella"
print(productos[estrella[0].item()], productos[estrella[1].item()])

# c) Mejor día con las dos sucursales juntas: junto sucursales y productos (7,)
por_dia_juntas = papeleria.sum(dim=(0, 2))
assert por_dia_juntas.shape == (7,), "Esperaba un total por día"
print(dias[por_dia_juntas.argmax().item()], por_dia_juntas.max().item())
assert por_dia_juntas.max().item() == 4900, "El mejor día debería sumar 4,900"
assert por_dia_juntas.argmax().item() == 4, "El mejor día debería ser viernes"
print("Ejercicio 16 superado")
```

Sortie :

```
tensor([1600,  400]) tensor([6, 6])
cuadernos cuadernos
viernes 4900
Ejercicio 16 superado
```

Le dimanche est le jour le plus faible des deux succursales (1 600 et 400), les cahiers sont le produit vedette des deux et le vendredi, en regroupant les succursales, on a vendu 4 900. Dans la question c), `argmax()` sans `dim` convient parce que `por_dia_juntas` n'a déjà plus qu'un seul axe.

**Exercice 17.**

```python
import torch

parciales = torch.tensor([
    [[8, 9, 9], [7, 8, 9], [9, 9, 10], [6, 7, 8]],
    [[10, 9, 10], [9, 9, 9], [8, 8, 9], [9, 10, 10]],
    [[7, 7, 8], [8, 8, 8], [6, 7, 7], [8, 9, 9]],
])                                   # alumno × materia × parcial
decimales = parciales.float()

por_alumno = decimales.mean(dim=(1, 2))           # a) quito materias y parciales
assert por_alumno.shape == (3,)
por_alumno_y_materia = decimales.mean(dim=2)      # b) quito parciales
assert por_alumno_y_materia.shape == (3, 4)
por_materia = decimales.mean(dim=(0, 2))          # d) quito alumnos y parciales
assert por_materia.shape == (4,)

mejor_alumno = por_alumno.argmax()                # c)
assert mejor_alumno.shape == ()
assert mejor_alumno.item() == 1, "El segundo alumno debería tener el mejor promedio"
assert torch.allclose(por_materia, torch.tensor([8.5556, 8.3333, 8.1111, 8.4444]), atol=1e-4)
assert por_materia.argmin().item() == 2, "La tercera materia debería ser la más difícil"  # d)

print(por_alumno)
print(por_alumno_y_materia)
print(mejor_alumno)
print(por_materia, por_materia.argmin())

suma = 0
cuantas = 0
for materia in parciales[0].tolist():
    for calificacion in materia:
        suma = suma + calificacion
        cuantas = cuantas + 1
assert round(por_alumno[0].item(), 2) == round(suma / cuantas, 2), "No coincide con el for"
print("Ejercicio 17 superado")
```

Sortie :

```
tensor([8.2500, 9.1667, 7.6667])
tensor([[8.6667, 8.0000, 9.3333, 7.0000],
        [9.6667, 9.0000, 8.3333, 9.6667],
        [7.3333, 8.0000, 6.6667, 8.6667]])
tensor(1)
tensor([8.5556, 8.3333, 8.1111, 8.4444]) tensor(2)
Ejercicio 17 superado
```

Le deuxième élève (indice 1) a la meilleure moyenne, 9,17, et la matière la plus difficile est la troisième (indice 2), avec 8,11. Si tu as oublié le `.float()`, tu as vu le même `RuntimeError` qu'à l'étape 5 : les notes aussi sont des entiers.

**Révision de la leçon A.**

1. Le nombre de dimensions est le nombre de données dont tu as besoin pour désigner un seul nombre.
2. `(4, 2, 7, 3)` : semaine, succursale, jour et produit. Il contient 4 × 2 × 7 × 3 = 168 nombres.
3. Si la condition est vraie, il ne fait rien et le programme continue. Si elle est fausse, il arrête le programme avec un `AssertionError` et affiche ton message.
4. PyTorch refuse de le créer, parce que la liste est irrégulière (une ligne de 3 et une autre de 2) : `ValueError: expected sequence of length 3 at dim 1 (got 2)`.
5. `ndim` donne 3 (trois axes), `numel()` donne 42 (le nombre total de valeurs, 2 × 7 × 3) et `dtype` donne `torch.int64` (entiers).
6. `torch.int64` pour les entiers et `torch.float32` pour les décimaux.
7. Il donne 200 : Nord, vendredi, sacs à dos. `.item()` le convertit d'un tenseur d'un seul nombre en nombre Python ordinaire, pour l'afficher ou le comparer dans un `assert`.
8. `(2, 7)` : les ventes de sacs à dos des deux succursales sur les sept jours.
9. `(7, 3)` : il regroupe les deux succursales ; il reste chaque jour et chaque produit, comme si doña Lupe n'avait qu'un seul magasin.
10. En plus de la valeur la plus haute, il renvoie **à quelle position** elle se trouve (les indices). `sum` ne renvoie que les totaux.

---

## La suite

Avec ceci, tu clôtures la leçon A : tu sais ce qu'est un tenseur, le créer, le parcourir selon ses axes et le résumer. Dans la session 5 commence la leçon B. Avant d'y arriver, fais l'autoévaluation complète ; si tout te vient sans regarder, tu pars sur une base solide.

---

*J'écris à partir de ce que nous voyons tous les jours chez Hábil, où nous concevons des systèmes pour le secteur bancaire, les assurances et les entreprises au Mexique. Si, dans ton travail ou dans l'entreprise de ta famille, tu rencontres un problème de ce genre, discutons-en : [habil.mx](https://www.habil.mx/fr/)*
