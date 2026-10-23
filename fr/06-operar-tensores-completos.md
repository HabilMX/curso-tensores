# Les tenseurs à partir de zéro (6) : opérer sur des tenseurs complets

*Multiplier toute une semaine de ventes par leurs prix en une seule ligne, la règle qui décide quand PyTorch « étire » un tenseur, et le piège de `*` face à `@` qui ne signale aucune erreur.*

*Les tenseurs à partir de zéro · leçon B, session 6 sur 8*

*Scénario : les personnages et l'histoire sont inventés. Les données techniques, historiques et les chiffres réels indiquent leur source.*

**Lundi 7 décembre, 19 h 40. Papeterie Herrera, succursale Centre.**

Le rideau métallique est à moitié baissé. À l'intérieur, ça sent le papier neuf et le café réchauffé. Doña Lupe compte des pièces de monnaie sur le comptoir. Mateo a son ordinateur portable ouvert entre un rouleau de papier glacé et une boîte de stylos.

Il reste seize jours avant le 23.

Depuis la semaine dernière, Mateo sait deux choses. Que Nord allait bien jusqu'en août. Et qu'en août, un homme en tee-shirt bleu a « changé le terminal » sans que personne l'ait appelé. Ce qu'il ne sait pas, c'est le rapport entre les deux.

« Je ne veux plus de pesos, ma tante », dit-il. « Je veux des quantités. »

« Des quantités de quoi ? »

« De tout. Combien de cahiers, combien de stylos, combien de sacs à dos. Chaque jour. De Centre et de Nord. »

Doña Lupe le regarde par-dessus ses lunettes.

« Pour ça, tu n'as pas besoin d'ordinateur, mon petit. Tu as besoin de quelqu'un qui compte. »

« On a déjà compté. Fer m'a envoyé ceux de Nord ce matin. »

Il y a deux semaines, personne dans la famille ne voulait prononcer le nom de Fer à voix haute. Centre vendait environ quatre fois plus que Nord, et c'était elle la responsable. Cet après-midi-là, elle n'avait pas répondu au téléphone. Mais aujourd'hui, elle a envoyé son comptage complet, ligne par ligne, sans qu'on ait eu à le lui demander deux fois.

Le téléphone de doña Lupe vibre. Un courriel de la banque. De monsieur Ortega.

« Chère madame Herrera : comme vous l'avez demandé, je vous communique le total encaissé par le terminal de votre succursale Nord pendant la semaine indiquée. Je vous rappelle que la date limite est le 23 décembre. Je reste à votre disposition. »

En dessous, un seul nombre : **16 850 $**.

Doña Lupe passe le téléphone à Mateo sans rien dire.

« C'est beaucoup ou c'est peu ? »

« Ça dépend de combien on a vendu. »

Et toute l'affaire est là. Pour savoir combien ils ont vendu, il faut multiplier chaque quantité par son prix. Vingt et une cases par succursale. Avec `for`, Mateo sait déjà le faire. Mais ce soir, il veut un calcul qui prenne le tableau entier d'un seul coup.

Parce que si les quantités multipliées par leurs prix donnent moins que ce qu'a encaissé le terminal, quelqu'un a encaissé quelque chose qui n'est jamais sorti de la papeterie.

Dans la session précédente, tu as changé la forme d'un tenseur sans toucher à ses nombres : 12 mois sur une ligne sont devenus 4 trimestres de 3 mois. Aujourd'hui, tu arrêtes de déplacer les nombres et tu commences à faire des calculs avec eux, sur le tenseur complet à la fois.

## Révision pour commencer

Réponds sans regarder la session précédente. Les réponses se trouvent dans les solutions, à la fin.

1. Quand un `reshape` est-il valide ?
2. Que signifie le `-1` dans `reshape` ?
3. Pourquoi faire un `reshape` d'un tableau, ce n'est pas la même chose que le faire pivoter ?

> **Le code de cette session** se trouve dans le dépôt public du cours, dans le dossier `sesion-06` : [github.com/HabilMX/curso-tensores](https://github.com/HabilMX/curso-tensores/tree/main/sesion-06). Écris-le toi-même d'abord, car c'est ainsi qu'on apprend ; le dépôt sert à comparer quand quelque chose ne donne pas le même résultat chez toi. Chaque programme est accompagné de la sortie qu'il doit te donner.

## Objectifs de cette session

À la fin, tu pourras faire ces quatre choses. À côté de chacune se trouve l'exercice qui la démontre.

| # | À la fin, tu peux… | Démontré par |
|---|---|---|
| 1 | **Opérer** sur deux tenseurs de même forme nombre par nombre, et **le vérifier** par rapport à un `for` | Exercice 23 |
| 2 | **Prédire**, avant d'exécuter, si deux formes peuvent être « étirées » pour opérer ensemble et quelle forme en sort | Exercices 22 et 25 |
| 3 | **Calculer** avec `@` un total par ligne et **le vérifier** de deux façons : avec un `for` et avec `sum(dim=1)` | Exercice 24 |
| 4 | **Reconnaître** quand `*` et `@` donnent des résultats différents sans signaler d'erreur | Exercice 22 |

**Ce que tu dois savoir avant :** les sessions 1 à 5 : créer un tenseur, lire son `shape` et son `dtype`, en extraire des morceaux avec des indices, additionner avec `sum(dim=)` et changer sa forme avec `reshape`. **Ce dont tu as besoin :** l'environnement de la session 2, avec PyTorch installé, et un nouveau fichier, `operar.py`, auquel tu vas ajouter du code étape par étape. Quand une étape te demande de provoquer une erreur, ajoute la ligne, exécute, lis l'erreur et supprime-la pour continuer.

---

## Pourquoi c'est important

Mateo pourrait faire le calcul comme dans les sessions précédentes : un `for` dans un autre `for`, case par case. Ça marche. Mais ce n'est pas ainsi que travaille un modèle d'intelligence artificielle. Ni le système qui a suspendu le terminal de Nord un dimanche à minuit.

Une bonne partie du travail d'un modèle passe dans deux opérations : **multiplier des tenseurs nombre par nombre** et **multiplier des matrices**. Quand une banque examine un achat, le modèle prend la liste de nombres de l'achat et la multiplie par un tableau de « poids », des nombres que le modèle a appris. C'est une multiplication de matrices. Et il ne le fait pas une fois : il le fait des millions de fois par seconde. Quelque part dans la banque, c'est un calcul de ce genre qui a marqué les débits de Nord comme « inhabituels ».

C'est pour ça qu'il existe des cartes graphiques pour l'IA. Une analogie utile : l'ordinateur normal, c'est une poignée de chefs très habiles ; la carte graphique, ce sont des milliers de commis qui ne savent que hacher, mais qui hachent tous en même temps. Multiplier des matrices, c'est « hacher » : énormément de multiplications et d'additions qui peuvent être réparties entre beaucoup de commis à la fois. AlexNet, le réseau de 2012 qui a lancé la vague actuelle de l'IA, selon l'article même de ses auteurs « met entre cinq et six jours à s'entraîner sur deux GPU GTX 580 de 3 Go » ([Krizhevsky, Sutskever et Hinton, 2012](https://papers.nips.cc/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html)). Deux cartes de jeux vidéo.

Aujourd'hui, tout tourne sur ton ordinateur, sans carte graphique. Les deux opérations sont les mêmes, et dans la prochaine session tu en auras besoin pour entraîner ton propre modèle. Ce soir, Mateo en a besoin pour une question plus simple : combien d'argent disent les quantités ?

## Le cas : maintenant, doña Lupe compte des quantités

Jusqu'ici, les nombres de doña Lupe étaient des pesos. Cette fois, Mateo lui a demandé autre chose : **combien d'unités** de chaque produit Centre a vendues chaque jour. Et, à part, sa liste de prix, qu'elle dicte de mémoire : le cahier à 40 $, le stylo à 10 $ et le sac à dos à 350 $.

« Ils n'ont pas changé depuis la rentrée des classes », dit-elle. « Et ils ne changeront pas tant que le papier n'augmente pas ou que je ne meurs pas, ce qui arrivera en premier. »

Avec les quantités et les prix, tout ce qui concerne l'argent sort d'un calcul. Mateo commence par Centre, la succursale que personne ne soupçonne : si le calcul tombe juste là, il pourra s'y fier quand il arrivera à Nord.

### Étape 1 — Les quantités et les prix

Écris ceci au début de `operar.py`. D'abord les listes Python, que tu connais déjà ; ensuite les tenseurs, avec ce que tu as vu dans la session 2 :

```python
import torch

lista_unidades = [
    [30, 80, 2],    # lunes
    [40, 90, 3],    # martes
    [25, 70, 2],    # miércoles
    [35, 90, 2],    # jueves
    [50, 120, 3],   # viernes
    [30, 80, 1],    # sábado
    [20, 50, 1],    # domingo
]
lista_precios = [40, 10, 350]   # cuaderno, pluma, mochila

unidades = torch.tensor(lista_unidades)
precios = torch.tensor(lista_precios)

print(unidades.shape, unidades.dtype)
print(precios.shape, precios.dtype)
```

Sortie :

```
torch.Size([7, 3]) torch.int64
torch.Size([3]) torch.int64
```

Lis-le comme dans la session 2 : `unidades` est un tableau de 7 jours sur 3 produits ; `precios` est une liste de 3 prix, un par produit. Les deux contiennent des entiers. Garde les listes Python : elles te serviront pour vérifier avec `for`.

Avant de continuer, une question à méditer : le lundi, 30 cahiers ont été vendus à 40 $. Combien d'argent est entré grâce aux cahiers le lundi ? Si tu as répondu 1 200 $, tu sais déjà faire le calcul d'aujourd'hui. La seule nouveauté, c'est de le faire avec les 21 nombres à la fois.

### Étape 2 — Élément par élément : deux tableaux identiques

Maintenant, le fichier de Fer. Mateo l'ouvre avec précaution : dans la session 2, un fichier de Nord avait une ligne de travers. Celui-ci est complet. Même forme que celui de Centre : 7 jours, 3 produits. Avant toute chose, Mateo veut voir les deux succursales ensemble, case contre case. Ajoute à ton fichier :

```python
lista_norte = [
    [10, 20, 1],
    [10, 25, 0],
    [5, 20, 1],
    [10, 15, 0],
    [15, 30, 1],
    [10, 20, 0],
    [5, 10, 1],
]
unidades_norte = torch.tensor(lista_norte)

total = unidades + unidades_norte
print(total)
```

Sortie :

```
tensor([[ 40, 100,   3],
        [ 50, 115,   3],
        [ 30,  90,   3],
        [ 45, 105,   2],
        [ 65, 150,   4],
        [ 40, 100,   1],
        [ 25,  60,   2]])
```

Le `+` a pris la case lundi-cahiers de Centre (30) et l'a additionnée à la case lundi-cahiers de Nord (10) : 40. Et ainsi pour les 21 cases, chacune avec sa partenaire. C'est ce qu'on appelle opérer **élément par élément** : chaque nombre est combiné avec celui qui se trouve à la même place dans l'autre tenseur.

C'est comme superposer deux feuilles de papier quadrillé contre la fenêtre : chaque petit carré tombe sur son partenaire. C'est pour ça que les deux feuilles doivent avoir la même forme.

Vérifie par rapport à la version avec `for`. Ajoute :

```python
total_for = []
for d in range(7):
    renglon = []
    for p in range(3):
        renglon.append(lista_unidades[d][p] + lista_norte[d][p])
    total_for.append(renglon)

assert total.shape == (7, 3), "Deberían salir 7 días × 3 productos"
assert total.tolist() == total_for, "El tensor y el for deberían dar lo mismo"
print("Todo cuadra")
```

```
Todo cuadra
```

Comme dans la session 5, `tolist()` reconvertit le tenseur en listes Python pour le comparer au `for`. D'abord la forme, ensuite la valeur.

**Attention : les listes Python ne font pas ça.** Essaie dans la console Python :

```python
print([1, 2, 3] + [10, 20, 30])
print(torch.tensor([1, 2, 3]) + torch.tensor([10, 20, 30]))
print([1, 2, 3] * 2)
print(torch.tensor([1, 2, 3]) * 2)
```

```
[1, 2, 3, 10, 20, 30]
tensor([11, 22, 33])
[1, 2, 3, 1, 2, 3]
tensor([2, 4, 6])
```

Avec des listes, `+` **colle** une liste derrière l'autre et `* 2` la **répète**. Avec des tenseurs, `+` additionne case par case et `* 2` multiplie chaque nombre. Si un jour ton résultat sort deux fois plus long au lieu de deux fois plus grand, c'est que tu opérais sur des listes.

Mateo reste les yeux fixés sur la dernière colonne de Fer : 1, 0, 1, 0, 1, 0, 1. Quatre sacs à dos dans toute la semaine. Le produit le plus cher du magasin, presque immobile, comme dans la session 3. Il le note sur un post-it. Il ne sait pas encore pourquoi.

### Étape 3 — L'étirement automatique : quantités par prix

Maintenant, le calcul qui compte. Les quantités sont un tableau `(7, 3)` ; les prix, une liste `(3,)`. Ils n'ont pas la même forme. Ajoute quand même ceci :

```python
ingresos = unidades * precios
print(ingresos)
print(ingresos.shape)
```

```
tensor([[1200,  800,  700],
        [1600,  900, 1050],
        [1000,  700,  700],
        [1400,  900,  700],
        [2000, 1200, 1050],
        [1200,  800,  350],
        [ 800,  500,  350]])
torch.Size([7, 3])
```

Ça a marché. Lundi : 30 cahiers × 40 $ = 1 200 $ ; 80 stylos × 10 $ = 800 $ ; 2 sacs à dos × 350 $ = 700 $. Et pareil pour les six autres jours.

Que s'est-il passé ? PyTorch a vu qu'il manquait à la liste de prix la dimension des jours, et il l'a **étirée** : il a fait comme si la liste de prix était écrite sept fois, l'une sous l'autre, une par jour.

```
   unidades (7, 3)          precios (3,)  →  estirado a (7, 3)
  30   80   2                40  10  350
  40   90   3                40  10  350
  25   70   2                40  10  350
  35   90   2      ×         40  10  350
  50  120   3                40  10  350
  30   80   1                40  10  350
  20   50   1                40  10  350
```

Et une fois les deux feuilles de la même taille, il a multiplié élément par élément, comme à l'étape 2.

En anglais, on appelle ça le ***broadcasting*** ; ici, nous allons l'appeler **étirement automatique**. L'important : PyTorch **ne copie pas** les prix sept fois en mémoire. Il fait seulement le calcul *comme si* ils étaient copiés. Avec 7 jours, ça ne change rien ; avec 7 millions d'achats, c'est cette différence qui évite que ton ordinateur se bloque.

C'est comme le carton de prix que doña Lupe a collé derrière le comptoir, écrit au feutre : il n'y en a qu'un, et il sert à encaisser toutes les ventes de la journée. Personne ne le réécrit pour chaque client.

Vérifie par rapport au `for` :

```python
ingresos_for = []
for d in range(7):
    renglon = []
    for p in range(3):
        renglon.append(lista_unidades[d][p] * lista_precios[p])
    ingresos_for.append(renglon)

assert ingresos.shape == (7, 3), "Los ingresos deberían ser 7 días × 3 productos"
assert ingresos.tolist() == ingresos_for, "El tensor y el for deberían dar lo mismo"
print("Ingresos correctos")
print(ingresos.sum())
```

```
Ingresos correctos
tensor(19900)
```

Regarde le `for` tranquillement : dans `lista_precios[p]`, seul le produit apparaît, pas le jour. C'est exactement ce qu'a fait l'étirement : utiliser le même prix pour tous les jours. Et le total de la semaine, 19 900 $, sort avec le `sum` que tu connais déjà depuis la session 4.

19 900 $ à Centre. Le calcul est vérifié deux fois : avec des tenseurs et avec `for`. Maintenant, Mateo a un outil auquel il peut se fier. Mais avant de l'utiliser avec Nord, il veut savoir où il casse.

### Étape 4 — La règle : quand peut-on étirer ?

PyTorch n'étire pas n'importe quoi. Il suit une règle, la même qu'utilise NumPy (la bibliothèque dont PyTorch a copié sa façon de travailler) et que suit aussi TensorFlow. La voici :

> **On compare les formes de droite à gauche. Deux tailles sont compatibles si elles sont égales ou si l'une des deux vaut 1.**

Et un complément : s'il manque des dimensions à gauche d'une forme, PyTorch y met des uns (1), et ces uns s'étirent.

Applique-la aux quantités par prix. Écris les formes alignées **à droite**, comme quand tu posais tes additions à l'école primaire :

```
unidades   7   3
precios        3
           ↑   ↑
           │   └─ 3 contra 3: iguales, compatibles
           └───── 7 contra nada: a precios le falta, se le pone 1 y se estira a 7
resultado  7   3
```

Maintenant, un cas qui **ne** marche **pas**. Doña Lupe a commencé à vendre des crayons de couleur à 25 $ et les a ajoutés à sa liste de prix, mais elle ne les compte pas encore dans son tableau de quantités.

« C'est que les enfants les demandent », se défend-elle. « Ils sont déjà sur le carton. Dans le comptage, pas encore. »

Ajoute ceci, exécute-le et lis l'erreur :

```python
precios_con_colores = torch.tensor([40, 10, 350, 25])   # agregó los colores
print(unidades * precios_con_colores)
```

```
Traceback (most recent call last):
  ...
RuntimeError: The size of tensor a (3) must match the size of tensor b (4) at non-singleton dimension 1
```

Lis-la mot à mot, comme dans la session 5 :

- *The size of tensor a (3)* : la taille du tenseur a, celui de gauche du `*`, `unidades`, est 3…
- *must match the size of tensor b (4)* : …et elle doit correspondre à la taille du tenseur b, `precios_con_colores`, qui est 4…
- *at non-singleton dimension 1* : …dans la dimension 1, qui n'est pas de taille 1. (*Singleton* veut dire « d'un seul élément ». Si l'un des deux mesurait 1, on pourrait l'étirer ; comme aucun ne mesure 1, il n'y a rien à faire.)

Avec la règle :

```
unidades               7   3
precios_con_colores        4
                           ↑
                           3 contra 4: ni iguales ni alguno es 1 → error
```

L'erreur te dit quelque chose de vrai sur ton commerce : tu as quatre prix et trois colonnes de quantités. À quelle colonne revient le prix des crayons de couleur ? PyTorch ne devine pas, et tant mieux. Supprime ces deux lignes pour continuer.

### Étape 5 — Un scalaire avec tout le monde : la TVA

Le cas d'étirement le plus simple, tu l'utilises depuis le début sans le savoir. Doña Lupe veut voir ses recettes avec la TVA de 16 % incluse, « pour que le comptable ne me gronde pas en janvier ». Ici, suppose que les prix de sa liste sont hors TVA ; dans un vrai magasin, le prix que voit le client doit déjà inclure les taxes ([Ley Federal de Protección al Consumidor, article 7 bis](https://www.diputados.gob.mx/LeyesBiblio/pdf/LFPC.pdf)). Ajoute :

```python
con_iva = ingresos * 1.16
print(con_iva)
print(con_iva.shape, con_iva.dtype)
print(ingresos.sum(), con_iva.sum())
```

```
tensor([[1392.,  928.,  812.],
        [1856., 1044., 1218.],
        [1160.,  812.,  812.],
        [1624., 1044.,  812.],
        [2320., 1392., 1218.],
        [1392.,  928.,  406.],
        [ 928.,  580.,  406.]])
torch.Size([7, 3]) torch.float32
tensor(19900) tensor(23084.)
```

Le 1.16 est un scalaire, un seul nombre ; dans la session 1, tu as vu qu'il n'a pas de dimensions, sa forme est `()`. D'après la règle, il lui manque toutes les dimensions, donc on lui met des uns et il s'étire en `(7, 3)`. C'est pour ça qu'il multiplie les 21 cases. La semaine avec TVA : 23 084 $.

Deux choses à remarquer dans la sortie :

- Les nombres ont maintenant un point (`1392.`) et le `dtype` est passé à `float32`. Multiplier des entiers par un nombre à décimales donne des décimales.
- La forme n'a pas changé. Un scalaire ne change jamais la forme de ce qu'il multiplie.

Ajoute sa vérification :

```python
assert con_iva.shape == ingresos.shape, "El IVA no cambia la forma"
assert con_iva[0][0] == 1392, "Lunes, cuadernos: 1,200 × 1.16 = 1,392"
print("IVA correcto")
```

```
IVA correcto
```

### Étape 6 — `@` : multiplier et additionner d'un seul coup

« Et combien on a vendu vendredi ? » demande doña Lupe. « Le vendredi, c'est toujours le bon jour. »

Elle veut la recette de chaque jour. Tu saurais déjà le faire en deux étapes : multiplier les quantités par les prix, puis additionner chaque ligne avec `sum(dim=1)`. Cette combinaison, « multiplie par paires et additionne », est si courante qu'elle a son propre symbole : `@`. On l'appelle **produit matriciel** (en anglais, *matrix multiplication* ; dans PyTorch, il existe aussi sous le nom `torch.matmul`).

Ajoute :

```python
por_dia = unidades @ precios
print(por_dia)
print(por_dia.shape)
```

```
tensor([2700, 3550, 2400, 3000, 4250, 2350, 1650])
torch.Size([7])
```

Sept totaux, un par jour, en une seule opération. Fais-le manuellement pour le lundi et tu verras ce qu'a fait `@` :

```
lunes:   30  80   2
precios: 40  10  350

30 × 40  +  80 × 10  +  2 × 350
 1200    +   800     +   700     =  2700
```

`@` prend **une ligne** du tableau de gauche et **la liste** de droite, multiplie les paires et **additionne le tout**. Une ligne, un nombre. Sept lignes, sept nombres. C'est pour ça que la forme est passée de `(7, 3)` à `(7,)` : la dimension des produits a été « consommée » par l'addition.

C'est exactement ce que fait la caissière avec le ticket : quantité par prix, ligne par ligne, et à la fin le total. `@` fait le ticket de chaque jour. Et le vendredi, le bon jour, a été le plus élevé : 4 250 $.

**La règle des formes pour `@`**, avec des tableaux et des listes comme ceux de cette session : la dernière taille de gauche doit être égale à la première de droite, et cette taille disparaît :

```
unidades @ precios
 (7, 3)    (3,)
     └──┬──┘
   3 y 3: iguales, se consumen
resultado: (7,)
```

Maintenant, vérifie-le de trois façons. C'est l'autocorrection la plus importante de la session :

```python
# 1) Con for, como lo harías sin tensores
por_dia_for = []
for d in range(7):
    suma = 0
    for p in range(3):
        suma = suma + lista_unidades[d][p] * lista_precios[p]
    por_dia_for.append(suma)

# 2) Multiplicar y luego sumar por renglón
por_dia_dos_pasos = (unidades * precios).sum(dim=1)

assert por_dia.shape == (7,), "Debería salir un número por día"
assert por_dia.tolist() == por_dia_for, "El @ y el for deberían dar lo mismo"
assert por_dia.tolist() == por_dia_dos_pasos.tolist(), "El @ y el sum(dim=1) deberían dar lo mismo"
print("Los tres caminos dan lo mismo")
```

```
Los tres caminos dan lo mismo
```

Regarde le `for` : il contient `suma = suma + …`. Cette addition est la différence avec le `for` de l'étape 3, qui ne faisait que multiplier. `*` multiplie ; `@` multiplie **et additionne**.

**L'ordre compte.** Essaie à l'envers :

```python
print(precios @ unidades)
```

```
Traceback (most recent call last):
  ...
RuntimeError: mat1 and mat2 shapes cannot be multiplied (1x3 and 7x3)
```

*mat1 and mat2 shapes cannot be multiplied* : les formes de la matrice 1 et de la matrice 2 ne peuvent pas être multipliées. PyTorch a traité les prix comme un tableau de 1 × 3 et l'a placé face à un tableau de 7 × 3 : la dernière taille de gauche (3) n'est pas égale à la première de droite (7). Avec `@`, celui qui fournit les lignes va à gauche.

### Étape 7 — Le piège le plus dangereux : `*` et `@` avec des tableaux carrés

Il est 21 h 15 et Mateo est sur le point de commettre l'erreur que presque tout le monde commet une fois.

Jusqu'ici, chaque fois que tu as utilisé le mauvais symbole, PyTorch s'est plaint. Avec des tableaux carrés, non.

Doña Lupe veut comparer deux succursales sur deux produits : les quantités de cahiers et de stylos, à Centre et à Nord. À Nord, elle vend un peu plus cher, donc elle a un tableau de prix par succursale. Ce qu'elle veut, c'est la recette de chaque case : quantité par prix, chacune avec sa partenaire. C'est élément par élément, `*`. Mais imagine que quelqu'un écrive `@` par habitude :

```python
import torch
# Renglones: Centro y Norte. Columnas: cuadernos y plumas.
piezas = torch.tensor([[30, 80],
                       [10, 20]])
precio = torch.tensor([[40, 10],    # en el Centro
                       [45, 12]])   # en Norte, un poco más caro
print(piezas * precio)
print(piezas @ precio)
```

```
tensor([[1200,  800],
        [ 450,  240]])
tensor([[4800, 1260],
        [1300,  340]])
```

**Aucun des deux n'a signalé d'erreur.** Les deux ont donné un tableau de 2 × 2. Mais seul le premier correspond à ce que voulait doña Lupe : 30 cahiers × 40 $ = 1 200 $. Le second dit qu'à Centre, 4 800 $ sont entrés grâce aux cahiers, quatre fois plus que la réalité. D'où ça sort ? Avec deux tableaux, `@` prend une ligne de gauche et une **colonne** de droite, et fait le « multiplie et additionne » du ticket : la ligne de Centre (30, 80) face à la colonne des prix des cahiers (40, 45) donne 30 × 40 + 80 × 45 = 4 800. Il a mélangé les quantités de stylos de Centre avec le prix des cahiers de Nord. Ça n'a aucun sens pour le commerce.

Pourquoi n'y a-t-il pas eu d'erreur ? Parce qu'avec des tableaux carrés, les deux règles sont respectées à la fois : `(2, 2)` et `(2, 2)` sont égales, donc `*` fonctionne ; et la dernière taille de gauche (2) est égale à la première de droite (2), donc `@` fonctionne aussi.

C'est le piège le plus dangereux parce qu'**une erreur qui plante se corrige en cinq minutes ; une erreur qui ne plante pas arrive jusqu'au rapport du directeur.** Ou jusqu'au bureau de monsieur Ortega. Mateo s'imagine arriver à la banque le 23 avec un tableau qui a l'air parfait et qui est faux. Il en a froid dans le dos.

La protection est celle de toute la série : compare au `for`, même sur une seule ligne :

```python
lista_piezas = [[30, 80], [10, 20]]
lista_precio = [[40, 10], [45, 12]]
ingreso_for = []
for s in range(2):
    renglon = []
    for p in range(2):
        renglon.append(lista_piezas[s][p] * lista_precio[s][p])
    ingreso_for.append(renglon)
print(ingreso_for)
```

```
[[1200, 800], [450, 240]]
```

Ça correspond à `*`, pas à `@`. Et pour choisir le symbole avant d'écrire, une seule question :

| Si tu veux… | Utilise | Ce qui arrive à la forme |
|---|---|---|
| Chaque nombre avec son partenaire (prix × quantité de la même case) | `*` | Elle reste la même (ou celle de l'étirement) |
| Multiplier par paires **et additionner** (le total du ticket) | `@` | La dimension additionnée disparaît |

---

**Et dans TensorFlow ?** L'étirement suit la même règle et donne le même résultat. Le `@` est plus strict : avec un tableau et une liste, il signale une erreur (*different ndims*, nombre de dimensions différent), et pour ce cas on utilise une autre fonction, `tf.linalg.matvec`. Mêmes idées, autres manières.

---

👉 **Si tu viens de la gestion :** tu as déjà fait ça dans ton tableur. Quand tu écris `=B2*$F$1` et que tu tires la formule vers le bas, le prix fixe de `$F$1` est utilisé sur toutes les lignes : c'est l'étirement. Et `@`, c'est la fonction `SOMMEPROD` d'Excel : elle multiplie par paires et additionne. Le piège de l'étape 7 a aussi sa version dans Excel : une formule qui n'affiche pas `#VALEUR!` mais qui additionne la mauvaise colonne. C'est pour ça que, dans n'importe quel rapport, tu vérifies une ligne manuellement avant de le présenter.

---

## 22 h 40 — Ce qui ne colle pas

Mateo fait avec Nord la même chose qu'avec Centre : les quantités de Fer par les mêmes prix, puis la somme. Une ligne. (Ce calcul, c'est toi qui le fais dans l'exercice 23 ; je ne te donne pas le nombre à l'avance.)

Il le compare au courriel d'Ortega. Il le refait. Il le compare au `for`. Même résultat.

« Ma tante. Ce que Nord a vendu sur toute la semaine n'arrive même pas à la moitié de ce qu'a encaissé le terminal. »

« Peut-être que presque tout le monde a payé par carte. »

« Même si *tout le monde* avait payé par carte. Même si personne n'avait laissé un seul peso en espèces. Le terminal a encaissé 16 850 $, et à Nord, il n'est pas sorti de marchandise pour autant d'argent. »

Doña Lupe pose les pièces de monnaie.

« Alors, qu'est-ce qu'ils ont encaissé ? »

« Rien. Des débits sans marchandise derrière. »

Dehors, un camion passe et le rideau vibre.

« Fer », dit doña Lupe, lentement. « C'est elle qui encaisse à Nord. »

Mateo y avait déjà pensé. C'est pour ça que, pendant que le `for` tournait, il avait ouvert la boîte mail de la papeterie. Celle que personne ne consulte. Celle dont doña Lupe dit que « ce n'est que de la pub ». Il avait cherché « Nord ». Il avait cherché « Fernanda ».

C'était là. Mardi 22 septembre, 18 h 12. Objet : **Écarts : le terminal ne correspond pas à l'inventaire**.

« Doña Lupe : j'ai fait la clôture de la semaine et le terminal indique plus d'encaissements que ce qui sort de mon inventaire. J'ai compté deux fois. D'après le terminal, il devrait me manquer des articles sur l'étagère, et il ne m'en manque pas. Je crois qu'il y a un problème avec le nouveau terminal. Je préviens la banque ou je vous préviens vous ? Fer. »

Sans réponse. Pas ouvert. Depuis septembre.

« Elle a fait le même calcul que moi », dit Mateo. « Quantités par prix face à ce qui a été encaissé. À la calculatrice. Il y a deux mois et demi. »

Doña Lupe retire ses lunettes et se frotte les yeux.

« Et moi qui pensais que c'était elle. »

« On pensait avec des soupçons, ma tante. Les nombres disent autre chose. Si Fer volait, elle ne t'écrirait pas pour te prévenir que ça ne colle pas. »

« Demain, je l'appelle. » Un silence. « Et je lui apporte des conchas. Au chocolat, ce sont celles qu'elle préfère. »

Ce soir-là, Mateo apprend ce qu'aucun `assert` n'allait lui apprendre : **on décide avec des données, pas avec des soupçons.**

---

## Exercices de la session 6

Les solutions se trouvent à la fin de cette livraison. Essaie d'abord ; le `assert` te dit si tu as réussi.

**Exercice 22 — Prédis avant d'exécuter.**

a) Pour chaque paire, écris sur papier si elles peuvent être étirées et, si oui, quelle forme en sort. Utilise la règle de l'étape 4 : aligne à droite et compare. Ensuite, exécute-le :

`torch.ones(...)` crée un tenseur rempli de uns avec la forme que tu indiques. Ici, on ne l'utilise que pour tester des formes : les valeurs n'ont pas d'importance.

```python
print((torch.ones(7, 3) * torch.ones(3)).shape)
print((torch.ones(2, 7, 3) * torch.ones(7, 1)).shape)
print((torch.ones(7, 1) * torch.ones(1, 3)).shape)
print((torch.ones(7, 3) * torch.tensor(2)).shape)
print((torch.ones(2, 7, 3) * torch.ones(2)).shape)
```

Voici à quoi ressemble une paire résolue : `(7, 3)` et `(3,)` → j'aligne : 3 avec 3, égales ; il manque le 7 à la liste, elle s'étire → on obtient `(7, 3)`.

b) Avec ce tableau, calcule **manuellement**, sur papier, `m * m` et `m @ m`. Ensuite, exécute-le et compare.

```python
m = torch.tensor([[2, 1],
                  [0, 3]])
print(m * m)
print(m @ m)
```

Indice pour `@` : la case en haut à droite vient de la ligne du haut, (2, 1), face à la colonne de droite, (1, 3) : 2 × 1 + 1 × 3.

**Exercice 23 — Les recettes de Nord.** Remplis les trous. Utilise `unidades_norte` de l'étape 2 et les mêmes `precios`.

```python
ingresos_norte = ____ * ____

ingresos_norte_for = []
for d in range(7):
    renglon = []
    for p in range(3):
        renglon.append(____)
    ingresos_norte_for.append(renglon)

assert ingresos_norte.shape == (7, 3), "Norte también es 7 días × 3 productos"
assert ingresos_norte.tolist() == ingresos_norte_for, "El tensor y el for deberían dar lo mismo"
con_iva = ____
print(ingresos_norte.sum(), con_iva.sum())
print("Ejercicio 23 superado")
```

Avant de l'exécuter, prédis : combien sont entrés grâce aux sacs à dos le mardi à Nord ? Regarde le tableau des quantités.

**Exercice 24 — Le Buen Fin.** Pour le Buen Fin, doña Lupe baisse le cahier de 5 $, ne touche pas au stylo et baisse le sac à dos de 50 $. Suppose qu'elle vend les mêmes quantités que cette semaine.

1. Crée un tenseur `descuento` avec les trois remises (en nombres négatifs) et additionne-le à `precios` pour obtenir `precios_bf`.
2. Avec `@`, calcule `por_dia_bf`, la recette de chaque jour avec les prix du Buen Fin.
3. Écris deux `assert` : celui de la forme, et un qui compare ton `@` à la version avec `*` et `sum(dim=1)`.
4. Calcule combien elle a cessé de gagner chaque jour à cause des remises, et combien sur toute la semaine.

Défi : l'argent qu'elle a cessé de gagner chaque jour peut s'obtenir avec un seul `@`, sans utiliser `por_dia_bf`. Lequel ?

**Exercice 25 — Écris toi-même la règle.** Sans PyTorch, seulement avec Python. Écris une fonction `se_pueden_estirar(forma_a, forma_b)` qui reçoit deux tuples et renvoie `True` s'ils peuvent être étirés pour opérer ensemble et `False` sinon. Elle doit passer ces `assert` :

```python
assert se_pueden_estirar((7, 3), (3,)) == True
assert se_pueden_estirar((7, 3), (7,)) == False
assert se_pueden_estirar((2, 7, 3), (7, 1)) == True
assert se_pueden_estirar((7, 3), (4,)) == False
assert se_pueden_estirar((7, 1), (1, 3)) == True
assert se_pueden_estirar((7, 3), ()) == True
print("Ejercicio 25 superado")
```

Indices : `forma[-1]` est la dernière taille d'un tuple et `forma[-2]` l'avant-dernière ; c'est ainsi que tu parcours de droite à gauche. Il suffit de comparer autant de tailles qu'en a la forme la plus courte.

---

## Erreurs courantes (et comment t'en rendre compte)

| Erreur | Ce que l'on observe | Comment t'en rendre compte | Ce qu'il faut faire |
|---|---|---|---|
| Une liste de prix avec un produit en trop (ou en moins) | `RuntimeError: The size of tensor a (3) must match the size of tensor b (4) at non-singleton dimension 1` | Les deux nombres entre parenthèses sont différents et aucun ne vaut 1 | Affiche le `.shape` des deux avant d'opérer ; corrige les données, pas le calcul |
| Diviser un tableau `(7, 3)` par un total par jour `(7,)` | `RuntimeError: The size of tensor a (3) must match the size of tensor b (7) at non-singleton dimension 1` | La règle aligne à droite : elle a mis les 7 jours sous les 3 produits | Donne-lui une forme de colonne : `por_dia.reshape(7, 1)` |
| Utiliser `@` là où tu voulais `*` avec des tableaux carrés | **Il n'y a pas d'erreur.** On obtient un tableau de la taille attendue avec des nombres qui ne collent pas | Calcule une case manuellement ou compare au `for` | Demande-toi : est-ce que je veux des paires (`*`) ou des paires additionnées (`@`) ? |
| Inverser les côtés de `@` | `RuntimeError: mat1 and mat2 shapes cannot be multiplied (1x3 and 7x3)` | La dernière taille de gauche n'est pas égale à la première de droite | Celui qui fournit les lignes va à gauche : `unidades @ precios`. Faire pivoter avec `.T` ne sert à rien ici : une liste `(3,)` reste `(3,)` |
| Mélanger entiers et décimaux dans `@` | `RuntimeError: addmv input tensors must have the same dtype, but got Float, Long, and Float` | Vérifie le `.dtype` des deux | Convertis celui des entiers avec `.float()` |
| S'attendre à ce qu'une liste Python opère comme un tenseur | `[1, 2, 3] * 2` donne `[1, 2, 3, 1, 2, 3]` | Le résultat sort plus long, pas plus grand | Convertis d'abord avec `torch.tensor(...)` |

---

## Révision pour la prochaine session

La session 7 commence par ces trois questions. Réponds-y sans regarder la leçon :

1. Quand deux formes peuvent-elles être « étirées » pour opérer ensemble ?
2. Quelle différence y a-t-il entre `*` et `@` ?
3. Quelle forme sort d'un tableau `(7, 3) @` un vecteur `(3,)` ?

🔁 **La semaine prochaine :** Mateo peut déjà prouver que ces débits ne sont pas sortis de la papeterie. Ce qu'il ne sait pas, c'est d'où ils sont sortis. Pour ça, il va construire un détecteur de fraude qui apprend à partir des données : tu lui donnes des milliers d'achats inventés avec une règle cachée, et le modèle la découvre tout seul, à force de `@` et d'étirement automatique, les deux opérations d'aujourd'hui.

*Les débits ne venaient pas de la papeterie. Ils venaient d'ailleurs. Et la banque ne voyait que le terminal.*

---

## Solutions (ne les regarde pas avant d'avoir essayé)

**Révision pour commencer.**

1. Quand le nombre total de valeurs est le même avant et après : une forme `(12,)` tient dans `(4, 3)` parce que 12 = 4 × 3, mais pas dans `(5, 3)`.
2. Que PyTorch calcule cette taille pour toi, avec les nombres qui restent. Il ne peut y avoir qu'un seul `-1` par `reshape`.
3. Parce que `reshape` réorganise les nombres dans l'ordre où ils sont stockés, ligne par ligne ; faire pivoter le tableau, pour que les lignes deviennent des colonnes, c'est `.T`.

**Exercice 22.**

a) Sortie réelle :

```
torch.Size([7, 3])
torch.Size([2, 7, 3])
torch.Size([7, 3])
torch.Size([7, 3])
Traceback (most recent call last):
  ...
RuntimeError: The size of tensor a (3) must match the size of tensor b (2) at non-singleton dimension 2
```

- `(7, 3)` et `(3,)` : 3 avec 3 ; on met un 1 à celui de droite et il s'étire à 7 → `(7, 3)`.
- `(2, 7, 3)` et `(7, 1)` : 3 avec 1, il s'étire ; 7 avec 7 ; il manque le 2 à celui de droite → `(2, 7, 3)`.
- `(7, 1)` et `(1, 3)` : les deux s'étirent → `(7, 3)`.
- `(7, 3)` et un scalaire `()` : il manque toutes les dimensions au scalaire, il s'étire → `(7, 3)`.
- `(2, 7, 3)` et `(2,)` : le 2 s'aligne sur le 3 de droite, pas sur le 2 de gauche. 3 face à 2 : erreur. C'est celui qui trompe : le 2 « ressemble » aux succursales, mais la règle compare par la droite. Remarque que l'erreur dit *dimension 2* : dans une forme à trois dimensions, la dimension 2 est la dernière.

b)

```
tensor([[4, 1],
        [0, 9]])
tensor([[4, 5],
        [0, 9]])
```

`m * m` : chaque nombre au carré. `m @ m` : en haut à gauche, 2 × 2 + 1 × 0 = 4 ; en haut à droite, 2 × 1 + 1 × 3 = 5 ; en bas à gauche, 0 × 2 + 3 × 0 = 0 ; en bas à droite, 0 × 1 + 3 × 3 = 9. Regarde comme c'est traître : les deux tableaux coïncident sur trois cases sur quatre. Si tu ne vérifiais que celle d'en haut à gauche, tu croirais qu'ils donnent la même chose.

**Exercice 23.** Les trous sont `unidades_norte` et `precios` ; `lista_norte[d][p] * lista_precios[p]`, et `ingresos_norte * 1.16`. La sortie :

```
tensor(5400) tensor(6264.)
Ejercicio 23 superado
```

Les sacs à dos du mardi à Nord : 0 unité, donc 0 $. Si tu affiches `ingresos_norte`, le mardi vaut `[400, 250, 0]`.

**Exercice 24.**

```python
descuento = torch.tensor([-5, 0, -50])
precios_bf = precios + descuento
por_dia_bf = unidades @ precios_bf
print(precios_bf)
print(por_dia_bf)

assert por_dia_bf.shape == (7,), "Debería salir un número por día"
assert por_dia_bf.tolist() == (unidades * precios_bf).sum(dim=1).tolist(), "El @ y el sum(dim=1) deberían dar lo mismo"

dejo_de_ganar = unidades @ precios - por_dia_bf
print(dejo_de_ganar)
assert dejo_de_ganar.tolist() == (-(unidades @ descuento)).tolist()
print(dejo_de_ganar.sum())
print("Ejercicio 24 superado")
```

```
tensor([ 35,  10, 300])
tensor([2450, 3200, 2175, 2725, 3850, 2150, 1500])
tensor([250, 350, 225, 275, 400, 200, 150])
tensor(1850)
Ejercicio 24 superado
```

`precios + descuento` se fait élément par élément : deux listes `(3,)`. Sur la semaine, elle a cessé de gagner 1 850 $. Le défi : `unidades @ descuento` donne ce qu'elle a cessé de gagner chaque jour, avec un signe négatif, parce que les remises sont négatives ; le `-` placé devant le rend positif. C'est ce que vérifie le dernier `assert`.

**Exercice 25.** Une solution possible :

```python
def se_pueden_estirar(forma_a, forma_b):
    cuantos = min(len(forma_a), len(forma_b))
    for i in range(1, cuantos + 1):
        a = forma_a[-i]
        b = forma_b[-i]
        if a != b and a != 1 and b != 1:
            return False
    return True
```

Avec les `assert` de l'énoncé en dessous, la sortie est :

```
Ejercicio 25 superado
```

Elle parcourt de droite à gauche avec des indices négatifs : `-1`, `-2`… Elle ne compare qu'autant de tailles qu'en a la forme la plus courte, parce qu'il manque des dimensions à l'autre et que celles-ci peuvent toujours s'étirer. Si elle trouve une paire qui n'est pas égale et dans laquelle aucune ne vaut 1, il n'y a plus rien à faire.

---

## La suite

Dans la session 7, tu rassembles tout ce que tu as appris : un tenseur d'achats, un `@` face à un tableau de poids et un étirement pour ajouter un ajustement, répétés jusqu'à ce que le modèle apprenne. Si tu veux prendre de l'avance, refais l'étape 6 manuellement avec le mardi : si tu obtiens 3 550 $ sans regarder, tu es prêt.

---

*J'écris à partir de ce que nous voyons tous les jours chez Hábil, où nous concevons des systèmes pour le secteur bancaire, les assurances et les entreprises au Mexique. Si, dans ton travail ou dans l'entreprise de ta famille, tu rencontres un problème de ce genre, discutons-en : [habil.mx](https://www.habil.mx/fr/)*
