# Les tenseurs à partir de zéro (1) : un nombre, une liste, un tableau, un cube

*Ce qu'est un tenseur, raconté avec les listes de Python que tu sais déjà écrire, et pourquoi Einstein a dû demander de l'aide à un ami pour le comprendre.*

*Les tenseurs à partir de zéro · leçon A, session 1 sur 8*

*Scénario : les personnages et l'histoire sont inventés. Les données techniques, historiques et les chiffres réels indiquent leur source.*

**Dimanche 1er novembre, 23 h 52**

Le téléphone portable de doña Lupe a vibré sur la table de nuit. Une seule fois. Elle qui ne répond jamais à rien après dix heures l'a retourné par pure habitude.

> *Nous avons détecté des débits inhabituels sur le terminal de votre succursale Nord. Le terminal est suspendu pendant sa vérification.*

Elle l'a lu trois fois. Puis elle s'est assise au bord du lit, les lunettes sur le front, et ne s'est plus rendormie.

**Lundi 2 novembre, 9 h 10**

— Bonjour, madame Herrera. Ici monsieur Ortega, de la banque.

Voix formelle. Pressé comme quelqu'un qui a quarante autres appels en attente.

— Vous avez jusqu'au 23 décembre pour nous prouver quelles transactions de ce terminal correspondent à de véritables ventes. Sinon, nous le résilions.

— Le 23 ? Monsieur Ortega, c'est la pleine saison ! Sans terminal à Nord, en décembre…

— C'est bien pour cela que je vous accorde ce délai. Bonne journée.

Clic.

**Lundi 2 novembre, 19 h 40. Papeterie, succursale Centre.**

Odeur de papier neuf et de gomme à effacer. Le rideau métallique à moitié baissé, qui racle chaque fois qu'un camion passe. Mateo Ríos, 20 ans, deuxième semestre d'informatique, entre en se baissant, le sac à dos sur l'épaule et l'ordinateur portable sous le bras.

— Ma tante, maman m'a dit que c'était urgent.

Doña Lupe ne le salue pas. Elle lui met le téléphone sous les yeux, avec le message de la banque.

— Toi qui t'y connais en ordinateurs.

Mateo le lit. Il sent son estomac se nouer.

— Ma tante, moi je connais les bases de Python. Ça, c'est… autre chose.

— Eh bien, apprends l'autre chose. Tu as jusqu'au 23 décembre. — Elle croise les bras. — Et ne me dis pas non, c'est moi qui ai payé tes frais d'inscription.

Mateo ouvre son ordinateur portable sur le comptoir, entre un pot à crayons et une pile de papier cartonné. Il ne sait qu'une seule chose avec certitude : la banque n'a pas vu une papeterie. Elle a vu des chiffres. S'il veut défendre sa tante, il doit apprendre à voir la même chose que la banque.

Et tu vas apprendre avec lui.

---

Cette série comporte deux leçons de quatre sessions chacune, à raison d'une session par semaine. Dans la première leçon, **De la liste au cube**, tu apprends ce qu'est un tenseur, à le créer, à le parcourir et à le résumer. Dans la deuxième, **Les tenseurs au travail**, tu modifies sa forme, tu effectues des opérations sur le tenseur complet, tu entraînes avec lui un modèle qui détecte les fraudes et tu découvres où il apparaît dans une banque, une compagnie d'assurance et un magasin.

| Session | Ce que tu vois | Pratique |
|---|---|---|
| **1 — Un nombre, une liste, un tableau, un cube** | Ce qu'est une dimension, comment les compter et ce qu'est la forme d'un tenseur, tout cela avec des listes Python | Exercices 1 à 5 |
| **2 — Ton premier tenseur** | Installer PyTorch, convertir tes listes en tenseurs, `shape` et `dtype` | Exercices 6 à 9 |
| **3 — Les axes ont un nom** | Extraire un morceau du tenseur selon n'importe lequel de ses axes | Exercices 10 à 13 |
| **4 — Additionner par onglet** | Totaux et moyennes par axe, comparés à un `for` | Exercices 14 à 17, auto-évaluation et révision de la leçon |

Chaque session dure environ 2 heures : à peu près une heure pour lire et suivre l'exemple, et une autre pour pratiquer. À partir de la session 2, chacune commence par **3 questions de révision** de la précédente, auxquelles répondre sans regarder.

**Ce que tu dois savoir avant :** les bases de Python : variables, `if`, `for` et `while`, fonctions, listes et tuples. Si tu viens du cours de Python de Hábil, ce sont ses leçons 0 à 4. **Ce que tu dois avoir :** Python 3 et un éditeur de texte. Aujourd'hui, tu n'installes rien d'autre.

> **Le code de cette session** se trouve dans le dépôt public du cours, dans le dossier `sesion-01` : [github.com/HabilMX/curso-tensores](https://github.com/HabilMX/curso-tensores/tree/main/sesion-01). Écris-le toi-même d'abord, car c'est ainsi qu'on apprend ; le dépôt sert à comparer quand quelque chose ne donne pas le même résultat chez toi. Chaque programme est accompagné de la sortie qu'il doit te donner.

## Objectifs de cette session

À la fin, tu seras capable de faire ces quatre choses. À côté de chacune se trouve l'exercice qui le démontre.

| # | À la fin, tu peux… | Démontré par |
|---|---|---|
| 1 | Étant donné un ensemble de données (ventes, notes, une photo), **dire** combien de dimensions il possède et ce que chacune représente | Exercice 1 |
| 2 | **Écrire** ces données sous forme de listes imbriquées Python et **désigner** un nombre grâce à ses indices | Exercices 2 et 4 |
| 3 | **Calculer** la forme d'un tenseur (combien d'éléments se trouvent dans chaque dimension), manuellement et avec une fonction | Exercices 1 et 3 |
| 4 | **Reconnaître** quand un tableau ne peut pas être un tenseur parce qu'il est bancal, et le vérifier avec du code | Exercice 5 |

---

## Pourquoi c'est important

Cette nuit-là, pendant que doña Lupe compte la monnaie de la caisse, Mateo cherche sur son téléphone portable comment s'y prend une banque pour qualifier une opération d'« inhabituelle ». La réponse lui fait un peu peur.

La banque ne lit pas un achat comme toi tu le lis. Elle le convertit en nombres : le montant, l'heure, la distance par rapport au dernier achat, le nombre d'achats effectués au cours de la dernière heure. Un achat devient une liste de nombres. Mille achats, un tableau. Les achats de chaque client au cours des 30 derniers jours, un cube.

Personne à la banque n'a vu le visage des clients de Nord. Une machine a vu des cubes de nombres, et quelque chose ne collait pas.

Et ce n'est pas seulement la banque. Quand une compagnie d'assurance reçoit la photo d'une voiture accidentée, elle ne voit pas non plus une voiture. Elle voit une grille de points, et chaque point comporte trois nombres : sa quantité de rouge, de vert et de bleu. Là encore, un cube de nombres.

Presque tous les systèmes d'intelligence artificielle dont tu entendras parler au cours de ta carrière fonctionnent ainsi : ils commencent par convertir le monde en boîtes de nombres bien ordonnées, puis ils effectuent des calculs avec ces boîtes. Ces boîtes s'appellent des **tenseurs**. Si tu comprends comment elles sont ordonnées, tu comprends la moitié de ce que fait un modèle. L'autre moitié, nous la verrons dans la deuxième leçon.

— Et ça se mange avec quoi ? demande doña Lupe sans lever les yeux de ses pièces.

— Avec des listes, ma tante. Des listes dans des listes.

C'est vrai, et c'est la bonne nouvelle : tu sais déjà écrire les données d'un tenseur. Ce sont des listes dans des listes, formées uniquement de nombres et bien régulières. Dans la session 2, tu vas les transformer en véritables tenseurs.

## Le cas : la papeterie de doña Lupe

Doña Lupe tient une papeterie avec deux succursales, Centre et Nord. Elle vend trois articles qui comptent pour elle : des cahiers, des stylos et des sacs à dos. Les chiffres de cette série sont donnés à titre d'exemple, mais la forme des données est celle de n'importe quel commerce.

Mateo ne sait pas encore ce qu'il cherche. Il fait donc la seule chose sensée : commencer par le plus facile et demander progressivement plus de détails. Il interroge sa tante sur les ventes de la semaine dernière à quatre reprises, en regardant chaque fois d'un peu plus près.

### Étape 1 — Un nombre

Doña Lupe baisse le rideau métallique jusqu'en bas. Le vacarme métallique résonne dans tout le local.

— Ma tante, combien as-tu vendu la semaine dernière ?
— 18 500 $, au Centre. Et ne me parle pas encore de Nord, ma tension monte rien que d'y penser.

```python
total = 18500
```

Un seul nombre. Pour le désigner, tu n'as besoin d'aucune donnée : il n'y en a qu'un. On dit qu'il a **zéro dimension**. Dans le langage des tenseurs, on l'appelle un **scalaire**.

Un scalaire dit beaucoup et n'explique pas grand-chose. Avec 18 500 $, personne ne peut rien défendre face à une banque.

### Étape 2 — Une liste

— Et jour par jour ?
— Lundi 2 500 ; mardi 3 100 ; mercredi 2 200 ; jeudi 2 800 ; vendredi 3 900 ; samedi 2 400 ; dimanche 1 600. — Elle récite cela de mémoire, comme une prière.

```python
por_dia = [2500, 3100, 2200, 2800, 3900, 2400, 1600]
```

Cette fois, tu as bien besoin d'une information pour désigner un nombre : **le jour**. `por_dia[4]` correspond au vendredi, 3 900 (rappelle-toi que les indices commencent à 0). Une direction, **une dimension**. On l'appelle un **vecteur**.

Vérifie que rien ne s'est perdu : `sum(por_dia)` donne 18500, le même total qu'à l'étape 1. Le détail ne change pas le total ; il ne fait que le répartir.

Mateo note quelque chose sur un post-it : *le détail ne ment pas ; le total dissimule.* Il ne sait pas encore à quel point cette phrase va lui être utile.

### Étape 3 — Un tableau

— Et par jour et par produit ?

Doña Lupe pousse un soupir, sort son vieil ordinateur portable, celui qui a l'autocollant de la Vierge, et ouvre sa feuille de calcul :

| Jour | Cahiers | Stylos | Sacs à dos |
|---|---|---|---|
| Lundi | 1 200 | 800 | 500 |
| Mardi | 1 500 | 900 | 700 |
| Mercredi | 1 000 | 700 | 500 |
| Jeudi | 1 300 | 900 | 600 |
| Vendredi | 2 000 | 1 200 | 700 |
| Samedi | 1 200 | 800 | 400 |
| Dimanche | 800 | 500 | 300 |

En Python, chaque ligne est une liste, et le tableau est une liste de lignes :

```python
por_dia_y_producto = [
    [1200, 800, 500],    # lunes
    [1500, 900, 700],    # martes
    [1000, 700, 500],    # miércoles
    [1300, 900, 600],    # jueves
    [2000, 1200, 700],   # viernes
    [1200, 800, 400],    # sábado
    [800, 500, 300],     # domingo
]
```

Pour désigner un nombre, tu as besoin de **deux informations : le jour et le produit**. `por_dia_y_producto[4][2]` correspond au vendredi (ligne 4), sacs à dos (colonne 2) : 700. Deux directions, **deux dimensions**. On l'appelle une **matrice**.

Remarque bien l'ordre : d'abord la ligne, ensuite la colonne. Cet ordre n'est pas un caprice ; c'est l'ordre dans lequel tu as écrit les listes. La liste extérieure correspond aux jours ; la liste intérieure, aux produits.

— Autrement dit, chaque nombre a son adresse, dit doña Lupe.

— Exactement, ma tante. Le jour et le produit. La rue et le numéro.

### Étape 4 — Un cube : la feuille avec des onglets

Alors Mateo pose la question que sa tante évitait.

— Et la succursale Nord ?

Silence. Doña Lupe clique sur l'autre onglet de la même feuille, lentement, comme quelqu'un qui ouvre une enveloppe contenant de mauvaises nouvelles. Même structure — 7 jours, 3 produits —, d'autres chiffres. Sa feuille comporte désormais **deux onglets, un par succursale**, et chaque onglet est un tableau.

```
        Pestaña «Centro»                Pestaña «Norte»
      Cuad.  Plumas  Moch.            Cuad.  Plumas  Moch.
Lun   1200    800    500        Lun    300    200    100
Mar   1500    900    700        Mar    400    250    150
Mié   1000    700    500        Mié    350    200    100
Jue   1300    900    600        Jue    300    150    100
Vie   2000   1200    700        Vie    500    300    200
Sáb   1200    800    400        Sáb    300    200    100
Dom    800    500    300        Dom    200    100    100
```

Mateo regarde les deux onglets côte à côte. Quelque chose lui trotte dans la tête. Mais il doit d'abord tout intégrer dans Python.

En Python, la feuille complète est une liste d'onglets :

```python
centro = por_dia_y_producto
norte = [
    [300, 200, 100],
    [400, 250, 150],
    [350, 200, 100],
    [300, 150, 100],
    [500, 300, 200],
    [300, 200, 100],
    [200, 100, 100],
]
papeleria = [centro, norte]
```

Pour désigner un nombre, tu as besoin de **trois informations : la succursale, le jour et le produit**. `papeleria[1][4][2]` correspond à Nord (onglet 1), vendredi (ligne 4), sacs à dos (colonne 2) : 200. Trois directions, **trois dimensions**. Il n'a plus de nom particulier : c'est un **tenseur à trois dimensions**.

Voici l'image qu'il convient de garder en tête durant toute la série : **un tenseur à trois dimensions est un classeur avec des onglets, où chaque onglet possède les mêmes colonnes et le même nombre de lignes.**

| Ce que tu as | Pour désigner un nombre, tu as besoin de | Dimensions | Nom |
|---|---|---|---|
| 18 500 $ | rien : il n'y en a qu'un | 0 | scalaire |
| Ventes de 7 jours | le jour | 1 | vecteur |
| Ventes par jour et par produit | le jour et le produit | 2 | matrice |
| La même chose, par succursale | la succursale, le jour et le produit | 3 | tenseur à 3 dimensions |

**La règle qui résume tout :** le nombre de dimensions est le nombre d'informations dont tu as besoin pour désigner un seul nombre. Chacune de ces directions s'appelle aussi un **axe** ; tu rencontreras les deux termes.

— Et c'est tout ? Un cube ? — Doña Lupe le regarde par-dessus ses lunettes. — Je pensais que la banque utilisait des trucs de la NASA.

— Elle utilise des cubes, ma tante. Simplement, ils sont très grands.

### Étape 5 — Compter les dimensions avec du code

Compter les crochets à l'œil nu fonctionne pour des petits exemples. Avec ce que la banque a entre les mains, non. Mateo le sait : s'il doit se battre contre une machine, la sienne doit compter toute seule. D'abord, le Python que tu vas utiliser à partir de maintenant, au cas où tes souvenirs seraient un peu flous :

- `type(x) == list` demande si `x` est une liste.
- `x[0]` prend le premier élément ; `len(x)` compte combien il y en a.
- `lista.append(valor)` ajoute une valeur à la fin d'une liste ; `tuple(lista)` la convertit en tuple.

Crée un fichier `dimensiones.py` contenant les listes des étapes 1 à 4 et, au-dessus d'elles, cette fonction :

```python
def contar_dimensiones(x):
    dimensiones = 0
    while type(x) == list:
        dimensiones = dimensiones + 1
        x = x[0]
    return dimensiones
```

Comment elle fonctionne : pense aux poupées russes. Tant que ce qu'elle tient en main est une liste, elle ouvre la boîte — prend le premier élément, `x[0]` — et en compte un de plus. Dès que ce qui en sort est un nombre, elle s'arrête. Une liste de listes de listes s'ouvre trois fois : trois dimensions. Un nombre ne s'ouvre aucune fois : zéro.

Ajoute à la fin du fichier :

```python
if __name__ == "__main__":
    print(contar_dimensiones(total))
    print(contar_dimensiones(por_dia))
    print(contar_dimensiones(por_dia_y_producto))
    print(contar_dimensiones(papeleria))
```

Exécute-le avec `python3 dimensiones.py` :

```
0
1
2
3
```

La ligne `if __name__ == "__main__":` est une particularité de Python : elle fait en sorte que le code situé en dessous ne s'exécute que lorsque tu lances ce fichier directement. Pour l'instant, recopie-la telle quelle.

Quatre chiffres sur l'écran. Doña Lupe jette un coup d'œil.

— Ça y est ? C'est ça qui sauve Nord ?

— Non, ma tante. Ça me dit simplement que je comprends ce que j'ai devant moi. Ce qui est déjà beaucoup.

### Étape 6 — La forme : combien d'éléments dans chaque direction

Savoir que la papeterie a trois dimensions ne suffit pas. Un détective ne note pas « il y a des suspects » : il note combien. Mateo veut aussi savoir **combien d'éléments se trouvent dans chacune** : 2 succursales, 7 jours, 3 produits. C'est ce qu'on appelle la **forme** (*shape*) du tenseur, et cela s'écrit sous forme de tuple : `(2, 7, 3)`.

La forme t'en apprend beaucoup d'un seul coup d'œil :

- Combien il y a de dimensions : combien de nombres comporte le tuple (3).
- La taille de chacune : 2, 7 et 3.
- Combien de nombres il y a au total : on les multiplie, 2 × 7 × 3 = 42.

Quarante-deux nombres. Toute une semaine de deux succursales tient en quarante-deux nombres.

La fonction est presque identique à la précédente. Au lieu de compter, elle note la longueur de chaque boîte qu'elle ouvre :

```python
def forma(x):
    tamanos = []
    while type(x) == list:
        tamanos.append(len(x))
        x = x[0]
    return tuple(tamanos)
```

Remplace le bloc final de ton fichier par celui-ci :

```python
if __name__ == "__main__":
    print("total", contar_dimensiones(total), forma(total))
    print("por_dia", contar_dimensiones(por_dia), forma(por_dia))
    print("por_dia_y_producto", contar_dimensiones(por_dia_y_producto), forma(por_dia_y_producto))
    print("papeleria", contar_dimensiones(papeleria), forma(papeleria))

    assert forma(papeleria) == (2, 7, 3), "La papelería debería ser 2 sucursales × 7 días × 3 productos"
    print("Todo cuadra")
```

Sortie :

```
total 0 ()
por_dia 1 (7,)
por_dia_y_producto 2 (7, 3)
papeleria 3 (2, 7, 3)
Todo cuadra
```

Deux nouveautés dans cette sortie :

- `(7,)` comporte une virgule parce que c'est ainsi que Python écrit un tuple d'un seul élément. Sans la virgule, `(7)` ne serait que le nombre 7.
- `()` est un tuple vide : le scalaire n'a aucune direction.

Et une nouvelle ligne dans le code : `assert`. Cela signifie **« ceci doit être vrai »**. C'est comme le vigile à l'entrée : si c'est vrai, rien ne se passe et le programme continue. Sinon, le programme s'arrête et t'indique pourquoi. Fais le test : remplace `(2, 7, 3)` par `(7, 2, 3)` et relance-le :

```
Traceback (most recent call last):
  ...
AssertionError: La papelería debería ser 2 sucursales × 7 días × 3 productos
```

Remets-le à `(2, 7, 3)`. C'est ainsi que tu vas travailler durant toute la série : chaque exercice comporte son propre `assert`, et le programme te signale tout seul si tu as commis une erreur, sans attendre que quelqu'un vienne te corriger.

— « Todo cuadra » — lit doña Lupe à voix haute. — Si seulement la banque pensait la même chose.

### Étape 7 — Quand une liste de listes n'est PAS un tenseur

Il est presque vingt-deux heures. Mateo bâille et une question embarrassante lui traverse l'esprit : et si un jour quelqu'un envoie un onglet incomplet ? Sa fonction s'en rendrait-elle compte ?

Il y a une condition non négociable : **toutes les boîtes d'un même niveau doivent mesurer la même chose.** Tous les onglets avec 7 lignes, toutes les lignes avec 3 colonnes. Un tenseur est rectangulaire.

Regarde cette liste :

```python
chueca = [[1, 2, 3], [4, 5]]
```

Python l'accepte sans problème : c'est une liste de deux listes. Mais la première ligne contient 3 éléments et la seconde, 2. Ce n'est pas un tableau, c'est un tableau bancal. Et nos fonctions te mentent effrontément :

```
>>> contar_dimensiones(chueca)
2
>>> forma(chueca)
(2, 3)
```

`forma` répond `(2, 3)` parce qu'elle n'a mesuré que la première ligne. Comme un témoin qui n'a vu que le premier pâté de maisons et jure pour toute la rue. Si tu lui fais confiance et que tu demandes `chueca[1][2]`, ça plante :

```
IndexError: list index out of range
```

Voilà pourquoi il est nécessaire de vérifier. Pour un tableau (deux dimensions), cette fonction compare la longueur de chaque ligne à celle de la première. Avec plus de dimensions, la vérification est plus longue ; ne t'inquiète pas, lors de la prochaine session, PyTorch s'en charge intégralement pour toi :

```python
def es_rectangular(tabla):
    largo = len(tabla[0])
    for fila in tabla:
        if len(fila) != largo:
            return False
    return True
```

```
>>> es_rectangular(por_dia_y_producto)
True
>>> es_rectangular(chueca)
False
```

Les bibliothèques d'IA, comme PyTorch, font cette vérification pour toi : lorsqu'on essaie de convertir une liste bancale en tenseur ordinaire, elles s'y refusent. Tu le verras lors de la prochaine session. (Il existe des tenseurs particuliers pour les données disparates, mais nous ne les utilisons pas dans cette série.)

Mateo reste à fixer ce `False`. Une donnée bancale ne prévient pas. Elle se dissimule, jusqu'à ce que tout plante.

### Au-delà de trois : l'adresse postale

— Et si je te demandais quatre semaines, ma tante, au lieu d'une seule ?

— Je te les donne. Mais tu m'aides à ranger les bristols.

Avec les ventes de 4 semaines, Mateo aurait besoin de quatre informations pour désigner un nombre : la semaine, la succursale, le jour et le produit. Quatre dimensions, forme `(4, 2, 7, 3)`.

Quatre dimensions ne se dessinent pas, mais se comprennent exactement comme une adresse postale. Pour arriver jusqu'à un appartement, tu indiques la région, la ville, l'immeuble, l'étage et le numéro de porte : cinq informations, cinq dimensions. Personne ne s'en effraie. Chaque donnée choisit une boîte à l'intérieur de la précédente, exactement comme chaque indice de `papeleria[1][4][2]`.

Quelques tenseurs réels que tu vas rencontrer :

| Ce que c'est | Dimensions | Exemple de forme |
|---|---|---|
| Une petite photo en couleur | hauteur × largeur × couleur | `(480, 640, 3)` |
| Un lot de 32 photos | photo × hauteur × largeur × couleur | `(32, 480, 640, 3)` |
| 10 secondes de vidéo à 30 images par seconde | image × hauteur × largeur × couleur | `(300, 720, 1280, 3)` |
| Bilan quotidien de 1 000 clients pendant 30 jours, avec 5 données par client et par jour | client × jour × donnée | `(1000, 30, 5)` |

Ici, nous écrivons les photos sous la forme hauteur × largeur × couleur parce que c'est ainsi que nos listes sont structurées. Cet ordre n'est pas universel : dans de nombreux programmes PyTorch, tu verras couleur × hauteur × largeur. L'important est de savoir ce que représente chaque axe.

Cette vidéo, enregistrée sans compression, représente 829 440 000 nombres (multiplie les dimensions de la forme). C'est pourquoi les vidéos sont enregistrées sous forme compressée (un fichier MP4 contient de la vidéo compressée), et pourquoi travailler avec de la vidéo non compressée exige énormément de mémoire.

Mateo regarde la dernière ligne du tableau : 1 000 clients, 30 jours, 5 données. C'est à peu près à cela que doit ressembler ce que la banque détient sur le terminal de Nord. Et lui n'en est encore qu'à 42 nombres.

---

## D'où vient le mot « tenseur » ?

Passé minuit, Mateo ne dort toujours pas dans sa chambre. Il ne peut pas s'arrêter de lire. Il s'avère que l'outil qu'il va utiliser contre la banque a une histoire peuplée de gens tout aussi bloqués que lui.

Personne n'a inventé les tenseurs tout seul.

**1846.** L'Irlandais William Rowan Hamilton emploie pour la première fois le mot *tenseur*, du latin *tendere*, « étirer ». Mais pour lui, cela signifiait autre chose : la norme d'un quaternion, un type de nombre qu'il avait lui-même inventé. Le mot est né, mais il ne sait pas encore ce qu'il va devenir.

**1898.** L'Allemand Woldemar Voigt lui donne son sens moderne. Il étudiait les cristaux, qui s'étirent et se déforment différemment selon la direction dans laquelle on tire dessus, et il avait besoin d'un seul objet capable d'englober toutes ces directions à la fois. Une boîte pour plusieurs directions. Cela te parle, n'est-ce pas ?

**1900.** Les Italiens Gregorio Ricci-Curbastro et Tullio Levi-Civita publient, méthodiquement organisées, les mathématiques que nous appelons aujourd'hui calcul tensoriel (eux parlaient de « calcul différentiel absolu »). Et voici un détail digne d'un roman à suspense : dans tout leur article, le mot « tenseur » n'apparaît pas une seule fois.

**1912.** Albert Einstein revient à Zurich, bloqué sur sa théorie de la gravitation. Les calculs ne tombent pas juste. Il va trouver son ami d'université, le mathématicien Marcel Grossmann. D'après ce que racontait des années plus tard un de leurs camarades, Einstein lui a dit :

— Grossmann, aide-moi ou je deviens fou.

Grossmann lui fait découvrir les travaux des Italiens. Ceux de Ricci-Curbastro et Levi-Civita. Ceux qui ne contenaient pas le mot « tenseur ».

**1915.** Mars. Une lettre arrive pour Einstein. Elle est signée Levi-Civita. Il lui signale qu'une démonstration de la version de sa théorie qu'il soutenait alors est erronée.

Imagine la scène : Einstein, corrigé par courrier.

Ils échangent des lettres jusqu'en mai. Einstein défend sa preuve avec obstination. Les courriers vont et viennent. Finalement, Einstein finit par admettre qu'elle était incomplète. Dans une lettre du 2 avril, il lui écrivit : « Je n'ai jamais eu de correspondance aussi intéressante ». Quelques mois plus tard, il change d'approche, et le 25 novembre, il présente les équations définitives de la relativité générale, écrites avec des tenseurs.

Il y a là une leçon qui dépasse les mathématiques : corriger quelqu'un, ce n'est pas chercher la dispute. Le « votre démonstration est erronée » de Levi-Civita n'a pas tourné à l'affrontement : il a montré à Einstein une faille bien réelle, des mois avant les équations définitives, et a donné naissance à une amitié.

*Sources : Jeff Miller, « Earliest Known Uses of Some of the Words of Mathematics », entrée TENSOR (Hamilton 1846, Voigt 1898, Ricci-Curbastro et Levi-Civita), sur MacTutor : [mathshistory.st-andrews.ac.uk/Miller/mathword/t/](https://mathshistory.st-andrews.ac.uk/Miller/mathword/t/). L'article de Ricci-Curbastro et Levi-Civita s'intitule « Méthodes de calcul différentiel absolu et leurs applications », Mathematische Annalen 54, p. 125–201 ; le volume porte la date de 1900 et certains catalogues le citent sous l'année 1901. Les lettres de 1915 : Galina Weinstein, « Einstein the Stubborn: Correspondence between Einstein and Levi-Civita », [arxiv.org/abs/1202.4305](https://arxiv.org/abs/1202.4305).*

Mateo éteint sa lampe en pensant à cela. Si l'on disait à Einstein « c'est faux », on le lui dira à lui aussi. Mieux vaut qu'un `assert` le lui dise aujourd'hui plutôt que la banque le 23 décembre.

### Une vérité pour ne pas être pris au dépourvu

Si un jour un physicien te dit qu'un tenseur « n'est pas un tableau de nombres », il a raison, de son point de vue. En physique et en mathématiques, un tenseur est une boîte de nombres qui obéit en outre à des règles précises lorsqu'on fait tourner les axes ou qu'on change d'unités. En programmation et en IA, un *tenseur* n'est que la boîte ordonnée. Cette boîte peut contenir les nombres d'un tenseur de physique, mais elle n'apporte pas d'elle-même ses règles ni sa signification. Les développeurs ont emprunté ce nom à la physique parce que cela sonne mieux que « tableau multidimensionnel », ce qu'il est pourtant.

Et un piège de vocabulaire : en anglais, le nombre de dimensions est parfois appelé *rank*. Ne le confonds pas avec le « rang » d'une matrice que tu as peut-être vu en algèbre linéaire ; ce sont deux choses distinctes.

---

👉 **Si tu viens de la gestion :** la partie chiffrée de ton classeur avec onglets (sans en-têtes ni annotations) constitue déjà un tenseur à trois dimensions, pourvu que tous les onglets possèdent les mêmes colonnes et les mêmes lignes. Doña Lupe l'avait sur son ordinateur portable depuis des années sans le savoir. Quand en réunion quelqu'un dira « les données entrent dans le modèle sous forme de tenseurs », pense à cela : des tableaux empilés, ayant tous la même forme. Et la règle de l'étape 7 s'applique aussi à tes rapports : si chaque succursale t'envoie son onglet avec des colonnes différentes, personne ne pourra les additionner sans les corriger au préalable.

---

## Exercices de la session 1

Les solutions se trouvent à la fin de cette livraison. Essaie d'abord par toi-même ; le `assert` t'indique si tu as vu juste.

**Exercice 1 — Prédis avant d'exécuter.** Pour chaque donnée, écris sur une feuille combien de dimensions elle possède, sa forme et ce que représente chaque dimension. Vérifie ensuite avec tes fonctions.

```python
temperatura = 31
calificaciones = [[9, 8, 10], [7, 9, 8], [10, 10, 9], [6, 8, 7]]
foto = [[[255, 0, 0], [0, 255, 0]], [[0, 0, 255], [255, 255, 255]]]
lote = [foto, foto]
```

Dès que tu as tes réponses sur papier, écris-les dans ces `assert` (remplace chaque `?`) et exécute :

```python
assert (contar_dimensiones(temperatura), forma(temperatura)) == (?, ?)
assert (contar_dimensiones(calificaciones), forma(calificaciones)) == (?, ?)
assert (contar_dimensiones(foto), forma(foto)) == (?, ?)
assert (contar_dimensiones(lote), forma(lote)) == (?, ?)
print("Ejercicio 1 superado")
```

Indice : `calificaciones` correspond à 4 élèves avec 3 matières chacun. `foto` est une image de 2 × 2 points, et chaque point apporte trois nombres : rouge, vert et bleu.

**Exercice 2 — Désigne le nombre.** Avec `papeleria` de l'étape 4, écris l'indice qui correspond à chaque donnée et vérifie-le avec un `assert` :

a) Les ventes du Centre le mercredi pour les stylos.
b) Les ventes de Nord le vendredi pour les sacs à dos.
c) Les ventes de Nord le dimanche pour les cahiers.

Exemple de résultat attendu : `assert papeleria[0][0][0] == 1200` (Centre, lundi, cahiers).

**Exercice 3 — Complète la fonction.** Cette version de `forma` contient deux trous. Remplis-les sans regarder l'étape 6 et vérifie avec les `assert` :

```python
def forma2(x):
    tamanos = []
    while type(x) == list:
        tamanos.append(____)
        x = ____
    return tuple(tamanos)

assert forma2(por_dia) == (7,)
assert forma2(papeleria) == (2, 7, 3)
assert forma2(31) == ()
print("Ejercicio 3 superado")
```

**Exercice 4 — Tes propres données.** Écris sous forme de listes imbriquées les notes de 3 élèves, dans 4 matières, sur 3 examens partiels. Invente les notes. Choisis toi-même l'ordre des dimensions et indique-le dans un commentaire. Ensuite :

a) Écris le `assert` de sa forme.
b) Écris l'indice correspondant au deuxième examen partiel de la troisième matière du premier élève, et vérifie-le par un `assert` par rapport à la note que tu as inventée à cet endroit.

Cet exercice n'a pas une seule réponse correcte : cela dépend de l'ordre que tu choisis. Ce qui doit obligatoirement être respecté, c'est que le `assert` passe et que ton commentaire indique ce que représente chaque dimension.

**Exercice 5 — Trouve le tableau bancal.** L'un de ces tableaux ne peut pas être un tenseur. Repère-le à l'œil nu et vérifie-le avec `es_rectangular` :

```python
a = [[1, 2], [3, 4], [5, 6]]
b = [[1, 2, 3], [4, 5, 6], [7, 8]]
c = [[0], [0], [0]]
```

Écris tes réponses dans ces `assert` (remplace chaque `?` par `True` ou `False`) :

```python
assert es_rectangular(a) == ?
assert es_rectangular(b) == ?
assert es_rectangular(c) == ?
print("Ejercicio 5 superado")
```

Que t'indiquerait `forma(b)` ? Pourquoi n'est-ce pas fiable ?

---

## Erreurs courantes (et comment t'en rendre compte)

| Erreur | Ce que l'on observe | Comment t'en rendre compte | Ce qu'il faut faire |
|---|---|---|---|
| Compter les indices à partir de 1 | Tu demandes le vendredi avec `por_dia[5]` et cela donne le samedi | Le nombre ne correspond pas au tableau | Les indices commencent à 0 : lundi est 0, vendredi est 4 |
| Inverser l'ordre des indices | `papeleria[4][1][2]` au lieu de `papeleria[1][4][2]` | `IndexError`, car il n'y a que 2 succursales | L'ordre des indices suit l'ordre des listes : de l'extérieur vers l'intérieur |
| Faire confiance à `forma` avec une liste bancale | `forma` indique `(2, 3)` puis un indice plante | `es_rectangular` renvoie `False` | Nettoie les données avant de les utiliser ; ne complète jamais avec des nombres inventés sans le préciser |
| Passer une liste vide à `forma` ou à `contar_dimensiones` | `IndexError: list index out of range` à l'intérieur de la fonction | Plante avec `[]` | Une boîte vide n'a pas de premier élément à ouvrir ; vérifie que tes données contiennent quelque chose |
| Oublier la virgule dans `(7,)` | `assert forma(por_dia) == (7)` échoue alors que la forme est correcte | Le `assert` plante sur quelque chose qui semble pourtant correct | Un tuple d'un seul élément prend une virgule : `(7,)` |

---

## Révision pour la prochaine session

La session 2 commence par ces trois questions. Réponds-y sans regarder la leçon :

1. Comment sais-tu combien de dimensions possède un ensemble de données, sans compter les crochets ?
2. Que t'indique la forme `(2, 7, 3)`, et combien de nombres y a-t-il au total ?
3. Quelle condition une liste de listes doit-elle remplir pour être un tenseur ?

🔁 **La semaine prochaine :** Mateo délaisse les listes pour prendre en main un véritable outil. Tu installes PyTorch et tu convertis la papeterie de doña Lupe en ton premier vrai tenseur. Tu verras que PyTorch intègre déjà la forme nativement (il l'appelle `shape`) et qu'il refuse d'accepter des listes bancales.

Cette nuit-là, avant de refermer son ordinateur portable, Mateo rouvrit les deux onglets. Centre, Nord. Centre, Nord. Les mêmes colonnes, les mêmes sept jours. Et pourtant, dans l'onglet de Nord, quelque chose clochait. Mais pour le voir, il avait besoin d'un outil digne de ce nom.

---

## Solutions (ne les regarde pas avant d'avoir essayé)

**Exercice 1.**

```
temperatura      0 ()             un solo número
calificaciones   2 (4, 3)         alumno × materia
foto             3 (2, 2, 3)      renglón de puntos × punto × color
lote             4 (2, 2, 2, 3)   foto × renglón × punto × color
```

Les `assert` se présentent ainsi : `(0, ())`, `(2, (4, 3))`, `(3, (2, 2, 3))` et `(4, (2, 2, 2, 3))`.

Le `lote` (lot) ajoute une dimension au début : quelle photo. C'est ainsi que fonctionnent presque toujours les modèles de traitement d'images : ils reçoivent des lots (*batches*), et même une photo isolée est généralement transmise comme un lot d'une seule image.

**Exercice 2.**

```python
assert papeleria[0][2][1] == 700   # a) Centro, miércoles, plumas
assert papeleria[1][4][2] == 200   # b) Norte, viernes, mochilas
assert papeleria[1][6][0] == 200   # c) Norte, domingo, cuadernos
```

**Exercice 3.** Les trous sont `len(x)` et `x[0]` : note la longueur de la boîte, puis ouvre-la.

**Exercice 4.** Une réponse possible, avec l'ordre élève × matière × partiel :

```python
# alumno × materia × parcial
parciales = [
    [[8, 9, 9], [7, 8, 9], [9, 9, 10], [6, 7, 8]],
    [[10, 9, 10], [9, 9, 9], [8, 8, 9], [9, 10, 10]],
    [[7, 7, 8], [8, 8, 8], [6, 7, 7], [8, 9, 9]],
]
assert forma(parciales) == (3, 4, 3)
assert parciales[0][2][1] == 9   # primer alumno, tercera materia, segundo parcial
```

Si tu as choisi matière × élève × partiel, ta forme est `(4, 3, 3)` et ton indice est `parciales[2][0][1]`. Les deux sont acceptables tant que le commentaire le précise. Choisir l'ordre des dimensions est une décision de conception, et tu verras dans la session 3 pourquoi cela compte.

**Exercice 5.** Les `assert` sont `True`, `False` et `True`. Le tableau bancal est `b` : sa dernière ligne contient 2 éléments tandis que les autres en ont 3. `es_rectangular(b)` renvoie `False`. `forma(b)` indiquerait `(3, 3)` car il ne mesure que la première ligne, et n'est pas fiable pour cette raison : il ne vérifie pas les autres.

---

## La suite

Dans la session 2, tu laisses de côté les listes pour utiliser un outil conçu pour cela. Si tu veux prendre de l'avance, vérifie que tu disposes de Python 3.11 ou d'une version plus récente avec `python3 --version`.

---

*J'écris à partir de ce que nous voyons tous les jours chez Hábil, où nous concevons des systèmes pour le secteur bancaire, les assurances et les entreprises au Mexique. Si, dans ton travail ou dans l'entreprise de ta famille, tu rencontres un problème de ce genre, discutons-en : [habil.mx](https://www.habil.mx/fr/)*
