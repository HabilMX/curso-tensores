# Les tenseurs à partir de zéro (3) : les axes ont un nom

*Comment extraire un morceau d'un tenseur selon n'importe lequel de ses axes — un nombre, un jour, un produit, le canal rouge d'une photo — et ce qui arrive à la forme quand tu le fais.*

*Les tenseurs à partir de zéro · leçon A, session 3 sur 8*

*Scénario : les personnages et l'histoire sont inventés. Les données techniques, historiques et les chiffres réels indiquent leur source.*

**Lundi 16 novembre, 19 h 40. Papeterie Centre.**

Le rideau métallique est baissé à mi-hauteur. À l'intérieur, ça sent le papier neuf et le café réchauffé. Doña Lupe compte des billets sur le comptoir. Mateo a son ordinateur portable ouvert à côté de la caisse enregistreuse, entre un pot de crayons et une pile de bristols.

Il reste 37 jours avant le 23 décembre.

La semaine dernière, ils ont réparé le fichier de Nord, celui à la ligne tronquée. Maintenant, PyTorch l'accepte : un tenseur de forme `(2, 7, 3)`. Deux succursales, sept jours, trois produits. Quarante-deux nombres.

« Alors, ça y est ? » demande doña Lupe sans lever les yeux. « Tu sais qui me vole ? »

« Pas encore, ma tante. J'ai les nombres. Il me reste à savoir les regarder. »

« Eh bien, regarde-les. C'est pour ça que tu as des yeux. »

Mateo rit, mais elle a raison. Quarante-deux nombres entassés ne disent rien. C'est comme un entrepôt où tout est empilé en vrac : la réponse est là, mais on ne la trouve pas. Ce qu'il lui faut, c'est extraire des **morceaux**. Un jour. Un produit. Une succursale. Les mettre côte à côte et voir ce qui ne colle pas.

Il écrit une ligne. Il obtient le vendredi des deux succursales.

Il reste immobile.

« Ma tante. Quel est ton meilleur jour de la semaine ? »

« Le vendredi, lequel d'autre ? Ils sortent de l'école, ils viennent avec les parents, ils achètent de tout. »

« À Centre, oui. Le vendredi, tu vends pour 2 000 en cahiers. Aucun autre jour n'en approche. »

« Et à Nord ? »

Mateo tourne l'écran. `500, 300, 200`.

Doña Lupe met ses lunettes. Les enlève à nouveau.

« Ça, Nord le vend un mardi quelconque. »

Presque. Le mardi de Nord, c'est 400, 250 et 150. Le vendredi, le jour fort, ne le dépasse que d'un cheveu.

Et il y a pire dans la dernière colonne. Les sacs à dos. Ce qu'il y a de plus cher dans la boutique. À Nord, presque tous les jours : 100. Comme si personne n'y touchait.

Mateo sent un creux à l'estomac. Il ne sait pas encore ce que ça signifie. Mais il sait qu'il vient de trouver la première vraie piste.

---

Dans la session précédente, tu as installé PyTorch et converti la papeterie de doña Lupe en ton premier véritable tenseur, de forme `(2, 7, 3)`. Aujourd'hui, tu apprends ce que Mateo a fait ce soir-là : en extraire des morceaux. Pas seulement un nombre, mais un jour complet, un produit complet ou une semaine découpée, selon l'axe que tu choisis.

## Révision pour commencer

Réponds sans regarder la session précédente. Les réponses se trouvent à la fin, dans les solutions.

1. Que t'indique `t.shape` et comment le vérifies-tu avec un `assert` ?
2. Pourquoi `torch.tensor([[1, 2, 3], [4, 5]])` échoue-t-il ?
3. Quel `dtype` sort de `torch.tensor([1, 2, 3])` et lequel de `torch.tensor([1.0, 2.0])` ?

> **Le code de cette session** se trouve dans le dépôt public du cours, dans le dossier `sesion-03` : [github.com/HabilMX/curso-tensores](https://github.com/HabilMX/curso-tensores/tree/main/sesion-03). Écris-le toi-même d'abord, car c'est ainsi qu'on apprend ; le dépôt sert à comparer quand quelque chose ne donne pas le même résultat chez toi. Chaque programme est accompagné de la sortie qu'il doit te donner.

## Objectifs de cette session

À la fin, tu seras capable de faire ces quatre choses. À côté de chacune se trouve l'exercice qui le démontre.

| # | À la fin, tu peux… | Démontré par |
|---|---|---|
| 1 | **Nommer** chaque axe d'un tenseur et l'**écrire** dans un commentaire au-dessus du code | Exercices 10 et 13 |
| 2 | **Extraire** un seul nombre avec un indice par axe, `papeleria[1, 4, 2]`, et le **convertir** en nombre Python avec `.item()` | Exercice 10 |
| 3 | **Extraire** un morceau selon n'importe quel axe avec `:` et avec une plage comme `1:5`, et **prédire** sa forme avant de l'exécuter | Exercices 11 et 12 |
| 4 | **Décider** de l'ordre des axes d'une photo et en **extraire** une seule couleur | Exercice 13 |

**Ce dont tu as besoin :** l'environnement de la session 2 avec PyTorch installé (active-le avant de commencer) et un éditeur de texte. Aujourd'hui, tu n'installes rien de nouveau.

---

## Pourquoi c'est important

Mateo n'a pas trouvé la piste en additionnant tout. Il l'a trouvée en découpant : un jour de deux succursales, une colonne d'une succursale. C'est ainsi que fonctionne presque n'importe quel système qui traite des données.

Un programme qui examine la photo d'une voiture accidentée ne l'utilise presque jamais en entier : il découpe la zone du pare-chocs, ou ne garde qu'une seule couleur. Une banque qui enregistre les achats de mille clients sur 30 jours, avec la forme `(1000, 30, 5)`, veut parfois un client complet et parfois seulement la journée d'hier de tous. C'est le même tenseur. Ce qui change, c'est l'axe selon lequel tu le découpes.

Et voici le piège. Beaucoup des erreurs que tu verras avec des modèles ne sont pas mathématiques : ce sont des erreurs de forme. Tu as demandé un morceau, tu croyais qu'il avait une forme et il en avait une autre. Rien ne plante. Tu obtiens simplement un nombre faux, et tu le crois. Aujourd'hui, tu apprends à prédire cette forme avant d'exécuter le code, pour que cela ne t'arrive pas.

## Étape 1 — Donne un nom à chaque axe

Avant de découper quoi que ce soit, Mateo fait une chose qui semble superflue et ne l'est pas : il écrit au-dessus du code ce qu'est chaque axe. Parce que PyTorch ne sait pas ce qu'est « Nord ». Il ne sait pas ce qu'est « vendredi ».

Pour PyTorch, `papeleria` est une boîte de `2 × 7 × 3` nombres, rien de plus. Celui qui sait que le premier axe correspond aux succursales, c'est toi. D'où la première règle du jour : **dis-le à voix haute et écris-le dans un commentaire**, comme l'étiquette d'un tiroir de cuisine.

Crée un fichier `ejes.py` et commence ainsi :

```python
import torch

# Ejes de la papelería:
#   eje 0 = sucursal  (0 Centro, 1 Norte)
#   eje 1 = día       (0 lunes ... 6 domingo)
#   eje 2 = producto  (0 cuadernos, 1 plumas, 2 mochilas)
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
print(papeleria.shape)
```

Exécute-le avec `python3 ejes.py` :

```
torch.Size([2, 7, 3])
```

Lis-la avec les noms : **2 succursales, 7 jours, 3 produits.** Les axes sont numérotés à partir de 0, comme les indices d'une liste : axe 0, axe 1, axe 2. Quand tu lis « selon l'axe 1 », traduis par « selon les jours ».

« Et ça sert à quoi ? » dit doña Lupe en se penchant. « Tu sais déjà lequel est lequel. »

« Aujourd'hui, oui. Dans trois semaines, à deux heures du matin, je ne m'en souviendrai plus. »

« Comme ton oncle avec les clés. C'est pour ça que je leur ai mis une étiquette. »

## Étape 2 — Un seul nombre : `papeleria[1, 4, 2]`

La première chose que Mateo veut voir, c'est un seul nombre : ce que Nord a vendu le vendredi en sacs à dos. S'il y a quelque chose de bizarre là, il veut pouvoir le toucher du doigt.

Dans la session 1, tu désignais un nombre ainsi : `papeleria[1][4][2]`. Avec un tenseur, ça fonctionne toujours, mais il existe une meilleure manière : **tous les indices dans les mêmes crochets, séparés par des virgules.**

À partir d'ici, ajoute chaque bloc à la fin de `ejes.py` et réexécute-le ; pour chaque sortie, je ne te montre que les nouvelles lignes. Ajoute :

```python
print(papeleria[1][4][2])
print(papeleria[1, 4, 2])
```

```
tensor(200)
tensor(200)
```

Les deux donnent la même chose : Nord, vendredi, sacs à dos, 200 $. Le meilleur jour de la semaine. Mateo note le nombre sur un post-it et le colle au bord de l'écran.

La différence, c'est la façon dont elles travaillent en interne :

- `papeleria[1][4][2]`, ce sont **trois étapes** : extraire l'onglet de Nord, puis le vendredi, puis, dans le vendredi, les sacs à dos. Une boîte dans une boîte.
- `papeleria[1, 4, 2]`, c'est **une seule étape** : tu lui donnes l'adresse complète et il y va directement.

À l'étape 7, tu verras un cas où la version à crochets séparés te trompe. À partir d'aujourd'hui, utilise celle à virgules.

### Ce qui en sort n'est pas un nombre : c'est un tenseur à zéro dimension

La sortie n'affiche pas `200`, mais `tensor(200)` : le nombre est enveloppé. Ajoute ceci :

```python
x = papeleria[1, 4, 2]
print(x.shape, x.ndim)
print(type(x))
print(x.item())
print(type(x.item()))
```

```
torch.Size([]) 0
<class 'torch.Tensor'>
200
<class 'int'>
```

Forme vide, zéro dimension : c'est un **scalaire**, comme le total de 18 500 $ de la session 1, mais en version tenseur. Un bonbon encore dans son emballage. `.item()` lui retire l'emballage et te donne un `int` Python ordinaire, pour l'afficher, l'enregistrer dans une liste ou l'envoyer dans un rapport.

Et l'autocorrection habituelle : d'abord la forme, ensuite la valeur.

```python
assert papeleria[1, 4, 2].shape == (), "Un solo número no tiene ejes"
assert papeleria[1, 4, 2].item() == 200, "Norte, viernes, mochilas: 200"
print("Paso 2 correcto")
```

```
Paso 2 correcto
```

`torch.Size([])` se compare correctement au tuple vide `()` : tes formes s'écrivent toujours sous forme de tuples.

Un nombre isolé ne prouve rien. 200 pesos de sacs à dos, ça peut être juste un mauvais vendredi. Mateo a besoin de voir toute la semaine de Nord, d'un seul coup.

## Étape 3 — Un seul indice : l'axe disparaît

Que se passe-t-il si tu donnes moins d'indices que d'axes ? Essaie avec un seul :

```python
print(papeleria[1])
print(papeleria[1].shape)
```

```
tensor([[300, 200, 100],
        [400, 250, 150],
        [350, 200, 100],
        [300, 150, 100],
        [500, 300, 200],
        [300, 200, 100],
        [200, 100, 100]])
torch.Size([7, 3])
```

L'onglet complet de Nord. Mateo parcourt du regard la dernière colonne : 100, 150, 100, 100, 200, 100, 100. Plat comme une table.

Mais regarde la forme, qui est la vedette du jour. Elle est passée de `(2, 7, 3)` à `(7, 3)` : **l'axe des succursales a disparu.** Tu en as déjà choisi une ; devant l'onglet de Nord, il n'y a plus de « succursales », seulement des jours et des produits.

Avec deux indices, même chose :

```python
print(papeleria[1, 4])
print(papeleria[1, 4].shape)
```

```
tensor([500, 300, 200])
torch.Size([3])
```

Nord, vendredi, les trois produits. Deux axes sont partis ; il reste celui des produits. Première moitié de la règle du jour :

> **Chaque indice que tu donnes mange son axe.** Trois axes moins un indice : il en reste deux. Trois axes moins trois indices : il n'en reste aucun, et c'est un scalaire.

## Étape 4 — Les deux-points : « tous, s'il te plaît »

Voici maintenant la ligne qui lui a donné ce creux à l'estomac. Mateo ne veut pas Nord seule : il veut Nord **à côté de** Centre, le même jour, pour que la différence n'ait nulle part où se cacher.

Et si tu veux **le vendredi des deux succursales** ? Tu dois sauter le premier axe sans rien choisir. C'est à cela que servent les deux-points, `:`, qui dans un indice veulent dire « tous ».

```python
viernes = papeleria[:, 4]
print(viernes)
print(viernes.shape)
```

```
tensor([[2000, 1200,  700],
        [ 500,  300,  200]])
torch.Size([2, 3])
```

Lis-le avec les noms des axes : **toutes les succursales, le jour 4 (vendredi), et comme tu n'as rien dit du produit, tous les produits.** Ligne du haut, Centre ; ligne du bas, Nord.

La voilà. En haut, 2000. En bas, 500. Même jour, même sortie d'école. Une ligne de code et la différence saute aux yeux, sans onglets ni copier-coller.

Forme `(2, 3)` : l'axe des jours est parti parce que tu en as choisi un ; celui des succursales est resté parce que tu les as toutes demandées ; celui des produits, parce que tu n'y as pas touché. Comme commander des tacos « avec tout » : tu ne choisis pas oignon ou coriandre, tu prends ce qu'il y a.

### Vérifie-le par rapport à la version avec `for`

Mateo ne se fie pas à une seule ligne. S'il doit montrer ça à la banque, il veut l'obtenir par deux chemins. Avec ce que tu savais déjà, ce serait nombre par nombre :

```python
a_mano = []
for s in range(2):
    a_mano.append([papeleria[s][4][p].item() for p in range(3)])
print(a_mano)
print(viernes.tolist())
```

```
[[2000, 1200, 700], [500, 300, 200]]
[[2000, 1200, 700], [500, 300, 200]]
```

`.tolist()` est le cousin de `.item()` : il reconvertit un tenseur complet en listes Python, qui se comparent avec `==` comme toujours. Les deux donnent la même chose ; mets-le par écrit :

```python
assert viernes.shape == (2, 3), "Viernes: 2 sucursales × 3 productos"
assert viernes.tolist() == [[2000, 1200, 700], [500, 300, 200]]
print("Paso 4 correcto")
```

```
Paso 4 correcto
```

## Étape 5 — Une plage : `papeleria[0, 1:5]`

« Voyons », dit doña Lupe. « Montre-moi à quoi ressemble une semaine normale. Celle de Centre. Mais sans le lundi, qui est toujours mort, ni le week-end. »

Mateo a besoin d'une plage : du mardi au vendredi. Les deux-points servent aussi à demander une plage, comme avec les listes Python : `lista[1:5]` donne les éléments 1, 2, 3 et 4. **Le premier est inclus, le dernier est exclu.** Avec les tenseurs, selon n'importe quel axe.

Doña Lupe veut Centre du mardi au vendredi :

```python
tramo = papeleria[0, 1:5]
print(tramo)
print(tramo.shape)
```

```
tensor([[1500,  900,  700],
        [1000,  700,  500],
        [1300,  900,  600],
        [2000, 1200,  700]])
torch.Size([4, 3])
```

Centre, jours 1 à 4 (le 5, samedi, est exclu), tous les produits. Forme `(4, 3)` : l'axe des succursales est parti à cause de l'indice ; celui des jours **est resté**, mais il mesure 4 au lieu de 7. Une plage raccourcit l'axe, elle ne le supprime pas. Compare-le à la liste Python `centro` :

```python
print(centro[1:5])
assert tramo.shape == (4, 3), "De martes a viernes: 4 días × 3 productos"
assert tramo.tolist() == centro[1:5]
print("Paso 5 correcto")
```

```
[[1500, 900, 700], [1000, 700, 500], [1300, 900, 600], [2000, 1200, 700]]
Paso 5 correcto
```

Ce morceau, celui qu'on extrait avec les deux-points, s'appelle une **tranche**. C'est le mot que tu rencontreras en français ; en anglais, c'est *slice*.

## Étape 6 — Le dernier axe : `papeleria[:, :, 2]`

Doña Lupe pose les billets. Elle ne compte plus. Maintenant, c'est elle qui pose la question : **combien de sacs à dos chaque succursale a-t-elle vendus chaque jour ?** Les sacs à dos sont ce qui rapporte le plus. Si quelque chose cloche, elle veut le voir là.

Le produit est l'axe 2, le dernier ; pour y arriver, tu passes par les deux autres avec « tous » :

```python
mochilas = papeleria[:, :, 2]
print(mochilas)
print(mochilas.shape)
```

```
tensor([[500, 700, 500, 600, 700, 400, 300],
        [100, 150, 100, 100, 200, 100, 100]])
torch.Size([2, 7])
```

Forme `(2, 7)` : l'axe des produits est parti ; chaque ligne est une succursale et chaque colonne, un jour.

Doña Lupe s'approche tellement de l'écran que Mateo doit reculer. En haut, Centre : 500, 700, 500, 600… En bas, Nord : 100, 150, 100, 100…

« Mon petit, à Nord, il passe plus de monde qu'ici. C'est juste à côté de l'arrêt de bus. »

Mateo ne répond pas. Il n'a encore rien à répondre.

Dans le tableur, tu devrais copier la dernière colonne de chaque onglet et les coller côte à côte. Ici, il a suffi d'une ligne. Avec `for`, ce sont deux boucles :

```python
m = []
for s in range(2):
    fila = []
    for d in range(7):
        fila.append(papeleria[s, d, 2].item())
    m.append(fila)
print(m)

assert mochilas.shape == (2, 7), "Mochilas: 2 sucursales × 7 días"
assert mochilas.tolist() == m
print("Paso 6 correcto")
```

```
[[500, 700, 500, 600, 700, 400, 300], [100, 150, 100, 100, 200, 100, 100]]
Paso 6 correcto
```

Sept lignes contre une. C'est tout l'intérêt des tenseurs : quand tu es pressé — et il reste 37 jours à Mateo —, la question et la réponse tiennent sur la même ligne.

## Étape 7 — Pourquoi les virgules : le piège de `papeleria[:][4]`

Ce même soir, Mateo a failli se tromper. Avant de trouver la bonne ligne de l'étape 4, il a essayé celle qui lui venait naturellement. Si tu demandais le vendredi des deux succursales avec des crochets séparés, le réflexe naturel serait ceci :

```python
print(papeleria[:][4])
```

```
Traceback (most recent call last):
  ...
IndexError: index 4 is out of bounds for dimension 0 with size 2
```

Lis l'erreur mot à mot : « l'indice 4 est hors des limites de la dimension 0, qui mesure 2 ». La dimension 0 ? Mais tu voulais le jour !

Les crochets séparés travaillent par étapes, et chaque étape repart du premier axe **de ce qui reste** de l'étape précédente, pas de l'axe que tu avais en tête :

1. `papeleria[:]` signifie « toutes les succursales » : il te renvoie le tenseur complet, tel quel.
2. `[4]` retombe sur le premier axe : tu lui demandes la succursale 4, alors qu'il n'y en a que deux.

Avec les virgules, chaque position correspond à un axe fixe. Et cette fois, tu as eu de la chance : ça a planté. Si doña Lupe avait 7 succursales, `papeleria[:][4]` te donnerait la succursale 4 sans se plaindre, et tu croirais que c'est le vendredi. **L'erreur qui ne plante pas est la plus dangereuse.**

Mateo imagine la scène : lui devant monsieur Ortega, en train de montrer « les vendredis » qui sont en réalité une autre succursale. La banque ne lui donnerait pas de seconde chance. À partir de ce soir-là, tout avec des virgules.

## Étape 8 — `papeleria[0]` face à `papeleria[0:1]`

Une dernière chose chiffonne Mateo. Il y a deux manières de demander une seule succursale, et on dirait qu'elles donnent la même chose. On dirait.

Utilise l'indice et la tranche pour demander **la même chose**, la succursale Centre :

```python
print(papeleria[0].shape)
print(papeleria[0:1].shape)
```

```
torch.Size([7, 3])
torch.Size([1, 7, 3])
```

Les mêmes 21 nombres, une forme différente. `papeleria[0]` est un indice : **l'axe disparaît** et tu te retrouves avec un tableau. `papeleria[0:1]` est une tranche d'une seule succursale : **l'axe reste**, même s'il mesure 1, et tu te retrouves avec une feuille à un seul onglet :

```python
print(papeleria[0:1])
```

```
tensor([[[1200,  800,  500],
         [1500,  900,  700],
         [1000,  700,  500],
         [1300,  900,  600],
         [2000, 1200,  700],
         [1200,  800,  400],
         [ 800,  500,  300]]])
```

Trois crochets au début : c'est toujours en trois dimensions. C'est la différence entre sortir une feuille du classeur et garder un classeur d'une seule feuille. Et cela compte pour PyTorch : un modèle qui attend « un classeur de photos », de forme `(N, C, H, W)`, peut refuser une photo isolée de forme `(C, H, W)`. La règle complète du jour :

> **Avec un indice, l'axe disparaît. Avec une tranche, l'axe reste** (même s'il mesure 1).

| Ce que tu écris | Quel axe part | Forme | Ce que c'est |
|---|---|---|---|
| `papeleria[1, 4, 2]` | les trois | `()` | Nord, vendredi, sacs à dos |
| `papeleria[1]` | succursale | `(7, 3)` | l'onglet de Nord |
| `papeleria[1, 4]` | succursale et jour | `(3,)` | Nord, vendredi, les 3 produits |
| `papeleria[:, 4]` | jour | `(2, 3)` | le vendredi des deux succursales |
| `papeleria[0, 1:5]` | succursale | `(4, 3)` | Centre du mardi au vendredi |
| `papeleria[:, :, 2]` | produit | `(2, 7)` | les sacs à dos, par succursale et par jour |
| `papeleria[0:1]` | aucun | `(1, 7, 3)` | une feuille avec seulement l'onglet Centre |

L'astuce pour prédire la forme : **écris la forme d'origine, barre chaque axe où tu as mis un indice et remplace par la longueur de la plage chaque axe où tu as mis une tranche.** Pour `papeleria[0, 1:5]` : `(2, 7, 3)` → barre le 2 → remplace le 7 par 4 (de 1 à 4) → `(4, 3)`.

## Étape 9 — Une photo de 2 × 3 points

**Mardi 17 novembre, 1 h 15. Chambre de Mateo.**

Il n'arrive pas à dormir. Sur son téléphone, il a une photo que doña Lupe lui a envoyée dans l'après-midi : le terminal de Nord, avec son câble emmêlé. « Pour que tu voies qu'il est bien branché », a-t-elle écrit.

Mateo se dit qu'une photo aussi est un tenseur. Et que s'il sait découper la papeterie, il sait découper une photo.

Pour le vérifier, il commence par une toute petite photo créée manuellement : 2 points de haut sur 3 de large. Chaque point contient trois nombres, **rouge, vert et bleu**, de 0 (rien) à 255 (le maximum). Le jaune, par exemple, c'est du rouge et du vert au maximum, sans bleu.

Crée un fichier `foto.py` :

```python
import torch

# Foto de 2 × 3 puntos, en orden alto × ancho × color
#   eje 0 = renglón (alto), eje 1 = punto en el renglón (ancho), eje 2 = color (0 rojo, 1 verde, 2 azul)
foto = torch.tensor([
    [[255, 0, 0], [0, 255, 0], [0, 0, 255]],          # arriba: rojo, verde, azul
    [[255, 255, 255], [0, 0, 0], [255, 255, 0]],      # abajo: blanco, negro, amarillo
])
print(foto.shape)
```

```
torch.Size([2, 3, 3])
```

La forme contient deux 3 : le premier correspond aux points en largeur ; le second, aux couleurs. Sans le commentaire, personne ne sait lequel est lequel, pas même toi dans deux semaines. Extrais **seulement le canal rouge**, comme pour les sacs à dos :

```python
rojo = foto[:, :, 0]
print(rojo)
print(rojo.shape)
assert rojo.shape == (2, 3), "El canal rojo es una tabla de 2 × 3"
assert rojo.tolist() == [[255, 0, 0], [255, 0, 255]]
print("Canal rojo correcto")
```

```
tensor([[255,   0,   0],
        [255,   0, 255]])
torch.Size([2, 3])
Canal rojo correcto
```

En haut, seul le point rouge contient du rouge ; en bas, le blanc et le jaune oui, le noir non. Ce tableau est une photo en noir et blanc de la quantité de rouge dans chaque point : c'est cela, « séparer les canaux ». Un point isolé, avec ses trois couleurs :

```python
print(foto[1, 2])
```

```
tensor([255, 255,   0])
```

En bas, à droite : le jaune.

Ça fonctionne exactement comme pour les sacs à dos. Mêmes deux-points, mêmes règles. Mateo sourit dans le noir. Ce qu'il a appris avec la papeterie lui sert pour n'importe quelle boîte de nombres.

## Étape 10 — L'ordre des axes est une décision de conception

Mais là, à une heure du matin, une question gênante lui vient : qui a décidé que la couleur serait le dernier axe ? Lui. Tu as enregistré la photo en **hauteur × largeur × couleur**, mais rien ne t'y obligeait. On peut l'enregistrer en **couleur × hauteur × largeur** : le tableau du rouge, puis celui du vert, puis celui du bleu.

```python
# La misma foto en orden color × alto × ancho
#   eje 0 = color, eje 1 = renglón, eje 2 = punto en el renglón
foto_cha = torch.tensor([
    [[255, 0, 0], [255, 0, 255]],      # canal rojo
    [[0, 255, 0], [255, 0, 255]],      # canal verde
    [[0, 0, 255], [255, 0, 0]],        # canal azul
])
print(foto_cha.shape)
print(foto_cha[0])
assert foto_cha[0].tolist() == foto[:, :, 0].tolist()
print("Mismo rojo, otro orden")
```

```
torch.Size([3, 2, 3])
tensor([[255,   0,   0],
        [255,   0, 255]])
Mismo rojo, otro orden
```

Les mêmes 18 nombres, une autre forme : `(3, 2, 3)`. Et le rouge n'est plus `foto[:, :, 0]`, c'est simplement `foto_cha[0]`. **Ce que tu places en premier est ce qui s'extrait le plus facilement.**

Lequel est le bon ? Les deux. C'est comme ranger la penderie par type de vêtement ou par tenue, ou comme doña Lupe range l'entrepôt : par succursale ou par produit. La marchandise est la même ; ce qui change, c'est ce que tu trouves vite. C'est pourquoi l'ordre des axes est une **décision de conception** : tu la prends en pensant à ce que tu demanderas le plus souvent.

Il existe déjà des habitudes. Dans PyTorch, la convention courante pour les images est **canaux d'abord**, que tu verras dans la documentation sous la forme **C × H × W** (*channels*, *height*, *width* : canaux, hauteur et largeur ; [documentation de TorchVision](https://docs.pytorch.org/vision/stable/transforms.html)) ; un lot de photos ajoute au début l'axe qui indique de quelle photo il s'agit. D'autres outils et beaucoup de programmes d'images utilisent hauteur × largeur × couleur. Aucun n'est faux ; ce qui est grave, c'est de ne pas savoir lequel tu as entre les mains. C'est pourquoi, avec les photos, **le commentaire avec les axes va toujours au-dessus du tenseur.**

| Ordre | Forme de ta photo | Le canal rouge | Un point avec ses 3 couleurs |
|---|---|---|---|
| hauteur × largeur × couleur | `(2, 3, 3)` | `foto[:, :, 0]` | `foto[1, 2]` |
| couleur × hauteur × largeur | `(3, 2, 3)` | `foto_cha[0]` | `foto_cha[:, 1, 2]` |

Si tu exécutes `foto_cha[:, 1, 2]`, tu obtiens `tensor([255, 255,   0])` : le même jaune.

Mateo referme son ordinateur portable à 2 h 03. Sur le post-it de l'écran, il y a toujours le 200. Ça ne lui semble plus un nombre. Ça lui semble une question.

---

👉 **Si tu viens de la gestion :** ce que Mateo a fait ce soir-là, tu le fais avec un tableur quand tu filtres ou masques des colonnes : « seulement le vendredi de toutes les succursales » ou « seulement les sacs à dos de tous les onglets ». Ici, chacune de ces opérations tient en une ligne. Et l'étape 10 te sert dans n'importe quel rapport : quand on te demande « les ventes par succursale et par mois », demande ce qui va en lignes et ce qui va en colonnes. C'est une décision de conception, et elle se prend en pensant à la personne qui va le lire.

---

## Exercices de la session 3

Les solutions se trouvent à la fin. Travaille dans ton `ejes.py`, qui contient déjà `papeleria` et le commentaire avec les axes. Dans l'exercice 12, les `assert` déjà écrits te disent si tu as réussi ; dans les exercices 10, 11 et 13, compare ton travail avec les solutions.

**Exercice 10 — Désigne le nombre, maintenant avec des virgules.** Avec un indice par axe et `.item()`, écris le `assert` de chaque donnée :

a) Ce que Centre a vendu le mercredi en stylos.
b) Ce que Nord a vendu le dimanche en cahiers.
c) Ce que Centre a vendu le samedi en sacs à dos.
d) Vérifie avec un `assert` que ta réponse à a) donne la même chose avec les crochets séparés de la session 1.

Exemple résolu : `assert papeleria[0, 0, 0].item() == 1200` (Centre, lundi, cahiers).

**Exercice 11 — Prédis la forme avant d'exécuter.** Sur papier, écris la forme de chaque expression et ce qu'elle représente, en utilisant l'astuce de l'étape 8. Ensuite, exécute et compare.

```python
for nombre, x in [
    ("a) papeleria[0]", papeleria[0]),
    ("b) papeleria[0, 6]", papeleria[0, 6]),
    ("c) papeleria[:, 6]", papeleria[:, 6]),
    ("d) papeleria[:, :, 0]", papeleria[:, :, 0]),
    ("e) papeleria[1, 0:3]", papeleria[1, 0:3]),
    ("f) papeleria[1, 0:3, 1]", papeleria[1, 0:3, 1]),
    ("g) papeleria[0:1, 6]", papeleria[0:1, 6]),
    ("h) papeleria[1, 4, 2]", papeleria[1, 4, 2]),
]:
    print(nombre, x.shape)
```

Si tu t'es trompé sur l'une d'elles, cherche quel axe tu croyais conservé alors qu'il est parti, ou l'inverse.

**Exercice 12 — Complète les trous.** Remplis chaque `____` et vérifie avec les `assert` :

```python
# a) El fin de semana (sábado y domingo) de Norte, los 3 productos
fin_norte = papeleria[____, ____]
assert fin_norte.shape == (2, 3), "Fin de semana de Norte: 2 días × 3 productos"
assert fin_norte.tolist() == [[300, 200, 100], [200, 100, 100]]

# b) Las plumas del Centro de lunes a miércoles
plumas = papeleria[0, ____, ____]
assert plumas.shape == (3,), "Plumas del Centro: 3 días"
assert plumas.tolist() == [800, 900, 700]

# c) Lo mismo que b), con un for
a_mano = []
for d in range(____):
    a_mano.append(papeleria[0, d, 1].____)
assert plumas.tolist() == a_mano
print("Ejercicio 12 superado")
```

Indice pour a) : le dimanche est le jour 6, et la tranche exclut son dernier nombre.

**Exercice 13 — Ta propre photo.** Crée manuellement une photo de **3 points de haut sur 2 de large**, enregistrée dans l'ordre habituel de PyTorch : **couleur × hauteur × largeur** (`C × H × W`). Choisis toi-même les couleurs des 6 points (rouge, vert, bleu, blanc, noir, jaune ou celles que tu veux) et écris-les dans un commentaire. Ensuite :

a) Écris le `assert` de sa forme.
b) Extrais le canal vert. Écris le `assert` de sa forme et celui de sa valeur par rapport à une liste écrite manuellement.
c) Extrais la ligne du haut avec ses trois couleurs. Quelle forme a-t-elle ? Écris ses deux `assert`.
d) Extrais le point en bas à droite avec ses trois couleurs. Écris ses deux `assert`.

Il n'y a pas une seule réponse ; ce qui doit être respecté, c'est que les `assert` passent et que le commentaire dise ce qu'est chaque axe.

---

## Erreurs courantes (et comment t'en rendre compte)

| Erreur | Ce que l'on observe | Comment t'en rendre compte | Ce qu'il faut faire |
|---|---|---|---|
| Demander un indice qui n'existe pas | `papeleria[1, 7, 2]` donne `IndexError: index 7 is out of bounds for dimension 1 with size 7` | Le message t'indique quel axe (*dimension 1*, les jours) et combien il mesure (*size 7*) | Les indices vont de 0 à la taille moins 1 : les jours, de 0 à 6 |
| Inverser l'ordre des axes | `papeleria[4, 1, 2]` au lieu de `papeleria[1, 4, 2]` donne `IndexError: index 4 is out of bounds for dimension 0 with size 2` | Ça plante sur *dimension 0*, qui n'a que 2 succursales | Relis ton commentaire d'axes : le premier indice est toujours la succursale |
| Enchaîner des crochets avec `:` | `papeleria[:][4]` donne la même erreur de *dimension 0*, ou pire, ne plante pas et te donne autre chose | Le résultat n'est pas ce que tu as demandé, ou ça plante sur un axe que tu n'as pas touché | Tous les indices dans les mêmes crochets : `papeleria[:, 4]` |
| Croire que la plage inclut le dernier | Tu demandes du mardi au vendredi avec `1:4` et il te manque le vendredi | Le `assert` de la forme indique 3 jours là où tu en attendais 4 | Le dernier nombre est exclu : du mardi au vendredi, c'est `1:5` |
| Confondre indice et tranche | `papeleria[0:1]` a la forme `(1, 7, 3)` alors que tu attendais `(7, 3)` | Le `assert` de la forme échoue avec un 1 en trop | Indice : l'axe part. Tranche : l'axe reste, même s'il mesure 1 |
| Utiliser `.item()` avec plusieurs nombres | `papeleria[0, 0].item()` donne `RuntimeError: a Tensor with 3 elements cannot be converted to Scalar` | Le message dit combien d'éléments il y avait (3) | `.item()` ne sert que pour un seul nombre ; pour plusieurs, `.tolist()` |
| Comparer un morceau à un nombre dans un `assert` | `assert papeleria[0, 0] == 1200` donne `RuntimeError: Boolean value of Tensor with more than one value is ambiguous` | C'est le `assert` qui plante, pas l'indice | Il te manquait un indice (tu voulais `papeleria[0, 0, 0]`), ou compare la forme puis `.tolist()` à une liste |

Un piège qui ne plante pas : `papeleria[0, 5:20]` ne signale aucune erreur alors qu'il n'y a que 7 jours ; il te donne samedi et dimanche, forme `(2, 3)`, exactement comme le feraient les listes Python. Si une plage te donne moins que prévu, vérifie ses nombres.

---

## Révision pour la prochaine session

La session 4 commence par ces trois questions. Réponds-y sans regarder la leçon :

1. Quelle forme a `papeleria[:, 4]` et que représente-t-il ?
2. Quelle différence de forme y a-t-il entre `papeleria[0]` et `papeleria[0:1]` ?
3. Comment extrais-tu les ventes de sacs à dos des deux succursales pour tous les jours ?

🔁 **La semaine prochaine :**

Mardi, 8 h 10 du matin. Mateo appelle doña Lupe avant d'entrer en cours. Il lui raconte l'histoire du vendredi. Celle des sacs à dos qui ne bougent pas.

À l'autre bout du fil, un long silence. On entend le rideau de Centre qui se lève.

« Mais si les sacs à dos ne se vendent pas », dit enfin doña Lupe, « qu'est-ce qu'on fait payer aux gens ? »

Mateo ne le sait pas. Mais il sait comment continuer : arrêter d'extraire des morceaux et commencer à les **résumer**. Des totaux et des moyennes par axe : combien chaque succursale a vendu dans la semaine, quelle a été la moyenne de chaque jour, le tout en une ligne et vérifié par rapport à un `for`. Quand il additionnera la semaine de Nord, le nombre désignera quelqu'un.

---

## Solutions (ne les regarde pas avant d'avoir essayé)

**Révision pour commencer.**

1. `t.shape` t'indique combien d'éléments il y a sur chaque axe : pour la papeterie, 2 succursales, 7 jours et 3 produits. On le vérifie avec `assert papeleria.shape == (2, 7, 3)`.
2. Parce qu'elle n'est pas rectangulaire : la première ligne contient 3 nombres et la deuxième, 2. PyTorch refuse et le dit : `ValueError: expected sequence of length 3 at dim 1 (got 2)`.
3. `torch.tensor([1, 2, 3])` sort en `torch.int64` (entiers) et `torch.tensor([1.0, 2.0])` sort en `torch.float32` (décimaux).

**Exercice 10.**

```python
assert papeleria[0, 2, 1].item() == 700    # a) Centro, miércoles, plumas
assert papeleria[1, 6, 0].item() == 200    # b) Norte, domingo, cuadernos
assert papeleria[0, 5, 2].item() == 400    # c) Centro, sábado, mochilas
assert papeleria[0][2][1].item() == papeleria[0, 2, 1].item()   # d)
print("Ejercicio 10 superado")
```

```
Ejercicio 10 superado
```

Les questions a) et b) portent sur les mêmes données que tu as cherchées aux questions a) et c) de l'exercice 2 de la session 1, maintenant avec des virgules.

**Exercice 11.**

```
a) papeleria[0] torch.Size([7, 3])
b) papeleria[0, 6] torch.Size([3])
c) papeleria[:, 6] torch.Size([2, 3])
d) papeleria[:, :, 0] torch.Size([2, 7])
e) papeleria[1, 0:3] torch.Size([3, 3])
f) papeleria[1, 0:3, 1] torch.Size([3])
g) papeleria[0:1, 6] torch.Size([1, 3])
h) papeleria[1, 4, 2] torch.Size([])
```

a) l'onglet Centre ; b) le dimanche de Centre ; c) le dimanche des deux succursales ; d) les cahiers des deux, tous les jours ; e) Nord du lundi au mercredi ; f) les stylos de Nord du lundi au mercredi ; g) le dimanche de Centre en conservant l'axe des succursales, qui mesure 1 ; h) Nord, vendredi, sacs à dos. La plus piégeuse est g) : elle ressemble à b), mais la succursale a été demandée avec une tranche.

**Exercice 12.**

```python
fin_norte = papeleria[1, 5:7]       # a)
plumas = papeleria[0, 0:3, 1]       # b)
for d in range(3):                  # c)
    a_mano.append(papeleria[0, d, 1].item())
```

```
Ejercicio 12 superado
```

Dans a), `5:7` prend le samedi (5) et le dimanche (6) et exclut le 7, qui de toute façon n'existe pas.

**Exercice 13.** Une réponse possible :

```python
import torch

# Foto de 3 × 2 puntos, en orden color × alto × ancho
#   eje 0 = color (0 rojo, 1 verde, 2 azul)
#   eje 1 = renglón (0 arriba ... 2 abajo)
#   eje 2 = punto en el renglón (0 izquierda, 1 derecha)
# Los puntos: arriba rojo y verde; en medio azul y blanco; abajo negro y morado
foto = torch.tensor([
    [[255, 0], [0, 255], [0, 128]],      # canal rojo
    [[0, 255], [0, 255], [0, 0]],        # canal verde
    [[0, 0], [255, 255], [0, 128]],      # canal azul
])
assert foto.shape == (3, 3, 2), "3 colores × 3 renglones × 2 puntos"

verde = foto[1]
assert verde.shape == (3, 2)
assert verde.tolist() == [[0, 255], [0, 255], [0, 0]]

arriba = foto[:, 0]
assert arriba.shape == (3, 2)
assert arriba.tolist() == [[255, 0], [0, 255], [0, 0]]

esquina = foto[:, 2, 1]
assert esquina.shape == (3,)
assert esquina.tolist() == [128, 0, 128]
print("Ejercicio 13 superado")
```

```
Ejercicio 13 superado
```

Dans c), la forme est `(3, 2)` et non `(2, 3)` : extraire un morceau ne réorganise pas les axes restants, il les laisse dans l'ordre où ils étaient, et la couleur vient en premier.

---

## La suite

Dans la session 4, le total de Centre sera la somme de `papeleria[0]`. Si tu veux prendre de l'avance, calcule avec un `for` combien chaque succursale a vendu dans la semaine. Pour Centre, tu le sais déjà depuis la session 1 : 18 500 $. Il te manque Nord.

---

*J'écris à partir de ce que nous voyons tous les jours chez Hábil, où nous concevons des systèmes pour le secteur bancaire, les assurances et les entreprises au Mexique. Si, dans ton travail ou dans l'entreprise de ta famille, tu rencontres un problème de ce genre, discutons-en : [habil.mx](https://www.habil.mx/fr/)*
