# Les tenseurs à partir de zéro (5) : changer de forme sans changer les données

*Douze mois sur une ligne, quatre trimestres ou deux semestres : les mêmes nombres rangés autrement, et comment lire l'erreur qui apparaît quand tu demandes une forme impossible.*

*Les tenseurs à partir de zéro · leçon B, session 5 sur 8*

*Scénario : les personnages et l'histoire sont inventés. Les données techniques, historiques et les chiffres réels indiquent leur source.*

**Lundi 30 novembre, 21 h 15. Cuisine de doña Lupe.**

La marmite de haricots était encore tiède sur la cuisinière. Sur la nappe en toile cirée, il y avait un classeur vert, épais, fermé par un élastique.

« Fer ne répond toujours pas », dit doña Lupe. « Une semaine, Mateo. Une semaine. »

« Elle est peut-être malade, ma tante. »

« Peut-être. » Elle n'avait pas l'air convaincue.

Elle lui tendit le classeur. Dedans, il y avait le cahier du comptable : ce que Centre avait vendu chaque mois de l'année précédente. Douze lignes manuscrites, au stylo bleu.

« Le comptable les veut par trimestre. Et monsieur Ortega, par semestre, pour “sa révision”. Chacun les veut dans sa petite boîte. »

« Ce sont les mêmes douze nombres. »

« Eh bien, dis-le-leur. »

Mateo les saisit sur son ordinateur portable. Douze nombres sur une ligne. Il n'avait à en changer aucun. Seulement à les ranger autrement. Et une idée lui trottait dans la tête : s'il voyait une année saine rangée par trimestres, il saurait quoi chercher à Nord. Quel mois devait être fort. Lequel ne l'était pas.

« Quel mois est-ce que tu vends le plus ? »

« Août, mon petit. La rentrée des classes. Les mamans font la queue jusque dans la rue. Presque toutes paient par carte. »

Mateo nota *août* sur un post-it et le colla au bord de l'écran. Il ne savait pas encore pourquoi.

Sur le téléphone de doña Lupe, une notification apparut. Un courriel de la banque. Elle l'ouvrit, le lut avec ses lunettes au bout du nez et le lui passa sans rien dire.

*« Nous vous informons que les premiers débits inhabituels enregistrés sur le terminal de votre succursale Nord datent de la seconde quinzaine d'août. »*

Le post-it était toujours là. Jaune. *Août*.

---

Avec cette session commence la deuxième leçon, **Les tenseurs au travail**. Jusqu'ici, Mateo a appris à regarder : créer un tenseur, en extraire des morceaux et le résumer. C'est ainsi qu'il a découvert que Centre vend environ quatre fois plus que Nord : 18 500 $ contre 4 600 $ sur la semaine. Maintenant, les tenseurs se mettent au travail, et chaque outil le rapproche un peu plus de ce qui s'est passé à Nord : tu changes la forme d'un tenseur sans toucher à ses données, tu fais des calculs sur des tenseurs complets sans un seul `for`, tu entraînes un modèle qui détecte les fraudes et tu vois où on les trouve dans une banque, une compagnie d'assurance et un magasin. Il reste 23 jours avant l'échéance de la banque.

| Session | Ce que tu vois | Pratique |
|---|---|---|
| **5 — Changer de forme sans changer les données** | `reshape` et `flatten`, et comment lire l'erreur de forme | Exercices 18 à 21 |
| **6 — Opérer sur des tenseurs complets** | Faire des calculs sur tout le tenseur à la fois, sans `for` | Exercices 22 à 25 |
| **7 — Un détecteur de fraude qui apprend à partir des données** | Entraîner un petit modèle avec des tenseurs | Exercices 26 à 29 |
| **8 — Les tenseurs à la banque, dans l'assurance et au magasin** | Lire un cas réel et donner la forme de son tenseur | Exercices 30 à 33, autoévaluation et révision de la leçon B |

Comme dans la leçon A, chaque session dure environ 2 heures : une pour lire et une autre pour pratiquer.

Dans la session précédente, tu as additionné par onglet avec `sum(dim=)` et tu as comparé le résultat à un `for`. Aujourd'hui, tu ne changes aucun nombre : tu changes la façon dont ils sont rangés. Comme Mateo ce soir-là, avec le cahier du comptable ouvert à côté de l'ordinateur.

## Révision pour commencer

Réponds sans regarder la session 4. Les réponses se trouvent dans les solutions, à la fin.

1. Si tu additionnes `papeleria` avec `dim=1`, quelle forme reste-t-il et que signifie-t-elle ?
2. Pourquoi `mean` échoue-t-il avec des entiers et comment le corriges-tu ?
3. Comment vérifies-tu que ta somme avec des tenseurs est correcte ?

> **Le code de cette session** se trouve dans le dépôt public du cours, dans le dossier `sesion-05` : [github.com/HabilMX/curso-tensores](https://github.com/HabilMX/curso-tensores/tree/main/sesion-05). Écris-le toi-même d'abord, car c'est ainsi qu'on apprend ; le dépôt sert à comparer quand quelque chose ne donne pas le même résultat chez toi. Chaque programme est accompagné de la sortie qu'il doit te donner.

## Objectifs de cette session

À côté de chaque objectif se trouve l'exercice qui le démontre.

| # | À la fin, tu peux… | Démontré par |
|---|---|---|
| 1 | **Dire**, sans rien exécuter, si une nouvelle forme (avec ou sans `-1`) est valide pour un tenseur, et **calculer** la dimension qui manque quand il y a un `-1` | Exercice 18 |
| 2 | **Réorganiser** un tenseur avec `reshape`, en utilisant `-1` pour que PyTorch calcule une taille, et **le vérifier** par rapport à un `for` | Exercices 19 et 20 |
| 3 | **Lire** mot à mot l'erreur de forme de PyTorch et **corriger** le code qui la provoque | Exercice 20 |
| 4 | **Distinguer** réorganiser un tableau et le faire pivoter, et **détecter** avec un `assert` quand quelqu'un a confondu les deux | Exercice 21 |
| 5 | **Aplatir** un tenseur avec `flatten` et **restaurer** sa forme sans perdre de valeurs | Exercice 19, partie c |

**Ce que tu dois savoir avant :** les sessions 1 à 4 : créer un tenseur, lire son `shape`, indexer, découper en tranches et additionner avec `sum(dim=)` ; et en Python, utiliser `for`, `range`, les listes, `append`, `tolist()` et `assert` (plus bas, à l'étape 3, il y a un rappel). **Ce dont tu as besoin :** l'environnement de la session 2 avec PyTorch, et le fichier `papeleria.py` de la session 4 (le tenseur `papeleria` de forme `(2, 7, 3)`). Les fichiers d'aujourd'hui vont dans ce même dossier.

**Périmètre d'aujourd'hui :** la nouvelle décision que tu pratiques, c'est de savoir si `reshape` conserve l'ordre dont tu as besoin. `flatten()` est le cas où l'on met tout sur une seule ligne. `.T` n'apparaît que pour contraster avec `reshape` : écris cette ligne et laisse les `assert` la vérifier. L'étape 10 et l'encadré sur TensorFlow sont à lire pour voir où cela sert en IA : exécute `foto.py` si tu veux, mais aucun exercice n'en a besoin, et il n'est pas nécessaire d'installer TensorFlow.

---

## Pourquoi c'est important

« Et comment tu les ranges sans qu'ils se mélangent ? » demanda doña Lupe.

Mateo montra la boîte d'œufs à côté de la cuisinière.

Une douzaine tient dans une boîte de 2 rangées de 6 ou dans une de 3 par 4 : les mêmes 12 œufs, une autre boîte. Et si on te demande de la mettre dans une boîte de 5 par 3, tu sais que c'est impossible : elle a 15 alvéoles et tu as 12 œufs.

« Ça, même moi je le sais », dit-elle. « Et s'il reste une alvéole vide, c'est que quelqu'un a mangé un œuf. »

Avec les tenseurs, cela arrive tout le temps : les données arrivent avec une forme et l'étape suivante en a besoin sous une autre. Doña Lupe a des ventes par mois et son comptable les veut par trimestre. Une photo arrive sous forme de grille et le modèle la veut sous forme d'une ligne de nombres. On n'invente ni ne perd un seul nombre : on change seulement de boîte. C'est ça, **reshape** (« changer de forme », en anglais).

Et une raison pratique : parmi les erreurs que tu verras le plus souvent avec les tenseurs, il y a les erreurs de forme : demander une boîte qui ne correspond pas à tes œufs. Aujourd'hui, tu apprends à les lire sans t'affoler. Et tu vas découvrir l'autre erreur, la dangereuse : celle qui ne plante pas. La boîte a l'air identique, tout tombe juste, personne ne se plaint… et les œufs ne sont plus ceux qu'ils étaient.

## Le cas : les ventes de l'année de doña Lupe

Ce sont les douze lignes du cahier du comptable : ce que Centre a vendu chaque mois de l'année précédente, en pesos. Une année saine, sans surprises. Août est son mois fort, à cause de la rentrée des classes, et décembre aussi, à cause des sacs à dos offerts en cadeau. Mateo va s'en servir comme modèle : voilà à quoi ressemble une année normale dans une papeterie de doña Lupe.

### Étape 1 — Les douze mois sur une ligne

Crée un fichier `cambiar_forma.py` :

```python
import torch

#        ene    feb    mar    abr    may    jun    jul    ago     sep    oct    nov    dic
meses = [60000, 55000, 60000, 50000, 65000, 50000, 70000, 100000, 70000, 60000, 55000, 90000]
ventas = torch.tensor(meses)
print(ventas.shape)
print(ventas.sum())
```

Exécute-le avec `python3 cambiar_forma.py` :

```
torch.Size([12])
tensor(785000)
```

Rien de nouveau : 12 nombres qui font 785 000 $ au total. Ce total est ton témoin : tu vas changer la forme de nombreuses fois, et il ne doit jamais bouger.

### Étape 2 — Quatre trimestres de trois mois

Le comptable demande les ventes par trimestre : de janvier à mars, d'avril à juin, et ainsi de suite. Tu veux un tableau de **4 lignes (trimestres) sur 3 colonnes (mois)**.

Ajoute à la fin du fichier :

```python
trimestres = ventas.reshape(4, 3)
print(trimestres)
print(trimestres.shape)
print(trimestres.sum())
```

Les nouvelles lignes de la sortie (au-dessus, les précédentes continuent de s'afficher) :

```
tensor([[ 60000,  55000,  60000],
        [ 50000,  65000,  50000],
        [ 70000, 100000,  70000],
        [ 60000,  55000,  90000]])
torch.Size([4, 3])
tensor(785000)
```

Quatre lignes, trois colonnes et le même total. `reshape(4, 3)` veut dire « range-les en 4 sur 3 ».

**Comment PyTorch décide-t-il quel nombre va à quel endroit ?** Il lit la ligne d'origine de gauche à droite, comme un livre, et remplit le nouveau tableau ligne par ligne, comme quand tu ranges des œufs dans une boîte : d'abord la première rangée, puis la deuxième.

Comme les mois étaient dans l'ordre, chaque ligne correspond exactement à un trimestre. Ce n'est pas de la chance : l'ordre de lecture coïncide avec ce que tu veux. Garde cette idée en tête ; à l'étape 9, tu la verras échouer.

Maintenant, tu peux utiliser tout ce que tu as vu dans la leçon A sur le nouveau tableau. Ajoute :

```python
print(trimestres.sum(dim=1))   # total de cada trimestre
print(trimestres[3, 2])        # cuarto trimestre, tercer mes: diciembre
print(ventas.shape)            # ¿se descompuso la original?
```

Nouvelle sortie :

```
tensor([175000, 165000, 240000, 205000])
tensor(90000)
torch.Size([12])
```

- Le troisième trimestre (de juillet à septembre) est le plus fort : 240 000 $, grâce à la rentrée des classes.
- `trimestres[3, 2]` est décembre, 90 000 $ : ligne 3 (quatrième trimestre), colonne 2 (troisième mois).
- `ventas` garde la forme `(12,)`. **`reshape` ne change pas la forme du tenseur d'origine : il t'en renvoie un avec la nouvelle forme.** C'est pour ça que tu le stockes dans une variable, comme `trimestres`. Attention : ce n'est pas toujours une copie séparée ; souvent, les deux partagent les mêmes nombres en interne. Aujourd'hui, ça n'a pas d'importance, parce que tu n'en modifies aucun.

> Doña Lupe se pencha par-dessus son épaule.
>
> « Et comment tu sais que la machine ne les a pas inventés ? »
>
> « Je ne le sais pas, ma tante. C'est pour ça que je vais le vérifier. »
>
> « Ça, ça me plaît. Même l'ordinateur, il ne faut pas le croire du premier coup. »

### Étape 3 — Le vérifier par rapport à un `for`

Ne crois pas PyTorch sur parole. Calcule les totaux avec un `for` et compare.

**Rappel de Python que tu vas utiliser :** `range(4)` produit 0, 1, 2 et 3 ; `lista.append(x)` ajoute `x` à la fin d'une liste ; `tensor.tolist()` convertit un tenseur en liste ordinaire pour la comparer à une autre liste.

Et un outil que tu utiliseras dans toute la série. Un `assert` est une vérification automatique : si la condition est vraie, il n'affiche rien et le programme continue ; si elle est fausse, le programme s'arrête et affiche le message entre guillemets. Ici, il sert de contrôle d'erreur : tu n'as pas à croire le résultat, tu peux le vérifier. Si tu veux voir à quoi ressemble un échec, essaie ceci dans une console Python :

```python
# Prueba opcional para reconocer un fallo:
assert 2 == 3, "2 no es igual a 3"
# Sale: AssertionError: 2 no es igual a 3
```

Maintenant, ajoute :

```python
a_mano = []
for t in range(4):
    total = 0
    for m in range(3):
        total = total + meses[t * 3 + m]
    a_mano.append(total)

assert trimestres.shape == (4, 3), "Deberían ser 4 trimestres × 3 meses"
assert trimestres.sum(dim=1).tolist() == a_mano, "Los totales por trimestre no cuadran con el for"
print("Trimestres comprobados")
```

Nouvelle sortie :

```
Trimestres comprobados
```

Regarde `meses[t * 3 + m]` : c'est le calcul que fait `reshape` en interne. Décembre (trimestre 3, mois 2) se trouve à la position 3 × 3 + 2 = 11 de la ligne d'origine. `reshape` t'épargne ce calcul. Et comme toujours : d'abord le `assert` de la forme, ensuite celui des valeurs.

### Étape 4 — Deux semestres, et même trois dimensions

Monsieur Ortega demande à doña Lupe ses ventes par semestre : deux moitiés de six mois. Ajoute :

```python
semestres = ventas.reshape(2, 6)
print(semestres)
print(semestres.sum(dim=1))

anio = ventas.reshape(2, 2, 3)   # semestre × trimestre × mes
print(anio)
print(anio[1, 0, 2])
```

Nouvelle sortie :

```
tensor([[ 60000,  55000,  60000,  50000,  65000,  50000],
        [ 70000, 100000,  70000,  60000,  55000,  90000]])
tensor([340000, 445000])
tensor([[[ 60000,  55000,  60000],
         [ 50000,  65000,  50000]],

        [[ 70000, 100000,  70000],
         [ 60000,  55000,  90000]]])
tensor(70000)
```

Premier semestre, 340 000 $ ; deuxième, 445 000 $. Ensemble, ils font les 785 000 $ habituels.

La forme `(2, 2, 3)` transforme l'année en **cube** : la feuille de calcul à onglets de la session 1, un onglet par semestre et, dans chacun, un tableau de trimestres par mois. `anio[1, 0, 2]`, c'est le deuxième semestre, son premier trimestre, son troisième mois : septembre, 70 000 $. Douze nombres, quatre formes, et rien ne s'est perdu.

> Mateo resta les yeux fixés sur le deuxième semestre. Là se trouvait le plus grand nombre de toute l'année, seul, au milieu de son trimestre : août, 100 000 $. Dans une année saine, août est le mois où le plus de gens paient par carte.
>
> Il regarda le post-it. Il regarda le courriel de la banque : *seconde quinzaine d'août*.
>
> Si quelqu'un voulait cacher de faux débits, quel mois choisirait-il ? Le mois où personne ne fait attention, parce que tout le monde est en train de payer.

### Étape 5 — La règle : multiplie la forme

> **Un `reshape` avec des tailles positives est valide quand la nouvelle forme, une fois multipliée, donne le même nombre total de valeurs que l'originale.** Avec `-1`, il y a une règle à part, à l'étape 6.

Vérifie-le avant d'exécuter :

| Forme demandée | Multipliée | Convient-elle pour 12 mois ? |
|---|---|---|
| `(4, 3)` | 4 × 3 = 12 | Oui |
| `(2, 2, 3)` | 2 × 2 × 3 = 12 | Oui |
| `(5, 3)` | 5 × 3 = 15 | Non : il te manque 3 œufs |
| `(5, 2)` | 5 × 2 = 10 | Non : il te reste 2 œufs en trop |

`reshape` ne complète jamais avec des zéros et ne jette jamais ce qui dépasse. Si ça ne tombe pas juste, il refuse.

### Étape 6 — Le `-1`, qui calcule tout seul

Parfois, tu sais combien tu veux dans une dimension et tu as la flemme de faire la division pour l'autre. Tu mets **`-1`** à l'endroit que tu ne veux pas calculer. Ajoute :

```python
print(ventas.reshape(-1, 3).shape)
print(ventas.reshape(6, -1).shape)
print(ventas.reshape(2, -1, 3).shape)
print(ventas.reshape(-1).shape)
```

Nouvelle sortie :

```
torch.Size([4, 3])
torch.Size([6, 2])
torch.Size([2, 2, 3])
torch.Size([12])
```

**`-1` veut dire « calcule toi-même combien il en faut ici pour que ça tombe juste »**. Dans `reshape(-1, 3)`, PyTorch se dit : « 12 nombres en colonnes de 3 : ça fait 4 lignes ». Dans `reshape(2, -1, 3)` : « 2 × quelque chose × 3 = 12 ; quelque chose vaut 2 ».

Et `reshape(-1)`, tout seul, veut dire « mets tout dans une seule dimension », quelle que soit la forme : une liste de nombres, comme le `(12,)` du début.

**La règle avec `-1` :** multiplie les tailles que tu as écrites ; ce produit doit diviser exactement le nombre total de valeurs (sans reste). Le `-1` vaut le total divisé par ce produit. Dans `reshape(2, -1, 3)` : 2 × 3 = 6, et 12 ÷ 6 = 2.

- **Il ne peut y en avoir qu'un.** Avec `ventas.reshape(-1, -1)`, tu obtiens `RuntimeError: only one dimension can be inferred` (« une seule dimension peut être déduite »).
- **Il ne fait pas de miracles.** Avec `reshape(-1, 5)`, aucun nombre multiplié par 5 ne donne 12, et ça plante. Le `-1` calcule ; il n'invente pas d'œufs.
- **C'est le seul nombre négatif accepté.** Avec `ventas.reshape(-2, -6)`, tu obtiens `RuntimeError: invalid shape dimension -2 at index 0 of shape [-2, -6]`, même si -2 × -6 donne 12.

### Étape 7 — L'erreur de forme, lue mot à mot

Provoquons l'erreur exprès. Un camarade voulait « 5 groupes de 3 mois ». Ajoute à la fin du fichier :

```python
quintos = ventas.reshape(5, 3)
```

Exécute-le (j'ai raccourci le chemin avec `...` ; sur ton ordinateur, il est différent) :

```
Traceback (most recent call last):
  File ".../cambiar_forma.py", line 42, in <module>
    quintos = ventas.reshape(5, 3)
              ^^^^^^^^^^^^^^^^^^^^
RuntimeError: shape '[5, 3]' is invalid for input of size 12
```

Un message d'erreur se lit **de bas en haut** : la dernière ligne dit ce qui s'est passé ; celles du dessus, où. La dernière, mot à mot :

| Morceau | Ce qu'il veut dire |
|---|---|
| `RuntimeError` | « Erreur d'exécution » : le code était correctement écrit, mais il a échoué pendant qu'il s'exécutait. |
| `shape '[5, 3]'` | « La forme [5, 3] ». C'est la forme que **tu as demandée**. |
| `is invalid` | « n'est pas valide ». |
| `for input of size 12` | « pour une entrée de taille 12 ». C'est ce que **tu as** : 12 nombres. |

Ensemble : **« La forme 5 × 3 ne convient pas pour quelque chose qui contient 12 nombres. »** 5 × 3 = 15, et 15, ce n'est pas 12. L'erreur ne te gronde pas : elle te donne ce que tu as demandé et ce que tu as.

Celles du dessus disent où : dans ton fichier, ligne 42 (tu peux obtenir un autre numéro si tu as laissé plus ou moins de lignes vides), et elles recopient la ligne fautive avec des `^^^^` sous le morceau qui a planté.

Si tu voulais des groupes de 3 mois, la correction est `reshape(4, 3)` ou `reshape(-1, 3)`. Si tu avais vraiment besoin de 5 groupes, le problème n'est pas le code : 12 ne se répartit pas en 5 parts égales. Supprime cette ligne avant de continuer.

**La recette pour n'importe quelle erreur de forme :** cherche dans la dernière ligne les deux tailles (ce que tu as demandé et ce que tu as), multiplie, et demande-toi ce que tu voulais vraiment.

> « Comme c'est correct », dit doña Lupe quand Mateo lui lut l'erreur. « Elle te dit ce que tu as demandé et ce que tu as. Si seulement la banque était comme ça. »
>
> Mateo rit. Puis il cessa de rire. *Ce que tu as demandé et ce que tu as.* Ce qui a été vendu et ce qui a été encaissé. Si quelqu'un mettait ces deux nombres ensemble, l'un à côté de l'autre…
>
> Il le nota sur un autre post-it. Ça devrait attendre.

### Étape 8 — Aplatir la papeterie avec `flatten`

Revenons à la papeterie `(2, 7, 3)`. Parfois, tu as besoin de ses 42 nombres sur une seule ligne, pour les enregistrer ou les donner à un modèle. C'est ce qu'on appelle **aplatir**, comme déplier une boîte en carton : le même carton, sans volume.

Dans PyTorch, cela s'appelle **`flatten`** (« aplatir »). Avant de créer le fichier, vérifie que ton `papeleria.py` peut être importé. Ouvre le Terminal dans le dossier où tu l'as enregistré et exécute :

```bash
python3 -c "from papeleria import papeleria; print(papeleria.shape)"
```

Cela doit afficher `torch.Size([2, 7, 3])`. Si tu obtiens `ModuleNotFoundError`, place `papeleria.py` dans ce même dossier et ouvre le Terminal à cet endroit. S'il s'importe mais indique qu'il ne trouve pas `papeleria`, retourne au fichier de la session 4 et vérifie que c'est bien la variable enregistrée.

Maintenant, crée `aplanar.py` :

```python
from papeleria import papeleria

print(papeleria.shape)
plana = papeleria.flatten()
print(plana.shape)
print(plana[:6])       # los primeros 6 números
print(plana[21:24])    # del 22 al 24
print(plana.sum(), papeleria.sum())

de_vuelta = plana.reshape(2, 7, 3)
assert de_vuelta.shape == (2, 7, 3), "Debería regresar a 2 sucursales × 7 días × 3 productos"
assert de_vuelta.tolist() == papeleria.tolist(), "Al regresar debería quedar idéntica"
print("Ida y vuelta sin perder nada")
```

Enregistre le fichier et exécute-le depuis le même dossier avec `python3 aplanar.py`. Si une erreur apparaît, n'avance pas : compare d'abord la dernière ligne avec la sortie attendue.

Sortie :

```
torch.Size([2, 7, 3])
torch.Size([42])
tensor([1200,  800,  500, 1500,  900,  700])
tensor([300, 200, 100])
tensor(23100) tensor(23100)
Ida y vuelta sin perder nada
```

- **2 × 7 × 3 = 42** nombres, qui font les mêmes 23 100 $ au total.
- **L'ordre de lecture est celui de toujours**, de la boîte extérieure vers la boîte intérieure : tout l'onglet Centre, jour par jour et produit par produit, puis tout l'onglet Nord. Les six premiers sont Centre lundi et mardi. Les 21 premiers appartiennent à Centre (7 × 3), donc les positions 21 à 23 sont Nord lundi : 300, 200, 100.
- **On peut revenir en arrière**, à condition de se souvenir de la forme qu'il avait.

`flatten()` fait la même chose que `reshape(-1)`, mais son nom dit ce qu'il fait.

### Étape 9 — Le piège : `reshape` ne fait pas pivoter le tableau

C'est le plus important de la session, parce qu'ici l'erreur **ne plante pas**.

`papeleria.sum(dim=1)` est le tableau de la semaine : 2 succursales × 3 produits. Doña Lupe le veut dans l'autre sens pour un rapport : **les produits en lignes, les succursales en colonnes**, 3 × 2. `reshape(3, 2)` ?

Crée `girar.py` :

```python
from papeleria import papeleria

semana = papeleria.sum(dim=1)    # sucursal × producto
print(semana)
print(semana.shape)

mal = semana.reshape(3, 2)
print(mal)
```

Exécute-le avec `python3 girar.py`. Sortie :

```
tensor([[9000, 5800, 3700],
        [2350, 1400,  850]])
torch.Size([2, 3])
tensor([[9000, 5800],
        [3700, 2350],
        [1400,  850]])
```

PyTorch ne s'est pas plaint : 3 × 2 = 2 × 3. Mais lis-le comme un rapport :

- La ligne des cahiers devrait contenir Centre 9 000 et Nord 2 350. Elle contient **9 000 et 5 800** : les cahiers et les stylos de Centre.
- Celle des stylos devrait contenir 5 800 et 1 400. Elle contient **3 700 et 2 350** : les sacs à dos de Centre et les cahiers de Nord.

Un vrai fouillis. `reshape` a fait la seule chose qu'il sait faire : lire dans l'ordre (9000, 5800, 3700, 2350, 1400, 850) et remplir les lignes deux par deux. **`reshape` change la boîte ; il ne sait pas à qui appartient chaque œuf.**

Ce que tu voulais, c'est **faire pivoter** le tableau : les lignes deviennent des colonnes. Dans PyTorch, c'est `.T` (de *transposer*, le nom mathématique). On écrit `semana.T`, **sans parenthèses** : on ne l'appelle pas comme une fonction, il décrit le tableau pivoté. Ajoute :

```python
bien = semana.T
print(bien)
print(bien.shape)
```

Nouvelle sortie :

```
tensor([[9000, 2350],
        [5800, 1400],
        [3700,  850]])
torch.Size([3, 2])
```

Cette fois, c'est bon : cahiers 9 000 et 2 350 ; stylos 5 800 et 1 400 ; sacs à dos 3 700 et 850. Même forme `(3, 2)` que le fouillis : **même taille, contenu différent.** C'est pour ça que le `assert` de la forme ne va jamais seul. De `.T`, tu n'as pas besoin de plus pour l'instant : c'est lui qui fait vraiment pivoter.

**Et pourquoi est-ce que ça a marché avec les trimestres ?** Parce que là, tu ne faisais rien pivoter : tu coupais une ligne en morceaux de 3, dans le même ordre. La question à te poser avant chaque `reshape` : **est-ce que je veux couper en morceaux, dans le même ordre de lecture ?** Si oui, `reshape`. Si tu veux que les lignes deviennent des colonnes, `reshape` te donne un fouillis sans te prévenir.

> *Même taille, contenu différent.* Mateo le dit à voix haute.
>
> « Quoi donc ? » demanda doña Lupe, qui était déjà en train de laver la marmite.
>
> « Quelque chose qui a l'air identique de l'extérieur peut être autre chose à l'intérieur. Comme le fouillis : il passe la vérification de la forme et personne ne s'en rend compte. »
>
> Doña Lupe ferma le robinet. Elle s'essuya les mains sur son tablier, lentement.
>
> « Le terminal de Nord, on l'a changé en août. »
>
> Mateo leva les yeux.
>
> « Qui ? »
>
> « Un technicien. Fer m'a envoyé une photo sur mon téléphone : “Le type de la banque est passé, le nouveau terminal est installé.” Pareil que l'autre. Et au bord de la photo, un homme en tee-shirt bleu, de dos. Moi, j'ai cru que c'était la banque qui l'envoyait. »
>
> « C'est toi qui l'as appelé ? »
>
> « Non. »
>
> « Et Fer ? »
>
> Doña Lupe ne répondit pas. Sur la table, le post-it disait toujours *août*.

### Étape 10 — Pourquoi l'IA passe son temps à changer de forme (lecture ; l'exécution est facultative)

**Une photo s'aplatit.** Une photo en noir et blanc de 28 × 28 points est un tableau, et beaucoup de modèles simples attendent une ligne de nombres par exemple.

**Un lot se réorganise.** Les modèles voient les photos par lots (*batches*), par exemple de 32 : un cube `(32, 28, 28)` que le modèle veut sous forme de 32 lignes, une par photo.

Si tu veux le voir s'exécuter, crée `foto.py` et exécute-le avec `python3 foto.py`. `torch.zeros` crée un tenseur de la forme que tu demandes, rempli de zéros, pour ne pas écrire des milliers de nombres :

```python
import torch

foto = torch.zeros(28, 28)           # una foto de 28 × 28 puntos, toda negra
print(foto.shape, foto.numel())
print(foto.flatten().shape)

lote = torch.zeros(32, 28, 28)       # 32 fotos
print(lote.reshape(32, -1).shape)
print(lote.flatten().shape)
```

Sortie :

```
torch.Size([28, 28]) 784
torch.Size([784])
torch.Size([32, 784])
torch.Size([25088])
```

- La photo aplatie : 784 nombres sur une ligne (28 × 28).
- `lote.reshape(32, -1)` : 32 lignes, une par photo ; le `-1` calcule les 784.
- Attention à `lote.flatten()` : il aplatit **tout**, y compris la séparation entre les photos. Il reste 25 088 nombres (32 × 784) et la forme ne dit plus où commence chaque photo : tu ne le sais que si tu te souviens qu'il y en avait 32 de 784. Avec des lots, tu veux `reshape(32, -1)`, qui conserve cette séparation dans la forme.

---

> **Encadré : la même chose dans TensorFlow.** Seulement pour comparer ; ne l'exécute pas et n'installe pas TensorFlow pour cette session : tous les objectifs et exercices se résolvent avec PyTorch. L'idée est identique : `tf.reshape(ventas, (4, 3))` au lieu de `ventas.reshape(4, 3)`, avec le même `-1`. Le message d'erreur contient : `Input to reshape is a tensor with 12 values, but the requested shape has 15`. TensorFlow fait la multiplication pour toi (5 × 3 = 15) ; PyTorch te donne la forme et c'est toi qui multiplies.

---

👉 **Si tu viens de la gestion :** c'est ce que fait le comptable de doña Lupe à chaque fin de trimestre, et ce que tu fais quand tu convertis un rapport mensuel en rapport trimestriel : les mêmes nombres regroupés autrement, et le total de l'année ne bouge pas. C'est ton contrôle de cohérence, comme le `assert` de l'étape 3. Et le piège de l'étape 9 existe aussi dans Excel : si tu réorganises un tableau manuellement au lieu d'utiliser « Collage spécial → Transposé », les nombres tombent dans les cellules d'à côté sans la moindre alerte. Après avoir réorganisé, vérifie une valeur que tu connais par cœur.

---

## Exercices de la session 5

Les solutions se trouvent à la fin. Essaie d'abord. Les `assert` vérifient les parties de code ; pour les prédictions et les explications, compare ta réponse avec la sortie attendue et avec la solution à la fin.

**Exercice 18 — Prédis avant d'exécuter.** Avec les 12 mois de `ventas`, écris sur papier si chaque forme est valide et quelle forme finale elle donnerait :

```
(3, 4)    (6, 2)    (5, 2)    (2, -1)    (-1, 5)    (2, 3, 2)    (12, 1)
```

Ensuite, vérifie-les une par une dans la console Python, avec `ventas` déjà créé. Si tu as encore dans `cambiar_forma.py` la ligne `quintos = ventas.reshape(5, 3)` de l'étape 7, supprime-la (sinon, le fichier s'arrête sur cette erreur) et ouvre la console avec :

```bash
python3 -i cambiar_forma.py
```

Quand `>>>` apparaît, écris une forme à la fois. Si `>>>` n'apparaît pas, le fichier s'est arrêté sur une autre erreur : corrige-la avant de faire l'exercice. Pour sortir, écris `exit()`. Ça ressemble à ceci (ne copie pas le `>>>` : c'est Python qui l'affiche, tu écris seulement ce qui suit) :

```text
>>> ventas.reshape(3, 4).shape
```

Voici à quoi ressemble une forme résolue : `(4, 3)` → 4 × 3 = 12 → valide → `torch.Size([4, 3])`. Pour les formes invalides, vérifie que l'erreur indique les deux tailles que tu attendais.

**Exercice 19 — Le tableau long.** Beaucoup de feuilles de calcul stockent les données dans un tableau long : une ligne par succursale et par jour, une colonne par produit. **Parties a) et b) :** remplis les trous et réponds à la question ci-dessous.

```python
from papeleria import papeleria

larga = papeleria.reshape(____, 3)
assert larga.shape == (14, 3), "Deberían ser 14 renglones (sucursal y día) × 3 productos"

a_mano = []
for s in range(2):
    for d in range(7):
        a_mano.append(papeleria[s, d].____())
assert larga.tolist() == a_mano, "La tabla larga no coincide con la armada a mano"
print("Ejercicio 19 superado")
```

Ensuite, sans rien exécuter, dis quelle ligne de `larga` occupe Nord mercredi. Vérifie-le avec `print(larga[9])`, recopie les trois nombres obtenus et ajoute :

```python
assert larga[9].tolist() == papeleria[1, 2].tolist(), "Ese renglón debería ser Norte miércoles"
```

**Partie c) — Aplatir.** Sans regarder l'étape 8, remplis les trous pour aplatir la papeterie et lui rendre sa forme :

```python
plana = papeleria.____()
de_vuelta = plana.____(2, 7, 3)

assert plana.shape == (42,), "Deberían quedar 42 números en una fila"
assert de_vuelta.tolist() == papeleria.tolist(), "La ida y vuelta debe conservar cada dato"
print("Ejercicio 19c superado")
```

**Exercice 20 — Lis l'erreur et corrige-la.** Les ventes quotidiennes de Centre pendant un mois de février de 28 jours qui a commencé un lundi : 4 semaines exactement. Un camarade a voulu les ranger par semaines :

```python
import torch

dias_febrero = [
    2500, 3100, 2200, 2800, 3900, 2400, 1600,
    2600, 3000, 2300, 2700, 4000, 2500, 1500,
    2400, 3200, 2100, 2900, 3800, 2300, 1700,
    2700, 3300, 2400, 3000, 4200, 2600, 1800,
]
febrero = torch.tensor(dias_febrero)
por_semana = febrero.reshape(7, 5)
```

a) Exécute-le, recopie la dernière ligne de l'erreur et explique ces quatre parties : `RuntimeError`, `shape '[7, 5]'`, `is invalid` et `input of size 28`. Ensuite, explique pourquoi 7 × 5 ne peut pas contenir 28 jours.

b) Corrige-le pour que `por_semana` soit de 4 semaines × 7 jours (du lundi au dimanche). Écris le `assert` de la forme et compare les totaux par semaine à un `for` sur la liste `dias_febrero`, comme à l'étape 3 (le jour `d` de la semaine `s` se trouve dans `dias_febrero[s * 7 + d]`).

c) Un autre camarade dit : « moi, je l'ai corrigé avec `reshape(7, 4)`, qui donne aussi 28 ». Ça ne plante pas. Pourquoi est-ce faux par rapport à ce qu'ils voulaient ? Pour le voir, ajoute :

```python
mal = febrero.reshape(7, 4)
print(mal[0])
```

Compare cette sortie avec les sept jours, du lundi au dimanche, de la première semaine.

**Exercice 21 — Attrape le piège.** Cache l'étape 9 avant d'essayer et écris un programme qui :

1. Calcule `semana = papeleria.sum(dim=1)`.
2. Construise correctement le tableau produits × succursales (3 × 2).
3. Contienne un `assert` de la forme et un autre qui vérifie que la ligne des cahiers contient `[9000, 2350]`.
4. Vérifie avec deux `for` que `girada[p, s]` est égal à `semana[s, p]` pour tous les éléments.

Ensuite, pour tester le piège, remplace **seulement** la ligne du tableau pivoté (dans la solution, c'est `girada = semana.T`) par :

```python
girada = semana.reshape(3, 2)
```

Ne change ni les `assert` ni les deux `for`. Exécute-le. Quel `assert` plante ? Pourquoi PyTorch ne t'a-t-il pas prévenu avant ?

---

## Erreurs courantes (et comment t'en rendre compte)

| Erreur | Ce que l'on observe | Comment t'en rendre compte | Ce qu'il faut faire |
|---|---|---|---|
| Demander une forme qui ne tombe pas juste | `RuntimeError: shape '[5, 3]' is invalid for input of size 12` | Le programme plante sur la ligne du `reshape` | Multiplie la forme demandée et compare-la à la taille qu'indique l'erreur |
| Croire que `reshape` a modifié l'original | Après `ventas.reshape(4, 3)`, `ventas.shape` reste `(12,)` | `ventas[3, 2]` plante | Stocke le résultat : `trimestres = ventas.reshape(4, 3)` |
| Utiliser deux `-1` | `RuntimeError: only one dimension can be inferred` | Ça plante alors que le total tomberait juste | Garde un seul `-1` et écris les autres tailles |
| Utiliser `reshape` pour faire pivoter un tableau | Des nombres dans les mauvaises lignes, sans aucune erreur | Le `assert` de la forme passe et celui des valeurs plante | Pour faire pivoter, `.T` ; `reshape` ne fait que couper dans l'ordre de lecture |

---

## Révision pour la prochaine session

La session 6 commence par ces trois questions. Réponds-y sans regarder la leçon :

1. Quand un `reshape` est-il valide ?
2. Que signifie le `-1` dans `reshape` ?
3. Pourquoi faire un `reshape` d'un tableau, ce n'est pas la même chose que le faire pivoter ?

🔁 **La semaine prochaine :** tu opères sur des tenseurs complets, sans `for`. Tu vas multiplier les prix par les quantités de toutes les succursales et de tous les jours d'un seul coup, et tu vas découvrir un piège qui, comme celui d'aujourd'hui, ne signale aucune erreur. Mateo va enfin mettre deux nombres côte à côte : ce qui a été vendu et ce qui a été encaissé.

Ce soir-là, avant de partir, il écrivit à monsieur Ortega : *La banque a-t-elle envoyé un technicien changer le terminal de la succursale Nord en août ?*

La réponse arriva à 9 h 03 le lendemain matin. Une seule ligne :

*« La banque n'a envoyé aucun personnel technique à votre succursale Nord depuis le début de l'année. »*

Mateo agrandit la photo que Fer avait envoyée à sa tante en août. Le nouveau terminal, brillant. Et au bord, coupé par le cadrage, un homme de dos.

**Qui était l'homme au tee-shirt bleu ?**

---

## Solutions (ne les regarde pas avant d'avoir essayé)

**Révision pour commencer.**

1. Il reste `(2, 3)` : les jours disparaissent parce qu'ils ont été additionnés. Chaque nombre est le total de la semaine d'une succursale pour un produit (tu l'as utilisé aujourd'hui à l'étape 9).
2. Parce que `mean` n'accepte que des tenseurs de décimaux (ou de complexes) : avec des entiers, il plante toujours, même si la moyenne tombait juste. La raison de cette règle est celle de la session 4 : une moyenne ne donne presque jamais un entier et PyTorch ne coupe pas les décimales en cachette. On convertit en décimaux avant : `papeleria.float().mean(dim=1)`.
3. En calculant la même chose avec un `for`, dont tu sais déjà qu'il fonctionne, et en comparant avec un `assert` : d'abord la forme, ensuite les valeurs.

**Exercice 18.**

| Forme | Multipliée | Résultat |
|---|---|---|
| `(3, 4)` | 12 | `torch.Size([3, 4])` |
| `(6, 2)` | 12 | `torch.Size([6, 2])` |
| `(5, 2)` | 10 | `RuntimeError: shape '[5, 2]' is invalid for input of size 12` |
| `(2, -1)` | 2 × 6 | `torch.Size([2, 6])` |
| `(-1, 5)` | aucun nombre × 5 ne donne 12 | `RuntimeError: shape '[-1, 5]' is invalid for input of size 12` |
| `(2, 3, 2)` | 12 | `torch.Size([2, 3, 2])` |
| `(12, 1)` | 12 | `torch.Size([12, 1])` |

La dernière est curieuse : 12 lignes d'un seul nombre chacune, une colonne. Contrairement à `(12,)`, qui n'a qu'une seule dimension, `(12, 1)` en a deux. Elle a l'air inutile, mais tu la rencontreras souvent dans les exemples d'IA.

**Exercice 19.** Les trous sont `-1` (ou `14`) et `tolist`. Ce bloc n'est pas à copier : il montre seulement les deux lignes à trous déjà remplies, et les `...` sont les lignes de ton fichier qui ne changent pas :

```text
larga = papeleria.reshape(-1, 3)
...
        a_mano.append(papeleria[s, d].tolist())
```

Nord mercredi est la ligne 9 : les 7 premières lignes (0 à 6) sont Centre, Nord lundi est la 7, mardi la 8 et mercredi la 9. `print(larga[9])` donne :

```
tensor([350, 200, 100])
```

Le calcul est le même qu'à l'étape 3 : succursale × 7 + jour = 1 × 7 + 2 = 9.

Partie c) : les trous sont `flatten` et `reshape` :

```python
plana = papeleria.flatten()
de_vuelta = plana.reshape(2, 7, 3)
```

Sortie : `Ejercicio 19c superado`.

**Exercice 20.**

a) La dernière ligne est :

```
RuntimeError: shape '[7, 5]' is invalid for input of size 28
```

`RuntimeError` est une erreur d'exécution : le code était correctement écrit, mais il a échoué pendant qu'il s'exécutait. `shape '[7, 5]'` est la forme qu'a demandée le camarade. `is invalid` dit qu'elle ne convient pas. `input of size 28` est ce qu'il a : les jours de février. 7 × 5 = 35, et 35, ce n'est pas 28, donc 28 jours ne tiennent pas dans 7 lignes de 5.

b) Remplace la dernière ligne par `por_semana = febrero.reshape(4, 7)` (ou `reshape(-1, 7)` : « de 7 en 7, calcule toi-même combien de semaines »). Le `for` est celui de l'étape 3, avec 4 et 7 au lieu de 4 et 3, sur la liste `dias_febrero` :

```python
por_semana = febrero.reshape(4, 7)

a_mano = []
for s in range(4):
    total = 0
    for d in range(7):
        total = total + dias_febrero[s * 7 + d]
    a_mano.append(total)

assert por_semana.shape == (4, 7), "Deberían ser 4 semanas × 7 días"
assert por_semana.sum(dim=1).tolist() == a_mano, "Los totales por semana no cuadran con el for"
print(por_semana.sum(dim=1))
print("Ejercicio 20 superado")
```

Sortie :

```
tensor([18500, 18600, 18400, 20000])
Ejercicio 20 superado
```

c) `reshape(7, 4)` coupe en morceaux de 4 jours. Sa première ligne est `tensor([2500, 3100, 2200, 2800])` : du lundi au jeudi de la première semaine. Encore un fouillis qui ne plante pas (la ligne suivante commence le vendredi). Que le total tombe juste dit seulement que le `reshape` est possible, pas que c'est celui que tu voulais.

**Exercice 21.**

```python
from papeleria import papeleria

semana = papeleria.sum(dim=1)        # sucursal × producto
girada = semana.T                    # producto × sucursal
assert girada.shape == (3, 2), "Deberían ser 3 productos × 2 sucursales"
assert girada[0].tolist() == [9000, 2350], "El renglón de cuadernos debería traer Centro y Norte"

for p in range(3):
    for s in range(2):
        assert girada[p, s] == semana[s, p], "Girar es cambiar renglones por columnas"
print("Ejercicio 21 superado")
```

Sortie :

```
Ejercicio 21 superado
```

Avec `semana.reshape(3, 2)`, le `assert` de la forme passe et celui des cahiers plante : la première ligne contient `[9000, 5800]`. PyTorch n'a rien signalé parce que, pour lui, il n'y a pas d'erreur : tu as demandé 6 nombres en 3 × 2 et c'est ce qu'il t'a donné. Si les données ont du sens, toi seul le vérifies, avec un `assert` des valeurs.

---

## La suite

*Sources des données techniques : documentation officielle de PyTorch pour [`reshape`](https://docs.pytorch.org/docs/stable/generated/torch.reshape.html), [`flatten`](https://docs.pytorch.org/docs/stable/generated/torch.flatten.html) et [`mean`](https://docs.pytorch.org/docs/stable/generated/torch.mean.html), [`Tensor.numel`](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.numel.html), [`torch.zeros`](https://docs.pytorch.org/docs/stable/generated/torch.zeros.html) et l'attribut `Tensor.T` (dans la [page des tenseurs](https://docs.pytorch.org/docs/stable/tensors.html)), et de TensorFlow pour [`tf.reshape`](https://www.tensorflow.org/api_docs/python/tf/reshape).*

Dans la session 6, tu arrêtes de faire les calculs nombre par nombre. Si tu veux prendre de l'avance, écris avec un `for` comment tu multiplierais par 2 chacun des 42 nombres de la papeterie. La semaine prochaine, tu le feras en une ligne.

---

*J'écris à partir de ce que nous voyons tous les jours chez Hábil, où nous concevons des systèmes pour le secteur bancaire, les assurances et les entreprises au Mexique. Si, dans ton travail ou dans l'entreprise de ta famille, tu rencontres un problème de ce genre, discutons-en : [habil.mx](https://www.habil.mx/fr/)*
