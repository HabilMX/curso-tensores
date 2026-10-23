# Les tenseurs à partir de zéro (2) : ton premier tenseur

*Tu installes PyTorch, tu convertis la papeterie de doña Lupe en un véritable tenseur et tu découvres que la fonction que tu as programmée la semaine dernière était déjà fournie d'origine.*

*Les tenseurs à partir de zéro · leçon A, session 2 sur 8*

*Scénario : les personnages et l'histoire sont inventés. Les données techniques, historiques et les chiffres réels indiquent leur source.*

**Lundi 9 novembre, 19 h 40. Arrière-boutique de la succursale Centre.**

Odeur de papier neuf et de café réchauffé. Doña Lupe baisse le rideau métallique à mi-hauteur, avec ce grincement qu'on entend jusqu'au coin de la rue, et s'assoit à côté de Mateo.

« Le fichier de Nord est arrivé », dit-elle en lui mettant le téléphone sous les yeux. « Celui de la semaine. C'est celui-là qui m'importe. »

Mateo le transfère sur son ordinateur portable. Sept jours, trois produits : cahiers, stylos et sacs à dos. Comme d'habitude. Sur le calendrier accroché au mur, quelqu'un a entouré en rouge le 23 décembre. Il reste quarante-quatre jours.

« Et maintenant, qu'est-ce que tu vas lui faire ? Encore tes petites listes ? »

La semaine dernière, Mateo a écrit la papeterie sous forme de listes dans des listes et a programmé manuellement deux fonctions : l'une qui compte les dimensions et l'autre qui calcule la forme. Elles ont fonctionné. Mais il a aussi découvert quelque chose de gênant : sa fonction `forma` se laisse tromper par une liste bancale. Et elle ne prévient pas.

« Les listes ne me suffisent plus, ma tante. J'ai besoin d'un véritable outil. »

« Et combien ça coûte ? »

« Rien. C'est gratuit. »

Doña Lupe le regarde par-dessus ses lunettes.

« Ce qui ne coûte rien finit toujours par coûter cher, mon petit. »

« Pas celui-là. C'est celui qu'utilisent les gens qui font de l'intelligence artificielle. »

« Alors c'est encore pire. »

Mateo rit et ouvre le terminal. Mais avant d'installer quoi que ce soit, il se fixe une règle : répondre aux trois questions de la semaine dernière sans regarder ses notes. S'il n'y arrive pas, inutile de continuer. Un outil neuf posé sur une base fragile ne fait que produire les erreurs plus vite.

## Révision pour commencer

Avant de commencer, réponds sans regarder la session 1. Les réponses se trouvent à la fin, dans les solutions.

1. Comment sais-tu combien de dimensions possède un ensemble de données, sans compter les crochets ?
2. Que t'indique la forme `(2, 7, 3)`, et combien de nombres y a-t-il au total ?
3. Quelle condition une liste de listes doit-elle remplir pour être un tenseur ?

Si tu as hésité sur l'une d'elles, retourne à la session 1 avant de continuer.

> **Le code de cette session** se trouve dans le dépôt public du cours, dans le dossier `sesion-02` : [github.com/HabilMX/curso-tensores](https://github.com/HabilMX/curso-tensores/tree/main/sesion-02). Écris-le toi-même d'abord, car c'est ainsi qu'on apprend ; le dépôt sert à comparer quand quelque chose ne donne pas le même résultat chez toi. Chaque programme est accompagné de la sortie qu'il doit te donner.

## Objectifs de cette session

À la fin, tu seras capable de faire ces cinq choses. À côté de chacune se trouve l'exercice qui le démontre.

| # | À la fin, tu peux… | Démontré par |
|---|---|---|
| 1 | **Installer** PyTorch dans un environnement dédié et **vérifier** qu'il fonctionne | Étape 1 et exercice 6 |
| 2 | **Convertir** tes listes imbriquées en tenseur avec `torch.tensor` | Exercices 7 et 8 |
| 3 | **Lire** la forme, les dimensions et le nombre total de nombres d'un tenseur (`.shape`, `.ndim`, `.numel()`) et les **vérifier** avec un `assert` par rapport à tes fonctions de la session 1 | Exercices 6, 7 et 8 |
| 4 | **Prédire** le type de nombre d'un tenseur (`.dtype`) et le **choisir** toi-même quand c'est nécessaire | Exercices 6 et 8 |
| 5 | **Lire** l'erreur que donne PyTorch avec une liste bancale et **dire** où se trouve le problème | Exercice 9 |

**Ce dont tu as besoin :** la session 1 terminée, avec ton fichier `dimensiones.py` (les listes de la papeterie et les fonctions `contar_dimensiones` et `forma`). Un ordinateur sous Linux Mint (ou n'importe quel Linux ou macOS avec Python 3.11 ou plus récent ; le cours n'a pas été testé sous Windows), une connexion internet pour télécharger PyTorch et un terminal ouvert.

---

## Pourquoi c'est important

Mateo relit sa fonction `forma`. Un `while`, un `append`, un `tuple`. Elle a fonctionné avec la papeterie. Mais elle a une faille : avec une liste bancale, elle ment. Et elle ment avec un aplomb parfait.

C'est ce qui l'empêche d'être tranquille. Si le fichier de Nord arrivait mal formé, sa fonction lui donnerait une forme sans sourciller, et il la croirait. Et si le fichier contenait un million de nombres, chaque calcul serait une nouvelle boucle `for` écrite manuellement.

Ceux qui travaillent en intelligence artificielle ne programment pas ces fonctions à chaque fois. Ils utilisent une bibliothèque qui range déjà les nombres, connaît leur forme, vérifie qu'ils sont rectangulaires et fait les calculs très vite. La plus utilisée dans la recherche s'appelle **PyTorch**.

Aujourd'hui, tu apprends la première chose que fait toute personne qui travaille en IA quand elle reçoit de nouvelles données : **créer le tenseur et lui demander comment il est.** Quelle forme a-t-il ? Combien de dimensions ? Combien de nombres ? De quel type ? Si ces réponses ne sont pas celles que tu attends, rien de ce qui suit ne se passera bien.

Mateo ne le sait pas encore, mais ce soir, l'une de ces questions va répondre « non ».

---

## Étape 1 — Installer PyTorch

### Une pièce à part pour tes outils

Avant d'installer quoi que ce soit, tu vas créer un **environnement virtuel**. Ça a l'air compliqué, mais ça ne l'est pas : c'est un dossier où tu ranges les bibliothèques d'un projet, séparées de celles du reste de ton ordinateur.

Pense à la caisse à outils de ton oncle plombier : s'il range ses clés dans le tiroir de la cuisine, un jour il ne trouve plus rien. Avec sa propre caisse, personne ne lui déplace rien. L'environnement virtuel est ta caisse : ce que tu y installes ne gêne pas Linux Mint, qui utilise son propre Python pour fonctionner.

Dans le terminal, vérifie d'abord ta version de Python :

```bash
python3 --version
```

Il te faut la 3.11 ou plus récente : c'est ce qu'exige PyTorch depuis sa version 2.15. Linux Mint 22 est livré avec Python 3.12, donc tout va bien.

**Sous Linux Mint et Ubuntu, d'abord ceci.** L'outil qui crée les environnements se trouve dans un paquet séparé, absent d'une installation neuve. Installe-le (il va te demander ton mot de passe) ; si tu l'avais déjà, aucun problème :

```bash
sudo apt install python3-venv
```

Maintenant, crée l'environnement et active-le :

```bash
python3 -m venv ~/tensores
source ~/tensores/bin/activate
```

La première ligne crée la caisse : un dossier appelé `tensores` dans ton dossier personnel (c'est ce que signifie `~`). La deuxième l'ouvre. Tu vas remarquer que ton terminal commence maintenant par `(tensores)`. C'est le signe que tu es à l'intérieur.

**Si tu utilises Windows :** ces deux lignes sont pour Linux et macOS. Sous Windows, depuis ton dossier personnel, crée l'environnement avec `python -m venv tensores` et ouvre-le avec `.\tensores\Scripts\activate` (c'est ce qu'indique la [documentation de Python](https://docs.python.org/3/library/venv.html)). Attention : le cours n'a pas été testé sous Windows ; si une sortie ne correspond pas, compare-la avec celle du dépôt avant de supposer que tu t'es trompé.

**Si tu obtiens cette erreur :** tu as sauté l'étape ci-dessus. Sous Ubuntu 24.04 (la même base que Linux Mint 22), le message ressemble à ceci :

```text
The virtual environment was not created successfully because ensurepip is not
available.  On Debian/Ubuntu systems, you need to install the python3-venv
package using the following command.

    apt install python3.12-venv

You may need to use sudo with that command.  After installing the python3-venv
package, recreate your virtual environment.
```

Remarque que le message propose `python3.12-venv`, avec le numéro de version, alors qu'ici on te demande `python3-venv`, sans numéro. Les deux conviennent : `python3-venv` installe en coulisse celui de ta version. On préfère celui sans numéro, car il continue de fonctionner quand ton système change de version de Python. Exécute `sudo apt install python3-venv` et relance les deux lignes.

Attention : **chaque nouveau terminal démarre avec l'environnement fermé.** Relance `source ~/tensores/bin/activate` et cherche le `(tensores)`. Si tu ne le vois pas, Python ne trouvera pas PyTorch.

### La commande officielle

Avec l'environnement activé, mets à jour l'installateur de paquets, puis installe PyTorch avec la commande officielle de son site pour Linux, sans carte graphique :

```bash
pip3 install --upgrade pip
pip3 install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cpu
```

Ce que dit cette commande, morceau par morceau :

- `torch==2.14.1`, c'est PyTorch, dans la version exacte avec laquelle ce cours a été écrit. On la fixe pour que tes sorties correspondent à celles de la leçon : avec une autre version, les nombres peuvent changer légèrement et tu ne saurais pas si l'erreur vient de toi.
- `torch`, c'est PyTorch. Le nom vient de **Torch**, une bibliothèque antérieure écrite dans un autre langage (Lua), dont la communauté a fourni une partie des personnes qui l'ont créé.
- `torchvision` est un paquet compagnon pour les images. Il fait partie de la commande officielle et ne gêne pas.
- `--index-url https://download.pytorch.org/whl/cpu` indique à `pip` où télécharger : l'index de l'édition **CPU uniquement**. Sans cette partie, sous Linux, il téléchargerait l'édition pour cartes graphiques NVIDIA, beaucoup plus lourde, dont tu n'as pas besoin dans ce cours.

C'est un gros téléchargement ; prends ton temps.

### Vérifier l'installation

Crée un fichier `verifica.py`, dans le même dossier que `dimensiones.py` :

```python
import torch
print(torch.__version__, torch.cuda.is_available())
```

Exécute-le avec `python3 verifica.py` :

```
2.14.1 False
```

La première information est la version de PyTorch. Si tu obtiens `2.14.1+cpu`, c'est bon aussi : le `+cpu` est l'étiquette de l'édition CPU uniquement. La deuxième demande si PyTorch peut utiliser une carte graphique NVIDIA (ce que PyTorch appelle CUDA). Comme tu as installé l'édition CPU uniquement, le `False` est exactement ce qu'on attend, même si ton ordinateur portable a une carte graphique.

**Qu'est-ce qu'un GPU ?** C'est la carte graphique : elle est née pour dessiner des millions de points à l'écran en même temps dans les jeux vidéo. Le CPU, c'est une poignée de chefs très habiles ; le GPU, une équipe de milliers de commis qui ne savent que hacher, mais qui hachent tous en même temps, et multiplier de grands tenseurs, c'est du hachage pur. Tout ce cours s'exécute sur le CPU d'un ordinateur portable ordinaire.

> Mateo voit le `False` et rit. Doña Lupe, qui ne quitte pas l'écran des yeux, demande si c'est grave.
> « Ça veut dire que PyTorch ne va pas utiliser de carte graphique sur mon ordinateur portable, ma tante. »
> « Ah, tant mieux. Comme ça, tu étudies. »

---

## Étape 2 — Du nombre et de la liste au tenseur

> **20 h 25.** Mateo a le fichier de Nord à un clic. Il ne l'ouvre pas. Il va d'abord s'entraîner avec les données de la semaine dernière, celles dont il sait qu'elles sont correctes. Si quelque chose échoue avec de bonnes données, l'erreur vient de lui. Si ça échoue avec des données nouvelles, l'erreur est dans les données. Mélanger les deux est le moyen le plus rapide de ne rien comprendre.

Il n'y a qu'une seule fonction à apprendre aujourd'hui pour créer des tenseurs : `torch.tensor`. Tu lui donnes tes listes et elle te renvoie un tenseur avec les mêmes nombres.

Crée `primer_tensor.py` dans le même dossier que `dimensiones.py`. La ligne `from dimensiones import ...` récupère les listes que tu as écrites la semaine dernière, sans avoir à les recopier :

```python
import torch
from dimensiones import total, por_dia

t_total = torch.tensor(total)
t_dia = torch.tensor(por_dia)

print(t_total)
print(t_dia)
print(t_total.shape, t_dia.shape)
```

Exécute-le avec `python3 primer_tensor.py` :

```
tensor(18500)
tensor([2500, 3100, 2200, 2800, 3900, 2400, 1600])
torch.Size([]) torch.Size([7])
```

Trois choses à lire dans cette sortie :

- PyTorch enveloppe tes nombres dans `tensor(...)` : ce n'est plus une liste Python, c'est un tenseur.
- Le scalaire s'affiche comme `tensor(18500)`, sans crochets. Zéro dimension, comme dans la session 1.
- `.shape` est la **forme**. Pour le scalaire, on obtient `torch.Size([])`, vide, parce qu'il n'a aucune direction. Pour la semaine, on obtient `torch.Size([7])` : sept jours.

Remarque : `.shape` ne prend pas de parenthèses. Ce n'est pas une fonction, c'est une information que le tenseur a déjà en mémoire, comme l'étiquette de taille d'une chemise : tu ne la mesures pas, tu la lis.

> « Et alors ? » dit doña Lupe en montrant le `18500`. « C'est la même chose que ce que j'avais déjà noté. »
> « C'est tout l'intérêt, ma tante. Que ce soit la même chose. Maintenant, l'ordinateur le sait aussi. »

---

## Étape 3 — La papeterie au complet

Maintenant, la feuille avec ses deux onglets. Crée `papeleria.py` dans le même dossier ; ce fichier va contenir le tenseur que tu utiliseras dans les sessions suivantes :

```python
import torch
from dimensiones import centro, norte

papeleria = torch.tensor([centro, norte])

# Esto corre al abrir papeleria.py, no cuando otra sesión lo importa.
if __name__ == "__main__":
    print(papeleria)
    print(papeleria.shape)
    print(papeleria.ndim)
    print(papeleria.numel())
    print(papeleria.dtype)
```

Sortie :

```
tensor([[[1200,  800,  500],
         [1500,  900,  700],
         [1000,  700,  500],
         [1300,  900,  600],
         [2000, 1200,  700],
         [1200,  800,  400],
         [ 800,  500,  300]],

        [[ 300,  200,  100],
         [ 400,  250,  150],
         [ 350,  200,  100],
         [ 300,  150,  100],
         [ 500,  300,  200],
         [ 300,  200,  100],
         [ 200,  100,  100]]])
torch.Size([2, 7, 3])
3
42
torch.int64
```

La condition `if __name__ == "__main__":` réserve ces cinq vérifications au moment où tu exécutes ce fichier directement. Les sessions suivantes pourront utiliser `from papeleria import papeleria` sans réafficher tout le tableau.

PyTorch aligne les colonnes et laisse une ligne vide entre l'onglet Centre et l'onglet Nord : la même feuille avec onglets que la semaine dernière. Lis maintenant les quatre réponses :

| Tu lui demandes | Tu écris | Il te répond | Cela signifie |
|---|---|---|---|
| Quelle forme as-tu ? | `papeleria.shape` | `torch.Size([2, 7, 3])` | 2 succursales, 7 jours, 3 produits |
| Combien de dimensions ? | `papeleria.ndim` | `3` | succursale, jour et produit |
| Combien de nombres contiens-tu ? | `papeleria.numel()` | `42` | 2 × 7 × 3 |
| De quel type sont-ils ? | `papeleria.dtype` | `torch.int64` | des entiers (on le voit à l'étape 4) |

Les noms viennent de l'anglais : `shape` signifie « forme » ; `ndim`, « nombre de dimensions » ; `numel`, *number of elements*, « nombre d'éléments » ; `dtype`, *data type*, « type de donnée ». Remarque un détail : `numel()` prend des parenthèses, parce que c'est une fonction qui compte. Les trois autres, non.

### `torch.Size([2, 7, 3])`, c'est ta `forma` de la semaine dernière

La session 1 s'est terminée avec cette sortie de ta fonction :

```
papeleria 3 (2, 7, 3)
```

Et aujourd'hui, PyTorch te dit `torch.Size([2, 7, 3])`. **C'est exactement la même chose**, simplement habillée autrement. `torch.Size` est la manière dont PyTorch enregistre une forme, et en interne il se comporte comme un tuple. C'est pourquoi tu peux le comparer directement à `(2, 7, 3)`.

Ton `while` avec `append` et `.shape` font le même travail. La différence : celle de PyTorch, comme tu le verras à l'étape 5, ne se laisse pas tromper par une liste bancale.

### Le vérifier avec `assert` : d'abord la forme, ensuite la valeur

Ne crois personne, pas même PyTorch : vérifie. Crée `compara.py` :

```python
import torch
from dimensiones import centro, norte, forma, contar_dimensiones

papeleria_lista = [centro, norte]
papeleria = torch.tensor([centro, norte])

# 1) Primero la forma: PyTorch contra tus funciones de la sesión 1
print(papeleria.shape, forma(papeleria_lista))
print(papeleria.ndim, contar_dimensiones(papeleria_lista))
assert papeleria.shape == (2, 7, 3), "La papelería debería ser 2 sucursales × 7 días × 3 productos"
assert papeleria.shape == forma(papeleria_lista), "PyTorch y tu función deberían dar la misma forma"
assert papeleria.ndim == contar_dimensiones(papeleria_lista), "Deberían contar las mismas dimensiones"

# 2) Después el total de números, contra un for
cuenta = 0
for pestana in papeleria_lista:
    for renglon in pestana:
        for numero in renglon:
            cuenta = cuenta + 1
print(papeleria.numel(), cuenta)
assert papeleria.numel() == cuenta, "Deberían ser los mismos números"

# 3) Al final, el valor: de regreso a listas, ¿quedó igual?
assert papeleria.tolist() == papeleria_lista, "Los datos no deberían cambiar al convertirlos"
print("Todo cuadra")
```

Sortie :

```
torch.Size([2, 7, 3]) (2, 7, 3)
3 3
42 42
Todo cuadra
```

Trois vérifications, dans cet ordre :

1. **La forme.** C'est le `assert` que tu écriras dans chaque exercice du cours. Si la forme est fausse, tout le reste est faux.
2. **Le total par rapport à un `for`.** Les trois boucles `for` imbriquées comptent un par un ce que `numel()` compte en un mot. Si les deux donnent 42, rien ne s'est perdu ni ajouté.
3. **La valeur.** `.tolist()` fait le trajet retour : il reconvertit le tenseur en listes. La liste s'appelle `papeleria_lista` pour ne pas entrer en conflit avec le tenseur `papeleria` ; si elles sont égales, les nombres n'ont pas changé en route.

Si tu remplaces `(2, 7, 3)` par `(7, 2, 3)`, le premier `assert` plante avec son message, comme dans la session 1. L'ordre compte : 2 onglets de 7 lignes, ce n'est pas la même chose que 7 onglets de 2.

> « Todo cuadra. » Mateo le lit deux fois. Avec des données connues, l'outil ne se trompe pas et ne le laisse pas se tromper. Il peut désormais lui faire confiance. Il reste une chose avant le fichier de Nord : savoir quel genre de nombres il va contenir.

---

## Étape 4 — Le type de nombre : `dtype`

Tous les nombres d'un tenseur sont du même type. Pas question de mélanger, comme dans une liste Python, un entier avec un texte. Ce type unique s'appelle **dtype**.

Pourquoi ? Parce qu'ainsi l'ordinateur sait exactement combien d'espace occupe chaque nombre et les aligne en rangée sans les vérifier un par un, comme des boîtes de même taille sur une étagère.

Pour commencer, deux types te suffisent :

| dtype | Ce qu'il contient | Exemple |
|---|---|---|
| `torch.int64` | des entiers : pièces, pesos sans centimes, comptages | 12 cahiers |
| `torch.float32` | des nombres décimaux : mesures, moyennes, pourcentages | 9.7 de moyenne |

Le nombre dans le nom indique l'espace qu'occupe chacun, en bits : 64 ou 32. Tu n'as pas besoin d'en savoir plus pour l'instant.

Un avertissement pour le jour où tu travailleras avec de l'argent réel : `float32` enregistre les décimales de façon approximative (le 0.1 devient 0.10000000149011612). Pour des montants exacts, les systèmes enregistrent les centimes sous forme d'entiers : 35,50 $ est enregistré comme `3550`. Dans l'exemple ci-dessous, on utilise des prix avec décimales uniquement pour voir quel `dtype` en sort.

Crée `tipos.py` :

```python
import torch

piezas = torch.tensor([12, 30, 7])
precios = torch.tensor([35.5, 12.0, 450.0])
mezcla = torch.tensor([12, 30.5, 7])

print(piezas, piezas.dtype)
print(precios, precios.dtype)
print(mezcla, mezcla.dtype)
```

Sortie :

```
tensor([12, 30,  7]) torch.int64
tensor([ 35.5000,  12.0000, 450.0000]) torch.float32
tensor([12.0000, 30.5000,  7.0000]) torch.float32
```

La règle que suit PyTorch :

- **Si tous sont des entiers, on obtient `int64`.** C'est pour cela que ta papeterie est sortie en `torch.int64` : doña Lupe enregistre des pesos sans centimes.
- **Si au moins un a un point décimal, on obtient `float32`,** et les entiers deviennent aussi décimaux : le 12 se transforme en `12.0000`. Un seul 30.5 suffit pour que toute la rangée change.

> « Moi, je ne note pas les centimes », dit doña Lupe. « Les centimes, c'est l'affaire de la banque. »
> « C'est pour ça que ta feuille est sortie en `int64`, ma tante. Des entiers, comme toi. »
> « Tu es en train de me traiter de carrée ? »
> « Je te dis que tu es exacte. »

### Choisir le type toi-même

Parfois, tu veux des décimales même si tes données sont entières. Par exemple, pour calculer des moyennes, comme tu le verras dans la session 4. Pour cela, `torch.tensor` accepte une deuxième information :

```python
import torch
from dimensiones import por_dia

dia_dec = torch.tensor(por_dia, dtype=torch.float32)
print(dia_dec)
print(dia_dec.dtype)
```

Sortie :

```
tensor([2500., 3100., 2200., 2800., 3900., 2400., 1600.])
torch.float32
```

Le point après chaque nombre (`2500.`) est l'indice qu'il s'agit désormais de décimaux. Mêmes nombres, autre type.

---

## Étape 5 — La liste bancale : PyTorch dit non

**22 h 15.** Cette fois, ça y est. Mateo copie les sept lignes du fichier de Nord dans une liste, écrit la ligne habituelle, `torch.tensor(...)`, et appuie sur Entrée.

Le terminal se remplit de lettres rouges.

« Tu l'as déjà cassé ? » demande doña Lupe.

« Non, ma tante. Il a refusé. »

Mateo parcourt le fichier, ligne par ligne, le doigt sur l'écran. Lundi : trois nombres. Mardi : deux. Mercredi : trois. Il en manque un au mardi.

Avant de tirer des conclusions, il reproduit le problème en miniature. Dans la session 1, tu as vu que ta fonction `forma` se laissait tromper par cette liste et que tu devais la vérifier avec `es_rectangular` :

```python
chueca = [[1, 2, 3], [4, 5]]
```

Donnons-la à PyTorch. Crée `chueca.py` :

```python
import torch

chueca = [[1, 2, 3], [4, 5]]
t = torch.tensor(chueca)
```

Exécute-le :

```
Traceback (most recent call last):
  File ".../chueca.py", line 4, in <module>
    t = torch.tensor(chueca)
        ^^^^^^^^^^^^^^^^^^^^
ValueError: expected sequence of length 3 at dim 1 (got 2)
```

Le même rouge qu'avec le fichier de Nord. PyTorch refuse de créer le tenseur. L'important, c'est la dernière ligne, et elle se lit mot à mot :

| Il dit | Traduction | Ce que cela signifie ici |
|---|---|---|
| `ValueError` | erreur de valeur | la donnée que tu lui as passée ne convient pas pour cela |
| `expected sequence of length 3` | j'attendais une liste de longueur 3 | la première ligne mesurait 3, donc toutes devraient mesurer 3 |
| `at dim 1` | à la dimension 1 | la dimension intérieure : les lignes. Celle de l'extérieur est la 0 |
| `(got 2)` | j'en ai reçu 2 | il a trouvé une ligne qui mesure 2 |

Remarque le `dim 1`. Les dimensions sont numérotées à partir de 0, comme les indices d'une liste : celle de l'extérieur est la 0, la suivante la 1. Dans la papeterie, la dimension 0 correspond aux succursales, la 1 aux jours et la 2 aux produits. Tu vas beaucoup utiliser cette numérotation dans la session 3.

C'est ton `es_rectangular`, en mieux : il vérifie toutes les boîtes de tous les niveaux et te dit où se trouve le problème. C'est pourquoi le tenseur que tu crées avec `torch.tensor` est toujours rectangulaire. (PyTorch possède des tenseurs particuliers pour les données disparates, mais nous ne les utilisons pas dans ce cours.)

Ta fonction de la semaine dernière aurait avalé le fichier de Nord sans rien dire. PyTorch, non : il préfère planter plutôt que te mentir.

**Comment on corrige :** pas en complétant avec un nombre inventé. Mateo pourrait mettre un zéro au mardi et continuer. Le tenseur serait parfait, et il mentirait : un zéro dit « rien n'a été vendu » ; un trou dit « je ne sais pas ». Ce n'est pas la même chose. Demande à la personne qui t'a fourni les données laquelle manque. S'il manque les sacs à dos du mardi, le tenseur ne se corrige pas avec un zéro ; il se corrige en posant la question.

> « Qui prépare ce fichier, ma tante ? »
> « C'est la succursale qui l'envoie. Chaque lundi. Il n'était jamais arrivé incomplet. »
> Jamais. Mateo reste les yeux fixés sur le mardi.

---

## Encadré : le combat de deux géants (PyTorch et TensorFlow)

Pendant que PyTorch se téléchargeait, Mateo s'est posé une question : si cet outil est si bon, pourquoi entend-il partout un autre nom ? Tu vas entendre deux noms tout le temps, et derrière eux il y a un combat entre deux géants. Les données proviennent de la recherche documentaire du cours, datées.

**Le premier arrivé.** **TensorFlow** appartient à Google. Google l'a publié en open source le **9 novembre 2015**, avec l'autorisation pour quiconque de l'utiliser : onze ans jour pour jour avant la nuit où Mateo a installé son rival. La version en vigueur en octobre 2026 est la **2.21.0**, publiée le 6 mars 2026.

**Le challenger.** **PyTorch** est né chez Facebook (aujourd'hui Meta), dans son laboratoire de recherche. Ses premières versions sur GitHub datent de **septembre 2016**, dix mois plus tard, et son lancement officiel, de janvier 2017. Depuis **septembre 2022**, il n'appartient plus à une seule entreprise : c'est la PyTorch Foundation, au sein de la Linux Foundation, qui en prend soin, et ses membres fondateurs étaient AMD, AWS (d'Amazon), Google Cloud, Meta, Microsoft Azure et NVIDIA. Remarque un nom dans cette liste : Google Cloud, de la même maison que le rival, siège là aussi. La version en vigueur au moment où nous écrivons cette session (octobre 2026) est la **2.14.1**, publiée le 30 septembre 2026. Si une version plus récente s'est installée chez toi, c'est bien.

**Qui est en train de gagner ?** Ça dépend du ring.

- **Dans les modèles que les gens publient, PyTorch, de loin.** Sur Hugging Face, le site où l'on publie des modèles d'IA, il y avait le 18 octobre 2026 258 777 modèles avec l'étiquette PyTorch contre 17 469 avec celle de TensorFlow : environ 15 pour un. Cela mesure ce que les gens mettent en ligne, pas ce qu'ils utilisent dans leur travail, mais l'écart est net. Et la bibliothèque de modèles la plus populaire de Hugging Face, Transformers, a annoncé que dans sa version 5 elle abandonne TensorFlow pour ne garder que PyTorch.
- **Chez les développeurs, presque à égalité.** Dans l'enquête Stack Overflow de 2024, 10,6 % des répondants ont déclaré utiliser PyTorch et 10,1 %, TensorFlow. L'enquête demande ce que chacun utilise, pas ce qui tourne en production. TensorFlow est arrivé plus tôt dans les entreprises et reste présent dans beaucoup d'entre elles, ainsi que sur les téléphones portables.

*Sources : [annonce de Google sur TensorFlow (2015)](https://blog.google/innovation-and-ai/products/tensorflow-smarter-machine-learning-for/) ; versions sur PyPI : [tensorflow](https://pypi.org/project/tensorflow/) et [torch](https://pypi.org/project/torch/) ; versions de PyTorch sur [GitHub](https://github.com/pytorch/pytorch/releases) ; [Linux Foundation, 12 septembre 2022](https://www.linuxfoundation.org/press/press-release/meta-transitions-pytorch-to-the-linux-foundation) ; [modèles PyTorch sur Hugging Face](https://huggingface.co/models?library=pytorch), consulté le 18 octobre 2026 (le chiffre change chaque jour) ; [enquête Stack Overflow 2024](https://survey.stackoverflow.co/2024/technology).*

**Une différence que tu remarqueras si un jour tu utilises les deux :** avec les mêmes entiers, PyTorch crée du `int64` et TensorFlow, du `int32`.

| | PyTorch | TensorFlow |
|---|---|---|
| Créer un tenseur | `torch.tensor([1, 2, 3])` | `tf.constant([1, 2, 3])` |
| Entier par défaut | `int64` | `int32` |
| Avec décimales | `float32` | `float32` |

**Lequel utiliser ?** Pour apprendre et lire du code de recherche, PyTorch. Si un jour ton travail utilise TensorFlow, tu l'apprends en un après-midi : forme, dimensions et type fonctionnent de la même façon.

---

👉 **Si tu viens de la gestion :** le `dtype`, c'est le format que tu donnes à une colonne dans ton tableur : « nombre entier » ou « monétaire avec deux décimales ». Et l'erreur de la liste bancale, c'est le rapport de la succursale qui arrive avec une colonne en moins : on ne peut pas l'additionner aux autres tant que quelqu'un n'a pas demandé ce qui manque. Quand un analyste te dira « le modèle n'a pas tourné parce que les données n'avaient pas la bonne forme », tu sauras de quoi il parle.

---

## Exercices de la session 2

Les solutions se trouvent à la fin de cette livraison. Travaille dans le même dossier que ton `dimensiones.py` et avec l'environnement activé.

**Exercice 6 — Prédis avant d'exécuter.** Écris sur une feuille la forme, les dimensions, le type et le nombre total de nombres de chaque tenseur. Exécute ensuite et compare.

```python
import torch

temperatura = torch.tensor(31)
calificaciones = torch.tensor([[9, 8, 10], [7, 9, 8], [10, 10, 9], [6, 8, 7]])
foto = torch.tensor([[[255, 0, 0], [0, 255, 0]], [[0, 0, 255], [255, 255, 255]]])
promedios = torch.tensor([9.0, 8.0, 9.7, 7.0])

for nombre, t in [("temperatura", temperatura), ("calificaciones", calificaciones),
                  ("foto", foto), ("promedios", promedios)]:
    print(nombre, t.shape, t.ndim, t.dtype, t.numel())
```

Ce sont les mêmes données que l'exercice 1 de la session 1, plus une avec des décimales. Indice : combien de nombres contient un scalaire ?

**Exercice 7 — Complète les trous.** L'onglet Nord seul, sous forme de tenseur. Remplis les `____` et exécute. Si le `print` final s'affiche, tu as réussi.

```python
import torch
from dimensiones import norte

norte_t = torch.____(norte)

assert norte_t.shape == ____, "Norte es una pestaña: 7 días × 3 productos"
assert norte_t.ndim == ____, "Una pestaña es una tabla"
assert norte_t.numel() == ____, "7 × 3"
assert norte_t.dtype == ____, "Son pesos enteros"
assert norte_t.tolist() == norte, "Los datos no deberían cambiar"
print("Ejercicio 7 superado")
```

**Exercice 8 — Tes propres données, maintenant en tenseur.** Reprends les notes que tu as écrites dans l'exercice 4 de la session 1 (élèves, matières et partiels) et :

a) Convertis-les en tenseur et écris le `assert` de leur forme par rapport à ta fonction `forma`.
b) Écris le `assert` de leurs dimensions par rapport à `contar_dimensiones`.
c) Compte leurs nombres avec des boucles `for` imbriquées et compare avec un `assert` par rapport à `numel()`.
d) Crée une deuxième version avec `dtype=torch.float32` et vérifie avec un `assert` que le type a changé, mais pas la forme.

Termine par `print("Ejercicio 8 superado")`.

**Exercice 9 — Trouve la bancale, maintenant avec PyTorch.** Avant d'exécuter, prédis lesquelles plantent et quel numéro de `dim` l'erreur va indiquer. Ensuite, teste chacune séparément avec `torch.tensor(...)`.

```python
a = [[1, 2], [3, 4], [5, 6]]
b = [[1, 2, 3], [4, 5, 6], [7, 8]]
c = [[[1, 2], [3, 4]], [[5, 6], [7]]]
d = [[1, 2], [3, 4, 5]]
```

Pour celles qui plantent, réponds : pourquoi, dans `d`, l'erreur dit-elle qu'elle attendait 2 et a reçu 3, alors que c'est la deuxième ligne qui « a un élément de trop » ? Et corrige `b` en lui ajoutant le nombre qui manque (suppose que c'est un 9).

---

## Erreurs courantes (et comment t'en rendre compte)

| Erreur | Ce que l'on observe | Comment t'en rendre compte | Ce qu'il faut faire |
|---|---|---|---|
| Oublier d'activer l'environnement | `ModuleNotFoundError: No module named 'torch'` | Ton terminal ne commence pas par `(tensores)` | Exécute `source ~/tensores/bin/activate` dans chaque nouveau terminal |
| Appeler ton fichier `torch.py` | `AttributeError: partially initialized module 'torch' has no attribute 'tensor' (most likely due to a circular import)` | Ça plante à la première ligne qui utilise `torch.` | Python a importé ton fichier à la place de PyTorch. Renomme-le, par exemple en `primer_tensor.py` |
| Mettre des parenthèses à `.shape` | `TypeError: 'torch.Size' object is not callable` | Tu as écrit `t.shape()` | `.shape`, `.ndim` et `.dtype` s'écrivent sans parenthèses ; seul `.numel()` en prend |
| Oublier les parenthèses de `numel` | Quelque chose comme `<built-in method numel of Tensor object at 0x...>` s'affiche au lieu d'un nombre | Le `print` n'affiche pas un nombre | Écris `t.numel()`, avec parenthèses |
| Écrire `torch.Tensor` avec une majuscule | `torch.Tensor([1, 2, 3])` donne `tensor([1., 2., 3.])`, des décimaux | Le `dtype` sort en `torch.float32` avec des données entières | Utilise `torch.tensor`, en minuscules : il respecte le type de tes données |

---

## Révision pour la prochaine session

La session 3 commence par ces trois questions. Réponds-y sans regarder la leçon :

1. Que t'indique `t.shape` et comment le vérifies-tu avec un `assert` ?
2. Pourquoi `torch.tensor([[1, 2, 3], [4, 5]])` échoue-t-il ?
3. Quel `dtype` sort de `torch.tensor([1, 2, 3])` et lequel de `torch.tensor([1.0, 2.0])` ?

🔁 **La semaine prochaine :** tu as déjà la papeterie sous forme de tenseur. Tu vas maintenant en extraire des morceaux selon n'importe lequel de ses axes : seulement l'onglet Nord, seulement les vendredis des deux succursales, seulement les sacs à dos de toute la semaine. Et tu verras qu'une photo se découpe exactement de la même façon.

Il est plus de onze heures quand Mateo referme son ordinateur portable. Doña Lupe a déjà compté la caisse et attend près du rideau, les clés à la main. Sur le téléphone, le fichier de Nord est toujours ouvert, avec son mardi à deux nombres. Ça ne ressemble pas à une simple faute de frappe : c'est la seule ligne bancale en des semaines de fichiers bien droits. Et elle est arrivée justement la semaine où la banque a suspendu le terminal.

Qui laisse une journée à moitié remplie ?

---

## Solutions (ne les regarde pas avant d'avoir essayé)

**Révision pour commencer.**

1. Tu comptes de combien d'informations tu as besoin pour désigner un seul nombre. Dans la papeterie, il te faut la succursale, le jour et le produit : trois informations, trois dimensions. Et si tu ne veux pas compter toi-même, `contar_dimensiones` le fait (et à partir d'aujourd'hui, `.ndim`).
2. Qu'il y a 2 éléments dans la première dimension (succursales), 7 dans la deuxième (jours) et 3 dans la troisième (produits). Au total, 2 × 7 × 3 = 42 nombres.
3. Qu'elle soit rectangulaire : toutes les boîtes d'un même niveau doivent mesurer la même chose. Tous les onglets avec 7 lignes, toutes les lignes avec 3 colonnes.

**Exercice 6.**

```
temperatura torch.Size([]) 0 torch.int64 1
calificaciones torch.Size([4, 3]) 2 torch.int64 12
foto torch.Size([2, 2, 3]) 3 torch.int64 12
promedios torch.Size([4]) 1 torch.float32 4
```

Le piège, c'était le scalaire : sa forme est vide et il a zéro dimension, mais il contient **un** nombre. `promedios` sort en `float32` parce qu'il contient des décimales. Remarque aussi que `calificaciones` et `foto` contiennent les mêmes 12 nombres avec des formes différentes : le total ne te donne pas la forme, mais la forme te donne le total.

**Exercice 7.**

```python
norte_t = torch.tensor(norte)

assert norte_t.shape == (7, 3), "Norte es una pestaña: 7 días × 3 productos"
assert norte_t.ndim == 2, "Una pestaña es una tabla"
assert norte_t.numel() == 21, "7 × 3"
assert norte_t.dtype == torch.int64, "Son pesos enteros"
```

Sortie :

```
Ejercicio 7 superado
```

**Exercice 8.** Une réponse possible, avec la liste `parciales` (élève × matière × partiel) de la solution de l'exercice 4 de la session 1 :

```python
import torch
from dimensiones import forma, contar_dimensiones

# alumno × materia × parcial
parciales = [
    [[8, 9, 9], [7, 8, 9], [9, 9, 10], [6, 7, 8]],
    [[10, 9, 10], [9, 9, 9], [8, 8, 9], [9, 10, 10]],
    [[7, 7, 8], [8, 8, 8], [6, 7, 7], [8, 9, 9]],
]

p = torch.tensor(parciales)
print(p.shape, p.ndim, p.numel(), p.dtype)
assert p.shape == forma(parciales), "PyTorch y tu función deberían coincidir"
assert p.ndim == contar_dimensiones(parciales)

cuenta = 0
for alumno in parciales:
    for materia in alumno:
        for calificacion in materia:
            cuenta = cuenta + 1
assert p.numel() == cuenta

p_dec = torch.tensor(parciales, dtype=torch.float32)
print(p_dec.dtype)
assert p_dec.dtype == torch.float32
assert p_dec.shape == p.shape, "Cambiar el dtype no cambia la forma"
print("Ejercicio 8 superado")
```

Sortie :

```
torch.Size([3, 4, 3]) 3 36 torch.int64
torch.float32
Ejercicio 8 superado
```

Si tes données ont un autre ordre ou d'autres tailles, ta forme sera différente, et c'est normal : ce qui compte, c'est que les `assert` passent.

**Exercice 9.**

- `a` fonctionne : `torch.Size([3, 2])`.
- `b` plante : `ValueError: expected sequence of length 3 at dim 1 (got 2)`. La dernière ligne mesure 2 et les autres, 3.
- `c` plante : `ValueError: expected sequence of length 2 at dim 2 (got 1)`. Ici, le problème se trouve plus à l'intérieur, dans la dimension 2 : la dernière paire, `[7]`, ne contient qu'un seul nombre.
- `d` plante : `ValueError: expected sequence of length 2 at dim 1 (got 3)`. PyTorch prend la **première** ligne comme règle. La première mesure 2, donc c'est la deuxième, qui mesure 3, qui « est fausse », même s'il te semble que c'est à la première qu'il en manque un. C'est pourquoi l'erreur t'indique où la règle a été enfreinte, mais décider quelle donnée est la bonne, c'est à toi de le faire, avec la personne qui t'a fourni les données.

`b` corrigé :

```python
b = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
print(torch.tensor(b).shape)
```

```
torch.Size([3, 3])
```

---

## La suite

Dans la session 3, tu cesses de regarder le tenseur complet et tu commences à en extraire des morceaux. Si tu veux prendre de l'avance, ouvre ton `papeleria.py` et prépare le tenseur `papeleria` : c'est la première chose que tu vas utiliser.

---

*J'écris à partir de ce que nous voyons tous les jours chez Hábil, où nous concevons des systèmes pour le secteur bancaire, les assurances et les entreprises au Mexique. Si, dans ton travail ou dans l'entreprise de ta famille, tu rencontres un problème de ce genre, discutons-en : [habil.mx](https://www.habil.mx/fr/)*
