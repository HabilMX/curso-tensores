# Les tenseurs à partir de zéro (8) : les tenseurs à la banque, dans l'assurance et au magasin

*Comment lire un cas réel et dire quelle forme a son tenseur, pourquoi, dans une compétition de prévision avec les données d'une chaîne de grande distribution américaine, c'est une méthode qui n'était pas un réseau de neurones qui a gagné, et quelles précautions ne sont pas facultatives.*

*Les tenseurs à partir de zéro · leçon B, session 8 sur 8*

*Scénario : les personnages et l'histoire sont inventés. Les données techniques, historiques et les chiffres réels indiquent leur source.*

**Mercredi 23 décembre, 9 h 40. Agence de la banque, deuxième étage.**

Le dernier jour du délai.

La salle d'attente sent le café réchauffé et le pin en plastique. Quelqu'un a posé un petit sapin à côté de la fontaine à eau. Il a trois boules, et l'une est en train de tomber.

Doña Lupe a son sac sur les genoux, serré à deux mains, comme si quelqu'un allait le lui prendre. Mateo a l'ordinateur portable. Tout est dedans : la feuille de Nord, les totaux, le changement de terminal en août, les débits sans marchandise et le détecteur de la semaine dernière.

« S'ils nous disent non », dit doña Lupe, « je ferme Nord le 2 janvier. J'y ai déjà réfléchi. »

« Ils ne vont pas nous dire non. »

« C'est ce que disait ton oncle de sa prime de fin d'année, et regarde. »

Mateo ne répond pas. Il repasse dans sa tête ce qui l'inquiète depuis neuf jours : le détecteur avait juste 91 % du temps. Et des fraudes lui échappaient. Quelle banque va croire un modèle fait sur un ordinateur portable, sur la table d'une papeterie ?

« Madame Herrera ? »

Monsieur Ortega est plus jeune qu'il ne paraissait au téléphone. Cravate rouge, badge qui pend, des chaussures pressées. Il les fait entrer dans un bureau dont la fenêtre donne sur le parking.

« J'ai vingt minutes », dit-il. « Montrez-moi ce que vous apportez. »

Mateo ouvre l'ordinateur. Il lui montre la chute de Nord en août. Les débits au petit matin. Les montants qui ne correspondent à aucune marchandise. Le détecteur.

Ortega ne dit rien pendant un long moment. Puis il tourne son propre écran vers eux.

« Vous savez ce que c'est ? »

À l'écran, il y a une liste. Des débits. Beaucoup. Certains marqués en rouge.

« C'est la même chose que ce que vous avez fait », dit Ortega. « Sauf que nous, on le fait avec des millions d'achats par jour. Asseyez-vous, jeune homme. Je vais vous montrer à quoi ça ressemble en grand. »

Mateo regarde l'horloge au mur. Les vingt minutes viennent de commencer à courir.

## Révision pour commencer

Réponds sans regarder la session 7. Les réponses se trouvent dans les solutions.

1. Quelle forme a le tenseur d'entrée du détecteur et que représente chaque axe ?
2. Que mesure la perte et pourquoi doit-elle baisser ?
3. Pourquoi avoir raison 91 % du temps peut-il ne pas suffire pour une banque ?

> **Le code de cette session** se trouve dans le dépôt public du cours, dans le dossier `sesion-08` : [github.com/HabilMX/curso-tensores](https://github.com/HabilMX/curso-tensores/tree/main/sesion-08). Écris-le toi-même d'abord, car c'est ainsi qu'on apprend ; le dépôt sert à comparer quand quelque chose ne donne pas le même résultat chez toi. Chaque programme est accompagné de la sortie qu'il doit te donner.

## Objectifs de cette session

| # | À la fin, tu peux… | Démontré par |
|---|---|---|
| 1 | **Lire** un cas réel et **écrire** la forme de son tenseur, en disant ce que représente chaque axe | Exercice 30 |
| 2 | **Créer** avec `torch.zeros` un tenseur ayant la forme d'un cas et **la vérifier** avec `assert` | Exercices 30 et 31 |
| 3 | **Expliquer** ce qu'est un embedding et **prédire** la forme de sa sortie | Exercice 32 |
| 4 | **Convertir** une séquence en tableau et **dire** ce qui se perd | Exercice 33 |

**Ce dont tu as besoin :** ton environnement PyTorch de la session 2 et les sessions 5 à 7. Aujourd'hui, tu n'installes rien.

---

## Pourquoi c'est important

Jusqu'ici, les données, c'est toi qui les inventais, ou elles sortaient de la feuille de doña Lupe. Aujourd'hui, tu vas voir ce que Mateo a vu sur l'écran d'Ortega : des cas réels de banques, de compagnies d'assurance et de magasins.

Personne ne va te les donner avec leur forme écrite. Ils t'arrivent avec une phrase comme « le modèle analyse l'historique du client ». Ton travail, c'est de la traduire en une forme : combien d'axes, ce que représente chacun et de quelle taille. Si tu sais le faire, aucune nouvelle sur l'IA ne te paraîtra magique. Elle te fera penser à une boîte avec des directions, comme la feuille de la papeterie.

Ortega l'a dit à Mateo avec d'autres mots : « Ici, personne ne s'impressionne du mot *intelligence*. Ici, on demande ce qui entre et ce qui sort. »

**Deux avertissements.** Premier : presque aucune entreprise ne publie les formes exactes de ses tenseurs ; elle publie la technique qu'elle utilise. Les formes d'aujourd'hui disent donc **à quoi ressemble ce type de donnée en général**, pas la recette secrète de chaque entreprise. Second : quand un chiffre est rapporté par l'entreprise elle-même, sans vérification extérieure, il est accompagné d'un **« selon l'entreprise »**. Comme ton cousin qui dit mesurer 1,80 m : c'est peut-être vrai, mais personne ne l'a mesuré.

---

## Étape 1 — Les trois questions pour lire un cas

« Avant que je vous montre quoi que ce soit », dit Ortega, « dites-moi une chose. Quand vous voyez un système de ce genre, quelle est la première question que vous posez ? »

Mateo réfléchit. En novembre, il aurait dit « à quel point il est bon ». Plus maintenant.

« Quelle forme il a. »

Ortega sourit pour la première fois de la matinée.

La compétence centrale nouvelle d'aujourd'hui, c'est de traduire un cas en axes et en forme, et elle tient en trois questions, dans cet ordre (tu découvriras aussi deux conventions nouvelles, l'ordre des couleurs dans PyTorch et les embeddings ; aujourd'hui, il suffit de reconnaître leurs formes) :

1. **Qu'est-ce qu'une chose ?** Un achat, un client, une photo. C'est presque toujours le premier axe : le lot.
2. **Qu'est-ce que chaque chose contient ?** Une liste de données ajoute un axe. Une histoire dans le temps en ajoute un autre avant les données. Une photo contient des couleurs, une hauteur et une largeur.
3. **Quel nombre va dans chaque case ?** Un montant, une couleur, le numéro d'un produit.

Avec les réponses, tu « dessines » la forme avec `torch.zeros`, qui crée un tenseur de zéros ayant la forme que tu lui demandes : les valeurs n'ont pas d'importance, c'est la boîte qui compte. Crée `casos.py` :

```python
import torch

# Banca: un lote de compras, como el detector de la sesión 7
compras = torch.zeros(64, 4)
print("compras:", compras.shape)

# Banca: el historial de cada cliente
historial = torch.zeros(32, 30, 4)
print("historial:", historial.shape)

# Seguros: las fotos de un siniestro
fotos = torch.zeros(8, 3, 480, 640)
print("fotos:", fotos.shape)
print("números en las fotos:", fotos.numel())

assert compras.shape == (64, 4), "Un lote de compras es (compras, datos)"
assert historial.shape == (32, 30, 4), "El historial es (clientes, compras en el tiempo, datos)"
assert fotos.shape == (8, 3, 480, 640), "Un lote de fotos es (fotos, colores, alto, ancho)"
print("Las tres formas cuadran")
```

Sortie :

```
compras: torch.Size([64, 4])
historial: torch.Size([32, 30, 4])
fotos: torch.Size([8, 3, 480, 640])
números en las fotos: 7372800
Las tres formas cuadran
```

`numel()` compte combien de nombres tiennent dans la boîte : 8 × 3 × 480 × 640 = 7 372 800. Huit photos moyennes, c'est déjà plus de sept millions de nombres.

Doña Lupe se penche vers l'ordinateur.

« Rien que des zéros ? »

« C'est le moule, ma tante. Comme quand tu achètes le cadre avant la photo. »

« Ah. Eh bien, le cadre m'a déjà coûté sept millions. »

---

## Étape 2 — La banque : cinq façons de traquer une fraude

Ortega ouvre un dossier sur son ordinateur. Dedans, il y a des coupures : des communiqués, des articles de presse, des articles de blogs d'entreprises.

« Je les rassemble pour les formations », dit-il. « Remarquez que chacun regarde quelque chose de différent. »

Toutes les banques veulent la même chose : détecter l'achat bizarre sans déranger le client honnête. Ce qui change, c'est **ce qu'elles regardent**, et cela change la forme. Voici ce qu'Ortega a ouvert, un par un :

- **Mastercard.** Son communiqué de février 2024 dit que Decision Intelligence aide à noter 143 milliards de transactions par an (la version en espagnol du même communiqué dit 125 milliards : c'est pour ça qu'on lit la source), et que la nouvelle version évalue **la relation entre plusieurs entités** autour de chaque achat. Selon l'entreprise, elle améliore la détection de 20 % en moyenne, et jusqu'à 300 % dans certains cas.
- **Visa.** En 2019, selon l'entreprise, elle analysait 100 % de 127 milliards de transactions par an, en à peu près une milliseconde chacune.
- **BBVA.** Dans une note de 2018 sur un travail avec le MIT, BBVA dit avoir généré **plus de 200 variables nouvelles par transaction** et que son modèle « peut réduire de 54 % le niveau de faux positifs » (des alertes qui dérangent un client honnête). Il s'agit de BBVA au niveau mondial, pas de BBVA Mexique.
- **HSBC.** Le cousin de la fraude : le blanchiment d'argent. Dans un texte de 2023 sur le blog de Google Cloud, HSBC dit examiner plus de 1 200 millions de transactions par mois en cherchant **des mouvements rapides entre comptes et des changements brusques de comportement**, et trouver de deux à quatre fois plus d'activité suspecte qu'avant, avec 60 % d'alertes en moins. Selon l'entreprise.
- **Nubank.** Sur son blog d'ingénierie, elle raconte qu'elle convertit les transactions de chaque client en **séquences de jetons (tokens)**, comme si c'étaient des mots, et qu'elle les lit avec le type de modèle utilisé pour le texte. Chaque transaction peut produire plusieurs jetons (montant, date, texte de la description). L'entreprise déclare 131 millions de clients au total fin 2025 ; cela ne veut pas dire que ce modèle les touche tous.

Mateo remarque quelque chose dans chaque coupure : presque toutes disent « selon l'entreprise ». Ortega le voit le remarquer.

« C'est ce qui me plaît le plus chez vous, jeune homme. Vous lisez les petits caractères. »

Maintenant, les trois questions :

| Cas | Qu'est-ce qu'une chose ? | Que contient-elle ? | Forme (type de donnée) |
|---|---|---|---|
| Visa : un achat à la fois | un achat | ses données | `(datos,)` |
| Mastercard, BBVA : beaucoup d'achats | un achat | ses données (chez BBVA, plus de 200) | `(compras, datos)` |
| HSBC : le comportement dans le temps | un compte | ses mouvements dans l'ordre, chacun avec ses données | `(cuentas, movimientos, datos)` |
| Nubank : l'historique comme un texte | un client | ses achats dans l'ordre, en jetons | modèle didactique, avec un jeton par achat : `(clientes, compras)` d'entiers → `(clientes, compras, números)` ; dans le cas réel, comme un achat donne plusieurs jetons : `(clientes, fichas)` → `(clientes, fichas, números)` |

Chez BBVA, l'axe des données est passé de 4 (ton détecteur) à plus de 200 : **la forme n'a pas changé d'idée, seulement de taille.** Chez HSBC, regarder le temps ajoute un axe : le passage du tableau au cube de la session 1, avec un onglet par compte.

Mateo reste à regarder la ligne de BBVA. Son détecteur avait quatre données par débit : montant, heure, distance et si le commerce était à l'étranger. Celui de la banque en a plus de deux cents. Et pourtant, c'est la même boîte : `(compras, datos)`.

Son ordinateur portable ne jouait pas à autre chose. Il jouait au même jeu, en petit.

---

## Étape 3 — L'assurance : l'accident en photos

« La banque n'est pas la seule à faire ça », dit Ortega, et il ouvre une autre coupure. « Mon beau-frère travaille dans l'assurance. Là-bas, on n'envoie plus un expert voir chaque bosse. On lui demande des photos. »

En mars 2020, la compagnie d'assurance britannique Ageas a annoncé qu'elle utilisait le système de l'entreprise Tractable pour évaluer les accidents de voiture. Selon Ageas, à partir des photos, le système identifie quelles pièces sont endommagées et à quel point, établit le devis complet et décide en quelques minutes, parfois pendant le premier appel.

Les trois questions : une chose, c'est **une photo** (un sinistre en apporte plusieurs : le lot) ; chaque photo contient **3 couleurs**, et chaque couleur est un tableau de **hauteur par largeur** ; dans chaque case va la quantité de cette couleur en ce point. Forme : `(fotos, 3, alto, ancho)`, le tenseur `fotos` de l'étape 1.

Doña Lupe ajuste ses lunettes.

« Donc, si j'ai un accident, je prends la voiture en photo et l'ordinateur me dit combien. »

« Quelque chose comme ça, ma tante. »

« Eh bien, qu'il prenne en photo ma facture d'électricité, pour voir ce qu'il en pense. »

**Le piège de l'ordre des couleurs.** Dans la session 1, tu as écrit une photo comme `(480, 640, 3)` : hauteur, largeur, couleur. Dans PyTorch, l'habitude est de mettre la couleur **avant** : `(3, 480, 640)`. Mêmes nombres, autre rangement. Si tu mélanges les habitudes, l'`assert` te prévient :

```python
fotos = torch.zeros(8, 3, 480, 640)
assert fotos.shape == (8, 480, 640, 3), "Un lote de fotos es (fotos, alto, ancho, colores)"
```

```
Traceback (most recent call last):
  ...
AssertionError: Un lote de fotos es (fotos, alto, ancho, colores)
```

Aucune habitude n'est mauvaise ; ce qui est mauvais, c'est de supposer.

Et la sortie ? Pour chaque pièce, la probabilité de « réparer » et celle de « remplacer » : `(piezas, 2)`. Quatre axes entrent, un petit tableau sort.

---

## Étape 4 — Le magasin : la leçon d'humilité de la prévision

Mateo pose la question qu'il veut poser depuis le début de la matinée :

« Alors, tout ça, on le fait avec des réseaux de neurones ? Le plus avancé gagne toujours ? »

Ortega rit.

« Je vais vous en montrer une qui va plaire à votre tante. Elle vient d'un magasin. »

En 2020 a eu lieu la compétition de prévision M5. Ces données viennent de l'article scientifique de ses organisateurs (Makridakis, Spiliotis et Assimakopoulos) :

- Il fallait prévoir **42 840 séries hiérarchiques** de ventes d'**une chaîne de grande distribution américaine** (données publiques de la compétition M5 de Kaggle) : 10 magasins en Californie, au Texas et au Wisconsin, des données quotidiennes sur environ 5,5 ans. Au niveau le plus désagrégé, il y avait 30 490 séries, chacune « combien ce produit s'est vendu dans ce magasin, jour après jour » ; les autres étaient des sommes par catégorie, rayon, magasin ou État.
- **7 092 personnes dans 5 507 équipes de 101 pays** ont participé.
- **La première place a utilisé LightGBM**, une méthode d'arbres de décision, pas des réseaux de neurones : un grand ensemble de 220 modèles, rien de trivial. La deuxième place a combiné LightGBM avec un réseau (N-BEATS). La troisième, elle, a utilisé des réseaux : une combinaison de 43 réseaux de neurones (l'équipe a utilisé DeepAR, qui lit des séquences).

La leçon d'humilité : **les réseaux de neurones ne gagnent pas toujours.** Avec des données en tableau, une méthode qui n'est pas un réseau peut les battre. Et même ainsi, LightGBM reçoit un tableau `(renglones, datos)` : **le tenseur est la façon de ranger les données, pas un type de modèle.**

| Qui | Comment il voit les données | Forme |
|---|---|---|
| Un réseau qui lit des séquences (troisième place) | chaque série est une histoire | `(series, días, datos)` |
| LightGBM (gagnant) | chaque jour de chaque série est une ligne | `(series × días, datos)` |

« Vous voyez ? » dit doña Lupe. « Le simple, ça marche aussi. Ça fait trente ans que je le dis. »

Passer de l'un à l'autre, c'est la session 5. Mateo le fait sur-le-champ, dans le bureau, avec la papeterie de doña Lupe : 2 succursales × 7 jours deviennent 14 lignes. Crée `tienda.py` :

```python
import torch

centro = [[1200, 800, 500], [1500, 900, 700], [1000, 700, 500], [1300, 900, 600],
          [2000, 1200, 700], [1200, 800, 400], [800, 500, 300]]
norte = [[300, 200, 100], [400, 250, 150], [350, 200, 100], [300, 150, 100],
         [500, 300, 200], [300, 200, 100], [200, 100, 100]]
papeleria = torch.tensor([centro, norte])
print("papeleria:", papeleria.shape)

tabla = papeleria.reshape(-1, 3)
print("tabla:", tabla.shape)

# Comprobación: con for, el total no cambia
total_for = 0
for sucursal in [centro, norte]:
    for dia in sucursal:
        for venta in dia:
            total_for = total_for + venta

assert tabla.shape == (14, 3), "La tabla debería ser 14 renglones (sucursal y día) × 3 productos"
assert tabla.sum().item() == total_for == 23100, "Cambiar de forma no cambia el total"
print("Total:", tabla.sum().item(), "| con for:", total_for)
```

Sortie :

```
papeleria: torch.Size([2, 7, 3])
tabla: torch.Size([14, 3])
Total: 23100 | con for: 23100
```

Les lignes 0 à 6 sont le Centre ; les lignes 7 à 13, Nord. Le total, 18 500 + 4 600 = 23 100, n'a pas changé. Qu'est-ce qui s'est perdu ? Que la ligne 7 soit « Nord, lundi », la forme ne le dit plus ; en pratique, on ajoute des colonnes comme « succursale » et « jour ».

Ortega regarde les deux nombres : 18 500 et 4 600.

« C'est cette différence qui nous a fait regarder Nord de plus près », dit-il. « Notre système a vu des débits bizarres sur votre terminal. Ce qu'il n'a pas vu, c'est ce que vous venez de me montrer : qu'il n'y avait pas de marchandise derrière. »

---

## Étape 5 — Le magasin qui te fait des recommandations : le dictionnaire qui s'apprend tout seul

Le téléphone de doña Lupe sonne. Une notification d'une boutique en ligne : « Vous aimerez peut-être aussi : kit de géométrie ».

« Et celle-là, comment elle sait que je vends des kits de géométrie ? »

Ortega et Mateo se regardent. Cette question a une réponse.

Derrière « vous aimerez peut-être aussi… », il y a souvent un **embedding**. **Facebook (aujourd'hui Meta)** a publié en 2019 un modèle de recommandation open source en PyTorch (DLRM) où « quel produit c'est » passe par des **tables d'embeddings**. **Alibaba** en a publié un la même année qui lit la séquence de ce qu'un utilisateur a vu et acheté ; selon l'entreprise, il fonctionne dans sa boutique Taobao.

Pense à un dictionnaire Python : `{"cuadernos": [1.5, -0.3, -2.2], ...}`, une liste de nombres par produit. La différence, c'est que **ces nombres, personne ne les écrit : le modèle les apprend tout seul**, comme il a appris les poids du détecteur, et ils s'ajustent avec les données et la perte. Dans une tâche de recommandation, des produits aux habitudes d'achat semblables **peuvent** finir avec des nombres proches, mais ce n'est pas une règle automatique.

**Un embedding est un dictionnaire qui s'apprend tout seul.** Dans PyTorch, la clé est un entier, alors tu numérotes d'abord ton catalogue. Crée `embedding.py` :

```python
import torch

torch.manual_seed(0)

# El catálogo: cada producto tiene un número
catalogo = {"cuadernos": 0, "plumas": 1, "mochilas": 2, "colores": 3, "tijeras": 4}

# La tabla de embeddings: 5 productos, 3 números por producto
diccionario = torch.nn.Embedding(5, 3)
print("tabla completa:", diccionario.weight.shape)

# Una clienta compró cuadernos, mochilas y otra vez cuadernos
compra = torch.tensor([catalogo["cuadernos"], catalogo["mochilas"], catalogo["cuadernos"]])
print("entrada:", compra.shape, compra.dtype)

salida = diccionario(compra)
print("salida:", salida.shape)
print(salida)

assert salida.shape == (3, 3), "3 productos comprados × 3 números por producto"
assert torch.equal(salida[0], salida[2]), "El mismo producto da el mismo renglón"
assert torch.equal(salida[1], diccionario.weight[2]), "Buscar un producto es sacar su renglón"
print("Forma y valores cuadran")

# Un lote: 2 clientas, 4 compras cada una
historiales = torch.tensor([[0, 1, 1, 3],
                            [2, 4, 0, 0]])
print("historiales:", historiales.shape, "->", diccionario(historiales).shape)
```

Sortie :

```
tabla completa: torch.Size([5, 3])
entrada: torch.Size([3]) torch.int64
salida: torch.Size([3, 3])
tensor([[ 1.5410, -0.2934, -2.1788],
        [ 0.4033,  0.8380, -0.7193],
        [ 1.5410, -0.2934, -2.1788]], grad_fn=<EmbeddingBackward0>)
Forma y valores cuadran
historiales: torch.Size([2, 4]) -> torch.Size([2, 4, 3])
```

`torch.equal(a, b)` répond `True` seulement si les deux tenseurs ont la même forme et les mêmes valeurs. C'est pour ça que les deux `assert` vérifient que chercher le même produit donne la même ligne.

Lis-le lentement :

- `Embedding(5, 3)` est une table de 5 lignes (une par produit) et 3 colonnes, au début avec des nombres au hasard.
- Chercher un produit, c'est **extraire sa ligne** : les cahiers apparaissent deux fois, identiques.
- `grad_fn`, c'est PyTorch qui prévient qu'il garde la trace des calculs pour pouvoir ajuster ces nombres pendant l'entraînement. C'est la partie « s'apprend tout seul ».
- **La règle de la forme :** la sortie a la forme de l'entrée **plus un axe à la fin**, de la taille du deuxième nombre de l'Embedding. `(3,)` → `(3, 3)` ; `(2, 4)` → `(2, 4, 3)`.

Cette dernière ligne, c'est le modèle didactique de Nubank : avec un jeton par achat, `(clientes, compras)` d'entiers devient `(clientes, compras, números)`. Dans le cas réel, un achat peut donner plusieurs jetons, et l'axe serait celui des jetons. Ton historique d'achats se lit comme un texte.

« Alors le magasin ne sait pas que je vends des kits de géométrie », dit doña Lupe. « Il sait que les gens qui achètent ce que j'achète achètent aussi ça. »

« Exactement, ma tante. »

« Eh bien, il est plus commère que ma voisine. »

Si tu demandes un produit qui n'est pas dans la table (le 5, alors qu'il n'y a que de 0 à 4), tu obtiens :

```
IndexError: index out of range in self
```

---

## Étape 6 — Le Mexique et les précautions qui ne sont pas facultatives

**10 h 05.** Ortega regarde l'horloge. Il a dépassé de cinq minutes les vingt prévues et ça n'a pas l'air de le gêner.

« Maintenant, je vais vous dire quelque chose que je ne vous ai pas dit au téléphone », dit-il. « Le message que votre tante a reçu le 1er novembre, c'est une machine qui l'a envoyé. Un système a vu des débits bizarres et a suspendu le terminal. Tout seul. »

Doña Lupe serre son sac.

« Et pourquoi on m'a laissé jusqu'à aujourd'hui ? »

« Parce qu'une machine peut signaler. Mais une décision comme celle-là ne doit pas rester entre les mains de la machine. Quelqu'un doit examiner. Aujourd'hui, ce quelqu'un, c'est moi. »

Mateo comprend d'un coup pourquoi il est assis là. Il n'est pas venu convaincre un modèle. Il est venu pour qu'une personne examine ce que le modèle a décidé.

Ce qui suit explique pourquoi ce « quelqu'un doit examiner » n'est pas une politesse de la banque.

**Banxico.** La Banque du Mexique a publié le 10 juin 2026 l'encadré « Inteligencia artificial y estabilidad financiera: canales potenciales de transmisión de riesgos » (Intelligence artificielle et stabilité financière : canaux potentiels de transmission des risques), où elle met en garde contre la concentration sur **peu de fournisseurs**, contre des modèles **opaques** et difficiles à superviser, et contre la possible **exclusion injustifiée de certains groupes de population**. Une note d'Expansión du 18 juin 2026 le commente aussi.

**Le biais : German Credit.** C'est une base publique très utilisée pour s'entraîner sur le crédit, issue du dépôt de l'Université de Californie à Irvine : 1 000 personnes avec 20 données chacune, de 1994. Comme tableau d'origine, `X` a 1 000 personnes et 20 attributs mixtes (nombres et catégories) ; avant de l'utiliser dans un tenseur numérique, il faut coder les catégories, et alors l'axe des données peut ne plus mesurer 20. Le problème est dans les colonnes : l'une mélange **état civil et sexe**, une autre est **l'âge** et une autre indique si la personne est un **« travailleur étranger »**. Entraîner avec ça tel quel, c'est demander au modèle de prêter ou non selon le sexe ou la nationalité. C'est pour ça que le cours ne l'a pas utilisée dans la session 7.

Et si tu supprimes ces colonnes ? Ça ne suffit pas : d'autres peuvent **trahir** la même donnée. C'est comme enlever le nom d'une copie d'examen en laissant l'écriture : le professeur sait quand même de qui elle est. Il faut mesurer comment le modèle traite chaque groupe.

**La loi européenne.** L'*AI Act* de l'Union européenne, annexe III, point 5(b), classe comme **à haut risque** les systèmes qui évaluent la solvabilité d'une personne ou établissent sa note de crédit, **à l'exception de ceux qui détectent la fraude financière**. Ton détecteur resterait en dehors ; un modèle qui note le crédit, comme ceux avec lesquels on s'entraîne sur German Credit, dedans. À méditer : pourquoi la loi traite-t-elle la fraude différemment du crédit ? Indice : si le détecteur se trompe, on te bloque un achat ; si celui du crédit se trompe, tu te retrouves sans la maison.

**La loi mexicaine.** La nouvelle Loi fédérale sur la protection des données personnelles détenues par les particuliers a été publiée le 20 mars 2025. Son **article 26, alinéa II**, te permet de **t'opposer** à ce que tes données passent par un **traitement automatisé** qui t'affecte de manière importante et qui t'évalue **sans intervention humaine** : ta situation économique, ta santé, ta fiabilité ou ton comportement, entre autres. Autrement dit : si un modèle décide seul si on t'accorde un crédit, dans ces cas-là tu peux t'y opposer. Cela ne veut pas dire que la loi impose, à elle seule, un examen humain pour chaque décision, et ce droit a des exceptions : il ne s'applique pas, par exemple, quand le traitement est nécessaire pour respecter une obligation légale. Pour qui conçoit le système, « le modèle décide, point » ne suffit pas : il vaut mieux un humain sur le chemin.

Attention à ce que dit cet article et à ce qu'il ne dit pas : il parle de **tes données personnelles**, de décisions qui **te** concernent. Le terminal d'une papeterie, c'est une autre histoire, et dans le récit, la banque l'examine manuellement selon sa propre règle. Mais l'idée est la même, et Mateo l'a vue dans le bureau d'Ortega : **la machine signale ; une personne examine ; et celui qu'elle affecte a un moyen de réclamer.**

> **10 h 20. Le bureau de monsieur Ortega.**
>
> Ortega tape quelque chose. Il imprime une feuille. Il la signe.
>
> « Le terminal de Nord est réactivé à partir d'aujourd'hui », dit-il. « Mais avant, un technicien va passer. Un vrai, avec un badge et un ordre d'intervention. Il va examiner l'appareil. »
>
> Doña Lupe reste silencieuse. Pour la première fois en sept semaines, elle n'a pas de dicton sous la main.
>
> **13 h 15. Succursale Nord.**
>
> Le technicien de la banque ouvre le terminal sur le comptoir. Fer est à côté, les bras croisés. Mateo, derrière.
>
> Dans la fente où entre la carte, il y a quelque chose qui ne devrait pas être là : une fine lamelle, avec un câble épais comme un cheveu qui file vers l'intérieur de l'appareil.
>
> « Un lecteur », dit le technicien. « Il copie la carte chaque fois que quelqu'un paie. Les gens ne s'en rendent même pas compte. »
>
> Fer expire lentement.
>
> « C'est celui au tee-shirt bleu qui l'a posé. En août. »
>
> Personne ne l'avait appelé. Personne ne lui avait demandé son nom. Les gens payaient à Nord, et leurs cartes étaient utilisées au petit matin, loin, ailleurs. C'est pour ça que la banque ne voyait que le terminal.
>
> Le technicien met la lamelle dans un sachet en plastique. Mateo pense au tableau des débits de la session 6 : ceux qui n'avaient pas de marchandise derrière. Voilà la main qui manquait.

---

👉 **Si tu viens de la gestion :** quand un fournisseur te présente son système d'IA, pose-lui les trois questions de l'étape 1 ; s'il ne sait pas y répondre, il ne comprend pas son produit. Demande d'où vient chaque chiffre. Et si le système décide au sujet de personnes (crédit, assurance, recrutement), demande où est l'humain qui examine (c'est utile même quand la loi ne l'exige pas dans tous les cas).

---

## Exercices de la session 8

Les derniers du cours. Les solutions se trouvent à la fin.

**Exercice 30 — Trois cas, trois formes.** Réponds aux trois questions sur papier, écris la forme, dessine-la avec `torch.zeros` et vérifie-la avec `assert`. Le point a) est résolu.

a) Un lot de 500 achats avec 200 données chacun, comme dans la note de BBVA :

```python
bbva = torch.zeros(500, 200)
assert bbva.shape == (500, 200), "a) compras × datos"
```

b) L'historique de 16 clients en séquences de 100 achats, chaque achat passant par un Embedding de 64 nombres, comme dans le modèle didactique de Nubank. Indice : `torch.zeros(16, 100, dtype=torch.long)` et `torch.nn.Embedding(1000, 64)`.
c) Un sinistre avec 5 photos de 720 de haut sur 1 280 de large, dans l'ordre de PyTorch.

Termine par `print("Ejercicio 30 superado")`.

**Exercice 31 — Le lot de photos.** Remplis les trous : un sinistre avec 6 photos de 224 × 224, et sa quantité de nombres calculée de deux façons qui doivent coïncider :

```python
siniestro = ____
assert siniestro.shape == ____, "6 fotos × 3 colores × 224 de alto × 224 de ancho"
cuenta = 1
for tamano in (6, 3, 224, 224):
    cuenta = ____
assert siniestro.numel() == cuenta, "numel debe dar lo mismo que multiplicar la forma con un for"
print("Ejercicio 31 superado:", siniestro.numel(), "números")
```

**Exercice 32 — Le dictionnaire de doña Lupe.** Sa boutique en ligne a 4 produits et elle veut 2 nombres par produit. Prédis les formes **avant** d'exécuter :

```python
torch.manual_seed(1)
catalogo = {"cuadernos": 0, "plumas": 1, "mochilas": 2, "colores": 3}
diccionario = torch.nn.Embedding(____, ____)
historiales = torch.tensor([[0, 1, 1, 2, 3],
                            [3, 3, 0, 1, 2],
                            [2, 2, 2, 0, 1]])
salida = diccionario(historiales)
assert diccionario.weight.shape == ____, "4 productos × 2 números"
assert historiales.shape == ____, "3 clientas × 5 compras"
assert salida.shape == ____, "3 clientas × 5 compras × 2 números"
assert torch.equal(salida[0][3], diccionario.weight[catalogo["mochilas"]]), "La cuarta compra de la primera clienta son mochilas"
print("Ejercicio 32 superado")
```

**Exercice 33 — De la séquence au tableau, tout seul.** Une M5 miniature : 100 séries, 28 jours, 3 données par jour (prix, promotion, jour férié). a) Crée avec `torch.zeros` ce que verrait un réseau qui lit des séquences. b) Convertis-le avec `reshape` et `-1` en tableau que verrait LightGBM. c) Écris trois `assert` : les deux formes, et le fait qu'elles ont la même quantité de nombres. d) Dans un commentaire, dis ce qu'est une ligne du tableau et ce que la forme ne dit plus.

---

## Erreurs courantes (et comment t'en rendre compte)

| Erreur | Ce que l'on observe | Comment t'en rendre compte | Ce qu'il faut faire |
|---|---|---|---|
| Mélanger l'ordre des couleurs | Tu écris `(fotos, alto, ancho, 3)` alors que PyTorch attend `(fotos, 3, alto, ancho)` | L'`assert` de la forme plante | Demande dans quel ordre arrivent les photos |
| Demander un produit hors de la table | `IndexError: index out of range in self` | Un des nombres est ≥ à la première taille de l'Embedding | N produits vont de 0 à N − 1 |
| Donner des décimaux à un Embedding | `RuntimeError: Expected tensor for argument #1 'indices' to have one of the following scalar types: Long, Int; but got torch.FloatTensor instead (while checking arguments for embedding)` | Il dit `FloatTensor` là où il attendait des entiers | `torch.tensor([0, 2])`, sans point |
| Répéter un chiffre de l'entreprise comme un fait | « Mastercard améliore la détection de 300 % » | La source est un communiqué de l'entreprise | « Selon l'entreprise », et précise si c'est « jusqu'à » ou « en moyenne » |

---

## Autoévaluation de la leçon B

Coche honnêtement. Si une ligne donne « pas encore », retourne à cette session.

| Je peux… | Session | Oui / Pas encore |
|---|---|---|
| Changer de forme avec `reshape`, utiliser `-1` et lire l'erreur quand ça ne tombe pas juste | 5 | |
| Prédire la forme qui sort quand on opère sur des tenseurs de formes différentes | 6 | |
| Distinguer `*` de `@` et savoir lequel utiliser | 6 | |
| Expliquer perte, gradient, optimiseur et époque | 7 | |
| Expliquer pourquoi l'exactitude seule est trompeuse, et ce que mesurent la précision et le rappel | 7 | |
| Lire un cas réel et écrire la forme de son tenseur | 8 | |

## Révision de la leçon B

Réponds sans regarder. Pour celles avec du code, prédis, puis exécute.

1. Si `meses = torch.arange(12)`, quelle forme donne `meses.reshape(4, -1)` ? Et `meses.reshape(-1, 6)` ?
2. Que se passe-t-il avec `meses.reshape(5, -1)`, et pourquoi ?
3. Avec la papeterie `(2, 7, 3)` et `precios = torch.tensor([45.0, 12.0, 350.0])`, quelle forme a `papeleria * precios` ?
4. Est-ce que `torch.zeros(2, 7, 3) * torch.zeros(7)` fonctionne ?
5. Avec `a = [[1, 2], [3, 4]]` et `b = [[10, 20], [30, 40]]` comme tenseurs, combien donnent `a * b` et `a @ b` ? Pourquoi `*` est-il un piège qui ne signale pas d'erreur ?
6. Si `tabla` est `(14, 3)` et `precios` est `(3,)`, quelle forme a `tabla @ precios` et que représente chaque nombre ?
7. Que font `perdida.backward()` et `optimizador.step()` ?
8. Qu'est-ce qu'une époque ?
9. Sur 1 000 achats, 50 sont des fraudes. Le modèle lance 40 alertes et 30 sont de vraies fraudes. Calcule l'exactitude, la précision et le rappel. Lequel inquiète le plus la banque ?
10. Écris la forme de : (a) 64 achats avec 4 données ; (b) 32 clients avec leurs 30 derniers achats de 4 données ; (c) 8 photos de 480 × 640 dans l'ordre de PyTorch.

---

🔁 **Pour clore le cours**

**Mercredi 23 décembre, 19 h 30. Papeterie, succursale Centre.**

Doña Lupe arrive avec une boîte de pâtisserie nouée d'un ruban. Un tres leches. Elle la pose sur le comptoir, entre les agrafeuses et les rouleaux de papier glacé.

« Celui-là, je l'ai payé par carte », dit-elle. « Sur le terminal de Nord. On va voir si, cette fois, on me fait payer le bon prix. »

Fer coupe le gâteau. C'est elle qui a la plus grosse part, et personne ne dit rien, parce que tout le monde sait pourquoi : c'est elle qui, en septembre, avait écrit qu'il manquait de la marchandise, et personne n'avait lu son courriel.

Mateo mange debout, appuyé contre la vitrine. Dehors, quelqu'un baisse un rideau métallique. Ça sent le papier neuf et la cannelle.

« Et toi, qu'est-ce que tu vas faire maintenant ? » lui demande doña Lupe. « Il n'y a plus rien à sauver. »

Mateo y a pensé toute la semaine. En novembre, il voulait finir ses études, et c'est tout. Maintenant, il sait ce qui lui plaît : qu'une pile de nombres dise qui ment. Il veut étudier ça. Les données. Les modèles. Et les petits caractères.

« Ça, ma tante. Ce qu'on a fait. »

« Surveiller des papeteries ? »

« Quelque chose comme ça. »

Ce soir-là, il ouvre l'ordinateur. Une feuille neuve, blanche. Aucune forme encore, aucun axe.

Il écrit une seule ligne :

`# ¿Y ahora qué?`

Tu en es au même point que lui. En huit sessions, tu es passé d'un nombre isolé à la lecture de la façon dont travaillent une banque, une compagnie d'assurance et un magasin. Il ne te reste pas une formule : il te reste une façon de regarder.

| Session | Ce que tu as appris |
|---|---|
| 1 | Ce qu'est une dimension et ce qu'est la forme, avec les listes de Python |
| 2 | Ton premier tenseur : `torch.tensor`, `shape` et `dtype` |
| 3 | Extraire un morceau du tenseur selon n'importe lequel de ses axes |
| 4 | Additionner et calculer la moyenne sur un axe, en comparant avec un `for` |
| 5 | Changer de forme avec `reshape` sans changer les données |
| 6 | Opérer sur des tenseurs complets, et la différence entre `*` et `@` |
| 7 | Entraîner un détecteur de fraude, et se méfier de l'exactitude |
| 8 | Lire des cas réels, les embeddings et les précautions du crédit automatisé |

---

## Solutions (ne les regarde pas avant d'avoir essayé)

### Révision pour commencer

1. `(compras, datos de cada compra)` : dans la session 7, `(4000, 4)`.
2. Elle mesure **à quel point le modèle prédit mal**, en un seul nombre ; si elle baisse, le modèle est en train d'apprendre.
3. Parce que presque tous les achats sont légitimes : un « détecteur » qui ne donne jamais l'alerte a déjà raison environ 84 % du temps. Avec 91 %, le nôtre laissait encore passer 1 fraude sur 3.

### Exercices

**30.** b) `(16, 100, 64)` : clients × achats × nombres de l'embedding. Le 1000, c'est combien de produits différents tiennent dans la table ; il ne change pas la forme de la sortie. c) `(5, 3, 720, 1280)`.

```python
historiales = torch.zeros(16, 100, dtype=torch.long)
embedding = torch.nn.Embedding(1000, 64)
salida = embedding(historiales)

assert historiales.shape == (16, 100), "clientes × compras"
assert salida.shape == (16, 100, 64), "clientes × compras × números"

fotos_siniestro = torch.zeros(5, 3, 720, 1280)
assert fotos_siniestro.shape == (5, 3, 720, 1280), "fotos × colores × alto × ancho"

print("Ejercicio 30 superado")
```

Sortie : `Ejercicio 30 superado`.

**31.** Les trous sont `torch.zeros(6, 3, 224, 224)`, `(6, 3, 224, 224)` et `cuenta * tamano`. Sortie : `Ejercicio 31 superado: 903168 números`.

**32.** `Embedding(4, 2)`, `(4, 2)`, `(3, 5)` et `(3, 5, 2)` : à la forme de l'entrée s'ajoute à la fin un axe de taille 2. Sortie : `Ejercicio 32 superado`.

**33.**

```python
series = torch.zeros(100, 28, 3)    # series × días × datos
tabla = series.reshape(-1, 3)       # (serie y día) × datos
assert series.shape == (100, 28, 3), "series × días × datos"
assert tabla.shape == (2800, 3), "(series y día) × datos"
assert tabla.numel() == series.numel() == 8400, "Cambiar de forma no cambia cuántos números hay"
print("Ejercicio 33 superado")
# Un renglón es un día de una serie; la forma ya no dice de cuál serie ni de cuál día.
```

### Révision de la leçon B

Sorties réelles des questions 1 à 6 :

```
1) torch.Size([4, 3]) torch.Size([2, 6])
2) shape '[5, -1]' is invalid for input of size 12
3) torch.Size([2, 7, 3])
4) The size of tensor a (3) must match the size of tensor b (7) at non-singleton dimension 2
5) a * b = [[10, 40], [90, 160]] | a @ b = [[70, 100], [150, 220]]
6) torch.Size([14])
```

1. `(4, 3)` et `(2, 6)` : le `-1`, c'est « calcule-le toi-même ».
2. Ça plante : 12 nombres ne se répartissent pas en 5 lignes égales.
3. `(2, 7, 3)` : les prix s'étirent à toutes les succursales et à tous les jours.
4. Non : PyTorch compare les formes de droite à gauche, et le dernier axe mesure 3 contre 7.
5. `*` opère case par case ; `@`, ligne par colonne. Avec ces formes, les deux fonctionnent : utiliser le mauvais ne plante pas, ça donne seulement un autre nombre. C'est pour ça qu'on compare avec un `for`.
6. `(14,)` : le total en pesos de chaque succursale pour chaque jour (quantités par prix, additionnées).
7. `perdida.backward()` calcule les gradients de la perte par rapport à chaque poids ; `optimizador.step()` utilise ces gradients pour déplacer les poids d'un petit pas et chercher à faire baisser la perte.
8. Un passage complet sur toutes les données d'entraînement.
9. Achats légitimes sans alerte : 1 000 − 30 − 10 − 20 = 940. Exactitude = 970 ÷ 1 000 = 0.97 ; précision = 30 ÷ 40 = 0.75 ; rappel = 30 ÷ 50 = 0.6. Les 97 % sonnent excellents, mais 4 fraudes sur 10 passent à travers.
10. `(64, 4)`, `(32, 30, 4)` et `(8, 3, 480, 640)` : les tenseurs de l'étape 1.

---

## Et ensuite

Un ordre possible pour continuer (en anglais, sauf le dernier point) :

- **Pour l'intuition :** la série *Neural Networks* de **3Blue1Brown**, des vidéos animées qui montrent ce que fait un réseau à l'intérieur. Commence par « But what is a neural network? ». Elles servent à comprendre, pas à pratiquer.
- **Pour construire un réseau à partir d'un seul nombre :** *Neural Networks: Zero to Hero* d'**Andrej Karpathy**. Sa première vidéo, *micrograd* (2 heures 25 minutes), construit de zéro ce que fait `perdida.backward()` ; il dit qu'il suffit d'un Python de base et d'un vague souvenir du calcul différentiel du lycée. Le rythme est celui d'un expert : avance avec des pauses.
- **Pour des réseaux plus grands en PyTorch :** *Learn PyTorch for Deep Learning*, de Daniel Bourke (learnpytorch.io), avec des exercices et des solutions pour comparer.
- **En espagnol :** les vidéos de **DotCSV** pour te motiver et, comme objectif, le cours de master *Aprendizaje Profundo* de l'UNAM, avec son matériel ouvert sur GitHub. Si un jour tu comprends ses notebooks, ce cours aura rempli sa mission.

Un dernier conseil : quand quelque chose ne fonctionne pas, affiche `.shape`. Tu sais déjà la lire.

---

*J'écris à partir de ce que nous voyons tous les jours chez Hábil, où nous concevons des systèmes pour le secteur bancaire, les assurances et les entreprises au Mexique. Si, dans ton travail ou dans l'entreprise de ta famille, tu rencontres un problème de ce genre, discutons-en : [habil.mx](https://www.habil.mx/fr/)*
