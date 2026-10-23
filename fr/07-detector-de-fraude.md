# Les tenseurs à partir de zéro (7) : un détecteur de fraude qui apprend à partir des données

*Tu lui donnes 4 000 achats par carte et la réponse pour chacun, mais tu ne lui écris jamais la règle. En moins d'une seconde, il en apprend une bonne partie. Ensuite, tu apprends pourquoi avoir raison 91 % du temps peut cacher qu'il laisse passer plus d'une fraude sur trois.*

*Les tenseurs à partir de zéro · leçon B, session 7 sur 8*

*Scénario : les personnages et l'histoire sont inventés. Les données techniques, historiques et les chiffres réels indiquent leur source.*

**Lundi 14 décembre, 20 h 18. Papeterie Herrera, succursale Centre.**

Le rideau métallique est déjà baissé et il tremble encore un peu, comme s'il refusait de rester immobile. L'ampoule de l'arrière-boutique bourdonne. Ça sent le papier neuf, le café réchauffé et la cannelle d'un petit pain que doña Lupe a oublié à côté de la caisse. Dehors, un camion passe et la vitre de la porte vibre.

Mateo a son ordinateur portable ouvert sur une boîte de bristol. À côté, deux feuilles imprimées. Sur l'une, les ventes qui sont vraiment sorties de Nord : des cahiers, des stylos, un sac à dos de temps en temps. Sur l'autre, les débits envoyés par la banque. Certains sont arrivés au petit matin. D'autres sont élevés. D'autres apparaissent très loin de la succursale, dans des endroits où la papeterie n'a jamais vendu le moindre trombone.

« Celui de trois heures du matin, ça ne peut pas être un sac à dos », dit Mateo, et il l'entoure au stylo rouge.

« À cette heure-là, je ne vends que du sommeil », répond doña Lupe. « Et pas cher, parce que personne n'en achète. »

Il y a une semaine, ils ont multiplié les quantités par les prix et le compte n'est pas tombé juste : il y a des débits sans marchandise derrière. Fer ne les a pas inventés ; elle avait signalé les manques en septembre et personne n'avait lu son courriel. Le problème n'était pas dans l'inventaire. Il était dans le terminal qu'avait changé cet homme en tee-shirt bleu. Un lecteur caché a pu copier des cartes à Nord. Ensuite, quelqu'un les a utilisées ailleurs.

Mais un soupçon ne suffit pas pour la banque. Monsieur Ortega l'a dit au téléphone, sans détour : « Madame, j'ai besoin de savoir quels débits sont vraiment les vôtres. Pas une intuition. Lesquels. »

Et c'est là que tout se noue. Un achat cher n'est pas une fraude : il y a des gens qui économisent toute l'année pour leur téléviseur. Un achat au petit matin non plus : il y a des pharmacies ouvertes et des gens qui ont des urgences. Un débit lointain peut venir de quelqu'un en vacances. Aucun indice seul ne condamne personne. Ce qui pèse, c'est quand ils se cumulent.

Mateo essaie de vérifier la feuille de la banque ligne par ligne et abandonne à la quarantième. À ce rythme, il n'aura pas fini en janvier.

Sur le calendrier accroché au mur, celui offert par la quincaillerie d'en face, quelqu'un a entouré le 23 en rouge. Il reste neuf jours.

« Je ne vais pas écrire une règle pour chaque cas, ma tante », dit Mateo. « Je vais donner des milliers d'exemples à l'ordinateur : combien ça a coûté, à quelle heure, à quelle distance et si le commerce était à l'étranger. Qu'il trouve le motif tout seul. »

« Et si l'ordinateur se trompe ? »

Mateo reste à fixer le cercle rouge du calendrier.

« Alors il nous dira où continuer à chercher. Mais il ne décidera pas seul. »

Dans la session précédente, tu as opéré sur des tenseurs complets d'un seul coup : les quantités par les prix de toute une semaine, d'un seul coup, avec le *broadcasting* et avec `@`. Aujourd'hui, ces deux outils se mettent à travailler seuls : tu vas voir un programme **apprendre**.

C'est la session la plus difficile du cours, alors on va doucement. Si quelque chose ne te paraît pas clair du premier coup, c'est normal : relis-le avec le code qui tourne sous tes yeux.

## Révision pour commencer

Réponds sans regarder la session 6. Les réponses se trouvent dans les solutions, à la fin.

1. Quand deux formes peuvent-elles être « étirées » pour opérer ensemble ?
2. Quelle différence y a-t-il entre `*` et `@` ?
3. Quelle forme sort d'un tableau `(7, 3) @` un vecteur `(3,)` ?

> **Le code de cette session** se trouve dans le dépôt public du cours, dans le dossier `sesion-07` : [github.com/HabilMX/curso-tensores](https://github.com/HabilMX/curso-tensores/tree/main/sesion-07). Écris-le toi-même d'abord, car c'est ainsi qu'on apprend ; le dépôt sert à comparer quand quelque chose ne donne pas le même résultat chez toi. Chaque programme est accompagné de la sortie qu'il doit te donner.

## Objectifs de cette session

| # | À la fin, tu peux… | Démontré par |
|---|---|---|
| 1 | **Nommer** dans l'ordre les cinq étapes de la boucle d'entraînement (prédire, mesurer la perte, effacer les anciennes pentes, calculer la pente, tourner les boutons) et **dire** quelle ligne de code fait chacune | Exercice 26 |
| 2 | **Prédire** ce qui arrive à la perte si tu changes le nombre d'époques ou le taux d'apprentissage, et **le vérifier** en exécutant le modèle | Exercices 26 et 27 |
| 3 | **Calculer** avec des tenseurs l'exactitude (*accuracy*), le rappel (*recall*) et la précision d'un détecteur, et **expliquer** pourquoi l'exactitude seule est trompeuse | Exercice 28 |
| 4 | **Relier** ce qu'a appris le modèle à la règle cachée qui a généré les données | Exercice 29 |

**Ce dont tu as besoin :** le même environnement que dans la session 2 (Python 3 avec PyTorch dans ton venv) et tout ce qui a été vu dans les sessions 1 à 6 : créer des tenseurs, `shape`, `dtype`, indexer, réduire avec `sum` et `mean`, `reshape`, le *broadcasting* et `@`. Aujourd'hui, tu n'installes rien.

Tu dois aussi reconnaître trois éléments de Python : `for` répète les lignes indentées ; `def nom(X):` donne un nom à une recette réutilisable ; et `with ...:` regroupe des lignes qui s'exécutent sous une même règle. Tu n'as pas encore besoin de les maîtriser : copie les blocs complets et ne change que ce qu'indique chaque exercice.

---

## Pourquoi c'est important

Jusqu'à aujourd'hui, c'est toi qui écrivais la règle : additionne cet axe, multiplie ces deux tableaux. Mateo a un autre genre de problème. Il voit des milliers de débits et sait lesquels se sont révélés frauduleux, mais il n'a pas de phrase courte qui explique tous les cas.

Chaque débit apporte des indices. Montant. Heure. Distance. Commerce étranger. Un seul indice ne condamne personne. Plusieurs ensemble peuvent justifier une vérification. Le travail du programme, c'est de trouver cette combinaison dans d'anciens exemples, pas de deviner l'histoire d'une personne.

Pense à la façon dont tu as appris à savoir si un marché est bon. Personne ne t'a donné de manuel. Tu y es allé souvent. Tu as vu où il y avait la queue, quel étal vendait des fruits mûrs et lequel cachait les abîmés en dessous. Avec le temps, sans pouvoir l'expliquer en une phrase, tu savais déjà de quel étal ne pas t'approcher. C'était apprendre à partir d'exemples.

Ou pense à ta grand-mère avec ses haricots. Elle ne mesure rien. Elle goûte, il manque du sel, elle en ajoute une pincée, elle goûte de nouveau. Personne ne lui a jamais donné la formule exacte : elle l'a trouvée à force de corriger. L'ordinateur va faire exactement ça, mais avec des nombres et des milliers de fois par seconde.

Alors on retourne la situation : **au lieu d'écrire la règle, tu donnes des exemples au programme et tu le laisses chercher une règle utile.** C'est ce qu'on appelle entraîner un modèle. C'est derrière une grande partie de ce qu'on appelle aujourd'hui l'intelligence artificielle.

La bonne nouvelle, c'est qu'aucune magie nouvelle n'apparaît. Entraîner, c'est **du pur calcul avec des tenseurs**. Les données sont un tenseur. Les boutons que le modèle apprend sont des tenseurs. La prédiction utilise `@`. Et à la fin, Mateo a encore besoin de comparer, de poser des questions et de réunir des preuves pour que quelqu'un examine le cas.

> **Dans le carnet de Mateo :** un modèle ne prouve pas qu'un débit était une fraude. Il ne fait que trier des milliers d'indices et signaler lesquels valent la peine d'être examinés en premier.

---

## Étape 1 — L'idée en une phrase

Mateo commence petit. S'il comprend un bouton, il peut en comprendre quarante-neuf. Si la perte baisse dans le petit exemple, il saura déjà quoi chercher quand on entraînera sur les achats.

Entraîner un modèle, c'est ça, répété de nombreuses fois :

1. **Prédire :** le modèle devine.
2. **Mesurer :** tu compares sa supposition à la bonne réponse et tu en tires un nombre qui dit à quel point il s'est trompé.
3. **Effacer les anciennes pentes :** tu nettoies le calcul du tour précédent (à l'étape 2, tu verras pourquoi c'est nécessaire).
4. **Calculer la pente :** tu découvres dans quel sens bouger chaque bouton pour qu'il se trompe un tout petit peu moins.
5. **Tourner les boutons :** tu les déplaces d'un petit pas dans cette direction.

Et on recommence : le `for` répète ces cinq étapes.

Comme les lancers francs sur le terrain : tu tires, tu vois que tu es parti à gauche, tu corriges un peu vers la droite et tu tires de nouveau. Personne ne t'a donné la formule du lancer parfait : tu l'as trouvée à force de corriger.

Avant de voir le détecteur de fraude, qui a 49 boutons, on va faire la boucle complète avec **un seul**.

> **20 h 24.** « Cinq étapes, et c'est tout ? » Mateo fait la moue. « Ça paraît trop facile. Si c'était comme ça, n'importe qui ferait de l'intelligence artificielle. »
>
> « N'importe qui fait des tortillas, mon petit », dit doña Lupe sans lever les yeux de ses pièces. « Bien les faire, c'est autre chose. La recette est courte ; ce qui est long, c'est la pratique. »
>
> Mateo rit, mais il reste songeur. Elle a raison. La recette de la boucle tient sur une serviette en papier. Le difficile sera de lui faire confiance quand il n'y aura plus un bouton, mais quarante-neuf, et qu'il ne pourra pas voir de ses propres yeux dans quel sens ils bougent. C'est pour ça qu'il commence avec un seul. S'il le voit fonctionner en petit, il y croira en grand.

## Étape 2 — Un seul bouton : le prix d'un cahier

Doña Lupe a perdu la liste des prix. La seule chose dont elle se souvient, c'est qu'un client a emporté 3 cahiers et a payé 60 $. Combien coûte un cahier ?

Toi, tu trouves 20 de tête. Mais on va laisser PyTorch le découvrir « à l'aveugle », avec la boucle en 5 étapes. Crée `cerro.py` :

```python
# cerro.py — una sola perilla: el precio de un cuaderno
import torch

precio = torch.tensor(0.0, requires_grad=True)   # empieza en $0: ni idea
tasa = 0.05                                      # qué tan largo es cada paso

for paso in range(1, 6):
    cobro = precio * 3                    # predicción: lo que cobraría por 3 cuadernos
    perdida = (cobro - 60) ** 2           # qué tan mal: la diferencia contra $60, al cuadrado
    perdida.backward()                    # PyTorch calcula la pendiente
    a_mano = 18 * precio.item() - 360     # la misma pendiente, sacada con lápiz
    assert abs(precio.grad.item() - a_mano) < 0.001, "PyTorch y el lápiz deberían coincidir"
    print(f"paso {paso}: precio {precio.item():6.2f}  pérdida {perdida.item():8.2f}  pendiente {precio.grad.item():8.2f}")
    with torch.no_grad():
        precio -= tasa * precio.grad      # un paso cuesta abajo
    precio.grad.zero_()                   # borrar la pendiente para el siguiente paso

assert abs(precio.item() - 20) < 0.1, "Tres cuadernos de $20 dan $60"
print("precio aprendido:", round(precio.item(), 2))
```

Exécute-le avec `python3 cerro.py` :

```
paso 1: precio   0.00  pérdida  3600.00  pendiente  -360.00
paso 2: precio  18.00  pérdida    36.00  pendiente   -36.00
paso 3: precio  19.80  pérdida     0.36  pendiente    -3.60
paso 4: precio  19.98  pérdida     0.00  pendiente    -0.36
paso 5: precio  20.00  pérdida     0.00  pendiente    -0.04
precio aprendido: 20.0
```

En cinq pas, il est arrivé à 20 $ sans que personne lui donne la réponse. Allons-y ligne par ligne, parce que toute la session est là, en miniature.

**Le bouton.** `precio` est le nombre que le modèle peut déplacer. Il commence à 0. Le `requires_grad=True` dit à PyTorch : « note tout ce que tu fais à ce nombre, parce que je vais ensuite te demander sa pente ». C'est mettre le magnétophone en marche avant la réunion.

Pourquoi « bouton » ? Pense à la cuisinière. Le bouton ne sait pas cuisiner. Il ne fait que monter ou baisser la flamme. Celui qui décide de combien le tourner, c'est celui qui goûte le plat. Ici, c'est pareil : `precio` ne sait rien des cahiers. C'est un nombre qu'on peut déplacer un tout petit peu d'un côté ou de l'autre. Celui qui décide de combien le déplacer, c'est la boucle.

**La prédiction.** `cobro = precio * 3` : avec le prix qu'il croit maintenant, combien encaisserait-il pour 3 cahiers ? À l'étape 1, 0 $.

**La perte.** `(cobro - 60) ** 2` : la différence entre ce qu'il a prédit et ce qui s'est réellement passé, au carré. Le carré fait que la perte n'est jamais négative (dépasser de 10 $ ou rester 10 $ en dessous coûte pareil) et que les grosses erreurs pèsent beaucoup plus. À l'étape 1, elle vaut 3 600 : il s'est très mal débrouillé. **La perte est une note à l'envers : plus elle est basse, mieux c'est.** Zéro, c'est parfait.

C'est comme au golf, où gagne celui qui joue le moins de coups. Ou comme les amendes du code de la route : moins tu en as, mieux tu t'es conduit. La perte ne te dit pas *ce que* tu as mal fait. Elle te dit seulement *de combien*. Ça suffit pour commencer à corriger.

**La pente.** Imagine-toi debout sur une colline **les yeux bandés**, en voulant arriver au fond de la vallée. Tu ne vois rien, alors tu tâtes le sol du pied : si ça descend devant, tu fais un pas en avant ; si ça monte, tu recules. C'est ça, la **pente** (dans les livres, le **gradient**) : de quel côté le sol descend et à quel point il est raide. La colline, c'est la perte ; l'endroit où tu te tiens, c'est la valeur du bouton ; et le fond, c'est la perte la plus basse possible.

Fais-le pour de vrai, même en imagination. Il fait nuit, il y a une coupure de courant dans tout le quartier et tu dois descendre la rue en pente de chez toi sans lampe. Tu ne vois pas le coin de la rue. Mais tes pieds sentent l'inclinaison. Là où ça paraît plus raide, tu fais un pas plus long. Là où le sol est presque plat, tu fais de petits pas, parce que tu es arrivé ou presque. Tu n'as jamais vu le plan de la rue. Il t'a suffi de sentir le sol, pas à pas. C'est ce que fait l'ordinateur : il ne voit pas la forme complète de la colline, il sent seulement la pente là où il se trouve.

`perdida.backward()`, c'est tâter le sol. PyTorch relit ce qu'a enregistré le magnétophone (tu as multiplié par 3, soustrait 60, élevé au carré) et calcule la pente tout seul. C'est ce qu'on appelle l'**autograd** : des pentes automatiques. Le résultat se retrouve dans `precio.grad`.

Tu ne le crois pas ? C'est pour ça qu'il y a l'`assert` avec `a_mano` : avec le calcul du lycée, la pente vaut `18 × precio − 360`, et l'`assert` vérifie à chaque étape que PyTorch dit la même chose. Facultatif, ne l'apprends pas par cœur : en dérivant `(3 × precio − 60)²`, on obtient `2 × (3 × precio − 60) × 3`, qui se simplifie en `18 × precio − 360`. Tu n'as pas besoin de savoir dériver ; cette ligne vérifie seulement que PyTorch a fait le même calcul, pour que tu voies qu'il n'y a pas de truc.

Et cet `assert` est une vérification automatique : `assert condition, "message"` n'affiche rien si la condition est vraie. Si elle est fausse, il arrête le programme et affiche le message. Pendant que tu apprends, ne supprime pas les `assert` : ce sont tes feux de signalisation.

**Le pas.** `precio -= tasa * precio.grad`. La pente vaut −360 : négative, ça veut dire que le sol descend si tu augmentes le prix. Soustraire un négatif, c'est additionner : 0 − 0.05 × (−360) = 18. Le prix saute à 18 $. Le `tasa` (*learning rate*, taux d'apprentissage) est **la longueur de chaque pas** : trop court, et tu mets une éternité ; trop long, et tu passes de l'autre côté de la vallée. C'est comme se garer en marche arrière dans une place étroite : si tu tournes le volant par à-coups, tu montes sur le trottoir ; si tu le tournes millimètre par millimètre, celui de derrière commence à klaxonner. Le `with torch.no_grad():` coupe le magnétophone pendant que tu tournes le bouton, parce qu'ajuster ne fait pas partie du calcul.

**Effacer la pente.** PyTorch **additionne** chaque nouvelle pente à celle qu'il avait déjà. Si tu ne l'effaces pas avec `precio.grad.zero_()`, le pas suivant utiliserait l'ancienne plus la nouvelle et irait trop loin.

Remarque : la perte passe de 3 600 à 36 puis à 0.36. Près du fond, la pente devient toute petite, donc les pas raccourcissent tout seuls. Personne ne lui a dit « freine » : c'est la colline qui le lui a dit.

> **20 h 36.** Doña Lupe se penche par-dessus l'épaule de Mateo et voit le prix arriver à 20.
>
> « Vingt pesos. Ça, je te le disais sans ordinateur. »
>
> « Je sais, ma tante. Mais lui ne le savait pas. Il a commencé à zéro, sans la moindre idée, et il y est arrivé tout seul. »
>
> « Comme ton cousin avec le vélo. Il est tombé une vingtaine de fois et, à la vingt et unième, il roulait tout seul. »
>
> Mateo rit et regarde de nouveau les cinq lignes. Il n'a pas trouvé une réponse cachée. Il a corrigé une mauvaise réponse jusqu'à ce qu'elle cesse d'être mauvaise. Et il comprend quelque chose qu'il n'attendait pas : le premier essai n'a pas d'importance. Il peut commencer très mal. Ce qui compte, c'est que chaque tour soit un tout petit peu moins mauvais. La suite, c'est la même chose, avec 4 000 achats et 49 boutons.

## Étape 3 — Les données : 4 000 achats inventés

Les achats, **nous les inventons** : il n'y a de données personnelles de personne, ça fonctionne sans internet et, surtout, **nous connaissons la règle qui décide ce qui est une fraude**, donc nous pouvons vérifier si le modèle l'a découverte. C'est la répétition générale de Mateo avant de reprendre les débits de Nord.

Pour chaque achat, nous gardons quatre données : le **montant** (de 0 $ à presque 20 000 $), l'**heure** (de 0 à 23), la **distance** jusqu'au domicile du client (de 0 à presque 500 km) et si le commerce est **étranger** (1 = oui, 0 = non ; 1 sur 10).

Et voici **la règle cachée**, que seuls toi et moi connaissons. Chaque achat déclenche des « voyants rouges » :

- un voyant si le montant dépasse 15 000 $,
- un voyant si l'achat a lieu au petit matin (avant 5 heures),
- un voyant s'il a lieu à plus de 400 km du domicile,
- un voyant si le commerce est étranger.

**Si un achat déclenche deux voyants ou plus, c'est une fraude.** Un seul ne suffit pas : acheter un téléviseur cher à midi, c'est normal ; l'acheter à 3 heures du matin, ça commence à sentir bizarre.

Le modèle ne voit **jamais** cette règle écrite : il reçoit seulement les quatre données de chaque achat et, pendant qu'il étudie, la bonne réponse (fraude ou non). Ce fichier fait deux travaux. Jusqu'ici, c'est **nous** qui fabriquons le laboratoire avec `focos` ; à partir de `predecir(X_ent)`, le modèle ne reçoit que `X_ent`. `focos` n'entre jamais dans `predecir`, et `y_ent` ne sert qu'à mesurer à quel point il a mal prédit. Avec des données réelles, `y` viendrait de cas déjà examinés.

Crée `detector.py`. On procède par parties. Copie les blocs 1 à 7 dans le même fichier, dans l'ordre où ils apparaissent (dans le code, ils sont numérotés par des commentaires). Voici la première :

```python
# detector.py — un detector de fraude que aprende de los datos
import torch

torch.manual_seed(42)          # semilla fija: los mismos números "al azar" cada vez

# 1) DATOS INVENTADOS: 4,000 compras con tarjeta
n = 4000
monto      = torch.rand(n) * 20_000               # pesos, de 0 a 20,000
hora       = torch.randint(0, 24, (n,)).float()   # hora del día, de 0 a 23
distancia  = torch.rand(n) * 500                  # km desde la casa del cliente
extranjero = (torch.rand(n) < 0.1).float()        # 1 = comercio de otro país

# 2) LA REGLA ESCONDIDA (solo nosotros la conocemos; el modelo nunca la ve)
focos = (monto > 15_000).float() + (hora < 5).float() + (distancia > 400).float() + extranjero
y = (focos >= 2).float()       # la etiqueta: 1 = fraude, 0 = normal

# 3) EL TENSOR DE ENTRADA: compras × datos, todo entre 0 y 1
X = torch.stack([monto / 20_000, hora / 23, distancia / 500, extranjero], dim=1)
print("X:", tuple(X.shape), "| y:", tuple(y.shape), "| % fraude:", round(y.mean().item() * 100, 1))
```

Des fonctions nouvelles, toutes simples :

- `torch.rand(n)` donne `n` nombres au hasard entre 0 et 1 ; multipliés par 20 000, des montants de 0 à 20 000 (*broadcasting*, comme dans la session 6). `torch.randint(0, 24, (n,))` donne des entiers au hasard de 0 à 23.
- `torch.manual_seed(42)` fixe le « hasard ». L'ordinateur ne lance pas de dés : il suit une recette et, avec la même graine, il donne les mêmes nombres chaque fois que tu l'exécutes sur le même ordinateur.
- `(monto > 15_000)` donne un `True` ou un `False` par achat ; `.float()` le transforme en 1.0 ou 0.0. En additionner quatre compte les voyants des 4 000 achats à la fois.
- `torch.stack([...], dim=1)` colle les quatre listes comme **colonnes** d'un tableau.

Diviser chaque colonne (`monto / 20_000`, `hora / 23`…) met tout entre 0 et 1, pour que le montant (des milliers) n'écrase pas l'heure (des dizaines) simplement parce que c'est un nombre plus grand.

La sortie de cette partie :

```
X: (4000, 4) | y: (4000,) | % fraude: 15.8
```

**`X` a la forme `(4000, 4)` : achats × données de chaque achat.** Une feuille de 4 000 lignes et 4 colonnes. Le premier axe s'appelle le **lot** (*batch*, combien d'exemples) et le second, les **caractéristiques** (*features*, ce que l'on sait de chacun). Cette forme, tu vas la voir partout.

**`y` a la forme `(4000,)` :** une réponse par achat. On les appelle des **étiquettes** (*labels*) : la colonne des bonnes réponses de l'examen. 15.8 % des achats sont des fraudes.

Pour le voir de près, ajoute `print(X[:3])` et `print(y[:3])` juste après ce `print`. C'est une inspection temporaire : exécute-les une fois ici et supprime-les avant de comparer ta sortie à la « sortie complète » de l'étape 8 :

```
tensor([[0.8823, 0.1304, 0.4309, 0.0000],
        [0.9150, 0.6087, 0.3986, 0.0000],
        [0.3829, 0.4348, 0.1907, 0.0000]])
tensor([1., 0., 0.])
```

L'achat 0 est de 17 645 $ (0.8823 × 20 000) à 3 heures du matin (0.1304 × 23) : deux voyants, fraude. L'achat 1 est plus cher, 18 300 $, mais à 2 heures de l'après-midi : un seul voyant, normal. C'est cette nuance que le modèle doit découvrir.

> **20 h 52.** « Ça y est », dit Mateo, très sûr de lui. « Tout ce qui dépasse quinze mille, c'est une fraude. Je le marque et c'est terminé. »
>
> Doña Lupe arrête de compter ses pièces.
>
> « Et la dame qui est venue en août payer d'un coup les fournitures scolaires de ses quatre enfants ? C'est une voleuse, elle aussi ? »
>
> Mateo ouvre la bouche et la referme. Il regarde l'achat 1 à l'écran : 18 300 $, à deux heures de l'après-midi, normal. Sa règle l'aurait accusée. **L'indice ne vit pas dans une seule colonne.** Ni le montant seul, ni l'heure seule. Ce qui est bizarre, c'est quand ils se cumulent.
>
> « Une hirondelle ne fait pas le printemps », conclut doña Lupe, et elle retourne à ses pièces.
>
> Mateo raye sa règle. C'est pour ça que ce n'est pas lui qui va l'écrire.

## Étape 4 — Étudier et passer l'examen

Si tu notes un élève avec les mêmes questions que celles qu'il a révisées, tu ne sais pas s'il a appris ou mémorisé. Avec les modèles, c'est pareil :

```python
# 4) Estudia con 3,200 compras; el examen es con las otras 800
X_ent, y_ent = X[:3200], y[:3200]
X_exa, y_exa = X[3200:], y[3200:]
```

C'est découper selon le premier axe, comme dans la session 3. Les 800 derniers achats sont mis sous clé : le modèle ne les voit jamais pendant qu'il apprend.

Mateo appelle les 3 200 premiers « étude » et les 800 autres « examen ». Ce n'est pas décoratif. S'il lui montrait les réponses de l'examen, le chiffre final sonnerait bien et ne dirait rien.

C'est la différence entre apprendre le corrigé par cœur et comprendre le sujet. Si l'examen reprend exactement les questions du corrigé, tu as dix sur dix même sans rien savoir. Si le professeur change les nombres, on voit qui a compris. Les 800 achats mis de côté, c'est l'examen avec des nombres changés.

Un détail honnête : dans les exercices 26 et 27, tu vas comparer des époques et des taux en regardant ces mêmes 800 achats. Dès que tu choisis avec eux, ce n'est plus un examen propre, mais un entraînement pour décider. Dans un vrai projet, on met de côté un troisième groupe, l'examen final, que personne ne consulte avant le dernier jour.

> **21 h 05.** Le téléphone de doña Lupe vibre sur le comptoir. C'est Fer, depuis Nord : « J'ai fermé. Aujourd'hui, tout s'est payé en espèces. Trois clients habituels sont repartis sans rien acheter parce qu'ils n'en avaient pas sur eux. Je vous envoie la photo des notes du jour. »
>
> Doña Lupe lit le message deux fois et le montre à Mateo.
>
> « Et moi qui, il y a un mois, ne voulais même pas prononcer son nom », dit-elle à voix basse. « On voit le mal partout, même là où il n'y en a pas. »
>
> Mateo ne répond pas. Il range la photo des notes dans un nouveau dossier. Si le modèle dit quelque chose sur les débits de Nord, il aura besoin de quoi le comparer. Des données réelles, mises de côté, que le modèle n'aura jamais vues. Comme l'examen.

## Étape 5 — Les boutons et la prédiction avec `@`

Maintenant, les boutons. Dans `cerro.py`, il y en avait un. Ici, il y a quatre tenseurs de boutons :

```python
# 5) LAS PERILLAS: empiezan al azar y el modelo las va girando
W1 = (torch.randn(4, 8) * 0.5).requires_grad_()
b1 = torch.zeros(8, requires_grad=True)
W2 = (torch.randn(8) * 0.5).requires_grad_()
b2 = torch.zeros((), requires_grad=True)   # () = un solo número: la perilla final

def predecir(X):
    oculta = torch.relu(X @ W1 + b1)   # (compras, 4) @ (4, 8) -> (compras, 8)
    return oculta @ W2 + b2            # (compras, 8) @ (8,)   -> (compras,)
```

Dans les livres, ces boutons s'appellent des **poids** (le `W`, de *weights*) et des **biais** (le `b`, de *bias*). `W1` en a 4 × 8 = 32, `b1` en a 8, `W2` en a 8 et `b2` en a 1 : **49 boutons**. Ils commencent au hasard (`torch.randn`) ou à zéro, comme le prix a commencé à 0 $. Deux détails de syntaxe : le `()` de `torch.zeros((), ...)` veut dire « un seul nombre », le bouton final ; et le tiret bas de `requires_grad_()` met en marche le « magnétophone » sur ce même tenseur, déjà créé.

Tu as déjà vu la console d'un DJ de quartier dans une fête ? Une table pleine de boutons : graves, aigus, volume de chaque micro, écho. Aucun bouton seul ne fait que ça sonne bien. Ce qui sonne bien, c'est la combinaison. Et le DJ ne la calcule pas : il en tourne un, écoute, en tourne un autre, écoute de nouveau. Les **poids**, c'est ça. Chacun dit combien une donnée compte dans un mélange. Chaque poids de `W1` règle combien une donnée compte dans l'un des soupçons intermédiaires : grand, « monte ça » ; proche de zéro, « ignore ça » ; négatif, « retire-le ». À lui seul, il ne mesure pas l'importance globale de la donnée, ni si c'est un signe de fraude ou de normalité : ça dépend aussi de `W2`, des biais et de `relu`. La différence, c'est que notre DJ a 49 boutons et les tourne tous à la fois, guidé par la perte plutôt que par l'oreille.

Lis `predecir` en ne regardant que les formes, comme dans la session 6 :

- `X @ W1` : `(compras, 4) @ (4, 8)` donne `(compras, 8)`. Chaque achat passe de 4 données à **8 « soupçons » intermédiaires**, chacun étant un mélange différent de montant, heure, distance et étranger. L'un pourrait finir par se concentrer sur « cher et au petit matin » ; un autre, sur « loin et étranger ». Personne ne lui dit lesquels : il les découvre en tournant les boutons.
- `+ b1` : un vecteur `(8,)` additionné à chaque ligne. Du *broadcasting*.
- `torch.relu(...)` : **si un nombre est négatif, il le met à zéro ; s'il est positif, il le laisse tel quel.** « Si ce soupçon ne s'applique pas, n'en parle même pas. » Sans `relu`, les deux couches se fondraient en un seul mélange, un seul calcul de multiplications et d'additions, qui ne parvient pas à imiter les seuils de la règle (plus de 15 000 $, avant 5 heures, plus de 400 km) ni la façon dont ils se combinent. (Si les voyants arrivaient déjà comptés comme entrées, « deux ou plus » serait bien un seul mélange.)
- `oculta @ W2 + b2` : `(compras, 8) @ (8,)` donne `(compras,)`, le calcul de la troisième question de la révision. **Un nombre par achat** : le **score**.

**Score supérieur à zéro : le modèle dit « fraude » ; inférieur ou égal à zéro : « normal ».** Plus il est loin de zéro, plus le modèle est sûr de lui, ce qui n'est pas une garantie. Dans les livres, ce score s'appelle un **logit** (d'où le nom de la perte de l'étape 6). Un score de 0 équivaut à « moitié-moitié » : 50 % de probabilité de fraude selon le modèle. C'est pour ça que le seuil est à zéro, et que le code tranche l'égalité exacte comme « normal » (`> 0`, pas `>= 0`). C'est ce qu'on appelle la **prédiction** ou la passe avant (*forward*) : les données entrent d'un côté et le score sort de l'autre, comme la pâte dans la machine à tortillas.

> **21 h 20.** Mateo fixe `W1` comme s'il lui devait de l'argent.
>
> « Et lequel des huit, c'est celui de “cher et au petit matin” ? » demande-t-il à voix haute, à personne.
>
> Il affiche `W1` avec `print(W1)`. Il obtient trente-deux nombres au hasard, aucun avec un nom. Il s'impatiente un peu. Il voulait voir la règle écrite, bien propre, dans une case. Ça n'arrivera pas. Les soupçons n'ont pas d'étiquette : ils se forment tout seuls pendant que les boutons tournent.
>
> « Tu t'es déjà disputé avec la machine ? » demande doña Lupe.
>
> « Pas encore. C'est juste qu'elle ne me dit pas ce qu'elle pense. »
>
> « Comme ton oncle. Trente ans de mariage et je ne sais toujours pas. »
>
> Mateo éclate de rire et décide de ne pas se battre contre ce qu'il ne peut pas voir. Il lui suffit de surveiller les formes : quatre données entrent, huit signaux sortent et, à la fin, il reste un score par débit.

## Étape 6 — La perte pour un « oui ou non »

Dans `cerro.py`, la perte était la différence au carré. Pour les questions de type « oui ou non », on en utilise une autre, toute prête dans PyTorch :

```python
perdida_fn = torch.nn.BCEWithLogitsLoss()
optimizador = torch.optim.SGD([W1, b1, W2, b2], lr=1.0)
```

Ne te laisse pas effrayer par le nom. `BCEWithLogitsLoss` joue le rôle de notre soustraction au carré, mais pour fraude/pas fraude : elle reçoit des scores et des étiquettes et renvoie **un seul nombre** qui dit à quel point il s'est trompé sur tous les achats ensemble. Sa règle relève du bon sens : si c'était une fraude et que le score était très positif, perte presque nulle ; si c'était une fraude et que le score était très négatif, perte énorme (être très sûr de soi et se tromper, c'est le pire) ; si le score est proche de zéro, perte moyenne.

Pense à l'ami qui te jure que le match est déjà gagné et qui, à la fin, perd 3 à 0. Ce n'est pas la même chose que l'ami qui a dit « qui sait ». Les deux se sont trompés, mais au premier, tu fais payer le pari plus cher. Cette perte fait payer de la même façon : douter coûte un tout petit peu ; se tromper en étant très sûr coûte énormément. Ainsi, le modèle apprend à ne pas afficher une assurance qu'il n'a pas.

La deuxième ligne, c'est l'**optimiseur**, celui qui tourne les boutons. `SGD` fait la même chose que ce que tu as fait manuellement dans `cerro.py`, `perilla -= tasa * pendiente`, mais avec les 49 à la fois. `lr=1.0`, c'est le taux.

> **21 h 41.** La machine a déjà les données, les boutons et une façon de noter ses erreurs. Il ne reste plus qu'à la lancer.
>
> Mateo a le doigt au-dessus de la touche et n'appuie pas. Et s'il n'apprend rien ? Et si la perte reste où elle est et qu'il perd sa soirée, puis la suivante, et que le 23 arrive les mains vides ?
>
> Doña Lupe arrête la cafetière, qui sent déjà le brûlé.
>
> « Qui ne risque rien n'a rien, mon petit. Et qui n'appuie pas sur le bouton ne saura jamais. »
>
> L'horloge murale indique presque dix heures. Il reste neuf jours, et chacun pèse. Mateo respire. Chaque tour de la boucle doit le rapprocher d'une piste, pas seulement d'un joli nombre.

## Étape 7 — La boucle complète et les époques

Maintenant, oui, le cœur de la session :

```python
# 6) EL CICLO DE ENTRENAMIENTO: una época = una vuelta a las 3,200 compras
for epoca in range(1, 301):
    puntaje = predecir(X_ent)               # 1. predecir
    perdida = perdida_fn(puntaje, y_ent)    # 2. medir qué tan mal le fue
    optimizador.zero_grad()                 # 3. borrar las pendientes de la vuelta anterior
    perdida.backward()                      # 4. calcular la pendiente de cada perilla
    optimizador.step()                      # 5. girar cada perilla un pasito cuesta abajo
    if epoca in (1, 10, 50, 100, 200, 300):
        print(f"época {epoca:3d}   pérdida {perdida.item():.4f}")
```

Compare-le avec `cerro.py`. C'est la même boucle, ligne par ligne :

| Étape | Dans `cerro.py` | Dans `detector.py` |
|---|---|---|
| Prédire | `cobro = precio * 3` | `puntaje = predecir(X_ent)` |
| Mesurer | `(cobro - 60) ** 2` | `perdida_fn(puntaje, y_ent)` |
| Effacer les anciennes pentes | `precio.grad.zero_()` | `optimizador.zero_grad()` |
| Calculer les pentes | `perdida.backward()` | `perdida.backward()` |
| Tourner les boutons | `precio -= tasa * precio.grad` | `optimizador.step()` |

Chaque tour du `for` montre au modèle **la totalité** des 3 200 achats d'étude. Un tour complet de toutes les données s'appelle une **époque** (*epoch*) : réviser tout le programme une fois. Ici, on lui en donne 300.

Pourquoi autant ? Parce qu'à chaque tour, les boutons ne bougent que d'un petit pas. C'est comme apprendre une chanson à la guitare : la première fois que tu la joues en entier, tu te trompes à chaque changement d'accord. La dixième, seulement sur le difficile. La centième, tu n'y penses même plus. Chaque époque, c'est jouer la chanson entière une fois de plus. Personne ne l'apprend du premier coup, ni toi ni l'ordinateur.

Exécute ce que tu as jusqu'ici et regarde la perte :

```
época   1   pérdida 0.6819
época  10   pérdida 0.4098
época  50   pérdida 0.2528
época 100   pérdida 0.2070
época 200   pérdida 0.1984
época 300   pérdida 0.1910
```

**La perte d'étude baisse :** le modèle réduit ses erreurs sur les 3 200 achats qu'il a déjà vus. C'est un bon signe, mais ça ne prouve pas encore qu'il fonctionne avec des achats nouveaux : c'est ce que vérifie l'examen de l'étape 8. Elle baisse vite au début, puis de plus en plus lentement, comme avec le cahier : près du fond, le sol est à peine incliné. Sur ma machine, les 300 époques ont pris moins d'un dixième de seconde.

> **Pourquoi 300 et pas plus ?** Nous l'avons mesuré. Chaque ordinateur arrondit les décimales dans un ordre un peu différent ; la différence est microscopique, de l'ordre de la septième décimale. Avec des pas longs comme `lr=1.0`, après environ 300 tours, ces différences commencent à grandir : avec 400, deux ordinateurs avec les mêmes versions ont obtenu des nombres différents de fraudes attrapées. Avec 300, le résultat n'a pas bougé, même quand nous avons introduit exprès des différences mille fois plus grandes que celles qu'il y a d'un ordinateur à l'autre. C'est pour ça que le cours utilise 300 : pour que, si ton nombre est différent, tu saches que l'erreur est dans ton code et pas dans ta machine.

**21 h 47.** Mateo appuie sur la touche.

Le terminal crache la première ligne avant qu'il ait le temps de cligner des yeux : `época 1 pérdida 0.6819`. Il a un nœud à l'estomac. Ce nombre ne lui dit encore rien. C'est le modèle qui devine à l'aveugle, avec les boutons au hasard, comme un chien qui entend son nom dans une autre langue.

Puis apparaît la deuxième : `0.4098`.

Mateo se penche vers l'écran.

`0.2528`.

« Ma tante. Ma tante, viens voir. »

Doña Lupe arrive en s'essuyant les mains sur son tablier.

« Qu'est-ce qui se passe ? Il est en panne ? »

« Non. Regarde. Elle baisse. »

`0.2070`. `0.1984`. `0.1910`.

Six nombres, l'un sous l'autre, chacun plus petit que le précédent. C'est déjà fini. Ça a duré moins de temps qu'il n'en faut à l'eau du café pour bouillir. Mateo reste à regarder la colonne comme on regarde le tableau d'affichage à la fin d'un match qu'on ne pensait pas gagner.

Personne n'a écrit à la machine la règle « deux voyants ou plus ». Personne ne lui a parlé de petit matin ni de commerces étrangers. On lui a bien donné 3 200 achats, leurs quatre données et la réponse fraude/pas fraude de chacun, et une façon de mesurer à quel point elle se trompait. Avec ces exemples, tour après tour, elle a déplacé ses 49 boutons jusqu'à se tromper moins. Mateo ressent quelque chose d'étrange, entre la fierté et le frisson. Il a écrit quelques lignes. Le reste, c'est elle qui l'a fait.

« Et c'est bien, ça ? » demande doña Lupe.

« C'est qu'elle est en train d'apprendre, ma tante. Je la vois apprendre. »

Doña Lupe regarde l'écran, puis son neveu, qui a les yeux brillants comme quand il était petit et que sa toupie tournait enfin.

« Eh bien, pourvu qu'elle apprenne vite. Le 23 n'attendra pas. »

Mateo acquiesce, mais il ne l'entend plus tout à fait. **La perte est un signal, pas une sentence.** Elle lui dit que le modèle se trompe moins sur l'étude. Il manque encore l'essentiel : voir ce qu'il fait quand on lui retire les réponses.

## Étape 8 — L'examen : 91 %

Il manque l'essentiel : est-ce qu'il s'en sort bien avec des achats qu'il **n'a jamais vus** ? Ajoute la dernière partie :

```python
# 7) EL EXAMEN: 800 compras que el modelo nunca vio
with torch.no_grad():
    dice_fraude = (predecir(X_exa) > 0).float()   # puntaje positivo = "esto es fraude"

exactitud = (dice_fraude == y_exa).float().mean().item()
atrapados = (dice_fraude * y_exa).sum().item()          # dijo fraude y sí era
falsas    = (dice_fraude * (1 - y_exa)).sum().item()    # dijo fraude y no era
escapados = ((1 - dice_fraude) * y_exa).sum().item()    # dijo normal y era fraude

print("exactitud:", round(exactitud, 3))
print("fraudes en el examen:", int(y_exa.sum().item()))
print("atrapados:", int(atrapados), "| se escaparon:", int(escapados), "| alarmas falsas:", int(falsas))
print("recall:", round(atrapados / (atrapados + escapados), 3),
      "| precisión:", round(atrapados / (atrapados + falsas), 3))
```

Les deux derniers chiffres qu'il affiche (rappel et précision) n'ont pas encore de nom pour toi : ignore-les jusqu'à l'étape 9, où ils sont expliqués et où tu les reliras. Pour l'instant, regarde l'exactitude et les trois comptes.

L'astuce, c'est `*` élément par élément, celui de la session 6 : `dice_fraude * y_exa` ne donne 1 que là où **les deux** valent 1 (il a dit fraude **et** c'en était une). `1 - y_exa` inverse les uns et les zéros, donc `dice_fraude * (1 - y_exa)` donne 1 là où il a dit fraude et que ce **n'en** était **pas** une. Tu additionnes et tu as combien de cas de chaque sorte.

Le script complet fait 61 lignes, 41 sans commentaires ni lignes vides. Exécute-le avec `python3 detector.py`. Sortie complète, identique les deux fois où je l'ai exécuté :

```
X: (4000, 4) | y: (4000,) | % fraude: 15.8
época   1   pérdida 0.6819
época  10   pérdida 0.4098
época  50   pérdida 0.2528
época 100   pérdida 0.2070
época 200   pérdida 0.1984
época 300   pérdida 0.1910
exactitud: 0.908
fraudes en el examen: 125
atrapados: 81 | se escaparon: 44 | alarmas falsas: 30
recall: 0.648 | precisión: 0.73
```

**Exactitude : 0.908.** Sur des achats qu'il n'a jamais vus, il a eu juste presque 91 fois sur 100, sans que personne lui donne la règle des voyants.

À propos de la graine : avec la même version de PyTorch et un ordinateur comme le mien, le plus probable est que tu obtiennes exactement ceci. Mais PyTorch ne garantit pas des nombres identiques d'une version à l'autre ni entre ordinateurs différents ([documentation de PyTorch sur la reproductibilité](https://docs.pytorch.org/docs/2.14/notes/randomness.html)) : les dernières décimales peuvent bouger un peu. Avec 300 époques, nous avons mesuré qu'elles ne bougent pas (voir « Pourquoi 300 et pas plus ? » à l'étape 7) ; dans les exercices avec beaucoup plus de tours, elles peuvent bouger. Ce qui **doit** toujours se produire : la perte baisse et l'exactitude reste autour de 91 %. Si tu obtiens quelque chose de très différent, consulte le tableau des erreurs.

**22 h 12.** Mateo voit le 91 et lève les bras comme s'il avait marqué un but.

« Quatre-vingt-onze, ma tante ! Il a eu juste quatre-vingt-onze fois sur cent ! »

Doña Lupe ne lève pas les bras. Elle ajuste ses lunettes et s'approche de l'écran avec le calme de quelqu'un qui a fait mille clôtures de caisse.

« Alors, il est bon à quel point ? »

« Eh bien… très bon. C'est un neuf sur dix. »

« Un neuf en quoi ? Qu'est-ce qu'elle dit, cette ligne en dessous ? »

Mateo la lit à voix haute, plus lentement : « se escaparon: 44 ».

Son enthousiasme retombe comme un ballon crevé. Quarante-quatre. Sur 125 fraudes dans l'examen, quarante-quatre sont passées comme si de rien n'était. Le modèle les a vues, examinées, et a dit « normal ».

Ce sont des achats inventés, il le sait. Mais il ne peut pas s'empêcher de leur mettre un visage. Il imagine l'une de ces quarante-quatre. Une dame qui a acheté des cahiers à Nord en septembre, a payé par carte, a dit merci et est partie. Et un matin, dans la file de la tortillería, elle ouvre l'appli de sa banque et voit un débit de dix-sept mille pesos dans un magasin où elle n'est jamais allée. Sa prime de fin d'année, ou ce qui allait être sa prime de fin d'année. Puis il imagine le garçon qui avait économisé pour un téléphone et dont on a vidé le compte un mardi à trois heures du matin. Et un autre. Et un autre. Quarante-quatre fois.

« Ce n'est pas un neuf, ma tante », dit-il, d'une toute petite voix. « Pour un devoir, neuf, c'est une bonne note. Pour 125 fraudes, quarante-quatre, c'est une porte grande ouverte. »

Doña Lupe lui pose une main sur l'épaule.

« Tu vois. Tout ce qui brille n'est pas or, mon petit. Maintenant, découvre de quoi c'est fait. »

> **Encadré d'alerte :** l'exactitude compte toutes les bonnes réponses ensemble. Elle ne te dit pas combien de fraudes il a laissé passer. C'est pour ça qu'une banque ne peut pas se contenter de ce nombre.

91 %, c'est bien ? Ça sonne comme un neuf sur dix. Voici maintenant le plus important de la session.

## Étape 9 — Le piège : le détecteur paresseux obtient 84

Imagine un « détecteur » qui dit « ce n'est pas une fraude » à tous les achats, sans les regarder.

Dans l'examen, il y a 800 achats et 125 sont des fraudes. Les 675 autres sont normaux, et le paresseux les a tous justes : 675 sur 800, c'est **84.4 %**. Tu le vérifieras avec un tenseur dans l'exercice 28.

Un programme d'une ligne qui ne sert à rien obtient 84 ; le nôtre, 91. La différence est beaucoup plus petite qu'il n'y paraissait. **L'exactitude est trompeuse quand ce qui compte est rare**, et dans une banque la fraude est généralement rare : presque tous les achats sont faits par des gens honnêtes.

Pense à un détecteur de métaux à l'entrée d'un concert. Presque personne ne porte quoi que ce soit de dangereux. Si le vigile laisse passer tout le monde sans contrôler, il « a raison » pour presque tout le public. Et pourtant c'est le pire vigile du monde, parce que le seul cas qui comptait lui a échappé. Avoir raison sur ce qui est courant, c'est facile. Le difficile, et ce qui a de la valeur, c'est d'attraper ce qui est rare.

> **22 h 31.** Mateo écrit le détecteur paresseux sur un papier, sans ordinateur : « à tout, non ». Il le regarde un moment. C'est une ligne. C'est ridicule. Et ça obtient 84.
>
> « Donc mon modèle à 49 boutons bat le fait de ne rien faire de moins de sept points », murmure-t-il.
>
> « Et c'est peu ? » demande doña Lupe.
>
> « C'est comme se vanter de courir plus vite que quelqu'un d'assis. »
>
> Doña Lupe éclate d'un rire qui fait tinter les verres de l'étagère.
>
> « Avec ça, tu ne bats même pas le camion-poubelle. »
>
> Mateo ne se vexe pas. Pour la première fois de la soirée, il sent qu'il comprend vraiment. Le nombre qui compte n'est pas le plus grand. C'est celui qui compte ce qui s'est échappé.

C'est pour ça que les banques regardent deux autres nombres. D'abord, les quatre cas possibles de l'examen :

| | C'était vraiment une fraude | C'était vraiment normal |
|---|---|---|
| **Le modèle a dit « fraude »** | 81 attrapées | 30 fausses alertes |
| **Le modèle a dit « normal »** | 44 échappées | 645 laissés passer à raison |

(Les quatre cases totalisent 800. Les bonnes réponses sont 81 + 645 = 726, et 726 / 800 = 0.9075, qui arrondi donne 0.908, l'exactitude ci-dessus.)

**Rappel (*recall*) : de toutes les fraudes, combien en a-t-il attrapé ?** Il y a eu 125 fraudes et il en a attrapé 81 : 81 / 125 = **0.648**. 44 sur 125 lui ont échappé : plus d'une fraude sur trois. Chacune de ces 44, c'est quelqu'un qui, demain, consulte son relevé de compte et trouve un débit de 17 000 $ dans un magasin où il n'est jamais allé. Il doit contester le débit, la banque examine le cas et finit souvent par lui rendre l'argent ; la peur, personne ne la lui rend. Le détecteur paresseux a un rappel **nul** : il n'en attrape pas une seule.

**Précision : de toutes les alertes qu'il a données, combien étaient fondées ?** Il a donné 81 + 30 = 111 alertes, et 81 étaient des fraudes : 81 / 111 = **0.73**. À peu près une alerte sur 4 était une fausse alerte. Chacune de ces 30, c'est une personne honnête à qui, selon la façon dont travaille la banque, on a bloqué l'achat à la caisse du supermarché, avec la file derrière, ou à qui on a demandé de confirmer que c'était bien elle avant de la laisser passer. Si ça arrive souvent, le client change de banque.

Les deux comptent. Si la banque veut attraper **toutes** les fraudes, elle marque tout comme suspect : rappel parfait et tous ses clients furieux. Si elle ne veut déranger **personne**, elle ne marque rien : zéro fausse alerte et toutes les fraudes passent, comme avec le paresseux. Le vrai travail se trouve au milieu, et il dépend de ce que coûte chaque erreur. C'est pour ça que, pour juger un système antifraude, combien de fraudes il attrape et combien de fausses alertes il donne comptent plus que l'« exactitude » seule.

**La grande leçon d'aujourd'hui :** avoir souvent raison ne suffit pas. Avant de croire un modèle, demande à quel point ce qu'il cherche est rare et combien de ces cas rares il attrape.

> **23 h 05.** Mateo dessine le tableau à quatre cases au dos d'une facture d'électricité. En haut à gauche, 81 : celles qu'il a attrapées. En bas à gauche, 44 : celles qui sont parties. En haut à droite, 30 : des gens honnêtes avec leur carte refusée à la caisse. Il reste à regarder cette case. Si la banque faisait ça aux clients de Nord, la papeterie perdrait les quelques-uns qui reviennent encore.
>
> « Ce n'est pas un nombre, ma tante. C'en est deux. Et ils se tirent l'un l'autre, comme une couverture trop courte : si tu te couvres les pieds, tu te découvres les épaules. »
>
> Doña Lupe acquiesce lentement. Celle-là, elle la comprend parfaitement. Mateo range la facture dans la poche de son blouson. Demain, il en aura besoin.

---

👉 **Si tu viens de la gestion :** tu n'as pas besoin d'entraîner le modèle pour poser la bonne question en réunion. Quand quelqu'un présente « notre modèle a raison à 91 % », demande : combien obtient un modèle qui ne fait rien ? Combien des cas qui nous importent attrape-t-il (rappel) ? Combien de ses alertes sont fondées (précision) ? Puis traduis chaque erreur en pesos : combien coûte une fraude qui s'échappe et combien coûte un client mécontent. Ce tableau à quatre cases est un outil de gestion, et il sert de la même façon pour décider à qui accorder un crédit ou quelle déclaration de sinistre examiner.

---

## Exercices de la session 7

Les solutions se trouvent à la fin. Avant chaque changement, **écris sur papier ce que tu crois qu'il va se passer**. Prédire avant d'exécuter, c'est la moitié de l'apprentissage. Remplis ce tableau sur papier (les exercices 26 et 27 te demandent ces quatre exécutions) :

| Configuration | Ma prédiction | Perte finale | Exactitude | Attrapées | Fausses alertes | Qu'ai-je appris ? |
|---|---|---:|---:|---:|---:|---|
| 300 époques, `lr=1.0` | | | | | | |
| 2000 époques, `lr=1.0` | | | | | | |
| 300 époques, `lr=0.1` | | | | | | |
| 300 époques, `lr=10` | | | | | | |

**Exercice 26 — Plus de tours, moins de tours.** Dans `detector.py`, le nombre d'époques se trouve dans `range(1, 301)`. Je l'ai déjà exécuté avec 100 époques (`range(1, 101)`) : la perte finit à 0.2070, l'exactitude à 0.902, il attrape 77 fraudes et donne 30 fausses alertes. À toi maintenant : exécute-le avec **2000** époques. Ajoute `1000, 2000` à la liste des époques affichées pour voir la perte.

a) Avant d'exécuter : la perte finale sera-t-elle supérieure ou inférieure à 0.1910 ?
b) Qu'est-il arrivé à l'exactitude, aux fraudes attrapées et aux fausses alertes ?
c) Indique dans ton code quelle ligne correspond à chacune des cinq étapes de la boucle (prédire, mesurer, effacer les pentes, calculer les pentes, tourner les boutons).

**Exercice 27 — La longueur du pas.** Reviens à 300 époques. Change maintenant le taux, `lr=1.0`, d'abord en `lr=0.1` (des pas de fourmi) puis en `lr=10` (des enjambées de géant). Avant d'exécuter chacun, prédis : la perte baisse-t-elle plus ou moins ? Pourquoi ?

Attention : avec l'un des deux, le programme plante à la fin avec une `ZeroDivisionError`. Ce n'est pas une erreur de ta part. Lis les lignes au-dessus de l'erreur et explique pourquoi c'est arrivé. Indice : regarde combien d'alertes il a données au total.

**Exercice 28 — Calcule le détecteur paresseux.** Ajoute ceci à la fin de `detector.py` et remplis les trous. Le détecteur paresseux est un tenseur de 800 zéros : « ce n'est pas une fraude » pour tous les achats de l'examen.

```python
# Ejercicio 28: el detector flojo que siempre dice "no es fraude"
siempre_no = torch.____(800)
assert siempre_no.shape == y_exa.shape, "Tiene que haber una respuesta por cada compra del examen"
exactitud_flojo = (siempre_no == ____).float().mean().item()
atrapados_flojo = (siempre_no * ____).sum().item()
print("exactitud del flojo:", round(exactitud_flojo, 3))
print("fraudes que atrapa:", int(atrapados_flojo), "de", int(y_exa.sum().item()))

# la misma cuenta con un for, como antes de los tensores
aciertos = 0
for i in range(800):
    if y_exa[i].item() == ____:
        aciertos = aciertos + 1
assert round(aciertos / 800, 3) == round(exactitud_flojo, 3), "El for y el tensor deberían coincidir"
print("El for dice lo mismo:", aciertos, "aciertos de 800")
```

**Deuxième partie.** Calcule maintenant le rappel et la précision du **détecteur entraîné**, sans regarder les deux dernières lignes de l'étape 8. Tu as déjà `atrapados`, `falsas` et `escapados` dans ton fichier :

```python
recall_modelo = ______________________________
precision_modelo = ___________________________

print("recall del modelo:", round(recall_modelo, 3))
print("precisión del modelo:", round(precision_modelo, 3))
assert round(recall_modelo, 3) == 0.648 and round(precision_modelo, 3) == 0.73, "Revisa cuál de las cuatro casillas falta en tu fracción"
```

Avec l'exécution de base (300 époques, `lr=1.0`), tu dois obtenir `0.648` et `0.73`. Si tu as changé les époques ou le taux, supprime l'`assert`.

Ensuite, réponds : quel est le rappel du détecteur paresseux ? Pourquoi ne peux-tu pas calculer sa précision ?

**Exercice 29 — Qu'a appris le modèle ?** La règle utilise les quatre données. Le modèle s'en est-il rendu compte ? Pose-lui la question **en mélangeant une colonne** : si tu mélanges l'heure entre les 800 achats de l'examen, chaque achat se retrouve avec l'heure d'un autre, au hasard. La colonne est toujours là, mais elle ne dit plus rien. Si l'exactitude chute beaucoup dans cette exécution, c'est un indice que le modèle utilisait cette colonne pour prédire. Cela ne prouve pas à soi seul qu'il a découvert la règle exacte, ni que cette colonne cause la fraude.

Tu vas le refaire pour chacune des 4 colonnes : copier l'examen, mélanger **seulement cette colonne**, prédire de nouveau et afficher l'exactitude et les fraudes attrapées. Deux nouveaux outils :

- `X_exa.clone()` fait une copie, pour ne pas abîmer l'examen original.
- `torch.randperm(800)` donne les nombres de 0 à 799 dans le désordre. `X_exa[torch.randperm(800), col]` est la colonne `col` mélangée.

Rappelle-toi de prédire à l'intérieur de `with torch.no_grad():`, comme dans l'examen. Un exemple résolu, avec la colonne `monto` (la colonne 0) :

```python
X_original = X_exa.clone()
X_revuelto = X_exa.clone()
X_revuelto[:, 0] = X_exa[torch.randperm(800), 0]

with torch.no_grad():
    dice = (predecir(X_revuelto) > 0).float()

print("monto: exactitud", round((dice == y_exa).float().mean().item(), 3))
```

Complète maintenant toi-même, avec un `for`, les colonnes 1, 2 et 3 (heure, distance et étranger), et affiche aussi les fraudes attrapées :

```python
for col in range(1, 4):
    X_revuelto = X_exa.________()
    X_revuelto[:, col] = X_exa[torch.________(800), col]
    with torch.no_grad():
        dice = (________(X_revuelto) > 0).float()
    # imprime la exactitud y los fraudes atrapados de esta columna
    assert torch.equal(X_exa, X_original), "No debes alterar el examen original"
```

Ensuite, réponds : quelle colonne le modèle a-t-il eu le plus de mal à perdre ? Y en a-t-il une qui lui était indifférente ? Est-ce cohérent avec la règle cachée des voyants ?

---

## Erreurs courantes (et comment t'en rendre compte)

| Erreur | Ce que l'on observe | Comment t'en rendre compte | Ce qu'il faut faire |
|---|---|---|---|
| Oublier `requires_grad=True` sur un bouton | Dans `cerro.py`, où c'est le seul bouton : `RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn` à l'appel de `backward()`. Dans `detector.py`, si tu l'oublies seulement sur `W1`, **ça ne plante pas** : `W1.grad` reste à `None`, `W1` ne bouge jamais, la perte reste bloquée à 0.3366 et il n'attrape que 13 fraudes sur 125 | Dans `cerro.py`, ça plante à la ligne du `backward` ; dans `detector.py`, la perte baisse beaucoup moins que la normale et `print(W1.grad)` affiche `None` | Mets en marche le « magnétophone » en créant chaque bouton : `requires_grad=True` |
| Oublier d'effacer les pentes (`zero_grad`) | Dans `cerro.py`, l'`assert` te sauve : ça plante à l'étape 2 avec `AssertionError: PyTorch y el lápiz deberían coincidir`. Sans les `assert`, ça ne plante pas, et c'est ça qui est dangereux : le prix saute à 37.80, puis à 41.58, la perte monte jusqu'à 4 191.27 (à l'étape 4) et, à la fin, le prix appris reste à 4.95 au lieu de 20 | La perte monte et descend au lieu de baisser | PyTorch additionne les nouvelles pentes aux anciennes ; efface-les à chaque tour |
| Étiquettes d'une autre forme que les scores | `ValueError: Target size (torch.Size([3200, 1])) must be the same as input size (torch.Size([3200]))` | Affiche `y.shape` et le `shape` du score : ils doivent être égaux | Laisse `y` avec la forme `(4000,)`, sans `reshape(4000, 1)` |
| Taux trop grand | L'entraînement ne plante pas. Avec `lr=10`, la perte reste bloquée à 0.4365 et le modèle cesse d'attraper des fraudes ; à la fin, c'est la précision qui plante avec `ZeroDivisionError`, parce qu'il n'a donné aucune alerte (exercice 27) | La perte cesse de baisser ou fait des bonds | Baisse le taux ; essaie de 10 en 10 fois (1, 0.1, 0.01…) |
| Appeler `backward()` deux fois sur la même perte | `RuntimeError: Trying to backward through the graph a second time (or directly access saved tensors after they have already been freed).` | Ça plante au deuxième `backward` | Un `backward` par tour ; le tour suivant prédit et mesure de nouveau |
| Croire l'exactitude seule | « 91 % ! » | Tu calcules le détecteur paresseux et il obtient 84 % | Regarde aussi le rappel et la précision |

---

## Révision pour la prochaine session

La session 8 commence par ces trois questions. Réponds-y sans regarder la leçon :

1. Quelle forme a le tenseur d'entrée du détecteur et que représente chaque axe ?
2. Que mesure la perte et pourquoi doit-elle baisser ?
3. Pourquoi avoir raison 91 % du temps peut-il ne pas suffire pour une banque ?

🔁 **La semaine prochaine :** les tenseurs à la banque, dans l'assurance et au magasin. Des cas réels, avec leur source, et le défi d'écrire toi-même la forme du tenseur de chacun. Divulgâchage : le détecteur d'aujourd'hui a exactement la forme des vrais.

---

## Solutions (ne les regarde pas avant d'avoir essayé)

**Révision pour commencer.**

1. On compare les formes **de droite à gauche**, axe par axe. Sur chaque axe, les deux tailles doivent être **égales, ou l'une d'elles doit valoir 1** (c'est le 1 qui s'« étire »). Si ce n'est pas le cas sur un axe, PyTorch signale une erreur.
2. `*` multiplie **élément par élément** : case par case, et les formes doivent coïncider ou s'étirer. `@` est le **produit matriciel** : ligne par colonne, en multipliant et en additionnant. Ce qui est trompeur, c'est que parfois les deux s'exécutent sans erreur et donnent des choses différentes.
3. `(7,)`. Le 3 du milieu est « mangé » (multiplié et additionné) et il reste un nombre pour chacune des 7 lignes. Aujourd'hui, tu l'as utilisé : `(compras, 8) @ (8,)` a donné `(compras,)`.

**Exercice 26.** Avec 2000 époques (les premières lignes sont identiques à celles de 300 ; je colle à partir de la 300) :

```
época 300   pérdida 0.1910
época 1000   pérdida 0.1641
época 2000   pérdida 0.1142
exactitud: 0.93
fraudes en el examen: 125
atrapados: 83 | se escaparon: 42 | alarmas falsas: 14
recall: 0.664 | precisión: 0.856
```

a) Inférieure : 0.1142. Plus de tours, plus de temps pour descendre la colline. Les premières époques donnent les mêmes nombres qu'avec 300 grâce à la graine fixe. À partir de la 1000, sur ton ordinateur, le dernier chiffre de la perte et un comptage peuvent être un peu différents : ce sont les différences d'arrondi de l'encadré « Pourquoi 300 et pas plus ? », qui, avec autant de tours, ont le temps de grandir.

b) L'exactitude monte à 93 %, il attrape 83 fraudes au lieu de 81 et les fausses alertes baissent de 30 à 14. Attention : plus d'époques n'aide pas toujours. Avec des données réelles, trop de tours peuvent amener le modèle à **mémoriser** les exemples d'étude et à moins bien réussir l'examen. C'est pour ça que l'examen se fait avec des données mises de côté.

c) `puntaje = predecir(X_ent)` prédit ; `perdida = perdida_fn(puntaje, y_ent)` mesure ; `optimizador.zero_grad()` efface les pentes ; `perdida.backward()` calcule les pentes ; `optimizador.step()` tourne les boutons.

**Exercice 27.** Avec `lr=0.1` (j'enlève quelques époques du milieu) :

```
época   1   pérdida 0.6819
época  50   pérdida 0.4238
época 300   pérdida 0.3194
exactitud: 0.859
fraudes en el examen: 125
atrapados: 12 | se escaparon: 113 | alarmas falsas: 0
recall: 0.096 | precisión: 1.0
```

La perte baisse, mais beaucoup plus lentement : en 300 époques, elle arrive à 0.3194, moins bien que ce que le taux 1.0 obtenait en 50. Des pas de fourmi : il faut beaucoup plus de tours. Et il n'attrape que 12 fraudes sur 125. Regarde la précision : 1.0, parfaite, parce que les 12 alertes qu'il a données étaient des fraudes. Un détecteur qui ne sonne presque jamais ne se trompe presque jamais quand il sonne, et pourtant il laisse passer 113 fraudes : une raison de plus de ne pas le juger sur un seul nombre.

Avec `lr=10` (pareil, raccourci) :

```
época   1   pérdida 0.6819
época  10   pérdida 0.8124
época  50   pérdida 0.4365
época 300   pérdida 0.4365
exactitud: 0.844
fraudes en el examen: 125
atrapados: 0 | se escaparon: 125 | alarmas falsas: 0
Traceback (most recent call last):
  ...
    "| precisión:", round(atrapados / (atrapados + falsas), 3))
                          ~~~~~~~~~~^~~~~~~~~~~~~~~~~~~~~~
ZeroDivisionError: float division by zero
```

Les enjambées étaient si longues qu'à l'époque 10 la perte a **monté** (de 0.6819 à 0.8124) : elle a sauté de l'autre côté de la vallée. Ensuite, elle est restée bloquée à 0.4365. Et comment ça s'est terminé ? Exactitude **0.844**, zéro fraude attrapée, zéro alerte : **le modèle a fini par se comporter comme le détecteur paresseux.** Avec `lr=10`, dans cette exécution, les mises à jour ont désactivé les huit signaux intermédiaires (`relu` a mis à zéro les 25 600 valeurs des achats d'étude) et le modèle s'est retrouvé avec un score constant et négatif, le même pour tous les achats (−1.67 dans l'examen). Avec le seuil à zéro, il ne donne jamais d'alerte. Attention : il minimise toujours la même perte que d'habitude ; il n'a pas « choisi » l'exactitude. S'il obtient 84 %, c'est parce que la fraude est rare.

La `ZeroDivisionError` apparaît parce que la précision, c'est les fraudes attrapées divisées par les alertes, et qu'il y a eu 0 alerte : 0 / 0. On ne peut pas mesurer l'adresse de quelqu'un qui n'a jamais tiré.

**Exercice 28.** Les trous sont `zeros`, `y_exa`, `y_exa` et `0`. Ce bloc n'est pas à copier : il montre seulement les lignes dont les trous sont déjà remplis, et les `...` sont les lignes de ton fichier qui ne changent pas :

```text
siempre_no = torch.zeros(800)
exactitud_flojo = (siempre_no == y_exa).float().mean().item()
atrapados_flojo = (siempre_no * y_exa).sum().item()
...
    if y_exa[i].item() == 0:
```

Sortie (les trois dernières lignes, après la sortie normale du détecteur) :

```
exactitud del flojo: 0.844
fraudes que atrapa: 0 de 125
El for dice lo mismo: 675 aciertos de 800
```

Le rappel est de 0 / 125 = **0** : il n'attrape pas une seule fraude. La précision ne peut pas être calculée pour la même raison que dans l'exercice 27 : il ne donne aucune alerte, et 0 / 0 n'est pas défini. Un détecteur avec 84 % d'exactitude et un rappel nul : voilà toute la leçon en deux nombres.

**Exercice 29.** Une solution :

```python
# Ejercicio 29: ¿qué dato le importa más al modelo?
nombres = ["monto", "hora", "distancia", "extranjero"]
for col in range(4):
    X_revuelto = X_exa.clone()                                  # copia del examen
    X_revuelto[:, col] = X_exa[torch.randperm(800), col]        # revuelve UNA columna
    with torch.no_grad():
        dice = (predecir(X_revuelto) > 0).float()
    exac = (dice == y_exa).float().mean().item()
    atrap = (dice * y_exa).sum().item()
    print(f"revolviendo {nombres[col]:10s} exactitud {exac:.3f}   atrapados {int(atrap)} de 125")
```

Sortie (à la fin du détecteur, avec la même graine) :

```
revolviendo monto      exactitud 0.848   atrapados 54 de 125
revolviendo hora       exactitud 0.873   atrapados 63 de 125
revolviendo distancia  exactitud 0.870   atrapados 64 de 125
revolviendo extranjero exactitud 0.849   atrapados 57 de 125
```

Sans mélange, il avait juste à 90.8 % et attrapait 81 fraudes. Mélanger **n'importe laquelle** des quatre le fait chuter : l'exactitude tombe entre 84.8 % et 87.3 %, à peine au-dessus du paresseux ou au même niveau, et il cesse d'attraper entre 17 et 27 fraudes. Celles qui lui ont le plus coûté : le montant et l'étranger.

C'est cohérent avec la règle cachée : les quatre colonnes déclenchent un voyant, et le modèle a reçu des exemples étiquetés produits avec ces quatre colonnes. C'est un indice qu'il les utilise, pas une preuve qu'il a retrouvé la règle exacte. Si tes nombres sont différents parce que ton `randperm` a mélangé autrement, ce n'est pas grave : ce qui doit se produire, c'est que les quatre baissent.

---

## La semaine prochaine

Dans la session 8, la dernière, Mateo quitte la papeterie et entre à la banque pour voir comment ces mêmes tenseurs travaillent avec des millions d'achats, et avec des personnes qui examinent la décision.

Si tu veux prendre de l'avance, réfléchis : si le détecteur ne voyait pas un achat, mais les 30 derniers achats de chaque client, quelle forme aurait son tenseur d'entrée ?

**23 h 48**

L'arrière-boutique est silencieuse. Doña Lupe s'est endormie sur la chaise en plastique, son tablier sur elle et le sac de pièces sur les genoux. Le petit pain à la cannelle est toujours à côté de la caisse, désormais dur.

Mateo n'exécute pas le détecteur sur les débits de Nord. Il y pense, le doigt sur la touche, et ne le fait pas : l'exercice a appris avec des achats et une règle inventés, et il ne sert pas à décider au sujet de personnes réelles. À la place, il sort le tableau à quatre cases et le pose à côté des notes que Fer a écrites à la main. Il ordonne les questions : quels débits n'ont pas de marchandise derrière, lesquels tombent au petit matin ou loin, quelles preuves il peut demander à la banque.

Il devrait se sentir content, et il l'est à moitié. Il a une méthode. Mais une méthode n'est pas une preuve.

Il ouvre sa messagerie pour écrire à monsieur Ortega. Il tape « Je vous transmets un modèle qui » et s'arrête. Il imagine la tête de monsieur Ortega. Un garçon de vingt ans, un ordinateur portable avec un autocollant décollé, un programme qui a raison 91 % du temps et laisse échapper plus d'une fraude sur trois. Et de l'autre côté, une banque avec ses propres systèmes, ses propres analystes, ses propres règles.

Il efface la phrase.

Par la fenêtre passe la dernière patrouille de la nuit. L'ampoule bourdonne. Sur le calendrier, le cercle rouge du 23 paraît plus grand que ce matin.

Mateo referme l'ordinateur doucement, pour ne pas réveiller sa tante. Il sait déjà ce qu'il a. Il sait aussi ce qui lui manque : pas un nombre plus élevé, mais quelqu'un de l'autre côté prêt à examiner.

Il restait neuf jours avant le 23. Et la banque n'allait pas croire un modèle fait sur un ordinateur portable.

---

*J'écris à partir de ce que nous voyons tous les jours chez Hábil, où nous concevons des systèmes pour le secteur bancaire, les assurances et les entreprises au Mexique. Si, dans ton travail ou dans l'entreprise de ta famille, tu rencontres un problème de ce genre, discutons-en : [habil.mx](https://www.habil.mx/fr/)*
