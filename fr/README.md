# Les tenseurs à partir de zéro

**Par Dorian Chávez, fondateur de Hábil et architecte d'intégration.**

De la liste Python que l'on sait déjà écrire à un détecteur de fraude qui apprend à partir des données.

**À qui il s'adresse :** aux étudiants en informatique qui programment déjà un peu en Python, et aux personnes de la gestion qui veulent
comprendre ce qu'il y a sous l'intelligence artificielle utilisée dans les banques, les compagnies d'assurance et les magasins. C'est un cours de
niveau intermédiaire : on n'y explique pas ce qu'est une variable ni une boucle, mais chaque mot nouveau est expliqué au moment où il apparaît. Chaque
session comprend un encadré « Si tu viens de la gestion » avec la question qui vaut la peine d'être posée en réunion.

**Ce que tu sauras faire à la fin :**

- Créer un tenseur PyTorch à partir de données réelles et donner sa forme avant d'exécuter le code.
- Extraire et résumer des données selon n'importe lequel de leurs axes, et vérifier chaque résultat par rapport au même calcul fait avec un `for`.
- Lire une erreur de forme de PyTorch, mot à mot, et la corriger.
- Entraîner un petit détecteur de fraude et expliquer pourquoi 91 % de bonnes prédictions ne suffisent pas.

**Ce dont tu as besoin avant de commencer :** des bases en Python (variables, listes, boucles `for` et fonctions ; les leçons 0 à 4
du [cours de Python](https://github.com/HabilMX/curso-python) suffisent), Python 3.11 ou plus récent et un terminal. La
première session n'installe rien ; PyTorch s'installe lors de la deuxième. Testé sous Linux (Mint 22.3 et Ubuntu 24.04) et sous
macOS ; non testé sous Windows. Pas besoin de carte graphique.

## Les huit sessions

Deux leçons de quatre sessions, d'environ 2 heures chacune : à peu près une pour lire et suivre l'exemple, et une autre pour
pratiquer. Toutes suivent la même histoire : une papeterie, un terminal bancaire qui facture en trop et un étudiant qui
a jusqu'au 23 décembre pour comprendre ce qui s'est passé.

| Session | | Ce que tu apprends |
|---|---|---|
| 1 | [Un nombre, une liste, un tableau, un cube](01-un-numero-una-lista-una-tabla-un-cubo.md) | ce qu'est un tenseur, avec les listes de Python ; compter les dimensions ; d'où vient ce mot |
| 2 | [Ton premier tenseur](02-tu-primer-tensor.md) | installer PyTorch ; `torch.tensor`, `shape` et `dtype` ; PyTorch et TensorFlow |
| 3 | [Les axes ont un nom](03-los-ejes-tienen-nombre.md) | extraire un morceau selon n'importe quel axe ; ce qui arrive à la forme |
| 4 | [Additionner par onglet](04-sumar-por-pestana.md) | additionner et calculer la moyenne sur un axe avec `dim` |
| 5 | [Changer de forme sans changer les données](05-cambiar-de-forma.md) | `reshape` et comment lire une erreur de forme |
| 6 | [Opérer sur des tenseurs complets](06-operar-tensores-completos.md) | le *broadcasting*, et le piège de `*` face à `@` |
| 7 | [Un détecteur de fraude qui apprend à partir des données](07-detector-de-fraude.md) | la boucle d'entraînement ; exactitude (*accuracy*), rappel (*recall*) et précision |
| 8 | [Les tenseurs à la banque, dans l'assurance et au magasin](08-banco-aseguradora-tienda.md) | lire un cas réel et donner la forme de son tenseur ; les précautions qui ne sont pas facultatives |

## Le code

Chaque session a son dossier dans ce dépôt ([`sesion-01`](../sesion-01) à [`sesion-08`](../sesion-08)), avec chaque
programme et la sortie qu'il doit te donner. `verificar.py` les exécute tous et les compare à ce qu'indique la leçon. L'installation
pas à pas se trouve dans le [README principal](../README.md).

## Où le cours est publié

Une session par semaine dans « En construcción », le Substack de Hábil, le mercredi à 8 h 00, du 4 novembre au 23
décembre 2026 : https://enconstruccionmx.substack.com
