# Ejercicio 1 — solución. Huecos llenados según la solución de la lección.
# (sesión 1; ver el README.md de la carpeta para saber de dónde sale este archivo)
# [repo] preludio añadido para que el archivo corra solo
from dimensiones import contar_dimensiones, forma, es_rectangular
# [/repo]

temperatura = 31
calificaciones = [[9, 8, 10], [7, 9, 8], [10, 10, 9], [6, 8, 7]]
foto = [[[255, 0, 0], [0, 255, 0]], [[0, 0, 255], [255, 255, 255]]]
lote = [foto, foto]

assert (contar_dimensiones(temperatura), forma(temperatura)) == (0, ())
assert (contar_dimensiones(calificaciones), forma(calificaciones)) == (2, (4, 3))
assert (contar_dimensiones(foto), forma(foto)) == (3, (2, 2, 3))
assert (contar_dimensiones(lote), forma(lote)) == (4, (2, 2, 2, 3))
print("Ejercicio 1 superado")
