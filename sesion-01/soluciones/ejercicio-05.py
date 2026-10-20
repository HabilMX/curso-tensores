# Ejercicio 5 — solución. Huecos llenados según la solución de la lección.
# (sesión 1; ver el README.md de la carpeta para saber de dónde sale este archivo)
# [repo] preludio añadido para que el archivo corra solo
from dimensiones import es_rectangular
# [/repo]

a = [[1, 2], [3, 4], [5, 6]]
b = [[1, 2, 3], [4, 5, 6], [7, 8]]
c = [[0], [0], [0]]

assert es_rectangular(a) == True
assert es_rectangular(b) == False
assert es_rectangular(c) == True
print("Ejercicio 5 superado")
