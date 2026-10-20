# Ejercicio 4 — solución (una respuesta posible), tal cual aparece en la lección.
# (sesión 1; ver el README.md de la carpeta para saber de dónde sale este archivo)
# [repo] preludio añadido para que el archivo corra solo
from dimensiones import forma
# [/repo]

# alumno × materia × parcial
parciales = [
    [[8, 9, 9], [7, 8, 9], [9, 9, 10], [6, 7, 8]],
    [[10, 9, 10], [9, 9, 9], [8, 8, 9], [9, 10, 10]],
    [[7, 7, 8], [8, 8, 8], [6, 7, 7], [8, 9, 9]],
]
assert forma(parciales) == (3, 4, 3)
assert parciales[0][2][1] == 9   # primer alumno, tercera materia, segundo parcial
