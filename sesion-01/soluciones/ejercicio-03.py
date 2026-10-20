# Ejercicio 3 — solución. Huecos llenados según la solución de la lección.
# (sesión 1; ver el README.md de la carpeta para saber de dónde sale este archivo)
# [repo] preludio añadido para que el archivo corra solo
from dimensiones import por_dia, papeleria
# [/repo]

def forma2(x):
    tamanos = []
    while type(x) == list:
        tamanos.append(len(x))
        x = x[0]
    return tuple(tamanos)

assert forma2(por_dia) == (7,)
assert forma2(papeleria) == (2, 7, 3)
assert forma2(31) == ()
print("Ejercicio 3 superado")
