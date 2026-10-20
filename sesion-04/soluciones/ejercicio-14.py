# Ejercicio 14 — solución. Huecos llenados según la solución; las tres líneas print(...) se añadieron para reproducir la salida que muestra la lección.
# (sesión 4; ver el README.md de la carpeta para saber de dónde sale este archivo)
from papeleria import papeleria

assert papeleria.sum(dim=0).shape == (7, 3)          # ejemplo resuelto
assert papeleria.sum(dim=1).shape == (2, 3)
assert papeleria.sum(dim=2).shape == (2, 7)
assert papeleria.sum(dim=(0, 1)).shape == (3,)
assert papeleria.sum(dim=(0, 2)).shape == (7,)
assert papeleria.sum(dim=(0, 1, 2)).shape == ()
print(papeleria.sum(dim=(0, 1)))
print(papeleria.sum(dim=(0, 2)))
print(papeleria.sum(dim=(0, 1, 2)))
print("Ejercicio 14 superado")
