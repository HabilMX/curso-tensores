# Ejercicio 19, parte c — solución. Huecos llenados según la solución de la lección.
# (sesión 5; ver el README.md de la carpeta para saber de dónde sale este archivo)
# [repo] preludio añadido para que el archivo corra solo
from papeleria import papeleria
# [/repo]

plana = papeleria.flatten()
de_vuelta = plana.reshape(2, 7, 3)

assert plana.shape == (42,), "Deberían quedar 42 números en una fila"
assert de_vuelta.tolist() == papeleria.tolist(), "La ida y vuelta debe conservar cada dato"
print("Ejercicio 19c superado")
