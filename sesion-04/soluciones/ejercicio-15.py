# Ejercicio 15 — solución. Huecos llenados según la solución de la lección.
# (sesión 4; ver el README.md de la carpeta para saber de dónde sale este archivo)
from papeleria import papeleria, norte

promedio = papeleria.float().mean(dim=1)
print(promedio)

assert promedio.shape == (2, 3), f"Esperaba (sucursal, producto) = (2, 3) y salió {tuple(promedio.shape)}"

suma_plumas_norte = 0
for renglon in norte:
    suma_plumas_norte = suma_plumas_norte + renglon[1]

assert round(promedio[1, 1].item(), 2) == round(suma_plumas_norte / 7, 2), "No coincide con el for"
print("Ejercicio 15 superado")
