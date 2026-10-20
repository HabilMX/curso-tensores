# Ejercicio 19 (a y b) — solución. Huecos llenados según la solución; al final, la comprobación del renglón 9 que pide la lección (assert + print).
# (sesión 5; ver el README.md de la carpeta para saber de dónde sale este archivo)
from papeleria import papeleria

larga = papeleria.reshape(-1, 3)
assert larga.shape == (14, 3), "Deberían ser 14 renglones (sucursal y día) × 3 productos"

a_mano = []
for s in range(2):
    for d in range(7):
        a_mano.append(papeleria[s, d].tolist())
assert larga.tolist() == a_mano, "La tabla larga no coincide con la armada a mano"
print("Ejercicio 19 superado")

assert larga[9].tolist() == papeleria[1, 2].tolist(), "Ese renglón debería ser Norte miércoles"
print(larga[9])
