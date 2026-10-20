# Ejercicio 21, parte de la trampa — la solución con SOLO la línea de la tabla girada cambiada por reshape(3, 2), como pide la lección.
# (sesión 5; ver el README.md de la carpeta para saber de dónde sale este archivo)
from papeleria import papeleria

semana = papeleria.sum(dim=1)        # sucursal × producto
girada = semana.reshape(3, 2)        # producto × sucursal
assert girada.shape == (3, 2), "Deberían ser 3 productos × 2 sucursales"
assert girada[0].tolist() == [9000, 2350], "El renglón de cuadernos debería traer Centro y Norte"

for p in range(3):
    for s in range(2):
        assert girada[p, s] == semana[s, p], "Girar es cambiar renglones por columnas"
print("Ejercicio 21 superado")
