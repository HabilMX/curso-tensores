# Ejercicio 16 — solución, tal cual aparece en la lección.
# (sesión 4; ver el README.md de la carpeta para saber de dónde sale este archivo)
from papeleria import papeleria, norte

dias = ["lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo"]
productos = ["cuadernos", "plumas", "mochilas"]

# a) Día más flojo: junto productos (2, 7) y busco el mínimo a lo largo de los días
valores, indices = papeleria.sum(dim=2).min(dim=1)
assert valores.shape == (2,), "Esperaba un valor por sucursal"
assert valores.tolist() == [1600, 400], "Los mínimos deberían ser 1,600 y 400"
assert indices.tolist() == [6, 6], "El domingo debería ser el día más flojo"
print(valores, indices)

peor_dia = 0
for d in range(7):
    if sum(norte[d]) < sum(norte[peor_dia]):
        peor_dia = d
assert indices[1].item() == peor_dia, "No coincide con el for"

# b) Producto estrella: junto días (2, 3) y busco la posición del máximo en productos
estrella = papeleria.sum(dim=1).argmax(dim=1)
assert estrella.shape == (2,), "Esperaba un producto por sucursal"
assert estrella.tolist() == [0, 0], "Los cuadernos deberían ser el producto estrella"
print(productos[estrella[0].item()], productos[estrella[1].item()])

# c) Mejor día con las dos sucursales juntas: junto sucursales y productos (7,)
por_dia_juntas = papeleria.sum(dim=(0, 2))
assert por_dia_juntas.shape == (7,), "Esperaba un total por día"
print(dias[por_dia_juntas.argmax().item()], por_dia_juntas.max().item())
assert por_dia_juntas.max().item() == 4900, "El mejor día debería sumar 4,900"
assert por_dia_juntas.argmax().item() == 4, "El mejor día debería ser viernes"
print("Ejercicio 16 superado")
