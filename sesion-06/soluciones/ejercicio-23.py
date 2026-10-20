# Ejercicio 23 — solución. Huecos llenados según la solución de la lección.
# (sesión 6; ver el README.md de la carpeta para saber de dónde sale este archivo)
# [repo] preludio añadido: los datos de los pasos 1 y 2 de la lección (sin sus print)
import torch

lista_unidades = [
    [30, 80, 2],    # lunes
    [40, 90, 3],    # martes
    [25, 70, 2],    # miércoles
    [35, 90, 2],    # jueves
    [50, 120, 3],   # viernes
    [30, 80, 1],    # sábado
    [20, 50, 1],    # domingo
]
lista_precios = [40, 10, 350]   # cuaderno, pluma, mochila

unidades = torch.tensor(lista_unidades)
precios = torch.tensor(lista_precios)

lista_norte = [
    [10, 20, 1],
    [10, 25, 0],
    [5, 20, 1],
    [10, 15, 0],
    [15, 30, 1],
    [10, 20, 0],
    [5, 10, 1],
]
unidades_norte = torch.tensor(lista_norte)
# [/repo]

ingresos_norte = unidades_norte * precios

ingresos_norte_for = []
for d in range(7):
    renglon = []
    for p in range(3):
        renglon.append(lista_norte[d][p] * lista_precios[p])
    ingresos_norte_for.append(renglon)

assert ingresos_norte.shape == (7, 3), "Norte también es 7 días × 3 productos"
assert ingresos_norte.tolist() == ingresos_norte_for, "El tensor y el for deberían dar lo mismo"
con_iva = ingresos_norte * 1.16
print(ingresos_norte.sum(), con_iva.sum())
print("Ejercicio 23 superado")
