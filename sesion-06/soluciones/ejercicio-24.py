# Ejercicio 24 — solución, tal cual aparece en la lección.
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

descuento = torch.tensor([-5, 0, -50])
precios_bf = precios + descuento
por_dia_bf = unidades @ precios_bf
print(precios_bf)
print(por_dia_bf)

assert por_dia_bf.shape == (7,), "Debería salir un número por día"
assert por_dia_bf.tolist() == (unidades * precios_bf).sum(dim=1).tolist(), "El @ y el sum(dim=1) deberían dar lo mismo"

dejo_de_ganar = unidades @ precios - por_dia_bf
print(dejo_de_ganar)
assert dejo_de_ganar.tolist() == (-(unidades @ descuento)).tolist()
print(dejo_de_ganar.sum())
print("Ejercicio 24 superado")
