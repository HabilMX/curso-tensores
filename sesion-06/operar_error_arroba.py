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

print(unidades.shape, unidades.dtype)
print(precios.shape, precios.dtype)

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

total = unidades + unidades_norte
print(total)

total_for = []
for d in range(7):
    renglon = []
    for p in range(3):
        renglon.append(lista_unidades[d][p] + lista_norte[d][p])
    total_for.append(renglon)

assert total.shape == (7, 3), "Deberían salir 7 días × 3 productos"
assert total.tolist() == total_for, "El tensor y el for deberían dar lo mismo"
print("Todo cuadra")

ingresos = unidades * precios
print(ingresos)
print(ingresos.shape)

ingresos_for = []
for d in range(7):
    renglon = []
    for p in range(3):
        renglon.append(lista_unidades[d][p] * lista_precios[p])
    ingresos_for.append(renglon)

assert ingresos.shape == (7, 3), "Los ingresos deberían ser 7 días × 3 productos"
assert ingresos.tolist() == ingresos_for, "El tensor y el for deberían dar lo mismo"
print("Ingresos correctos")
print(ingresos.sum())

con_iva = ingresos * 1.16
print(con_iva)
print(con_iva.shape, con_iva.dtype)
print(ingresos.sum(), con_iva.sum())

assert con_iva.shape == ingresos.shape, "El IVA no cambia la forma"
assert con_iva[0][0] == 1392, "Lunes, cuadernos: 1,200 × 1.16 = 1,392"
print("IVA correcto")

por_dia = unidades @ precios
print(por_dia)
print(por_dia.shape)

# 1) Con for, como lo harías sin tensores
por_dia_for = []
for d in range(7):
    suma = 0
    for p in range(3):
        suma = suma + lista_unidades[d][p] * lista_precios[p]
    por_dia_for.append(suma)

# 2) Multiplicar y luego sumar por renglón
por_dia_dos_pasos = (unidades * precios).sum(dim=1)

assert por_dia.shape == (7,), "Debería salir un número por día"
assert por_dia.tolist() == por_dia_for, "El @ y el for deberían dar lo mismo"
assert por_dia.tolist() == por_dia_dos_pasos.tolist(), "El @ y el sum(dim=1) deberían dar lo mismo"
print("Los tres caminos dan lo mismo")

print(precios @ unidades)
