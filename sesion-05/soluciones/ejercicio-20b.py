# Ejercicio 20, inciso b — solución. Datos del enunciado + el bloque de la solución, tal cual.
# (sesión 5; ver el README.md de la carpeta para saber de dónde sale este archivo)
import torch

dias_febrero = [
    2500, 3100, 2200, 2800, 3900, 2400, 1600,
    2600, 3000, 2300, 2700, 4000, 2500, 1500,
    2400, 3200, 2100, 2900, 3800, 2300, 1700,
    2700, 3300, 2400, 3000, 4200, 2600, 1800,
]
febrero = torch.tensor(dias_febrero)

por_semana = febrero.reshape(4, 7)

a_mano = []
for s in range(4):
    total = 0
    for d in range(7):
        total = total + dias_febrero[s * 7 + d]
    a_mano.append(total)

assert por_semana.shape == (4, 7), "Deberían ser 4 semanas × 7 días"
assert por_semana.sum(dim=1).tolist() == a_mano, "Los totales por semana no cuadran con el for"
print(por_semana.sum(dim=1))
print("Ejercicio 20 superado")
