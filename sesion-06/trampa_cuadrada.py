import torch
# Renglones: Centro y Norte. Columnas: cuadernos y plumas.
piezas = torch.tensor([[30, 80],
                       [10, 20]])
precio = torch.tensor([[40, 10],    # en el Centro
                       [45, 12]])   # en Norte, un poco más caro
print(piezas * precio)
print(piezas @ precio)

lista_piezas = [[30, 80], [10, 20]]
lista_precio = [[40, 10], [45, 12]]
ingreso_for = []
for s in range(2):
    renglon = []
    for p in range(2):
        renglon.append(lista_piezas[s][p] * lista_precio[s][p])
    ingreso_for.append(renglon)
print(ingreso_for)
