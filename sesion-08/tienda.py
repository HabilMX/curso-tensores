import torch

centro = [[1200, 800, 500], [1500, 900, 700], [1000, 700, 500], [1300, 900, 600],
          [2000, 1200, 700], [1200, 800, 400], [800, 500, 300]]
norte = [[300, 200, 100], [400, 250, 150], [350, 200, 100], [300, 150, 100],
         [500, 300, 200], [300, 200, 100], [200, 100, 100]]
papeleria = torch.tensor([centro, norte])
print("papeleria:", papeleria.shape)

tabla = papeleria.reshape(-1, 3)
print("tabla:", tabla.shape)

# Comprobación: con for, el total no cambia
total_for = 0
for sucursal in [centro, norte]:
    for dia in sucursal:
        for venta in dia:
            total_for = total_for + venta

assert tabla.shape == (14, 3), "La tabla debería ser 14 renglones (sucursal y día) × 3 productos"
assert tabla.sum().item() == total_for == 23100, "Cambiar de forma no cambia el total"
print("Total:", tabla.sum().item(), "| con for:", total_for)
