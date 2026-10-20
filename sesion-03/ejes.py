import torch

# Ejes de la papelería:
#   eje 0 = sucursal  (0 Centro, 1 Norte)
#   eje 1 = día       (0 lunes ... 6 domingo)
#   eje 2 = producto  (0 cuadernos, 1 plumas, 2 mochilas)
centro = [
    [1200, 800, 500],    # lunes
    [1500, 900, 700],    # martes
    [1000, 700, 500],    # miércoles
    [1300, 900, 600],    # jueves
    [2000, 1200, 700],   # viernes
    [1200, 800, 400],    # sábado
    [800, 500, 300],     # domingo
]
norte = [
    [300, 200, 100],
    [400, 250, 150],
    [350, 200, 100],
    [300, 150, 100],
    [500, 300, 200],
    [300, 200, 100],
    [200, 100, 100],
]
papeleria = torch.tensor([centro, norte])
print(papeleria.shape)

print(papeleria[1][4][2])
print(papeleria[1, 4, 2])

x = papeleria[1, 4, 2]
print(x.shape, x.ndim)
print(type(x))
print(x.item())
print(type(x.item()))

assert papeleria[1, 4, 2].shape == (), "Un solo número no tiene ejes"
assert papeleria[1, 4, 2].item() == 200, "Norte, viernes, mochilas: 200"
print("Paso 2 correcto")

print(papeleria[1])
print(papeleria[1].shape)

print(papeleria[1, 4])
print(papeleria[1, 4].shape)

viernes = papeleria[:, 4]
print(viernes)
print(viernes.shape)

a_mano = []
for s in range(2):
    a_mano.append([papeleria[s][4][p].item() for p in range(3)])
print(a_mano)
print(viernes.tolist())

assert viernes.shape == (2, 3), "Viernes: 2 sucursales × 3 productos"
assert viernes.tolist() == [[2000, 1200, 700], [500, 300, 200]]
print("Paso 4 correcto")

tramo = papeleria[0, 1:5]
print(tramo)
print(tramo.shape)

print(centro[1:5])
assert tramo.shape == (4, 3), "De martes a viernes: 4 días × 3 productos"
assert tramo.tolist() == centro[1:5]
print("Paso 5 correcto")

mochilas = papeleria[:, :, 2]
print(mochilas)
print(mochilas.shape)

m = []
for s in range(2):
    fila = []
    for d in range(7):
        fila.append(papeleria[s, d, 2].item())
    m.append(fila)
print(m)

assert mochilas.shape == (2, 7), "Mochilas: 2 sucursales × 7 días"
assert mochilas.tolist() == m
print("Paso 6 correcto")

print(papeleria[0].shape)
print(papeleria[0:1].shape)

print(papeleria[0:1])
