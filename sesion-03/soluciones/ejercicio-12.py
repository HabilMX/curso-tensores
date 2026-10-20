# Ejercicio 12 — solución. Huecos llenados según la solución de la lección.
# (sesión 3; ver el README.md de la carpeta para saber de dónde sale este archivo)
# [repo] preludio añadido: los datos del paso 1 de la lección (sin su print)
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
# [/repo]

# a) El fin de semana (sábado y domingo) de Norte, los 3 productos
fin_norte = papeleria[1, 5:7]
assert fin_norte.shape == (2, 3), "Fin de semana de Norte: 2 días × 3 productos"
assert fin_norte.tolist() == [[300, 200, 100], [200, 100, 100]]

# b) Las plumas del Centro de lunes a miércoles
plumas = papeleria[0, 0:3, 1]
assert plumas.shape == (3,), "Plumas del Centro: 3 días"
assert plumas.tolist() == [800, 900, 700]

# c) Lo mismo que b), con un for
a_mano = []
for d in range(3):
    a_mano.append(papeleria[0, d, 1].item())
assert plumas.tolist() == a_mano
print("Ejercicio 12 superado")
