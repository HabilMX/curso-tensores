# Ejercicio 10 — solución, tal cual aparece en la lección.
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

assert papeleria[0, 2, 1].item() == 700    # a) Centro, miércoles, plumas
assert papeleria[1, 6, 0].item() == 200    # b) Norte, domingo, cuadernos
assert papeleria[0, 5, 2].item() == 400    # c) Centro, sábado, mochilas
assert papeleria[0][2][1].item() == papeleria[0, 2, 1].item()   # d)
print("Ejercicio 10 superado")
