import torch
from dimensiones import centro, norte, forma, contar_dimensiones

papeleria_lista = [centro, norte]
papeleria = torch.tensor([centro, norte])

# 1) Primero la forma: PyTorch contra tus funciones de la sesión 1
print(papeleria.shape, forma(papeleria_lista))
print(papeleria.ndim, contar_dimensiones(papeleria_lista))
assert papeleria.shape == (2, 7, 3), "La papelería debería ser 2 sucursales × 7 días × 3 productos"
assert papeleria.shape == forma(papeleria_lista), "PyTorch y tu función deberían dar la misma forma"
assert papeleria.ndim == contar_dimensiones(papeleria_lista), "Deberían contar las mismas dimensiones"

# 2) Después el total de números, contra un for
cuenta = 0
for pestana in papeleria_lista:
    for renglon in pestana:
        for numero in renglon:
            cuenta = cuenta + 1
print(papeleria.numel(), cuenta)
assert papeleria.numel() == cuenta, "Deberían ser los mismos números"

# 3) Al final, el valor: de regreso a listas, ¿quedó igual?
assert papeleria.tolist() == papeleria_lista, "Los datos no deberían cambiar al convertirlos"
print("Todo cuadra")
