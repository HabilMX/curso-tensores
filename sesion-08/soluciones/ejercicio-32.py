# Ejercicio 32 — solución. Huecos llenados según la solución de la lección.
# (sesión 8; ver el README.md de la carpeta para saber de dónde sale este archivo)
# [repo] preludio
import torch
# [/repo]

torch.manual_seed(1)
catalogo = {"cuadernos": 0, "plumas": 1, "mochilas": 2, "colores": 3}
diccionario = torch.nn.Embedding(4, 2)
historiales = torch.tensor([[0, 1, 1, 2, 3],
                            [3, 3, 0, 1, 2],
                            [2, 2, 2, 0, 1]])
salida = diccionario(historiales)
assert diccionario.weight.shape == (4, 2), "4 productos × 2 números"
assert historiales.shape == (3, 5), "3 clientas × 5 compras"
assert salida.shape == (3, 5, 2), "3 clientas × 5 compras × 2 números"
assert torch.equal(salida[0][3], diccionario.weight[catalogo["mochilas"]]), "La cuarta compra de la primera clienta son mochilas"
print("Ejercicio 32 superado")
