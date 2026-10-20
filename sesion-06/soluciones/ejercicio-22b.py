# Ejercicio 22, inciso b — código del enunciado; la salida es la de la solución.
# (sesión 6; ver el README.md de la carpeta para saber de dónde sale este archivo)
# [repo] preludio
import torch
# [/repo]

m = torch.tensor([[2, 1],
                  [0, 3]])
print(m * m)
print(m @ m)
