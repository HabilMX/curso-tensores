# Ejercicio 22, inciso a — código del enunciado; la salida es la de la solución.
# (sesión 6; ver el README.md de la carpeta para saber de dónde sale este archivo)
# [repo] preludio
import torch
# [/repo]

print((torch.ones(7, 3) * torch.ones(3)).shape)
print((torch.ones(2, 7, 3) * torch.ones(7, 1)).shape)
print((torch.ones(7, 1) * torch.ones(1, 3)).shape)
print((torch.ones(7, 3) * torch.tensor(2)).shape)
print((torch.ones(2, 7, 3) * torch.ones(2)).shape)
