# [repo] se añadió import torch; el resto es el bloque de la lección (paso 2, «Ojo: las listas de Python no hacen esto»)
import torch

print([1, 2, 3] + [10, 20, 30])
print(torch.tensor([1, 2, 3]) + torch.tensor([10, 20, 30]))
print([1, 2, 3] * 2)
print(torch.tensor([1, 2, 3]) * 2)
