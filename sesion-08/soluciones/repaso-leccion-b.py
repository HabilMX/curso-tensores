# Repaso de la lección B (preguntas 1 a 6) — escrito para el repositorio.
# La lección da las «salidas reales» de las preguntas de código pero no el programa.
# (sesión 8; ver el README.md de la carpeta)
import torch

meses = torch.arange(12)
print("1)", meses.reshape(4, -1).shape, meses.reshape(-1, 6).shape)

try:
    meses.reshape(5, -1)
except RuntimeError as error:
    print("2)", error)

papeleria = torch.zeros(2, 7, 3)
precios = torch.tensor([45.0, 12.0, 350.0])
print("3)", (papeleria * precios).shape)

try:
    torch.zeros(2, 7, 3) * torch.zeros(7)
except RuntimeError as error:
    print("4)", error)

a = torch.tensor([[1, 2], [3, 4]])
b = torch.tensor([[10, 20], [30, 40]])
print("5) a * b =", (a * b).tolist(), "| a @ b =", (a @ b).tolist())

tabla = torch.zeros(14, 3)
precios = torch.tensor([45.0, 12.0, 350.0])
print("6)", (tabla @ precios).shape)
