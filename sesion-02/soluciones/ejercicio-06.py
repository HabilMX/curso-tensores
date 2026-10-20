# Ejercicio 6 — código del enunciado, tal cual; la salida es la de la solución.
# (sesión 2; ver el README.md de la carpeta para saber de dónde sale este archivo)
import torch

temperatura = torch.tensor(31)
calificaciones = torch.tensor([[9, 8, 10], [7, 9, 8], [10, 10, 9], [6, 8, 7]])
foto = torch.tensor([[[255, 0, 0], [0, 255, 0]], [[0, 0, 255], [255, 255, 255]]])
promedios = torch.tensor([9.0, 8.0, 9.7, 7.0])

for nombre, t in [("temperatura", temperatura), ("calificaciones", calificaciones),
                  ("foto", foto), ("promedios", promedios)]:
    print(nombre, t.shape, t.ndim, t.dtype, t.numel())
