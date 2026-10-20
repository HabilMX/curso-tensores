# Ejercicio 20, inciso a — el código del enunciado, que falla a propósito.
# (sesión 5; ver el README.md de la carpeta para saber de dónde sale este archivo)
import torch

dias_febrero = [
    2500, 3100, 2200, 2800, 3900, 2400, 1600,
    2600, 3000, 2300, 2700, 4000, 2500, 1500,
    2400, 3200, 2100, 2900, 3800, 2300, 1700,
    2700, 3300, 2400, 3000, 4200, 2600, 1800,
]
febrero = torch.tensor(dias_febrero)
por_semana = febrero.reshape(7, 5)
