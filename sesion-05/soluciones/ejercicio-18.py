# Ejercicio 18 — solución escrita para el repositorio.
# La lección da la tabla de resultados en prosa. Este script prueba las siete formas
# sobre los 12 meses y imprime lo que dice la tabla de la solución.
# (sesión 5; ver el README.md de la carpeta)
import torch

meses = [60000, 55000, 60000, 50000, 65000, 50000, 70000, 100000, 70000, 60000, 55000, 90000]
ventas = torch.tensor(meses)

for forma in [(3, 4), (6, 2), (5, 2), (2, -1), (-1, 5), (2, 3, 2), (12, 1)]:
    try:
        print(forma, "->", ventas.reshape(*forma).shape)
    except RuntimeError as error:
        print(forma, "-> RuntimeError:", error)
