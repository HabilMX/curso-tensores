# Ejercicio 9 — solución escrita para el repositorio.
# La lección da el resultado en prosa (no trae un archivo): este script prueba
# las cuatro listas del enunciado y el arreglo de b, y imprime lo que la solución afirma.
# (sesión 2; ver el README.md de la carpeta)
import torch

a = [[1, 2], [3, 4], [5, 6]]
b = [[1, 2, 3], [4, 5, 6], [7, 8]]
c = [[[1, 2], [3, 4]], [[5, 6], [7]]]
d = [[1, 2], [3, 4, 5]]

for nombre, lista in [("a", a), ("b", b), ("c", c), ("d", d)]:
    try:
        print(nombre, "funciona:", torch.tensor(lista).shape)
    except ValueError as error:
        print(nombre, "truena: ValueError:", error)

b = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
print(torch.tensor(b).shape)
