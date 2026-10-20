# Ejercicio 33 — solución, tal cual aparece en la lección.
# (sesión 8; ver el README.md de la carpeta para saber de dónde sale este archivo)
# [repo] preludio
import torch
# [/repo]

series = torch.zeros(100, 28, 3)    # series × días × datos
tabla = series.reshape(-1, 3)       # (serie y día) × datos
assert series.shape == (100, 28, 3), "series × días × datos"
assert tabla.shape == (2800, 3), "(series y día) × datos"
assert tabla.numel() == series.numel() == 8400, "Cambiar de forma no cambia cuántos números hay"
print("Ejercicio 33 superado")
# Un renglón es un día de una serie; la forma ya no dice de cuál serie ni de cuál día.
