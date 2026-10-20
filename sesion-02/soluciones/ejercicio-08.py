# Ejercicio 8 — solución, tal cual aparece en la lección.
# (sesión 2; ver el README.md de la carpeta para saber de dónde sale este archivo)
import torch
from dimensiones import forma, contar_dimensiones

# alumno × materia × parcial
parciales = [
    [[8, 9, 9], [7, 8, 9], [9, 9, 10], [6, 7, 8]],
    [[10, 9, 10], [9, 9, 9], [8, 8, 9], [9, 10, 10]],
    [[7, 7, 8], [8, 8, 8], [6, 7, 7], [8, 9, 9]],
]

p = torch.tensor(parciales)
print(p.shape, p.ndim, p.numel(), p.dtype)
assert p.shape == forma(parciales), "PyTorch y tu función deberían coincidir"
assert p.ndim == contar_dimensiones(parciales)

cuenta = 0
for alumno in parciales:
    for materia in alumno:
        for calificacion in materia:
            cuenta = cuenta + 1
assert p.numel() == cuenta

p_dec = torch.tensor(parciales, dtype=torch.float32)
print(p_dec.dtype)
assert p_dec.dtype == torch.float32
assert p_dec.shape == p.shape, "Cambiar el dtype no cambia la forma"
print("Ejercicio 8 superado")
