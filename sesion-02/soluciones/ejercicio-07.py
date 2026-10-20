# Ejercicio 7 — solución. Huecos llenados según la solución de la lección.
# (sesión 2; ver el README.md de la carpeta para saber de dónde sale este archivo)
import torch
from dimensiones import norte

norte_t = torch.tensor(norte)

assert norte_t.shape == (7, 3), "Norte es una pestaña: 7 días × 3 productos"
assert norte_t.ndim == 2, "Una pestaña es una tabla"
assert norte_t.numel() == 21, "7 × 3"
assert norte_t.dtype == torch.int64, "Son pesos enteros"
assert norte_t.tolist() == norte, "Los datos no deberían cambiar"
print("Ejercicio 7 superado")
