# Ejercicio 31 — solución. Huecos llenados según la solución de la lección.
# (sesión 8; ver el README.md de la carpeta para saber de dónde sale este archivo)
# [repo] preludio
import torch
# [/repo]

siniestro = torch.zeros(6, 3, 224, 224)
assert siniestro.shape == (6, 3, 224, 224), "6 fotos × 3 colores × 224 de alto × 224 de ancho"
cuenta = 1
for tamano in (6, 3, 224, 224):
    cuenta = cuenta * tamano
assert siniestro.numel() == cuenta, "numel debe dar lo mismo que multiplicar la forma con un for"
print("Ejercicio 31 superado:", siniestro.numel(), "números")
