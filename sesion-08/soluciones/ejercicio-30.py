# Ejercicio 30 — inciso a del enunciado + la solución de b y c, tal cual aparece en la lección.
# (sesión 8; ver el README.md de la carpeta para saber de dónde sale este archivo)
# [repo] preludio
import torch
# [/repo]

bbva = torch.zeros(500, 200)
assert bbva.shape == (500, 200), "a) compras × datos"

historiales = torch.zeros(16, 100, dtype=torch.long)
embedding = torch.nn.Embedding(1000, 64)
salida = embedding(historiales)

assert historiales.shape == (16, 100), "clientes × compras"
assert salida.shape == (16, 100, 64), "clientes × compras × números"

fotos_siniestro = torch.zeros(5, 3, 720, 1280)
assert fotos_siniestro.shape == (5, 3, 720, 1280), "fotos × colores × alto × ancho"

print("Ejercicio 30 superado")
